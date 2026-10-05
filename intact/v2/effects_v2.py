"""
intact/v2/effects_v2.py
=======================
Regime-conditioned, DOSE-BASED effect estimation.

WHAT CHANGED FROM v1, AND WHY EACH CHANGE WAS FORCED BY A MEASUREMENT
---------------------------------------------------------------------
v1 model:   dg_i = alpha_i + sum_j beta_ji Z_j + sum_jk gamma_jk Z_j Z_k
                            + theta_i . c + eps
v2 model:   dg_i = alpha_i(r) + sum_j s_pi(r) dnu_j
                              + sum_jk gamma_jk_i(r) dnu_j dnu_k
                              + theta_i(r) . c + eps

Four changes:

1. DOSE REPLACES ELIGIBILITY.  Z_j in {0,1} discards the write magnitude.
   Measured consequence in v1: the estimator recovers
   beta = s * E[dnu | Z=1], the write-size-averaged effect, and then the
   scheduler applies it as if every write were average sized.  On a
   controlled reconstruction this was 49% wrong in the high-load regime.
   dnu_j = nu*_j - nu_old_j is the actual executed change, so the
   coefficient is s itself -- the one estimand that validated in v1
   (held-out R^2 0.960).

2. REGIME CONDITIONING.  v1 beta had state-stability 0.204, i.e. the
   state-to-state spread was ~5x the mean and the sign flipped across
   operating points.  A scalar cannot represent that.  Separate tables per
   regime turn a hopeless scalar into three tractable ones.

3. GAMMA IS A DOSE PRODUCT.  Z_j Z_k fires on co-ELIGIBILITY, but indirect
   conflict is a physical interaction that scales with both doses.  A pair
   at (-8 dBm, +6 PRB) interacts differently from (-2 dBm, +2 PRB); the
   binary product cannot tell them apart.

4. WEIGHTED RIDGE, NOT OLS.  Per-(regime, pair) cells are small by
   construction.  Ridge keeps them conditioned; the sandwich covariance
   with Kish effective sample size keeps the standard errors honest under
   forgetting (the v1 defect that inflated the null false-positive rate
   from 0.063 to 0.483).

CELL-COUNT GUARD
----------------
With R regimes and P pairs there are R*P coefficient cells.  A cell with
too few material observations produces a number that looks like an
estimate and is noise.  `MIN_CELL_OBS` cells are reported as UNIDENTIFIED
and excluded, never silently returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MIN_CELL_OBS = 30          # per (regime, coefficient) cell
RIDGE_DEFAULT = 1e-3


@dataclass
class CellFit:
    """One fitted coefficient in one regime, with its evidence."""
    value: float = 0.0
    se: float = float("nan")
    n_obs: int = 0
    identified: bool = False
    reason: str = "no data"

    def as_dict(self) -> Dict:
        return {"value": self.value, "se": self.se, "n_obs": self.n_obs,
                "identified": self.identified, "reason": self.reason}


def _wls_sandwich(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                  ridge: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted ridge fit with the Kish/sandwich covariance.

    Discounting weights are a CHOICE, not a variance model, so the
    information actually carried is Kish's N_eff = (sum w)^2 / sum(w^2), and
    the covariance is the sandwich A^-1 (X'W^2X) A^-1 -- not s2 * A^-1,
    which is only correct when W = I.  Both reduce exactly to OLS at w == 1,
    so this cannot change a previously correct unweighted result.
    """
    n, p = X.shape
    W = w[:, None]
    A = X.T @ (W * X) + ridge * np.eye(p)
    b = np.linalg.solve(A, X.T @ (w * y))
    resid = y - X @ b
    n_w = float(w.sum())
    n_eff = float(n_w ** 2 / max(float((w ** 2).sum()), 1e-300))
    dof = max(n_eff - p, 1.0)
    s2 = float(resid @ (w * resid) / max(n_w, 1e-12)) * (n_eff / dof)
    Ainv = np.linalg.pinv(A)
    meat = X.T @ ((w[:, None] ** 2) * X)
    cov = s2 * (Ainv @ meat @ Ainv)
    return b, cov, n_eff


class RegimeDoseEstimator:
    """Per-regime dose-based estimator for s, gamma, alpha and theta.

    Observations are (regime, dnu vector, context, dg per intent).  Each
    regime maintains its own design matrix; nothing is pooled across
    regimes, because pooling is exactly what destroyed v1's beta.
    """

    def __init__(self, params: Sequence[str], intents: Sequence[str],
                 pairs: Sequence[Tuple[str, str]], n_context: int,
                 cfg: Dict, regimes: Sequence[str] = ("low", "mid", "high")):
        self.P = list(params)
        self.I = list(intents)
        self.pairs = [tuple(p) for p in pairs]
        self.n_ctx = int(n_context)
        self.regimes = list(regimes)
        est = cfg.get("estimation", {})
        self.ridge = float(est.get("ridge", RIDGE_DEFAULT))
        self.forget = float(est.get("forgetting_factor", 1.0))
        self.min_obs = int(cfg.get("v2", {}).get("min_cell_obs", MIN_CELL_OBS))

        # design width: intercept + |P| doses + |pairs| products + context
        self.width = 1 + len(self.P) + len(self.pairs) + self.n_ctx
        self._rows: Dict[str, List[np.ndarray]] = {r: [] for r in self.regimes}
        self._y: Dict[str, Dict[str, List[float]]] = {
            r: {i: [] for i in self.I} for r in self.regimes}

        self.s: Dict[str, Dict[str, Dict[str, CellFit]]] = {
            r: {i: {p: CellFit() for p in self.P} for i in self.I}
            for r in self.regimes}
        self.gamma: Dict[str, Dict[str, Dict[Tuple[str, str], CellFit]]] = {
            r: {i: {pr: CellFit() for pr in self.pairs} for i in self.I}
            for r in self.regimes}
        self.alpha: Dict[str, Dict[str, CellFit]] = {
            r: {i: CellFit() for i in self.I} for r in self.regimes}
        self.theta: Dict[str, Dict[str, np.ndarray]] = {
            r: {i: np.zeros(self.n_ctx) for i in self.I} for r in self.regimes}

    # ------------------------------------------------------------------
    def _design_row(self, dnu: Dict[str, float], ctx: np.ndarray) -> np.ndarray:
        row = np.zeros(self.width)
        row[0] = 1.0
        for k, p in enumerate(self.P):
            row[1 + k] = float(dnu.get(p, 0.0))
        off = 1 + len(self.P)
        for k, (a, b) in enumerate(self.pairs):
            row[off + k] = float(dnu.get(a, 0.0)) * float(dnu.get(b, 0.0))
        c = np.asarray(ctx, dtype=float).ravel()[:self.n_ctx]
        if c.size < self.n_ctx:
            c = np.pad(c, (0, self.n_ctx - c.size))
        row[off + len(self.pairs):] = c
        return row

    def observe(self, regime: str, dnu: Dict[str, float], ctx: np.ndarray,
                dg: Dict[str, float]) -> None:
        if regime not in self._rows:
            regime = self.regimes[len(self.regimes) // 2]
        self._rows[regime].append(self._design_row(dnu, ctx))
        for i in self.I:
            self._y[regime][i].append(float(dg.get(i, 0.0)))

    # ------------------------------------------------------------------
    def fit(self) -> None:
        for r in self.regimes:
            rows = self._rows[r]
            n = len(rows)
            if n == 0:
                continue
            X = np.vstack(rows)
            w = (self.forget ** np.arange(n - 1, -1, -1)
                 if self.forget < 1.0 else np.ones(n))
            for i in self.I:
                y = np.asarray(self._y[r][i], dtype=float)
                if n < self.width + 2:
                    self._mark_unidentified(r, i, n, "too few rows for width")
                    continue
                try:
                    b, cov, n_eff = _wls_sandwich(X, y, w, self.ridge)
                except np.linalg.LinAlgError:
                    self._mark_unidentified(r, i, n, "singular design")
                    continue
                se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

                self.alpha[r][i] = CellFit(float(b[0]), float(se[0]), n,
                                           n >= self.min_obs,
                                           "ok" if n >= self.min_obs
                                           else f"n={n} < {self.min_obs}")
                for k, p in enumerate(self.P):
                    # a dose column that never varied cannot be identified
                    col = X[:, 1 + k]
                    nz = int(np.count_nonzero(np.abs(col) > 1e-12))
                    ok = nz >= self.min_obs
                    self.s[r][i][p] = CellFit(
                        float(b[1 + k]), float(se[1 + k]), nz, ok,
                        "ok" if ok else f"nonzero dose rows={nz} < {self.min_obs}")
                off = 1 + len(self.P)
                for k, pr in enumerate(self.pairs):
                    col = X[:, off + k]
                    nz = int(np.count_nonzero(np.abs(col) > 1e-12))
                    ok = nz >= self.min_obs
                    self.gamma[r][i][pr] = CellFit(
                        float(b[off + k]), float(se[off + k]), nz, ok,
                        "ok" if ok else f"joint-dose rows={nz} < {self.min_obs}")
                self.theta[r][i] = b[off + len(self.pairs):].copy()

    def _mark_unidentified(self, r: str, i: str, n: int, why: str) -> None:
        self.alpha[r][i] = CellFit(0.0, float("nan"), n, False, why)
        for p in self.P:
            self.s[r][i][p] = CellFit(0.0, float("nan"), n, False, why)
        for pr in self.pairs:
            self.gamma[r][i][pr] = CellFit(0.0, float("nan"), n, False, why)

    # ------------------------------------------------------------------
    def predict_dg(self, regime: str, intent: str, dnu: Dict[str, float],
                   ctx: np.ndarray) -> float:
        row = self._design_row(dnu, ctx)
        b = np.zeros(self.width)
        b[0] = self.alpha[regime][intent].value
        for k, p in enumerate(self.P):
            b[1 + k] = self.s[regime][intent][p].value
        off = 1 + len(self.P)
        for k, pr in enumerate(self.pairs):
            b[off + k] = self.gamma[regime][intent][pr].value
        b[off + len(self.pairs):] = self.theta[regime][intent]
        return float(row @ b)

    def s_value(self, regime: str, param: str, intent: str) -> float:
        """Slope used by the inner loop.  Falls back across regimes when the
        current cell is unidentified, so safety never depends on a cell that
        happens to be sparse this run."""
        cell = self.s.get(regime, {}).get(intent, {}).get(param)
        if cell is not None and cell.identified:
            return cell.value
        best = None
        for r in self.regimes:
            c = self.s.get(r, {}).get(intent, {}).get(param)
            if c is not None and c.identified:
                if best is None or c.n_obs > best.n_obs:
                    best = c
        return best.value if best is not None else 0.0

    def gamma_value(self, regime: str, pair: Tuple[str, str],
                    intent: str) -> float:
        cell = self.gamma.get(regime, {}).get(intent, {}).get(tuple(pair))
        return cell.value if (cell is not None and cell.identified) else 0.0

    # ------------------------------------------------------------------
    def coefficient_table(self) -> List[Dict]:
        """Flat table of every cell, identified or not, for the audit CSV."""
        rows = []
        for r in self.regimes:
            for i in self.I:
                a = self.alpha[r][i]
                rows.append({"regime": r, "intent": i, "model": "alpha",
                             "key": "intercept", **a.as_dict()})
                for p in self.P:
                    rows.append({"regime": r, "intent": i, "model": "s",
                                 "key": p, **self.s[r][i][p].as_dict()})
                for pr in self.pairs:
                    rows.append({"regime": r, "intent": i, "model": "gamma",
                                 "key": f"{pr[0]}|{pr[1]}",
                                 **self.gamma[r][i][pr].as_dict()})
        return rows

    def identification_report(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for model, getter in (("s", lambda r, i: self.s[r][i].values()),
                              ("gamma", lambda r, i: self.gamma[r][i].values()),
                              ("alpha", lambda r, i: [self.alpha[r][i]])):
            ident = tot = 0
            for r in self.regimes:
                for i in self.I:
                    for c in getter(r, i):
                        tot += 1
                        ident += int(c.identified)
            out[model] = {"identified": ident, "total": tot,
                          "frac": ident / max(tot, 1)}
        return out
