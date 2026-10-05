"""
intact/mild/calibrate.py
========================
Per-intent probability calibration and operating-point alignment.

WHY BOTH, AND WHY IN THIS ORDER
-------------------------------
`p_hat` is consumed in two different ways, and each needs something the
other does not:

  OUTER   u(p_hat) = 1 + lambda_u * p_hat / (1 - p_hat + eps0)
          This reads the VALUE.  It must therefore be a genuine probability,
          or the urgency multiplier is an arbitrary monotone function of an
          arbitrary score.  -> needs CALIBRATION.

  INNER   mediate iff p_hat_i >= tau_risk
          This reads only whether one GLOBAL threshold is crossed.  A raw
          network head has a different natural scale per intent, so a single
          tau cannot mean the same thing for a common intent and a rare one.
          -> needs OPERATING-POINT ALIGNMENT.

Calibration alone breaks the gate: once p_hat is an honest probability of a
crossing, it rarely exceeds 0.65 for a rare intent, so `p_hat >= tau_risk`
almost never fires and recall collapses.  Measured on this scenario:
macro recall fell from 0.337 to 0.001 after Platt scaling alone.

So we do both, in order:

  1. PLATT on the logit, fitted on VALIDATION, making p_hat calibrated.
  2. A monotone piecewise-linear map sending each intent's chosen operating
     threshold t_k to exactly tau_risk.

Step 2 is monotone and fixes the endpoints, so it preserves ranking, keeps
p_hat in [0,1], and leaves the calibrated ORDER intact while making the
single global tau_risk mean "the operating point we selected for this
intent" for every intent at once.
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_platt(p: Sequence[float], y: Sequence[float],
              l2: float = 1e-3, iters: int = 200) -> Dict[str, float]:
    """Newton-fitted Platt scaling  p_cal = sigmoid(a * logit(p) + b).

    Ridge-regularised so a validation split containing very few positives
    cannot drive `a` to an extreme value.
    """
    z = _logit(np.asarray(p, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    if len(z) < 10 or y.sum() < 1 or y.sum() == len(y):
        return {"a": 1.0, "b": 0.0}          # nothing to fit against
    # The former unbounded Newton update had no line search and did not
    # regularise the intercept.  On rare RAN failures it could diverge to
    # values such as a=205, b=-740805, mapping every held-out score to zero.
    # Bounded optimisation keeps calibration monotone and numerically stable.
    from scipy.optimize import minimize

    prior = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    x0 = np.array([1.0, np.log(prior / (1.0 - prior))], dtype=np.float64)

    def objective(x):
        a, b = float(x[0]), float(x[1])
        q = np.clip(_sigmoid(a * z + b), 1e-9, 1.0 - 1e-9)
        nll = -float(np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))
        return nll + 0.5 * float(l2) * (a * a + 0.01 * b * b)

    opt = minimize(objective, x0, method="L-BFGS-B",
                   bounds=[(1e-4, 20.0), (-20.0, 20.0)],
                   options={"maxiter": int(iters), "ftol": 1e-12})
    a, b = (opt.x if opt.success and np.all(np.isfinite(opt.x)) else x0)
    return {"a": float(a), "b": float(b)}


def apply_platt(p: np.ndarray, par: Dict[str, float]) -> np.ndarray:
    return _sigmoid(par["a"] * _logit(p) + par["b"])


def choose_threshold(p: np.ndarray, y: np.ndarray, target_recall: float,
                     max_false_alarm: float) -> float:
    """Highest threshold reaching ``target_recall`` within the FA cap.

    If the joint target is infeasible, maximise recall among thresholds that
    still respect the false-alarm cap.  Only fall back to best F1 when no
    threshold satisfies the cap.  This ordering matches the documented safety
    asymmetry: a missed warning is costlier than an unnecessary mediation.

    The previous fallback went directly to best F1.  On a weak validation
    cell that could choose a very conservative threshold with poor recall even
    while most of the permitted false-alarm budget remained unused.  Changing
    ``target_recall`` or ``max_false_alarm`` then appeared to do nothing.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.sum() < 1:
        return 0.5
    grid = np.unique(np.clip(np.quantile(p, np.linspace(0.001, 0.999, 400)),
                             1e-4, 1 - 1e-4))
    best_f1, best_t = -1.0, float(np.median(p))
    feasible = []
    within_cap = []
    for t in grid:
        yh = p >= t
        tp = float((yh & (y > 0)).sum())
        fp = float((yh & (y == 0)).sum())
        fn = float((~yh & (y > 0)).sum())
        rec = tp / max(tp + fn, 1.0)
        prec = tp / max(tp + fp, 1.0)
        fa = fp / max(float((y == 0).sum()), 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
        if fa <= max_false_alarm:
            within_cap.append((float(rec), float(fa), float(t)))
        if rec >= target_recall and fa <= max_false_alarm:
            feasible.append(float(t))
    if feasible:
        # p >= t, so the highest feasible t gives the lowest false-alarm rate
        # while still attaining the requested recall.
        return max(feasible)
    if within_cap:
        # The requested recall cannot be attained under the cap.  Spend the
        # available alarm budget on safety: maximise recall first, then choose
        # the highest threshold among ties (the least trigger-happy tie).
        best_recall = max(row[0] for row in within_cap)
        return max(row[2] for row in within_cap
                   if abs(row[0] - best_recall) <= 1e-12)
    return best_t


def align_operating_point(p: np.ndarray, t: float, tau: float) -> np.ndarray:
    """Monotone piecewise-linear map with t -> tau, 0 -> 0, 1 -> 1."""
    p = np.asarray(p, dtype=np.float64)
    t = float(min(max(t, 1e-6), 1.0 - 1e-6))
    lo = tau * (p / t)
    hi = tau + (1.0 - tau) * (p - t) / (1.0 - t)
    return np.clip(np.where(p < t, lo, hi), 0.0, 1.0)


class IntentCalibrator:
    """Platt scaling plus operating-point alignment, one pair per intent."""

    def __init__(self, intents: List[str], tau: float):
        self.intents = list(intents)
        self.tau = float(tau)
        self.platt: List[Dict[str, float]] = [{"a": 1.0, "b": 0.0}] * len(intents)
        self.thresholds: List[float] = [0.5] * len(intents)

    def fit(self, p_val: np.ndarray, y_val: np.ndarray,
            target_recall: float = 0.80,
            max_false_alarm: float = 0.25,
            safe_mask: np.ndarray | None = None) -> "IntentCalibrator":
        """Platt on ALL validation rows; threshold on the SAFE subset.

        The two populations are deliberately different.  Calibration must
        describe the whole distribution, or u(p_hat) is wrong wherever the
        model spends its time.  The THRESHOLD, however, only ever matters
        where the intent still looks healthy: once g < epsilon the intent is
        already unsafe and mediating it is not a judgement call.  Choosing
        the threshold on the full population lets a 50-95% positive base
        rate satisfy the recall target while the ratchet cases -- the ones
        the gate exists to catch -- are still missed.  Measured: full-
        population recall 0.75 with safe-subset recall 0.35.
        """
        self.platt, self.thresholds = [], []
        for k in range(len(self.intents)):
            par = fit_platt(p_val[:, k], y_val[:, k])
            pc = apply_platt(p_val[:, k], par)
            self.platt.append(par)
            m = (safe_mask[:, k] if safe_mask is not None
                 else np.ones(len(pc), bool))
            if m.sum() < 20 or y_val[m, k].sum() < 1 or y_val[m, k].sum() == m.sum():
                m = np.ones(len(pc), bool)      # degenerate subset -> fall back
            self.thresholds.append(
                choose_threshold(pc[m], y_val[m, k], target_recall,
                                 max_false_alarm))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        cols = []
        for k in range(len(self.intents)):
            pc = apply_platt(p[:, k], self.platt[k])
            cols.append(align_operating_point(pc, self.thresholds[k], self.tau))
        return np.column_stack(cols)

    # -- persistence ---------------------------------------------------
    def to_dict(self) -> Dict:
        return {"intents": self.intents, "tau": self.tau,
                "platt": self.platt, "thresholds": self.thresholds}

    @staticmethod
    def from_dict(d: Dict) -> "IntentCalibrator":
        c = IntentCalibrator(d["intents"], d["tau"])
        c.platt = d["platt"]
        c.thresholds = d["thresholds"]
        return c
