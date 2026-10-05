"""
intact/estimation/effects.py   --  INTACT v11 §7.3
=================================================
beta_{j,i}  and  gamma_{jk,i}  by RANDOMISED EFFECT ESTIMATION.

THE MODEL
---------
    Delta g_i(tau) = alpha_i                       <- baseline drift (NUISANCE)
                   + sum_j   beta_{j,i}  Z_j(tau)  <- individual claim effects
                   + sum_jk  gamma_{jk,i} Z_j Z_k  <- pair interactions
                   + theta_i . c(tau)              <- network context
                   + eps                           <- unexplained

WHAT EACH PIECE IS FOR
    alpha  : how margins move when nobody acts.  NEVER used by the scheduler.
             It is a nuisance parameter and it is here for ONE reason: it is
             what forces E[eps] = 0.  Drop it and the mean drift leaks into
             the slopes and BIASES beta.  (Simulated: dropping alpha gives
             bias +0.004 AND doubles the standard deviation.)
    beta   : the marginal causal effect of letting claim j act on intent i.
             THIS IS ATTRIBUTION, as a measured number.  Naturally
             multi-cause and multi-victim.  Feeds V_j.
    gamma  : what a PAIR does beyond the sum of the parts.  Feeds H(j,k).
    theta  : context covariates.  Stops "load rose and margins fell" being
             misread as "claim j hurt intent i".

IDENTIFICATION -- THE LOAD-BEARING PART
---------------------------------------
On purely observational data Z is chosen by the scheduler based on what it
expects to happen, so Z correlates with the outcome for reasons other than
causation.  Simulated, that bias is -0.042 against a true effect of +0.120:
THIRTY-FIVE PERCENT of the effect destroyed.  The defence is randomisation:

  * FORCED EXPLORATION (eps_exp): with probability eps_exp per epoch,
    perturb the chosen set, subject to a safety veto.
  * PAIRED BLOCKS: randomise ON/OFF within adjacent-epoch pairs.  Drift over
    1-2 s is negligible compared with drift over minutes, so it differences
    out almost entirely.  Simulated: 36% variance reduction, FREE.

VARIANCE REDUCTION -- CUPED
---------------------------
Deng, Xu, Kohavi & Walker, "Improving the Sensitivity of Online Controlled
Experiments by Utilizing Pre-Experiment Data", WSDM 2013.  Adjust the
outcome by a pre-period covariate X that is correlated with the outcome but
independent of the treatment:

        Delta g_cuped = Delta g - theta_c (X - Xbar),
        theta_c = Cov(Delta g, X) / Var(X)

Variance falls by rho^2.  Bias is provably unchanged.  See also
Lin, Ann. Appl. Stat. 7(1):295-318, 2013.

NOTE: a traffic forecaster can be plugged in as the CUPED COVARIATE, where a
good forecast reduces variance and a bad one is automatically down-weighted
-- with no bias either way.  Subtracting a forecast directly would instead
propagate every forecaster error straight into beta.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


class EffectEstimator:
    """
    Recursive least squares over the design matrix
        [1, Z_1..Z_J, (selected pairs), context...]
    with one model per intent.
    """

    def __init__(self, claim_ids: List[str], intent_ids: List[str],
                 pair_list: List[Tuple[str, str]], n_context: int, cfg: Dict):
        self.J = claim_ids
        self.I = intent_ids
        self.pairs = pair_list
        self.nc = n_context
        c = cfg["estimation"]
        self.use_cuped = c["use_cuped"]
        self.ridge = c["ridge"]
        self.min_samples = c["min_samples"]
        self.forget = c["forgetting_factor"]

        # column layout: [intercept, claims..., pairs..., context...]
        self.p = 1 + len(self.J) + len(self.pairs) + n_context
        # store raw rows so we can refit with CUPED and compute honest CIs
        self.X: List[np.ndarray] = []
        self.Y: Dict[str, List[float]] = {i: [] for i in self.I}
        self.prev_dg: Dict[str, float] = {i: 0.0 for i in self.I}
        self.Xpre: Dict[str, List[float]] = {i: [] for i in self.I}  # CUPED covariate

        self.beta: Dict[str, Dict[str, float]] = {i: {j: 0.0 for j in self.J} for i in self.I}
        self.gamma: Dict[str, Dict[Tuple[str, str], float]] = {i: {p: 0.0 for p in self.pairs} for i in self.I}
        self.alpha: Dict[str, float] = {i: 0.0 for i in self.I}
        # Standard error of the baseline-drift intercept.  Previously fitted
        # but discarded, so alpha could never be validated the way beta and
        # gamma are -- and alpha is what keeps mean drift OUT of the slopes,
        # so an unchecked alpha means an unchecked beta.
        self.se_alpha: Dict[str, float] = {i: 0.0 for i in self.I}
        self.sigma: Dict[str, float] = {i: 1.0 for i in self.I}   # residual sd
        self.se_beta: Dict[str, Dict[str, float]] = {i: {j: np.inf for j in self.J} for i in self.I}
        # Gamma uncertainty used to be discarded even though it is available
        # from the same covariance matrix as beta.  Retaining it is necessary
        # for an honest null false-positive audit of pair interactions.
        self.se_gamma: Dict[str, Dict[Tuple[str, str], float]] = {
            i: {p: np.inf for p in self.pairs} for i in self.I}
        self.r2: Dict[str, float] = {i: 0.0 for i in self.I}
        # Effective sample size actually carried by the discounted history.
        # Reported so a reader can see that 3000 stored epochs supply ~399
        # rows' worth of information at forgetting_factor = 0.995.
        self.n_eff: Dict[str, float] = {i: 0.0 for i in self.I}

    # ------------------------------------------------------------------
    def design_row(self, S: set, context: np.ndarray) -> np.ndarray:
        """Build one row of the design matrix from an eligibility set."""
        row = np.zeros(self.p)
        row[0] = 1.0                                   # intercept -> alpha
        for k, j in enumerate(self.J):
            row[1 + k] = 1.0 if j in S else 0.0        # Z_j
        off = 1 + len(self.J)
        for k, (a, b) in enumerate(self.pairs):
            row[off + k] = 1.0 if (a in S and b in S) else 0.0   # Z_j Z_k
        row[off + len(self.pairs):] = context
        return row

    def observe(self, S: set, context: np.ndarray, dg: Dict[str, float]) -> None:
        """Record one epoch: which claims acted, the context, and the outcome."""
        self.X.append(self.design_row(S, context))
        for i in self.I:
            self.Y[i].append(dg.get(i, 0.0))
            # CUPED covariate = the PREVIOUS epoch's margin change.  It is
            # correlated with this epoch's outcome (drift persists) and is
            # independent of THIS epoch's randomised treatment.
            self.Xpre[i].append(self.prev_dg[i])
            self.prev_dg[i] = dg.get(i, 0.0)

    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Refit every intent's model.  Cheap enough to do once per epoch."""
        if len(self.X) < self.min_samples:
            return
        X = np.array(self.X)
        # exponential forgetting so old regimes fade out
        n = len(X)
        w = self.forget ** np.arange(n - 1, -1, -1)
        Wm = np.diag(w)

        for i in self.I:
            y = np.array(self.Y[i])
            if self.use_cuped:
                # ---- CUPED adjustment (Deng et al. 2013) ----------------
                xp = np.array(self.Xpre[i])
                var = np.var(xp)
                if var > 1e-12:
                    th = np.cov(y, xp)[0, 1] / var
                    y = y - th * (xp - xp.mean())
            A = X.T @ Wm @ X + self.ridge * np.eye(self.p)
            b = np.linalg.solve(A, X.T @ Wm @ y)

            resid = y - X @ b
            # ---- INFERENCE UNDER EXPONENTIAL FORGETTING -----------------
            # *** BUG FIX. ***  The old code divided the WEIGHTED residual
            # sum of squares by the RAW row count (n - p).  Under forgetting
            # the weighted sum only ever accumulates to sum(w) ~= 1/(1-lam)
            # -- 200 rows at lam = 0.995, no matter how many epochs are
            # stored -- so sigma^2 was deflated by roughly n / N_eff and
            # every standard error shrank by its square root (2.7x at
            # n = 3000).  Measured consequence: the controlled benchmark's
            # null false-positive rate rose 0.063 -> 0.483 and CI coverage
            # collapsed 0.94 -> 0.52 for beta, which made an ARITHMETIC
            # defect look like a failed specification.
            #
            # Two corrections, both of which are identities when lam = 1:
            #
            # 1. EFFECTIVE SAMPLE SIZE.  Discounting weights are a CHOICE,
            #    not a variance model, so the information actually carried is
            #    Kish's  N_eff = (sum w)^2 / sum(w^2).  For lam = 0.995 that
            #    is ~399 regardless of n.
            #
            # 2. SANDWICH COVARIANCE.  With homoskedastic errors and
            #    arbitrary weights, Var(X'Wy) = sigma^2 X'W^2X, hence
            #        Cov(b) = s2 * A^-1 (X'W^2X) A^-1
            #    NOT s2 * A^-1.  The two coincide only when W = I, so this
            #    reproduces the previously-passing lam = 1.0 numbers exactly
            #    while fixing the lam < 1 case.
            n_w = float(w.sum())
            n_eff = float(n_w ** 2 / max(float((w ** 2).sum()), 1e-300))
            dof = max(n_eff - self.p, 1.0)
            s2 = float(resid @ (w * resid) / n_w) * (n_eff / dof)
            Ainv = np.linalg.pinv(A)
            meat = X.T @ (w[:, None] ** 2 * X)
            cov = s2 * (Ainv @ meat @ Ainv)
            self.n_eff[i] = n_eff

            self.alpha[i] = float(b[0])
            self.se_alpha[i] = float(np.sqrt(max(cov[0, 0], 0.0)))
            for k, j in enumerate(self.J):
                self.beta[i][j] = float(b[1 + k])
                self.se_beta[i][j] = float(np.sqrt(max(cov[1 + k, 1 + k], 0)))
            off = 1 + len(self.J)
            for k, pr in enumerate(self.pairs):
                self.gamma[i][pr] = float(b[off + k])
                self.se_gamma[i][pr] = float(
                    np.sqrt(max(cov[off + k, off + k], 0)))
            self.sigma[i] = float(np.sqrt(max(s2, 0)))
            ss_tot = float(((y - y.mean()) ** 2).sum())
            self.r2[i] = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        """
        Design-matrix health.  Run this BEFORE interpreting any beta.
        If a claim is ON almost always, or two columns move together, the
        regression CANNOT separate their effects and the coefficients are
        meaningless regardless of how good they look.
        """
        import numpy as _np
        if not self.X:
            return {"n": 0}
        X = _np.array(self.X)
        Z = X[:, 1:1 + len(self.J)]
        on = Z.mean(axis=0)
        # correlation between treatment columns
        with _np.errstate(invalid="ignore"):
            corr = _np.corrcoef(Z.T)
        corr = _np.nan_to_num(corr)
        off = corr - _np.eye(len(self.J))
        A = X.T @ X
        ev = _np.linalg.eigvalsh(A)
        cond = float(ev.max() / max(ev.min(), 1e-12))
        return {
            "n_epochs": len(self.X),
            "on_fraction": {j: float(on[k]) for k, j in enumerate(self.J)},
            "max_abs_offdiag_corr": float(_np.abs(off).max()) if len(self.J) > 1 else 0.0,
            "worst_correlated_pair": (
                [self.J[a] for a in _np.unravel_index(_np.abs(off).argmax(), off.shape)]
                if len(self.J) > 1 else []),
            "rank": int(_np.linalg.matrix_rank(X)),
            "n_columns": int(X.shape[1]),
            "condition_number": cond,
            "rank_deficient": bool(_np.linalg.matrix_rank(X) < X.shape[1]),
        }

    def identified(self, i: str, j: str, k: float = 2.0) -> bool:
        """|beta| > k * SE  -- otherwise it is noise, not an effect."""
        se = self.se_beta[i].get(j, float("inf"))
        return bool(abs(self.beta[i].get(j, 0.0)) > k * se)

    def table(self) -> list:
        """beta +- 1.96 SE per (claim, intent), with the identification flag."""
        rows = []
        for i in self.I:
            for j in self.J:
                b, se = self.beta[i][j], self.se_beta[i][j]
                rows.append({"intent": i, "claim": j, "beta": b, "se": se,
                             "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
                             "identified": self.identified(i, j),
                             "r2_intent": self.r2[i], "sigma_intent": self.sigma[i]})
        return rows

    def sigma_min(self, i: str, j: str, k: float = 2.0) -> float:
        """
        The evidentiary threshold, DERIVED rather than chosen (v11 §7.4.6,
        §12).  An effect smaller than k standard errors is not evidence.
        """
        se = self.se_beta[i].get(j, np.inf)
        return k * se if np.isfinite(se) else np.inf


def reconstruct_beta(s_pi: float, dnu_when_eligible, se_s: float = 0.0):
    """
    beta^elig_{j,i}  =  s_{p,i} * E[ d nu_p | j eligible ]        (v11 s7.4.4)

    WHY THIS EXISTS.  The direct estimator regresses a DIFFERENCE (dg) on an
    ELIGIBILITY indicator (Z).  That is the right transform for an impulse
    treatment and the wrong one for a PERSISTENT control: it discards the
    signal (only epochs where the knob actually moved carry information) and
    it roughly doubles the noise (var(dg) = 2 var(g) for an uncorrelated
    series).  Reconstructing from s instead avoids both.

    NO q_j FACTOR IS NEEDED, AND ADDING ONE WOULD BE WRONG.
    E[d nu | eligible] already averages over the epochs in which the xApp
    declined to write, so inactivity is carried exactly once.  A converged
    controller therefore gets beta -> 0 automatically, which is the correct
    marginal value of granting it authority.

    Returns (beta, se_beta).
    """
    import numpy as _np
    a = _np.asarray(list(dnu_when_eligible), dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    m = float(a.mean())
    return float(s_pi * m), float(abs(m) * se_s)


class WriteEffectEstimator:
    """
    WRITE-LEVEL interaction  eta_{jk,i}.

        Delta g_i = a_i + sum_j s_ji dnu_j + sum_jk eta_jki dnu_j dnu_k + eps

    The outer regression's gamma answers "what do these two CLAIMS do when
    both are ELIGIBLE"; eta answers "what do these two KNOB MOVEMENTS do when
    they coincide".  A write-level optimiser (B7+) needs the second, and
    substituting the first is a category error even though both are called
    'the interaction term'.
    """

    def __init__(self, claim_ids, intent_ids, pair_list, cfg):
        self.J, self.I, self.pairs = list(claim_ids), list(intent_ids), list(pair_list)
        self.p = 1 + len(self.J) + len(self.pairs)
        self.ridge = cfg["estimation"]["ridge"]
        self.min_samples = cfg["estimation"]["min_samples"]
        self.X = []
        self.Y = {i: [] for i in self.I}
        self.eta = {i: {pr: 0.0 for pr in self.pairs} for i in self.I}
        self.se_eta = {i: {pr: np.inf for pr in self.pairs} for i in self.I}
        self.s_hat = {i: {j: 0.0 for j in self.J} for i in self.I}
        self.r2 = {i: 0.0 for i in self.I}

    def observe(self, dnu, dg):
        """dnu: {jid: applied delta}  (0 when the claim did not write)"""
        row = np.zeros(self.p); row[0] = 1.0
        for k, j in enumerate(self.J):
            row[1 + k] = float(dnu.get(j, 0.0))
        off = 1 + len(self.J)
        for k, (a, b) in enumerate(self.pairs):
            row[off + k] = float(dnu.get(a, 0.0)) * float(dnu.get(b, 0.0))
        self.X.append(row)
        for i in self.I:
            self.Y[i].append(dg.get(i, 0.0))

    def fit(self):
        if len(self.X) < self.min_samples:
            return
        X = np.array(self.X)
        A = X.T @ X + self.ridge * np.eye(self.p)
        for i in self.I:
            y = np.array(self.Y[i])
            b = np.linalg.solve(A, X.T @ y)
            resid = y - X @ b
            dof = max(len(y) - self.p, 1)
            s2 = float(resid @ resid / dof)
            cov = s2 * np.linalg.pinv(A)
            for k, j in enumerate(self.J):
                self.s_hat[i][j] = float(b[1 + k])
            off = 1 + len(self.J)
            for k, pr in enumerate(self.pairs):
                self.eta[i][pr] = float(b[off + k])
                self.se_eta[i][pr] = float(
                    np.sqrt(max(cov[off + k, off + k], 0.0)))
            ss_tot = float(((y - y.mean()) ** 2).sum())
            self.r2[i] = (float(1.0 - (resid ** 2).sum() / ss_tot)
                          if ss_tot > 1e-12 else 0.0)
