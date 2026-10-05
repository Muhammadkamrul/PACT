"""
intact/v2/validate_v2.py
========================
HELD-OUT validation for every fitted coefficient.

WHAT "HELD OUT" MEANS HERE
--------------------------
Splits are BY WHOLE PROBE (probes.split_by_probe), never by row.  A probe's
four arms are matched counterfactuals of one another; putting the 10 arm in
train and the 11 arm in test leaks the matched baseline across the split and
flatters the test score.  v1 split chronologically by row, which for the
coefficient models allowed exactly that leak.

WHAT IS SCORED, AND WITH WHICH STATISTIC
----------------------------------------
Different estimands need different metrics, and using the wrong one was a
real v1 problem:

  s      PREDICTION on unseen doses     -> held-out R^2 is meaningful
  gamma  RECOVERY against a differenced
         ground truth                   -> Pearson r + normalised MAE
  alpha  NUISANCE INTERCEPT             -> bias and interval coverage ONLY

R^2 is inappropriate for gamma recovery: the truth values cluster near zero,
so SS_total is tiny and R^2 goes negative for an estimator that is merely
imprecise.  Sign accuracy and normalised MAE are appropriate; R^2 is not.

alpha is never scored on sign accuracy or normalised MAE.  Its planted
magnitude sits well below the noise floor by construction, so those
statistics measure noise.  What matters is whether it is unbiased, whether
its interval is honest, and whether removing it MOVES BETA -- the last being
the only question whose answer changes what the paper may claim.

NORMALISED MAE IS THE HEADLINE
------------------------------
    nMAE = E|hat - true| / E|true|
Error as a multiple of the effect being estimated.  nMAE < 1 means the
estimate is smaller than the thing it measures; nMAE >> 1 means it is not.
One column, and the whole learnability argument is in it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
def pearson_r(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-15 or y.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r2_heldout(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 3:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-18:
        return float("nan")     # undefined, not zero -- do not fake a score
    return 1.0 - ss_res / ss_tot


def normalised_mae(hat, true) -> float:
    hat, true = np.asarray(hat, float), np.asarray(true, float)
    m = np.isfinite(hat) & np.isfinite(true)
    hat, true = hat[m], true[m]
    if len(hat) == 0:
        return float("nan")
    scale = float(np.mean(np.abs(true)))
    if scale < 1e-15:
        return float("nan")
    return float(np.mean(np.abs(hat - true)) / scale)


def materiality_threshold(truths: Sequence[float], pct: float = 75.0) -> float:
    """Below this, a pair is NULL and is scored for false positives instead.

    Grading sign accuracy on a near-zero truth grades the sign of noise.
    """
    a = np.abs(np.asarray(truths, float))
    a = a[np.isfinite(a)]
    return float(np.percentile(a, pct)) if len(a) else 0.0


# ----------------------------------------------------------------------
def validate_s(est, trials, intents, params) -> List[Dict]:
    """s is validated by PREDICTION on held-out probe arms."""
    rows = []
    for i in intents:
        yt, yp, regs = [], [], []
        for t in trials:
            pred = est.predict_dg(t.regime, i, t.dnu, t.ctx)
            yt.append(float(t.dg.get(i, 0.0)))
            yp.append(pred)
            regs.append(t.regime)
        yt, yp = np.asarray(yt), np.asarray(yp)
        if len(yt) < 5:
            continue
        err = yp - yt
        rows.append({
            "model": "s", "intent": i, "regime": "all", "n": len(yt),
            "heldout_R2": r2_heldout(yt, yp),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "pearson_r": pearson_r(yt, yp),
            "mean_abs_truth": float(np.mean(np.abs(yt))),
            "normalised_MAE": normalised_mae(yp, yt),
            "sign_accuracy": float(np.mean(np.sign(yp) == np.sign(yt))),
        })
        for r in sorted(set(regs)):
            m = np.asarray([g == r for g in regs])
            if m.sum() < 5:
                continue
            e = yp[m] - yt[m]
            rows.append({
                "model": "s", "intent": i, "regime": r, "n": int(m.sum()),
                "heldout_R2": r2_heldout(yt[m], yp[m]),
                "MAE": float(np.mean(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e ** 2))),
                "pearson_r": pearson_r(yt[m], yp[m]),
                "mean_abs_truth": float(np.mean(np.abs(yt[m]))),
                "normalised_MAE": normalised_mae(yp[m], yt[m]),
                "sign_accuracy": float(np.mean(np.sign(yp[m]) == np.sign(yt[m]))),
            })
    return rows


def validate_gamma(est, trials, intents, pairs, gamma_truth=None) -> List[Dict]:
    """gamma is validated by RECOVERY against the differenced estimate.

    The C - A - B difference is model-free: it needs no regression and makes
    no functional-form assumption.  If the regression and the difference
    disagree, the bilinear form is wrong rather than merely noisy, which is
    a distinction v1 could not draw.
    """
    from .probes import gamma_by_differencing
    rows = []
    for pr in pairs:
        pr = tuple(pr)
        sel = [t for t in trials if tuple(t.pair) == pr]
        if not sel:
            continue
        for i in intents:
            for r in sorted({t.regime for t in sel}):
                d = gamma_by_differencing(sel, i, r)
                fit = est.gamma.get(r, {}).get(i, {}).get(pr)
                truth = (gamma_truth or {}).get((pr, i, r))
                rows.append({
                    "model": "gamma", "pair": f"{pr[0]}|{pr[1]}", "intent": i,
                    "regime": r,
                    "gamma_regression": fit.value if fit else float("nan"),
                    "gamma_se": fit.se if fit else float("nan"),
                    "identified": bool(fit.identified) if fit else False,
                    "n_obs": fit.n_obs if fit else 0,
                    "gamma_differenced": d.get("gamma", float("nan")),
                    "gamma_double_arm": d.get("gamma_double_arm", float("nan")),
                    "bilinear_consistent": d.get("bilinear_consistent", None),
                    "gamma_true": truth if truth is not None else float("nan"),
                    "n_trials": d.get("n", 0),
                })
    return rows


def validate_alpha(est, trials, intents) -> List[Dict]:
    """alpha: bias, coverage, and whether it is LOAD-BEARING for s.

    Never sign accuracy, never normalised MAE -- see the module docstring.
    """
    rows = []
    for r in est.regimes:
        base = [t for t in trials if t.regime == r
                and all(abs(v) < 1e-12 for v in t.dnu.values())]
        if len(base) < 5:
            continue
        for i in intents:
            obs = np.asarray([t.dg.get(i, 0.0) for t in base], float)
            cell = est.alpha[r][i]
            sem = float(obs.std(ddof=1) / max(np.sqrt(len(obs)), 1.0))
            err = cell.value - float(obs.mean())
            se_tot = float(np.sqrt(max(cell.se, 0.0) ** 2 + sem ** 2))
            # IDENTIFIABILITY GUARD.  alpha is the drift when nobody acts.
            # In the OPERATIONAL loop that is a well-posed per-epoch quantity.
            # In a probe design with RANDOMISED starting states it is not: each
            # probe begins at a different distance from equilibrium, so the
            # "baseline drift" is a property of the jitter, not of the system,
            # and the fitted intercept absorbs it.  Measured here: bias -17
            # against margins of order 0.1.  Report that as NOT IDENTIFIABLE
            # rather than printing a number that looks like an estimate.
            scale = float(np.median(np.abs(obs))) if len(obs) else 0.0
            ill_posed = bool(abs(err) > 10.0 * max(scale, 1e-9))
            rows.append({
                "model": "alpha", "intent": i, "regime": r,
                "identifiable_from_probes": not ill_posed,
                "n_baseline_arms": len(base),
                "alpha_hat": cell.value, "se_alpha": cell.se,
                "heldout_mean_drift": float(obs.mean()), "sem": sem,
                "bias": err, "abs_error": abs(err),
                "z": err / se_tot if se_tot > 0 else float("nan"),
                "covers_95": bool(abs(err) <= 1.96 * se_tot),
                "identified": cell.identified,
            })
    return rows


def alpha_leakage(trials, params, pairs, n_ctx, ridge=1e-3) -> List[Dict]:
    """Refit WITHOUT the intercept and measure how far every s moves.

    This must be a genuine no-intercept refit.  Zeroing alpha after fitting
    leaves s numerically identical by construction and would report "alpha
    is decorative" no matter what the truth is -- a defect the first version
    of this check actually had.
    """
    from .effects_v2 import _wls_sandwich
    rows = []
    regimes = sorted({t.regime for t in trials})
    intents = sorted({i for t in trials for i in t.dg})
    for r in regimes:
        sel = [t for t in trials if t.regime == r]
        if len(sel) < 20:
            continue
        X = []
        for t in sel:
            row = [1.0] + [t.dnu.get(p, 0.0) for p in params]
            row += [t.dnu.get(a, 0.0) * t.dnu.get(b, 0.0) for a, b in pairs]
            c = np.asarray(t.ctx, float).ravel()[:n_ctx]
            row += list(np.pad(c, (0, max(0, n_ctx - c.size))))
            X.append(row)
        X = np.asarray(X, float)
        w = np.ones(len(X))
        for i in intents:
            y = np.asarray([t.dg.get(i, 0.0) for t in sel], float)
            b_full, cov, _ = _wls_sandwich(X, y, w, ridge)
            b_noint, _, _ = _wls_sandwich(X[:, 1:], y, w, ridge)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            for k, p in enumerate(params):
                s1, s0 = float(b_full[1 + k]), float(b_noint[k])
                sek = float(se[1 + k])
                rows.append({"regime": r, "intent": i, "param": p,
                             "s_with_alpha": s1, "s_alpha_dropped": s0,
                             "shift": s0 - s1, "se_s": sek,
                             "shift_in_se": (s0 - s1) / sek if sek > 0 else float("nan")})
    return rows
