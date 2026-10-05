"""Paired scenario-level statistics for the v10 benchmark.

The statistical unit is the SCENARIO.  Seeds are averaged inside a scenario
before any test, because seeds of one scenario share topology, contracts and
targets and are therefore not independent.

Functions
---------
paired_summary   mean/median/sd, wins/ties/losses, bootstrap CI, Wilcoxon,
                 sign test and paired t-test for one vector of differences
holm             Holm step-down adjustment of a family of p-values
tost             two one-sided tests for practical equivalence within +-m
decide_superiority_or_equivalence
                 the pre-registered decision rule used for MILD vs analytical
required_n, mde  normal-approximation power calculations
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

TIE_TOL = 1e-12


def _norm_ppf(q: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(q))


def bootstrap_distribution(x: Sequence[float], n_boot: int = 100000,
                           seed: int = 20260910, stat: str = "mean"
                           ) -> np.ndarray:
    """Bootstrap distribution of the mean (or median) across scenarios."""
    x = np.asarray(list(x), float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([np.nan])
    if len(x) == 1:
        return np.array([x[0]])
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    chunk = 20000
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, len(x), size=(m, len(x)))
        sample = x[idx]
        out[start:start + m] = (sample.mean(axis=1) if stat == "mean"
                                else np.median(sample, axis=1))
    return out


def bootstrap_ci(x: Sequence[float], level: float = 0.95, n_boot: int = 100000,
                 seed: int = 20260910, stat: str = "mean") -> List[float]:
    """Percentile bootstrap CI of the mean (or median) across scenarios."""
    dist = bootstrap_distribution(x, n_boot, seed, stat)
    a = (1.0 - level) / 2.0
    return [float(np.nanquantile(dist, a)), float(np.nanquantile(dist, 1 - a))]


def wilcoxon_p(x: Sequence[float]) -> float:
    x = np.asarray(list(x), float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.all(np.abs(x) <= TIE_TOL):
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(x, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def sign_test_p(x: Sequence[float]) -> float:
    x = np.asarray(list(x), float)
    wins = int((x > TIE_TOL).sum())
    losses = int((x < -TIE_TOL).sum())
    n = wins + losses
    if n == 0:
        return float("nan")
    from scipy.stats import binomtest
    return float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)


def ttest_p(x: Sequence[float]) -> float:
    x = np.asarray(list(x), float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return float("nan")
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(x, 0.0).pvalue)


def min_wilcoxon_p(n: int) -> float:
    """Smallest two-sided exact Wilcoxon p-value attainable with n pairs."""
    return float(min(1.0, 2.0 / (2 ** n))) if n > 0 else float("nan")


def paired_summary(x: Sequence[float], n_boot: int = 100000,
                   seed: int = 20260910) -> Dict[str, float]:
    x = np.asarray(list(x), float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    dist = bootstrap_distribution(x, n_boot, seed)
    q = lambda a: float(np.nanquantile(dist, a))
    return {
        "n": int(n),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": sd,
        "wins": int((x > TIE_TOL).sum()),
        "ties": int((np.abs(x) <= TIE_TOL).sum()),
        "losses": int((x < -TIE_TOL).sum()),
        "ci95_low": q(0.025), "ci95_high": q(0.975),
        "ci90_low": q(0.05), "ci90_high": q(0.95),
        "wilcoxon_p": wilcoxon_p(x),
        "sign_test_p": sign_test_p(x),
        "ttest_p": ttest_p(x),
        "cohen_dz": (float(x.mean() / sd) if n > 1 and sd > 0
                     else float("nan")),
        "min_attainable_wilcoxon_p": min_wilcoxon_p(n),
    }


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (NaN entries are ignored)."""
    p = np.asarray(list(pvals), float)
    out = np.full(len(p), np.nan)
    idx = [i for i in range(len(p)) if np.isfinite(p[i])]
    m = len(idx)
    if m == 0:
        return out.tolist()
    order = sorted(idx, key=lambda i: p[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out.tolist()


def tost(x: Sequence[float], margin: float, alpha: float = 0.05
         ) -> Dict[str, float]:
    """Two one-sided t-tests for equivalence of the mean within +-margin.

    Equivalence is supported when BOTH one-sided nulls are rejected, which
    is the same as the (1 - 2 alpha) confidence interval lying inside
    (-margin, +margin).
    """
    x = np.asarray(list(x), float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"tost_p": float("nan"), "tci_low": float("nan"),
                "tci_high": float("nan")}
    from scipy.stats import t
    mean = float(x.mean())
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    if se == 0:
        inside = abs(mean) < margin
        return {"tost_p": 0.0 if inside else 1.0, "tci_low": mean,
                "tci_high": mean}
    df = n - 1
    p_lower = 1.0 - t.cdf((mean + margin) / se, df)   # H0: mu <= -m
    p_upper = t.cdf((mean - margin) / se, df)         # H0: mu >= +m
    q = t.ppf(1.0 - alpha, df)
    return {"tost_p": float(max(p_lower, p_upper)),
            "tci_low": mean - q * se, "tci_high": mean + q * se}


def decide_superiority_or_equivalence(x: Sequence[float], margin: float,
                                      n_boot: int = 100000,
                                      seed: int = 20260910) -> Dict[str, object]:
    """Pre-registered decision for "candidate minus reference".

    IMPROVES            95% bootstrap CI of the mean > 0 and Wilcoxon p < .05
    WORSE               95% bootstrap CI of the mean < 0 and Wilcoxon p < .05
    EQUIVALENT (+-m)    90% bootstrap CI inside (-m, +m) and TOST p < .05
    INCONCLUSIVE        otherwise
    """
    s = paired_summary(x, n_boot, seed)
    e = tost(x, margin)
    verdict = "INCONCLUSIVE"
    if s.get("n", 0) >= 2:
        wp = s["wilcoxon_p"]
        if s["ci95_low"] > 0 and np.isfinite(wp) and wp < 0.05:
            verdict = "IMPROVES"
        elif s["ci95_high"] < 0 and np.isfinite(wp) and wp < 0.05:
            verdict = "WORSE"
        elif (s["ci90_low"] > -margin and s["ci90_high"] < margin
              and np.isfinite(e["tost_p"]) and e["tost_p"] < 0.05):
            verdict = f"EQUIVALENT within +-{margin:g}"
    sd = s.get("sd", float("nan"))
    return {**s, **e, "margin": margin, "verdict": verdict,
            "mde_80pct_power": mde(sd, s.get("n", 0)),
            "n_needed_for_observed_mean": required_n(sd, s.get("mean", 0.0))}


def required_n(sd: float, delta: float, alpha: float = 0.05,
               power: float = 0.80) -> float:
    """Scenarios needed to detect a mean paired difference ``delta``."""
    if not (np.isfinite(sd) and np.isfinite(delta)) or delta == 0 or sd <= 0:
        return float("nan")
    z = _norm_ppf(1 - alpha / 2) + _norm_ppf(power)
    n = (z * sd / abs(delta)) ** 2
    # One t-correction step keeps the estimate honest for small n.
    from scipy.stats import t
    for _ in range(3):
        df = max(n - 1, 1)
        zt = t.ppf(1 - alpha / 2, df) + t.ppf(power, df)
        n = (zt * sd / abs(delta)) ** 2
    return float(math.ceil(n))


def mde(sd: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable mean paired difference with n scenarios."""
    if not np.isfinite(sd) or sd <= 0 or n < 2:
        return float("nan")
    from scipy.stats import t
    df = n - 1
    return float((t.ppf(1 - alpha / 2, df) + t.ppf(power, df)) * sd
                 / math.sqrt(n))
