"""Where the offered-load forecast comes from, and how good it is.

THE PROBLEM THIS SOLVES
-----------------------
The analytical RAN publishes, in every KPM report, the exogenous offered
load that each tenant WILL have ``risk.horizon`` epochs from now
(``offered_input_ratio_forecast`` and its mean/peak/min over the horizon).
It is read straight off the load schedule, so it is EXACT.  Four consumers
use it:

  1. forecast-aware xApps (every method, baselines included) -- they scale
     their proposal by the predicted load change;
  2. the outer-loop sensitivity blend of INTACTv2/v3
     ``s_bar = (1-p) s_current + p s_future`` -- the future regime label
     comes from this forecast;
  3. MILD's input features (endpoint / mean / peak / min demand);
  4. the ForecastRiskPredictor diagnostic (INTACTv3-F).

A perfect forecast is an assumption, not a measurement.  This module makes
that assumption an explicit, swappable component so a paper can report how
much of the result survives when the forecast is only estimated.

HOW IT WORKS
------------
``ForecastProvider.apply(kpm)`` is called once per epoch, immediately after
the RAN produces the report and before ANY consumer reads it.  It

  * records the currently observed offered load of every tenant (this is a
    measurement, always legitimate);
  * computes its own prediction for t+H from that history alone (or
    degrades the oracle value, for the ``noisy`` mode);
  * OVERWRITES the four forecast fields in the report with its prediction,
    so every consumer -- xApps, blend, MILD, INTACTv3-F -- sees the same
    forecast quality;
  * scores its prediction against the oracle value it replaced, giving a
    free, exact accuracy measurement (MAE, RMSE, bias, correlation and
    regime hit-rate) reported in the run summary.

MODES (``v2.forecast_source``)
------------------------------
``oracle``       leave the exact schedule value in place.  DEFAULT: this
                 reproduces every earlier result bit-for-bit.
``noisy``        oracle x (1 + bias + N(0, sd)), a controlled degradation
                 with ``v2.forecast_noise_sd`` and ``v2.forecast_bias``.
                 Answers "how accurate must a forecast be?"
``persistence``  predict that the load stays where it is now.  The naive
                 estimator: no anticipation at all, only measurements.
``holt``         Holt's linear (double exponential) smoothing on the
                 observed load: level + H x trend, clipped to the observed
                 range.  A real, causal estimator; it lags at turning
                 points, which is what a deployed predictor would do.
``auto``         run persistence, holt and seasonal side by side, score each
                 one against what actually happened H epochs later, and
                 publish the candidate with the lowest recent error.  This
                 is the "no assumption" setting: the forecaster is chosen
                 by measured accuracy at run time.
``seasonal``     seasonal-naive with online period detection: once at least
                 ``1 + v2.forecast_seasonal_min_cycles`` periods of history
                 exist, predict the value one period back at the same
                 phase; fall back to ``holt`` before that.  Diurnal traffic
                 is periodic, so this is the estimator an operator would
                 actually build.

Every mode is causal: it reads only the present and the past.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import numpy as np

FIELDS = ("offered_input_ratio_forecast",
          "offered_input_ratio_forecast_mean",
          "offered_input_ratio_forecast_peak",
          "offered_input_ratio_forecast_min")
MODES = ("oracle", "noisy", "persistence", "holt", "seasonal", "auto")


class _Series:
    """One tenant's observed offered-load history and its estimators."""

    def __init__(self, maxlen: int, alpha: float, beta: float):
        self.hist: Deque[float] = deque(maxlen=maxlen)
        self.alpha, self.beta = float(alpha), float(beta)
        self.level: Optional[float] = None
        self.trend: float = 0.0
        self.period: Optional[int] = None

    def observe(self, x: float) -> None:
        x = float(x)
        self.hist.append(x)
        if self.level is None:
            self.level, self.trend = x, 0.0
            return
        prev = self.level
        self.level = self.alpha * x + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend

    # -- estimators ----------------------------------------------------
    def persistence(self) -> float:
        return float(self.hist[-1]) if self.hist else 1.0

    def holt_path(self, horizon: int) -> List[float]:
        if self.level is None:
            return [1.0] * max(horizon, 1)
        lo = min(self.hist) if self.hist else self.level
        hi = max(self.hist) if self.hist else self.level
        span = max(hi - lo, 1e-6)
        lo, hi = lo - 0.25 * span, hi + 0.25 * span
        return [float(min(max(self.level + self.trend * k, lo), hi))
                for k in range(1, max(horizon, 1) + 1)]

    def detect_period(self, min_p: int, max_p: int) -> Optional[int]:
        """Cheap autocorrelation period detection on the observed history."""
        n = len(self.hist)
        if n < 2 * min_p + 4:
            return None
        x = np.asarray(self.hist, float)
        x = x - x.mean()
        if float(np.dot(x, x)) < 1e-12:
            return None
        best, best_r = None, 0.50          # require a real periodic signal
        for lag in range(min_p, min(max_p, n // 2) + 1):
            a, b = x[:-lag], x[lag:]
            den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
            if den < 1e-12:
                continue
            r = float(np.dot(a, b)) / den
            if r > best_r:
                best, best_r = lag, r
        return best

    def seasonal_path(self, horizon: int, min_p: int, max_p: int,
                      min_cycles: float) -> Optional[List[float]]:
        if self.period is None or len(self.hist) % max(min_p // 2, 1) == 0:
            p = self.detect_period(min_p, max_p)
            if p is not None:
                self.period = p
        p = self.period
        if p is None or len(self.hist) < min_cycles * p + horizon:
            return None
        h = list(self.hist)
        out = []
        for k in range(1, max(horizon, 1) + 1):
            # value one full period before the target time t+k
            idx = len(h) - p + k
            while idx >= len(h):          # more than one period ahead
                idx -= p
            out.append(float(h[max(idx, 0)]))
        return out


class ForecastProvider:
    """Swaps the exact schedule forecast for an estimated or noisy one."""

    def __init__(self, cfg: Dict, log=None):
        v2 = (cfg.get("v2", {}) or {})
        self.mode = str(v2.get("forecast_source", "oracle")).lower()
        if self.mode not in MODES:
            raise ValueError(f"v2.forecast_source must be one of {MODES}")
        self.noise_sd = float(v2.get("forecast_noise_sd", 0.0))
        self.bias = float(v2.get("forecast_bias", 0.0))
        self.alpha = float(v2.get("forecast_holt_alpha", 0.25))
        self.beta = float(v2.get("forecast_holt_beta", 0.10))
        self.min_cycles = float(v2.get("forecast_seasonal_min_cycles", 1.0))
        ran = cfg.get("ran", {}) or {}
        spe = max(int(ran.get("slots_per_epoch", 1)), 1)
        risk = cfg.get("risk", {}) or {}
        self.horizon = int(risk.get(
            "horizon_epochs",
            max(1, int(math.ceil(float(risk.get("horizon_slots", 16)) / spe)))))
        # search band for the period, in epochs
        self.min_period = int(v2.get("forecast_seasonal_min_period",
                                     max(4, self.horizon)))
        self.max_period = int(v2.get("forecast_seasonal_max_period", 2000))
        self.maxlen = int(v2.get("forecast_history_epochs",
                                 max(4 * self.max_period, 4096)))
        seed = int(v2.get("forecast_seed", 20260912)) ^ int(ran.get("seed", 0))
        self.rng = np.random.default_rng(seed & 0xFFFFFFFF)
        # NOTE: a plain dict, not defaultdict(lambda: ...).  Some run
        # folders pickle the experiment for checkpointing, and a lambda
        # defined in __init__ is not picklable.
        self.series: Dict[str, _Series] = {}
        self.err: Dict[str, list] = defaultdict(list)
        self.pending: Dict[str, Dict[str, Deque]] = defaultdict(dict)
        self.cand_err: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.chosen_count: Dict[str, int] = defaultdict(int)
        self.pred_hist: List[float] = []
        self.true_hist: List[float] = []
        self.n_epochs = 0
        self.edges = list((cfg.get("v2", {}) or {}).get(
            "regime_load_edges", [0.9, 1.25]))
        if log is not None and self.mode != "oracle":
            log.info("forecast source: %s (the exact schedule forecast is "
                     "replaced for EVERY consumer)", self.mode)

    # ------------------------------------------------------------------
    def _regime(self, load: float) -> int:
        return min(sum(float(load) >= float(e) for e in self.edges),
                   len(self.edges))

    def _candidate_paths(self, key: str, now: float) -> Dict[str, List[float]]:
        s = self._series_for(key)
        out = {"persistence": [now] * self.horizon, "holt": s.holt_path(self.horizon)}
        seas = s.seasonal_path(self.horizon, self.min_period, self.max_period,
                               self.min_cycles)
        if seas is not None:
            out["seasonal"] = seas
        return out

    def _score_pending(self, key: str, now: float) -> None:
        """Compare predictions made H epochs ago with what actually happened."""
        pend = self.pending[key]
        for cand, dq in pend.items():
            keep = deque()
            for k_left, pred in dq:
                if k_left <= 1:
                    e = abs(pred - now)
                    prev = self.cand_err[key].get(cand)
                    self.cand_err[key][cand] = (e if prev is None
                                                else 0.9 * prev + 0.1 * e)
                else:
                    keep.append((k_left - 1, pred))
            pend[cand] = keep

    def _series_for(self, key: str) -> _Series:
        s = self.series.get(key)
        if s is None:
            s = _Series(self.maxlen, self.alpha, self.beta)
            self.series[key] = s
        return s

    def _predict(self, key: str, row: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Return replacement values, or None to keep the oracle values."""
        s = self._series_for(key)
        now = float(row.get("offered_input_ratio", 1.0))
        s.observe(now)
        if self.mode == "oracle":
            return None
        if self.mode == "noisy":
            eps = float(self.rng.normal(self.bias, self.noise_sd))
            return {f: max(0.0, float(row.get(f, now)) * (1.0 + eps))
                    for f in FIELDS if f in row}
        cands = self._candidate_paths(key, now)
        if self.mode == "auto":
            self._score_pending(key, now)
            for cand, path in cands.items():
                self.pending[key].setdefault(cand, deque())
                self.pending[key][cand].append((self.horizon, float(path[-1])))
            scored = {c: e for c, e in self.cand_err[key].items() if c in cands}
            chosen = (min(scored, key=scored.get) if scored else "holt")
            self.chosen_count[chosen] += 1
            path = cands[chosen]
        elif self.mode == "persistence":
            path = cands["persistence"]
        elif self.mode == "seasonal":
            path = cands.get("seasonal", cands["holt"])
            self.chosen_count["seasonal" if "seasonal" in cands else "holt"] += 1
        else:
            path = cands["holt"]
        arr = np.asarray(path, float)
        return {"offered_input_ratio_forecast": float(arr[-1]),
                "offered_input_ratio_forecast_mean": float(arr.mean()),
                "offered_input_ratio_forecast_peak": float(arr.max()),
                "offered_input_ratio_forecast_min": float(arr.min())}

    def apply(self, kpm: Dict[str, Dict[str, float]]
              ) -> Dict[str, Dict[str, float]]:
        """Rewrite the forecast fields in place and score the prediction."""
        self.n_epochs += 1
        for key, row in kpm.items():
            if not isinstance(row, dict) or "offered_input_ratio" not in row:
                continue
            truth = float(row.get("offered_input_ratio_forecast",
                                  row.get("offered_input_ratio", 1.0)))
            new = self._predict(key, row)
            if new is None:
                continue
            row.update(new)
            pred = float(new["offered_input_ratio_forecast"])
            self.err[key].append(pred - truth)
            if key != "_cell":
                self.pred_hist.append(pred)
                self.true_hist.append(truth)
        return kpm

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        out = {"forecast_source": self.mode,
               "forecast_horizon_epochs": float(self.horizon)}
        if self.mode == "oracle" or not self.true_hist:
            out.update({"forecast_mae": 0.0, "forecast_rmse": 0.0,
                        "forecast_bias": 0.0, "forecast_corr": 1.0,
                        "forecast_regime_hit_rate": 1.0,
                        "forecast_mape_pct": 0.0})
            return out
        p = np.asarray(self.pred_hist, float)
        t = np.asarray(self.true_hist, float)
        e = p - t
        hit = np.mean([self._regime(a) == self._regime(b) for a, b in zip(p, t)])
        corr = (float(np.corrcoef(p, t)[0, 1])
                if p.std() > 1e-12 and t.std() > 1e-12 else float("nan"))
        out.update({
            "forecast_mae": float(np.abs(e).mean()),
            "forecast_rmse": float(np.sqrt((e ** 2).mean())),
            "forecast_bias": float(e.mean()),
            "forecast_mape_pct": float(100 * np.abs(e / np.maximum(t, 1e-9)).mean()),
            "forecast_corr": corr,
            "forecast_regime_hit_rate": float(hit)})
        tot = sum(self.chosen_count.values())
        for cand in ("persistence", "holt", "seasonal"):
            out[f"forecast_share_{cand}"] = (self.chosen_count.get(cand, 0) / tot
                                             if tot else 0.0)
        return out
