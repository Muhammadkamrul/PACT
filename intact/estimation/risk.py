"""
intact/estimation/risk.py   --  INTACT v11 §7.1 (L1), §10.2
===========================================================
p_hat_i = P(intent i breaches its fulfilment requirement within H slots).

WHY THIS IS THE LABELABLE PROBLEM (and attribution is not):
You can replay a trace, see whether rho_i dropped below eta_i, and stamp
the window that preceded it.  That is a supervised learning problem with
real labels.  "WHICH xApp caused it" is NOT -- a trace records one world,
and you would need a world where that xApp did not act.  That is why the
framework uses RANDOMISED EFFECT ESTIMATION for attribution (effects.py)
and supervised prediction only here.

THE IMPLEMENTATION HERE is an analytical stand-in with the same INTERFACE
as a trained MILD model: margin level + trend + volatility -> probability.
When you port MILD, replace `predict` and change nothing else.

CRITICAL CONSTRAINT (v11 §10.2):
    H must exceed the fastest observed erosion time from tau_risk to breach.
    Otherwise a sequence of individually-safe writes walks an intent into
    failure before the gate ever fires.  config.py checks this.
"""
from __future__ import annotations
import math
from typing import Dict
import numpy as np

from .margins import MarginTracker


class RiskPredictor:
    """Interface-compatible stand-in for the MILD intent-failure predictor."""

    def __init__(self, cfg: Dict):
        self.source = "analytical"
        # predict() is called once per scheduler epoch and tracker.trend() is
        # therefore a change *per epoch*.  Multiplying it by horizon_slots was
        # a unit error whenever one epoch contained more than one radio slot.
        sample_slots = (int(cfg["ran"].get("pre_slots", cfg["ran"]["slots_per_epoch"]))
                        + int(cfg["ran"].get("post_slots", cfg["ran"]["slots_per_epoch"])))
        self.h = int(cfg["risk"].get(
            "horizon_epochs",
            max(1, int(math.ceil(cfg["risk"]["horizon_slots"] /
                                 max(sample_slots, 1))))))
        self.k = cfg["risk"]["logistic_gain"]
        self.trend_w = cfg["risk"]["trend_weight"]

    def predict(self, tracker: MarginTracker) -> Dict[str, float]:
        """
        Returns p_hat in [0,1] for every intent.

        Model: project the margin forward by H slots using the observed
        trend, then squash the projected shortfall through a logistic.
        A margin that is comfortable but FALLING gets a high risk BEFORE it
        crosses zero -- which is exactly the ratchet protection.
        """
        out = {}
        for iid in tracker.intents:
            h = list(tracker.hist[iid])
            if not h:
                out[iid] = 0.0
                continue
            g_now = h[-1]
            trend = tracker.trend(iid)
            # projected margin at the end of the horizon
            g_proj = g_now + self.trend_w * trend * self.h
            # volatility widens the risk: a noisy margin near zero is riskier
            vol = float(np.std(h[-min(len(h), 16):])) if len(h) > 2 else 0.0
            z = -(g_proj) / max(vol + 0.02, 1e-3)
            out[iid] = float(1.0 / (1.0 + math.exp(-self.k * z)))
        return out


class ForecastRiskPredictor(RiskPredictor):
    """Transparent forecast-risk diagnostic used to falsify placement.

    It sees only the current margin history and an exogenous offered-traffic
    forecast supplied in the current KPM.  It never reads future KPI values,
    fading, actions, or realised margins.  This is intentionally simpler than
    MILD: if even this high-quality forecast cannot improve an arbitration
    challenge, the problem is the *decision placement*, not MILD training.
    """

    def __init__(self, cfg: Dict, intents):
        super().__init__(cfg)
        self.source = "forecast_diagnostic"
        self.intents = intents
        self.last_kpm = None
        r = cfg.get("risk", {}) or {}
        self.forecast_gain = float(r.get("forecast_logistic_gain", 7.0))
        self.forecast_center = float(r.get("forecast_load_center", 1.05))
        self.delta_gain = float(r.get("forecast_delta_gain", 5.0))

    def observe(self, epoch, g, rho, kpm, controls):
        self.last_kpm = kpm

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-60.0, min(60.0, float(x)))
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, tracker: MarginTracker) -> Dict[str, float]:
        base = super().predict(tracker)
        if not self.last_kpm:
            return base
        out = {}
        for iid, intent in self.intents.items():
            row = self.last_kpm.get(intent.tenant, {})
            now = float(row.get("offered_input_ratio", 1.0))
            future = float(row.get("offered_input_ratio_forecast", now))
            p_level = self._sigmoid(
                self.forecast_gain * (future - self.forecast_center))
            p_rise = self._sigmoid(self.delta_gain * (future - now) - 1.0)
            # Noisy-OR: either an eroding margin or a forecast load shock is
            # enough to raise risk.  Host intents do not belong to a traffic
            # slice and retain the ordinary analytical risk.
            if intent.tenant in self.last_kpm and intent.tenant != "H":
                p_forecast = p_level * p_rise
                out[iid] = float(1.0 - (1.0 - base[iid]) * (1.0 - p_forecast))
            else:
                out[iid] = float(base[iid])
        return out


class ZeroRiskPredictor:
    """Registered ablation: p_hat = 0 for every intent at every epoch.

    With this predictor INTACTv3 receives NO risk information at all:
    the (1 + g*p) primary weight is 1, the pre-failure reserve is 0, the
    forecast-sensitivity blend reduces to the current slope, the urgency
    multiplier u(p) is 1, eps_eff equals eps and no intent ever reaches
    tau_risk, so the joint protection check never fires.  It is therefore
    the clean "INTACTv3 minus urgency/risk" ablation.  Select it with
    ``risk.predictor: zero`` and ``mild.enabled: false``.
    """

    def __init__(self, cfg: Dict, intents=None):
        self.source = "zero"
        self.intents = intents

    def predict(self, tracker: MarginTracker) -> Dict[str, float]:
        return {iid: 0.0 for iid in tracker.intents}


class ConstantRiskPredictor:
    """Registered ablation: p_hat = c for every intent at every epoch.

    A constant probability switches ON every risk-driven lever of
    INTACTv3 (forecast-sensitivity blend, pre-failure reserve, weight
    scaling, and -- when c >= tau_risk -- the joint protection check) but
    carries NO information about WHICH intent is at risk.  Comparing it
    with the analytical predictor and with MILD separates "anticipation is
    switched on" from "risk is discriminated correctly".  Select it with
    ``risk.predictor: constant`` and ``risk.constant_p: <c>``.
    """

    def __init__(self, cfg: Dict, intents=None):
        self.source = "constant"
        self.intents = intents
        self.c = float(min(max(float(
            (cfg.get("risk", {}) or {}).get("constant_p", 0.5)), 0.0), 1.0))

    def predict(self, tracker: MarginTracker) -> Dict[str, float]:
        return {iid: self.c for iid in tracker.intents}


def make_risk_predictor(cfg, intents, tenants, claims, log):
    """
    Factory.  ONE config flag decides which predictor INTACT uses, and
    nothing else in the framework changes -- both expose
    `predict(tracker) -> {iid: p_hat}`.

        mild.enabled: false  -> the analytical stand-in above
        mild.enabled: true   -> the trained MILD mixture-of-experts

    A trained model that has never seen the current scenario is WORSE than
    the stand-in.  Therefore ``mild.enabled: true`` is a strict request: a
    missing or mismatched model fails fast instead of silently changing the
    experiment back to the analytical predictor.
    """
    if not cfg.get("mild", {}).get("enabled", False):
        kind = str(cfg.get("risk", {}).get("predictor", "analytical"))
        if kind == "analytical":
            log.info("risk predictor: ANALYTICAL stand-in (mild.enabled = false)")
            return RiskPredictor(cfg)
        if kind == "forecast_diagnostic":
            log.info("risk predictor: EXOGENOUS-FORECAST diagnostic "
                     "(not trained MILD)")
            return ForecastRiskPredictor(cfg, intents)
        if kind == "zero":
            log.info("risk predictor: ZERO (ablation: no risk information)")
            return ZeroRiskPredictor(cfg, intents)
        if kind == "constant":
            log.info("risk predictor: CONSTANT p=%s (ablation: anticipation "
                     "without discrimination)",
                     (cfg.get("risk", {}) or {}).get("constant_p", 0.5))
            return ConstantRiskPredictor(cfg, intents)
        raise ValueError("risk.predictor must be analytical, "
                         "forecast_diagnostic, zero or constant when "
                         "mild.enabled=false")
    from pathlib import Path
    d = Path(cfg["mild"]["model_dir"])
    if not (d / "meta.json").exists():
        raise FileNotFoundError(
            f"mild.enabled is true but no trained model exists at {d}; "
            "run scripts/train_mild.py for this exact scenario first")
    from ..mild.predictor import MildRiskPredictor
    log.info("risk predictor: TRAINED MILD from %s", d)
    return MildRiskPredictor(str(d), cfg, intents, tenants, claims, log)
