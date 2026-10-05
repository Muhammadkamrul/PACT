"""
intact/xapps.py
===============
Stand-in tenant xApps.  BLACK BOXES from INTACT's point of view: we see what
they write, never how they decide (v11 §3 assumptions).

Each xApp has a private objective and a private trigger condition.  They do
NOT coordinate, do NOT know about each other, and are NEVER retrained.  That
is the whole premise: INTACT must work with unmodified third-party
controllers.

The trigger matters for q_hat in H(j,k) (v11 §8.2): an xApp that only writes
when a buffer threshold is crossed cannot conflict while the buffer is low.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np


class XApp:
    """Base: a simple proportional controller toward a private target."""

    def __init__(self, name, tenant, param, domain, step, gain, cfg,
                 forecast_aware=False, forecast_gain=1.0):
        self.name, self.tenant, self.param = name, tenant, param
        self.lo, self.hi = domain
        self.step = step
        self.gain = gain
        self.cfg = cfg
        self.forecast_aware = bool(forecast_aware)
        self.forecast_gain = float(forecast_gain)

    def _future_load_factor(self, kpm) -> float:
        """Forecasted/current offered traffic, clipped for robustness.

        The xApp is allowed to consume a traffic forecast just as a real
        proactive controller can consume a planned event/diurnal forecast.
        Every scheduler sees the same proposal, so this does not give INTACT
        private future information; only MILD's *ranking* differs from B3.
        """
        if not self.forecast_aware or self.tenant not in kpm:
            return 1.0
        row = kpm[self.tenant]
        now = max(float(row.get("offered_input_ratio", 1.0)), 1e-6)
        future = float(row.get("offered_input_ratio_forecast", now))
        raw = 1.0 + self.forecast_gain * (future / now - 1.0)
        return float(np.clip(raw, 0.65, 2.0))

    def _quantise(self, v):
        v = min(max(v, self.lo), self.hi)
        return round(round((v - self.lo) / self.step) * self.step + self.lo, 6)

    def propose(self, kpm, controls, rng) -> Optional[float]:
        raise NotImplementedError


class ThroughputXApp(XApp):
    """Wants more resource for its own tenant when its throughput is low."""
    def __init__(self, *a, target_mbps, **kw):
        super().__init__(*a, **kw); self.target = target_mbps

    def propose(self, kpm, controls, rng):
        tput = kpm[self.tenant]["throughput_mbps"]
        # A higher future offered load divides the service available per bit.
        # Acting on this projected error gives the arbiter a proposal before
        # the failure; it does not disclose any future realised KPI.
        projected = tput / self._future_load_factor(kpm)
        err = self.target - projected
        if abs(err) < 0.15:                       # trigger: only act on real error
            return None
        cur = controls.get(self.param, (self.lo + self.hi) / 2)
        return self._quantise(cur + self.gain * err)


class LatencyXApp(XApp):
    """Raises its slice's scheduler weight when delay is above target."""
    def __init__(self, *a, target_ms, **kw):
        super().__init__(*a, **kw); self.target = target_ms

    def propose(self, kpm, controls, rng):
        d = kpm[self.tenant]["delay_ms"]
        projected = d * self._future_load_factor(kpm)
        err = projected - self.target
        if abs(err) < 1.0:
            return None
        cur = controls.get(self.param, 1.0)
        return self._quantise(cur + self.gain * err)


class RobustnessXApp(XApp):
    """Lowers the MCS ceiling when the buffer grows (retransmissions hurt)."""
    def __init__(self, *a, buf_threshold_kb, **kw):
        super().__init__(*a, **kw); self.thr = buf_threshold_kb

    def propose(self, kpm, controls, rng):
        buf = kpm[self.tenant]["buffer_kb"]
        projected = buf * self._future_load_factor(kpm)
        if projected < self.thr:                  # trigger: projected buffer
            return None
        cur = controls.get(self.param, 20.0)
        return self._quantise(cur - self.gain * (projected - self.thr))


class EnergyXApp(XApp):
    """
    HOST-owned, CELL-SCOPED.  Pushes cell TX power / caps down whenever
    utilisation is below its target -- which is exactly the behaviour that
    damages every tenant at once (v11 §2.1 mechanism (a)).
    """
    def __init__(self, *a, util_target_pct, target_alt=None, period=0, **kw):
        super().__init__(*a, **kw)
        self.target = util_target_pct
        self.target_alt = target_alt        # second setpoint of the duty cycle
        self.period = int(period)           # epochs per half-cycle; 0 = static
        self._t = 0

    def propose(self, kpm, controls, rng):
        # *** Do NOT advance _t here. ***  The experiment sets _t = epoch at the
        # top of every loop (see Experiment._sync_xapp_clocks).  Incrementing
        # here as well makes the phase depend on WHETHER THIS xApp WAS POLLED,
        # so an architecture that polls every claim ends one tick ahead of one
        # that polls only the selected claims -- measured as
        # value_only (151,151,151,151,151) vs b6plus (151,151,150,150,150).
        # A duty cycle is a function of time, never of who asked.
        cur = controls.get(self.param, (self.lo + self.hi) / 2)
        tgt = self.target
        if self.period > 0 and self.target_alt is not None:
            # duty cycle: the setpoint alternates, so the controller NEVER
            # converges and never stops writing
            if (self._t // self.period) % 2 == 1:
                tgt = self.target_alt
        # measure the knob THIS xApp writes -- not always TX power.  A tilt
        # controller compared against TX power sees a constant huge error,
        # saturates at its domain floor and never writes again.
        meas = controls.get(self.param, kpm["_cell"].get("txpower_dbm", 43.0))
        err = meas - tgt
        # Deadband proportional to the knob's own step, not an absolute 2.
        # A duty-cycled setpoint means the target keeps moving, so this
        # controller never converges and never falls silent -- which is what
        # keeps its knob EXCITED for beta / s estimation.
        if abs(err) < max(self.step, 1e-9):
            return None
        # move a FRACTION of the way to the target (proportional control),
        # so the knob sweeps its domain instead of slamming to a rail
        frac = min(max(self.gain, 0.05), 1.0) if self.gain <= 1.0 else 0.5
        return self._quantise(cur - frac * err)


def build_xapps(cfg) -> Dict[str, XApp]:
    """Instantiate from config.  One xApp may hold SEVERAL claims -- which is
    why the schedulable unit is the claim, not the xApp (v11 §3.1)."""
    out = {}
    for x in cfg["xapps"]:
        kind = x["kind"]
        common = dict(name=x["name"], tenant=x["tenant"], param=x["param"],
                      domain=tuple(x["domain"]), step=x["step"],
                      gain=x["gain"], cfg=cfg,
                      forecast_aware=x.get("forecast_aware", False),
                      forecast_gain=x.get("forecast_gain", 1.0))
        if kind == "throughput":
            out[x["name"]] = ThroughputXApp(**common, target_mbps=x["target"])
        elif kind == "latency":
            out[x["name"]] = LatencyXApp(**common, target_ms=x["target"])
        elif kind == "robustness":
            out[x["name"]] = RobustnessXApp(**common, buf_threshold_kb=x["target"])
        elif kind == "energy":
            out[x["name"]] = EnergyXApp(**common, util_target_pct=x["target"],
                                        target_alt=x.get("target_alt"),
                                        period=x.get("period", 0))
        else:
            raise ValueError(f"unknown xApp kind {kind}")
    return out
