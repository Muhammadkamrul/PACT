"""
intact/ran/analytic.py
======================
A multi-slice analytical RAN model.

WHAT IT IS FOR
--------------
INTACT does not need a bit-accurate PHY.  It needs a RAN in which
(i)  control knobs move intent margins in a measurable way, and
(ii) SOME of those movements cross tenant boundaries.
Without (ii) there is no problem to solve, so the three coupling mechanisms
of v11 §2.1 are modelled EXPLICITLY and can each be switched off to prove
they are what generates the cross-tenant beta:

  (a) CELL-SCOPED CONTROL.  There is no per-slice transmit power in a shared
      cell -- the amplifier is shared.  Lowering cell TX power drops every
      tenant's SINR at unchanged PRB entitlement.        [coupling_txpower]

  (b) QUANTITY != QUALITY.  A PRB quota guarantees a COUNT, not an SINR.
      A slice whose scheduler weight rises steers its UEs onto better
      subbands, leaving worse ones for everyone else.  Modelled as an
      explicitly ZERO-SUM quality factor.                [coupling_subband]

  (c) SHARED PHY/PROCESSING.  An MCS ceiling set above what the channel
      supports causes retransmissions, and those retransmissions consume
      PRBs from the shared pool that other slices were relying on.
                                                          [coupling_retx]

TIME
----
slot  = ran.slot_ms          (one KPM reporting period; the finest thing observed)
epoch = ran.slots_per_epoch slots  (how often the outer loop redecides)
"""
from __future__ import annotations
import math
from typing import Dict, List
import numpy as np

from .base import RANBackend


def _mcs_to_se(mcs: float) -> float:
    """Very rough MCS index -> spectral-efficiency ceiling (bit/s/Hz).
    Index 0 ~ QPSK 1/8, index 28 ~ 256QAM 0.93."""
    return 0.15 + (mcs / 28.0) ** 1.35 * 7.2


class AnalyticRAN(RANBackend):
    def __init__(self, cfg: Dict, tenants: List[str]):
        # Keep the complete configuration.  v2 interaction and workload
        # specifications live beside ``ran`` rather than inside it.
        self.full_cfg = cfg
        self.cfg = cfg["ran"]
        self.tenants = tenants
        self.n_prb = self.cfg["n_prb"]
        self.w_prb_hz = self.cfg["prb_bandwidth_hz"]
        self.slot_s = self.cfg["slot_ms"] / 1000.0

        # coupling switches: set any to false for the ablation that proves
        # the cross-tenant effects come from these mechanisms and nowhere else
        self.k_tx = self.cfg.get("coupling_txpower", 1.0)
        self.k_sb = self.cfg.get("coupling_subband", 1.0)
        self.k_rx = self.cfg.get("coupling_retx", 1.0)
        self.f_ghz = self.cfg.get("carrier_ghz", 3.5)
        self.base_delay_ms = self.cfg.get("base_delay_ms", 4.0)
        self.max_queue_bits = self.cfg.get("max_queue_kb", 120.0) * 8e3
        # reference per-PRB TX power, used by the coupling-(a) switch
        self.k_cio = self.cfg.get("cio_load_gain", 0.06)
        self.k_tilt = self.cfg.get("tilt_penalty_db", 9.0)
        self.tilt_ref_m = self.cfg.get("tilt_ref_m", 200.0)
        self.tilt_nominal = self.cfg.get("tilt_nominal", 6.0)
        self.tx_nominal = (self.cfg["initial_controls"].get("txpower", 43.0)
                           - 10 * math.log10(self.cfg["n_prb"]))

        self.slices = {t: self.cfg["slices"][t] for t in tenants
                       if t in self.cfg["slices"]}
        # These declarations used to be generated into every v2 scenario but
        # were never connected to AnalyticRAN.  Loading them in the common
        # backend is essential: otherwise v1 baselines and v2 candidates see
        # different physics, or (as in the previous release) nobody sees the
        # declared indirect-conflict physics at all.
        from ..v2.ran_v2 import load_dependencies, load_interactions
        self.interactions = load_interactions(cfg)
        self.dependencies = load_dependencies(cfg)
        self._binding_edges = []
        self._intent_specs = {}
        for it in cfg.get("intents", []) or []:
            self._intent_specs.setdefault((it.get("tenant"), it.get("kpi")), []).append(it)
        self.rng = np.random.default_rng(0)
        self._controls: Dict[str, float] = {}
        self._command_controls: Dict[str, float] = {}
        self.reset(seed=self.cfg.get("seed", 0))

    # ------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.ue = {}          # per-tenant UE state
        for tid, sl in self.slices.items():
            n = sl["n_ue"]
            self.ue[tid] = {
                # distance drives large-scale path loss
                "dist_m": self.rng.uniform(80, 380, n),
                # log-normal shadowing, fixed per UE per run
                "shadow_db": self.rng.normal(0, 6.0, n),
                # queue backlog in bits
                "queue_bits": np.zeros(n),
                # offered load per UE (Mb/s)
                "load_mbps": np.full(n, sl["load_mbps_per_ue"]),
            }
        # initial control values = the "no intervention" defaults
        self._controls = dict(self.cfg["initial_controls"])
        self._command_controls = dict(self._controls)

    # ------------------------------------------------------------------
    def _load_multiplier(self, tenant: str, slot: int | None = None) -> float:
        """Return the exogenous offered-load multiplier for one slot.

        The original simulator's offered load was constant for the complete
        experiment.  Once an xApp found a useful control value, it normally
        became silent forever; the benchmark consequently rewarded policies
        that barely acted.  A configured profile supplies reproducible load
        regimes without making traffic depend on the scheduler under test.

        Example with ``levels=[0.9, 1.2, 1.5]`` and ``hold_slots=160``:
        every 160 slots the whole cell moves to the next load level.  Optional
        tenant phases produce heterogeneous slice shocks while remaining
        deterministic and identical across methods.
        """
        p = self.cfg.get("load_profile", {}) or {}
        if not p.get("enabled", False):
            return 1.0
        levels = [float(x) for x in p.get("levels", [1.0])]
        if not levels:
            return 1.0
        hold = max(int(p.get("hold_slots", 160)), 1)
        now = self.t if slot is None else int(slot)
        phase = 0
        if p.get("per_tenant_phase", False):
            order = sorted(self.slices)
            phase = order.index(tenant) * int(p.get("tenant_phase_stride", 1))
        absolute = max(now - 1, 0)
        index = ((absolute // hold) + phase) % len(levels)
        value = levels[index]
        # Forecastable ramps create a genuine lead-time problem: a reactive
        # B3 sees only the current marginal value, whereas MILD can reserve
        # headroom before the next high-load regime.  The schedule is still
        # exogenous and identical for every method.
        ramp = min(max(int(p.get("ramp_slots", 0)), 0), hold)
        pos = absolute % hold
        if ramp > 0 and pos >= hold - ramp:
            nxt = levels[(index + 1) % len(levels)]
            frac = (pos - (hold - ramp) + 1) / float(ramp)
            value = (1.0 - frac) * levels[index] + frac * nxt
        return max(float(value), 0.0)

    def current_offered_load_mbps(self) -> float:
        """Mean current exogenous load, used by the pre-treatment context."""
        vals = []
        for tenant, st in self.ue.items():
            vals.extend((np.asarray(st["load_mbps"], dtype=float)
                         * self._load_multiplier(tenant)).tolist())
        return float(np.mean(vals)) if vals else 0.0

    def baseline_offered_load_mbps(self) -> float:
        vals = []
        for st in self.ue.values():
            vals.extend(np.asarray(st["load_mbps"], dtype=float).tolist())
        return float(np.mean(vals)) if vals else 1.0

    def forecast_offered_load_ratio(self, tenant: str, horizon_slots: int) -> float:
        """Exogenous traffic forecast exposed to the Non-RT/RIC layer.

        In the analytical challenge the load schedule represents a forecastable
        diurnal/event pattern, not clairvoyance about fading or scheduler
        outcomes.  The forecast is deliberately limited to offered input; it
        contains no future KPI, margin, action, or random-noise information.
        """
        return float(self._load_multiplier(tenant, self.t + int(horizon_slots)))

    def forecast_offered_load_summary(self, tenant: str, horizon_slots: int,
                                      points: int = 21) -> Dict[str, float]:
        """Summarise exogenous demand over the forecast horizon.

        This is schedule knowledge only: it contains no future KPI, fading,
        controller action, or realised margin.
        """
        offsets = np.linspace(0, max(int(horizon_slots), 1),
                              max(int(points), 2)).astype(int)
        values = np.asarray([
            self._load_multiplier(tenant, self.t + int(q)) for q in offsets
        ], dtype=float)
        return {"mean": float(values.mean()), "peak": float(values.max()),
                "minimum": float(values.min())}

    def _effective_controls(self) -> Dict[str, float]:
        """Apply declared RCP dependency edges to the physical controls."""
        eff = dict(self._controls)
        binding = []
        for edge in self.dependencies:
            if edge.source not in eff or edge.target not in eff:
                continue
            value, is_binding = edge.apply(eff)
            eff[edge.target] = float(value)
            if is_binding:
                binding.append((edge.source, edge.target))
        self._binding_edges = binding
        return eff

    def binding_edge_count(self) -> int:
        return len(self._binding_edges)

    # ------------------------------------------------------------------
    def clone(self):
        """
        Exact copy INCLUDING the RNG state, so two clones advanced by the
        same number of slots see the IDENTICAL fading and traffic draw.
        That is what makes a paired counterfactual a counterfactual rather
        than two different worlds.
        """
        import copy as _c
        return _c.deepcopy(self)

    def current_controls(self) -> Dict[str, float]:
        return dict(self._controls)

    def commanded_controls(self) -> Dict[str, float]:
        """Last requested setpoints (separate from physical readback)."""
        return dict(self._command_controls)

    def apply(self, param: str, value: float) -> None:
        value = float(value)
        tau = float(self.cfg.get("actuation_time_constant_epochs", 0.0))
        self._command_controls[param] = value
        if tau <= 0.0:
            self._controls[param] = value

    def _advance_actuators(self, n_slots: int) -> None:
        """First-order actuator response, disabled unless explicitly set.

        Slow power/tilt/resource-policy reconfiguration is the scoped reason
        a 20-epoch failure forecast can matter.  It is not enabled in legacy
        scenarios, so their numerical results remain reproducible.
        """
        tau_epochs = float(self.cfg.get("actuation_time_constant_epochs", 0.0))
        if tau_epochs <= 0.0:
            return
        epoch_slots = (int(self.cfg.get("pre_slots", self.cfg["slots_per_epoch"]))
                       + int(self.cfg.get("post_slots", self.cfg["slots_per_epoch"])))
        frac = 1.0 - math.exp(-max(int(n_slots), 0) /
                              max(tau_epochs * epoch_slots, 1e-9))
        for p, target in self._command_controls.items():
            old = float(self._controls.get(p, target))
            self._controls[p] = old + frac * (float(target) - old)

    # ------------------------------------------------------------------
    def headroom(self, tenant: str, resource: str) -> float:
        """
        PRBs still available inside this tenant's envelope, i.e. the envelope
        minus what its OTHER allocative claims currently hold.  This is what
        F_feasible is tested against in the inner loop (v11 §10.3).
        """
        env = self.cfg["envelopes"].get(tenant, {}).get(resource, float("inf"))
        pre = f"quota_{tenant}"
        used = sum(v for k, v in self._controls.items()
                   if k == pre or (k.startswith(pre) and not k[len(pre):][:1].isdigit()))
        # the claim being tested will replace its own current value, so we
        # report the envelope minus the OTHER quota holders
        return env, used

    # ------------------------------------------------------------------
    def _instantaneous(self, controls: Dict[str, float] | None = None
                       ) -> Dict[str, Dict[str, np.ndarray]]:
        """Per-UE SINR and spectral efficiency for the current slot."""
        c = self._controls if controls is None else controls
        tx_dbm = c.get("txpower", 43.0)
        noise_dbm = self.cfg["noise_dbm"]
        inter_dbm = self.cfg["intercell_interf_dbm"]

        # ---- ANTENNA TILT (cell-scoped, host-owned) ----------------------
        # A single electrical tilt cannot be optimal for every UE: tilting
        # down helps near UEs and starves far ones.  Modelled as a
        # distance-dependent penalty that is zero at the matched range and
        # grows quadratically away from it.  This is a genuine cross-tenant
        # externality with a DIFFERENT shape from TX power -- power shifts
        # everyone the same way, tilt shifts near and far UEs OPPOSITELY.
        tilt = c.get("tilt", self.cfg.get("tilt_nominal", 6.0))

        # ---- (b) zero-sum subband quality from scheduler weights ---------
        # q_n rises for a slice whose weight rises, and is renormalised so
        # the TOTAL quality across the cell is conserved.  That is what makes
        # this a genuine cross-tenant externality rather than free lunch.
        w = {t: max(c.get(f"schedw_{t}", 1.0), 1e-6) for t in self.slices}
        n_ue = {t: self.slices[t]["n_ue"] for t in self.slices}
        raw = {t: w[t] ** (0.5 * self.k_sb) for t in self.slices}
        tot_ue = sum(n_ue.values())
        norm = sum(raw[t] * n_ue[t] for t in self.slices) / max(tot_ue, 1)
        q = {t: raw[t] / max(norm, 1e-9) for t in self.slices}

        out = {}
        for tid, st in self.ue.items():
            # ---- link budget -----------------------------------------
            # 3GPP UMi-NLOS style path loss, including the carrier term.
            pl_db = (32.4 + 21 * np.log10(np.maximum(st["dist_m"], 10))
                     + 20 * np.log10(self.f_ghz) + st["shadow_db"])
            # small-scale Rayleigh fading, resampled every slot
            fade_db = 10 * np.log10(
                np.maximum(self.rng.exponential(1.0, len(st["dist_m"])), 1e-3))

            # TX power is the CELL total: spread it over the PRBs.  Without
            # this the "power" knob is meaningless, because a per-PRB budget
            # is what actually reaches a UE.
            tx_per_prb = tx_dbm - 10 * np.log10(self.n_prb)

            # coupling switch (a): when k_tx = 0 the cell power knob is
            # decoupled from SINR, so the host's cell-scoped claim can no
            # longer hurt anybody.  That is the ablation.
            tx_eff = self.tx_nominal + self.k_tx * (tx_per_prb - self.tx_nominal)

            # tilt penalty: matched range moves inward as tilt increases
            d_match = self.tilt_ref_m * (self.tilt_nominal / max(tilt, 0.5))
            tilt_pen = self.k_tilt * ((np.log10(np.maximum(st["dist_m"], 10))
                                       - np.log10(d_match)) ** 2)
            rx_dbm = tx_eff - pl_db - tilt_pen + fade_db
            interf_lin = 10 ** (inter_dbm / 10) + 10 ** (noise_dbm / 10)
            sinr_lin = np.maximum(10 ** (rx_dbm / 10) / interf_lin * q[tid], 1e-4)

            se_shannon = np.log2(1 + sinr_lin)
            se_cap = _mcs_to_se(c.get(f"mcs_{tid}", 20.0))
            se = np.minimum(se_shannon, se_cap)

            # ---- (c) retransmissions when the MCS ceiling over-reaches ----
            # If the ceiling permits more than the channel supports, blocks
            # fail and must be resent.  Those resends eat shared PRBs.
            over = np.maximum(se_cap - se_shannon, 0.0)
            bler = np.clip(1 - np.exp(-0.55 * over * self.k_rx), 0.0, 0.75)

            out[tid] = {"se": se, "bler": bler, "sinr": sinr_lin}
        return out

    # ------------------------------------------------------------------
    def step(self, n_slots: int) -> Dict[str, Dict[str, float]]:
        """Advance n_slots and return averaged KPMs per tenant plus _cell."""
        self._advance_actuators(n_slots)
        c = self._effective_controls()
        acc = {t: {"tput": [], "delay": [], "buf": [], "deliv": []} for t in self.slices}
        prb_used_hist, retx_hist, demand_hist = [], [], []
        offered_input_hist, budget_scale_hist = [], []

        for _ in range(n_slots):
            self.t += 1
            inst = self._instantaneous(c)

            # ---- PASS 1: requested PRB allocation ---------------------
            # First compute every slice's desired allocation.  The previous
            # implementation computed and REPORTED service before discovering
            # that aggregate demand exceeded n_prb; it scaled only the cell
            # accounting counters and added a queue penalty afterward.  That
            # allowed reported throughput to exceed the physical allocation.
            plans = {}
            total_requested, total_retx_requested = 0.0, 0.0
            for tid, st in self.ue.items():
                # A tenant may hold SEVERAL allocative knobs on the same
                # resource (e.g. quota_T1 and quota_T1b).  They SUM.  That is
                # exactly why C2 is a sum over co-eligible claims and why a
                # per-parameter envelope would double-count (v11 §8.3.1).
                # exact tenant match: "quota_T1" must NOT capture "quota_T10".
                # sub-slice knobs are "quota_T1b", so accept a non-digit suffix.
                pre = f"quota_{tid}"
                qk = [v for k, v in c.items()
                      if k == pre or (k.startswith(pre) and not k[len(pre):].isdigit()
                                      and not k[len(pre):][:1].isdigit())]
                quota = sum(qk) if qk else self.n_prb / max(len(self.slices), 1)
                cap = c.get(f"prbcap_{tid}", self.n_prb)

                se, bler = inst[tid]["se"], inst[tid]["bler"]
                # SCHEDULING POLICY (slice-scoped): the PF exponent.
                #   0.0 = round robin (equal PRBs, fairness first)
                #   0.5 = proportional fair
                #   1.0 = max-CQI (throughput first, starves cell edge)
                # Changes WHO inside the slice gets served, so it moves a
                # slice's own tail latency without touching its PRB count.
                pol = c.get(f"schedpol_{tid}", 0.5)
                pfw = np.maximum(se, 1e-3) ** float(pol)
                share = pfw / pfw.sum()
                # prbcap_<tid> is declared and claimed as a SLICE cap, but
                # was compared against share*quota, which is PER-UE PRBs.  A
                # cap of 20-100 therefore never bound anything and the whole
                # prbcap lever was inert.  Cap the SLICE total, then split.
                eff_quota = min(float(quota), float(cap))
                prb_share = share * eff_quota
                # A UE cannot use more PRBs than its backlog needs.  Without
                # this the cell always reports full quota utilisation and the
                # host's PRB-utilisation intent could never move.
                cio_pk = c.get(f"cio_{tid}", 0.0)
                traffic_mult = self._load_multiplier(tid)
                arrivals_pk = (st["load_mbps"] * traffic_mult
                               * (1.0 + self.k_cio * cio_pk)
                               * 1e6 * self.slot_s)
                need_bits = st["queue_bits"] + arrivals_pk
                per_prb_bits = np.maximum(self.w_prb_hz * se * (1 - bler) * self.slot_s, 1.0)
                prb_needed = need_bits / per_prb_bits
                prb_requested = np.minimum(prb_share, prb_needed)
                total_requested += float(prb_requested.sum())

                # retransmission overhead consumes EXTRA shared PRBs
                retx_requested = float((
                    prb_requested * bler / np.maximum(1 - bler, 0.05)).sum())
                total_retx_requested += retx_requested
                plans[tid] = {
                    "se": se, "bler": bler,
                    "prb_requested": prb_requested,
                    "arrivals": arrivals_pk,
                    "retx_requested": retx_requested,
                }

            # ---- CELL BUDGET: scale before computing any service -------
            demand = total_requested + total_retx_requested
            scale = 1.0 if demand <= self.n_prb else self.n_prb / demand
            total_used = total_requested * scale
            total_retx = total_retx_requested * scale

            # ---- PASS 2: service, queueing and KPMs from ACTUAL PRBs ----
            for tid, st in self.ue.items():
                plan = plans[tid]
                se, bler = plan["se"], plan["bler"]
                prb = plan["prb_requested"] * scale
                cap_bps = prb * self.w_prb_hz * se * (1 - bler)
                arrivals = plan["arrivals"]
                backlog = st["queue_bits"] + arrivals
                served_bits = np.minimum(cap_bps * self.slot_s, backlog)
                # Finite buffer: real queues do not grow without bound, they
                # DROP.  Without this, delay diverges and the latency intent
                # becomes meaningless rather than merely stressed.
                q = np.maximum(backlog - served_bits, 0.0)
                dropped = np.maximum(q - self.max_queue_bits, 0.0)
                st["queue_bits"] = np.minimum(q, self.max_queue_bits)
                st["dropped_bits"] = st.get("dropped_bits", 0.0) + dropped
                served_mbps = served_bits / self.slot_s / 1e6

                # delay: queueing (Little) + a fixed processing floor, so a
                # healthy slice reports a small non-zero latency
                delay_ms = (self.base_delay_ms
                            + 1000.0 * st["queue_bits"] / np.maximum(cap_bps, 1e3))

                acc[tid]["tput"].append(float(np.mean(served_mbps)))
                acc[tid]["delay"].append(float(np.mean(delay_ms)))
                # DELIVERY RATIO: what fraction of offered traffic actually
                # got through rather than being dropped at the buffer.  This
                # is the reliability / robustness dimension, and it is
                # genuinely different from throughput -- a slice can hold
                # throughput up while quietly dropping the tail.
                off = float(np.sum(backlog))
                drop = float(np.sum(dropped))
                acc[tid]["deliv"].append(100.0 * (1.0 - drop / max(off, 1e-9)))
                acc[tid]["buf"].append(float(np.mean(st["queue_bits"]) / 8e3))  # kB

            prb_used_hist.append(total_used)
            retx_hist.append(total_retx)
            demand_hist.append(demand)
            budget_scale_hist.append(scale)
            offered_input_hist.append(float(np.mean([
                np.mean(st["load_mbps"]) * self._load_multiplier(tid)
                for tid, st in self.ue.items()])))

        # ---- assemble KPMs ------------------------------------------
        kpm: Dict[str, Dict[str, float]] = {}
        for tid in self.slices:
            kpm[tid] = {
                "throughput_mbps": float(np.mean(acc[tid]["tput"])),
                "delay_ms": float(np.mean(acc[tid]["delay"])),
                "buffer_kb": float(np.mean(acc[tid]["buf"])),
                "delivery_ratio": float(np.mean(acc[tid]["deliv"])),
                "delivery_pct": float(np.mean(acc[tid]["deliv"])) if acc[tid]["deliv"] else 100.0,
                "offered_input_ratio": float(self._load_multiplier(tid)),
            }
        used = float(np.mean(prb_used_hist)) + float(np.mean(retx_hist))
        util_pct = 100.0 * min(used / self.n_prb, 1.0)

        # Apply configured dose-product interactions in NORMALISED MARGIN
        # units.  The generator sizes gamma against a margin-noise floor, so
        # adding it directly to a raw KPI (e.g. milliseconds) is a unit bug.
        # For higher-better g=y/target-1; for lower-better g=1-y/target.
        # Thus a desired delta-g maps to delta-y = sign*target*delta-g.
        if self.interactions:
            edges = self.full_cfg.get("v2", {}).get(
                "regime_util_edges", [70.0, 88.0])
            regime = ("low" if util_pct < float(edges[0]) else
                      "mid" if util_pct < float(edges[1]) else "high")
            for term in self.interactions:
                if term.tenant not in kpm or term.kpi not in kpm[term.tenant]:
                    continue
                specs = self._intent_specs.get((term.tenant, term.kpi), [])
                if specs:
                    target = float(np.mean([float(x["target"]) for x in specs]))
                    higher = specs[0].get("direction") == "higher_better"
                    raw_delta = (1.0 if higher else -1.0) * target * term.value(c, regime)
                else:
                    raw_delta = term.value(c, regime)
                value = float(kpm[term.tenant][term.kpi]) + raw_delta
                if term.kpi in ("throughput_mbps", "delay_ms", "buffer_kb"):
                    value = max(value, 0.0)
                elif term.kpi in ("delivery_pct", "delivery_ratio"):
                    value = float(np.clip(value, 0.0, 100.0))
                kpm[term.tenant][term.kpi] = value
                if term.kpi == "delivery_pct":
                    kpm[term.tenant]["delivery_ratio"] = value

        base_load = max(self.baseline_offered_load_mbps(), 1e-12)
        epoch_slots = int(self.cfg.get("pre_slots", self.cfg["slots_per_epoch"])) \
            + int(self.cfg.get("post_slots", self.cfg["slots_per_epoch"]))
        horizon_epochs = int(self.full_cfg.get("risk", {}).get(
            "horizon_epochs", self.full_cfg.get("mild", {}).get(
                "forecast_horizon_epochs", 20)))
        horizon_slots = max(horizon_epochs, 1) * max(epoch_slots, 1)
        future_ratios = [self.forecast_offered_load_ratio(t, horizon_slots)
                         for t in self.slices]
        summaries = {
            tid: self.forecast_offered_load_summary(tid, horizon_slots)
            for tid in self.slices}
        for tid in self.slices:
            kpm[tid]["offered_input_ratio_forecast"] = \
                self.forecast_offered_load_ratio(tid, horizon_slots)
            kpm[tid]["offered_input_ratio_forecast_mean"] = summaries[tid]["mean"]
            kpm[tid]["offered_input_ratio_forecast_peak"] = summaries[tid]["peak"]
            kpm[tid]["offered_input_ratio_forecast_min"] = summaries[tid]["minimum"]
        kpm["_cell"] = {
            "txpower_dbm": float(self._controls.get("txpower", 43.0)),
            "tilt_deg": float(self._controls.get("tilt", self.tilt_nominal)),
            # TWO DISTINCT QUANTITIES, previously conflated into one:
            #   prb_util_pct     ACTUAL PRBs allocated.  Bounded by 100% by
            #                    physics.  This is what a "utilisation" KPI
            #                    means and what a host energy intent is
            #                    written against.
            #   offered_load_pct DEMAND including resends.  May exceed 100%;
            #                    that is congestion, not a bug, and it is the
            #                    quantity that predicts queue growth.
            "prb_util_pct": util_pct,
            "offered_load_pct": 100.0 * float(np.mean(demand_hist)) / self.n_prb,
            # Exogenous traffic state, unlike offered_load_pct above which is
            # PRB demand and therefore also depends on controls and BLER.
            "offered_input_ratio": float(np.mean(offered_input_hist)) / base_load,
            "offered_input_ratio_forecast": float(np.mean(future_ratios)),
            "offered_input_ratio_forecast_mean": float(np.mean(
                [x["mean"] for x in summaries.values()])),
            "offered_input_ratio_forecast_peak": float(np.mean(
                [x["peak"] for x in summaries.values()])),
            "offered_input_ratio_forecast_min": float(np.mean(
                [x["minimum"] for x in summaries.values()])),
            "forecast_horizon_epochs": horizon_epochs,
            "prb_used": used,
            "retx_prb": float(np.mean(retx_hist)),
            "binding_dependency_edges": len(self._binding_edges),
            "cell_budget_scale": float(np.mean(budget_scale_hist)),
        }
        return kpm
