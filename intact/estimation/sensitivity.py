"""
intact/estimation/sensitivity.py   --  INTACT v11 §7.4
======================================================
s_{p,i}(c) = d g_i / d p   evaluated in the CURRENT operating regime.

Plain English: "if I move this knob by one unit right now, roughly how much
does this intent's margin move?"   s = +0.0125 means one extra PRB of
headroom buys the stream intent 0.0125 of margin.

*** s IS NOT beta.  DO NOT CONFLATE THEM. ***
    beta : effect of ENABLING A CLAIM        -> binary treatment  -> outer loop
    s    : effect of a SPECIFIC VALUE        -> continuous dose   -> inner loop
beta cannot distinguish "cap to 94" from "cap to 82"; that resolution was
averaged away the moment the treatment became binary.

WHY YOU CANNOT JUST CORRELATE nu AGAINST g
------------------------------------------
Four confounders, and the naive fit gets the SIGN wrong:
 1. The xApp lowers the cap WHEN IT OBSERVES low congestion, and low
    congestion also raises margins.  So low cap co-occurs with high margin.
 2. Lag: a change takes T_set to settle and several KPM periods to show.
 3. Simultaneity: another knob may have moved in the same window.
 4. Range restriction: the xApp only ever writes 78-86, so the log contains
    NO evidence about 94 -- exactly the value the projector wants.
All four are fixed by the same thing: variation in nu that the xApp did not
choose.  Hence:

SOURCE A (primary)  OFFLINE SWEEP.  Hold the operating point fixed, FREEZE
                    EVERY OTHER KNOB, sweep p across its full domain,
                    record every intent's margin.  You control nu, so there
                    is no confounding by construction, and you cover values
                    no xApp ever writes.
SOURCE B (runtime)  The inner loop's OWN overrides and rejects are
                    host-chosen knob movements -- uncontaminated by the
                    xApp's private information.  The mediator generates its
                    own value-level identification as a by-product.
SOURCE C (check)    beta ~= s * E[delta nu | j acts].  Two independently
                    estimated quantities; agreement is evidence the
                    local-linear model holds in that regime.

"LOCAL" MEANS LOCAL.  The same knob, same intent, can have a 23x different
slope between load regimes.  Report s as a TABLE INDEXED BY REGIME, never a
scalar.  REPLICATES ARE NOT OPTIONAL: a single-pass sweep gives a
suspiciously perfect line and no way to set sigma_min -- which IS the
residual standard error of the fit.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def regime_key(kpm: Dict, cfg: Dict) -> str:
    """
    Bucket the current operating point.  Reuse the SAME partition the outer
    loop uses for its context covariates -- two partitions means two sets of
    thresholds and no way to reconcile disagreements (v11 §7.4.4 ii).
    """
    cell = kpm["_cell"]
    # Prefer the pre-treatment exogenous offered-input signal.  Utilisation
    # saturates at 100% in every congested regime, collapsing three genuinely
    # different operating states into one sensitivity table.  The generated
    # predictive scenarios expose a demand ratio precisely to avoid that.
    if "offered_input_ratio" in cell:
        value = float(cell["offered_input_ratio"])
        edges = (cfg.get("v2", {}) or {}).get(
            "regime_load_edges", [0.9, 1.25])
    else:
        value = float(cell["prb_util_pct"])
        edges = cfg["sensitivity"]["load_regime_edges"]
    for k, edge in enumerate(edges):
        if value < float(edge):
            return f"load{k}"
    return f"load{len(edges)}"


class SensitivityTable:
    """
    s[(regime, param, intent)] -> (slope, standard_error, n_samples)

    Populated offline by `sweep()` and updated at runtime from host-chosen
    knob movements.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.tab: Dict[Tuple[str, str, str], Tuple[float, float, int]] = {}
        self.sigma_min = cfg["sensitivity"]["sigma_min"]
        self.delta_max = cfg["sensitivity"]["max_disagreement"]
        self.degraded: Dict[Tuple[str, str], bool] = {}

    # ------------------------------------------------------------------
    def get(self, regime: str, param: str, iid: str) -> float:
        """Slope, or 0.0 if we have no reliable evidence."""
        v = self.tab.get((regime, param, iid))
        if v is None:
            return 0.0
        slope, se, n = v
        # v11 §7.4.6: a small |s| means NO RELIABLE EVIDENCE at the present
        # estimation precision.  It is NOT proof of structural independence.
        return slope if abs(slope) >= self.sigma_min else 0.0

    def is_degraded(self, param: str, regime: str) -> bool:
        """If runtime disagrees with the sweep prior by more than delta_max,
        the value model is untrustworthy for that knob: DISABLE OVERRIDING
        and fall back to admit/reject only (v11 §7.4.6).  Rejection needs no
        value model at all, so the inner loop degrades to a coarser
        granularity rather than failing."""
        return self.degraded.get((param, regime), False)

    def affected_intents(self, regime: str, param: str, intents: List[str]) -> List[str]:
        """
        I_p  =  { i : |s_{p,i}| >= sigma_min }     (v11 §10.1)

        THIS IS A TABLE LOOKUP, NOT A COMPUTATION.  Typically 2-5 intents
        out of possibly dozens.  It is also NOT ownership: I_p spans tenant
        boundaries by construction, because the MEASUREMENT says so.
        """
        return [i for i in intents if abs(self.get(regime, param, i)) >= self.sigma_min]


def sweep(ran, cfg, intents, params: List[str], log) -> SensitivityTable:
    """
    SOURCE A: the offline parameter sweep.

    For each parameter and each load regime:
      * freeze every other knob at its default
      * step the parameter across its domain
      * let the RAN settle, then measure every intent's margin
      * repeat `replicates` times with different seeds
      * least-squares slope of margin vs value, with standard error
    """
    from .margins import margin
    sc = cfg["sensitivity"]
    tab = SensitivityTable(cfg)
    base_controls = dict(ran.current_controls())

    for load_index, load_scale in enumerate(sc["sweep_load_scales"]):
        # set the operating point for this regime
        for tid in ran.slices:
            ran.ue[tid]["load_mbps"] = np.full(
                len(ran.ue[tid]["load_mbps"]),
                ran.slices[tid]["load_mbps_per_ue"] * load_scale)
        ran.step(sc["settle_slots"])
        probe = ran.step(sc["measure_slots"])
        # The sweep controls this regime by construction.  Inferring it from
        # PRB utilisation is wrong when all stressed states saturate at 100%,
        # and recomputing offered_input_ratio after scaling the stored base
        # load would trivially return 1.0.  Indexing the declared ordered
        # sweep levels is exact and matches runtime load0/load1/load2.
        regime = f"load{load_index}"

        for prm in params:
            lo, hi = cfg["sweep_domains"][prm]
            if sc.get("local_fit", False):
                # TRUST-REGION fit: sweep only the neighbourhood the inner
                # loop can actually reach in one epoch, centred on the
                # current operating point.  A local derivative should be
                # measured locally.
                cur = base_controls.get(prm, (lo + hi) / 2)
                half = sc.get("local_frac",
                              cfg["inner"].get("max_step_frac", 0.25)) * (hi - lo)
                glo, ghi = max(lo, cur - half), min(hi, cur + half)
                if ghi - glo < 1e-9:
                    glo, ghi = lo, hi
                grid = np.linspace(glo, ghi, sc["sweep_points"])
            else:
                grid = np.linspace(lo, hi, sc["sweep_points"])
            samples = {i: [] for i in intents}
            xs = []
            for rep in range(sc["replicates"]):
                for val in grid:
                    # A fresh common-random-number world for every value.
                    # Sequentially sweeping one live RAN confounds knob value
                    # with queue history and time/order; resetting to the same
                    # seed and advancing the same number of slots isolates p.
                    ran.reset(seed=sc["base_seed"] + rep * 101)
                    for tid in ran.slices:
                        ran.ue[tid]["load_mbps"] = np.full(
                            len(ran.ue[tid]["load_mbps"]),
                            ran.slices[tid]["load_mbps_per_ue"] * load_scale)
                    for k, v in base_controls.items():
                        ran.apply(k, v)
                    ran.step(sc["settle_slots"])
                    ran.apply(prm, float(val))
                    ran.step(sc["settle_slots"])          # let it settle
                    kpm = ran.step(sc["measure_slots"])   # then measure
                    xs.append(val)
                    for iid, it in intents.items():
                        samples[iid].append(margin(it, kpm))
            X = np.array(xs)
            for iid in intents:
                Y = np.array(samples[iid])
                if len(X) < 3 or np.std(X) < 1e-9:
                    continue
                # OLS slope + standard error
                A = np.column_stack([np.ones_like(X), X])
                coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
                resid = Y - A @ coef
                dof = max(len(X) - 2, 1)
                s2 = float(resid @ resid / dof)
                se = float(np.sqrt(s2 * np.linalg.pinv(A.T @ A)[1, 1]))
                tab.tab[(regime, prm, iid)] = (float(coef[1]), se, len(X))
                log.debug("sweep %s | %-16s -> %-4s  s=%+.5f  se=%.5f",
                          regime, prm, iid, coef[1], se)

    # restore
    ran.reset(seed=cfg["ran"].get("seed", 0))
    for k, v in base_controls.items():
        ran.apply(k, v)
    log.info("sensitivity sweep complete: %d (regime,param,intent) entries",
             len(tab.tab))
    return tab
