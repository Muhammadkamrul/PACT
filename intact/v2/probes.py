"""
intact/v2/probes.py
===================
Balanced four-arm factorial probes with FORCED dose levels.

WHY v1's PROBE DESIGN COULD NOT IDENTIFY GAMMA
----------------------------------------------
v1 drew eligibility Z_j independently per claim and let the xApp choose the
magnitude.  Three consequences, each fatal on its own:

 1. UNBALANCED ARMS.  For a specific pair the four cells (00, 10, 01, 11)
    appeared with probabilities ~(0.25, 0.25, 0.25, 0.25) only in
    expectation.  With ~4-6 of 12 claims eligible per epoch the joint arm
    (11) actually occurred in roughly 11% of epochs, so across 800 epochs a
    pair had ~88 joint observations before burn-in -- below what is needed
    to separate an interaction from two main effects.

 2. UNCONTROLLED DOSE.  The magnitude was whatever the xApp's controller
    happened to want.  A converged controller writes ~0, so many "eligible"
    rows carried no dose at all and contributed nothing to identification
    while still counting as support.

 3. CORRELATED DOSES.  Both xApps react to the same cell state, so their
    write magnitudes are correlated with each other AND with the outcome.
    That is confounding, not identification.

THE FIX
-------
For every configured pair, run an equal number of trials in each of four
arms, with the dose FORCED to a specified level:

    arm   dnu_a     dnu_b      identifies
    ----------------------------------------------------
    00    0         0          alpha (baseline drift)
    10    a         0          s_a
    01    0         b          s_b
    11    a         b          gamma, via C - A - B

and a fifth "double" arm (2a, b) that over-identifies gamma so the
bilinear form itself can be falsified: if gamma is real and bilinear,
the 11 and the double arm must agree after scaling by the dose product.

SPLITTING
---------
Train/test splits are BY WHOLE PROBE, never by row.  A probe's four arms
are matched by construction; putting the 10 arm in train and the 11 arm in
test leaks the matched baseline across the split and inflates test
performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ARMS = ("00", "10", "01", "11", "2a_b")


@dataclass
class ProbeTrial:
    """One measured arm of one probe."""
    probe_id: int
    pair: Tuple[str, str]
    arm: str
    regime: str
    dnu: Dict[str, float]
    ctx: np.ndarray
    dg: Dict[str, float]
    dose_a: float
    dose_b: float

    def as_row(self) -> Dict:
        return {"probe_id": self.probe_id, "pair": f"{self.pair[0]}|{self.pair[1]}",
                "arm": self.arm, "regime": self.regime,
                "dose_a": self.dose_a, "dose_b": self.dose_b}


@dataclass
class ProbePlan:
    """Dose levels for one pair, sized to clear the noise floor."""
    pair: Tuple[str, str]
    dose_a: float
    dose_b: float
    ref_a: float
    ref_b: float


def plan_doses(pair: Tuple[str, str], domains: Dict[str, Tuple[float, float]],
               current: Dict[str, float], frac: float = 0.35) -> ProbePlan:
    """Choose a dose that is large enough to measure and inside the domain.

    A dose that is too small is invisible under sigma ~ 0.015; a dose that
    leaves the domain cannot be applied.  `frac` of the half-range from the
    current value, clipped to the domain, satisfies both.
    """
    a, b = pair
    lo_a, hi_a = domains[a]
    lo_b, hi_b = domains[b]
    cur_a = float(current.get(a, 0.5 * (lo_a + hi_a)))
    cur_b = float(current.get(b, 0.5 * (lo_b + hi_b)))
    # prefer the direction with more headroom, so 2a stays in domain
    up_a, dn_a = hi_a - cur_a, cur_a - lo_a
    up_b, dn_b = hi_b - cur_b, cur_b - lo_b
    da = frac * up_a if up_a >= dn_a else -frac * dn_a
    db = frac * up_b if up_b >= dn_b else -frac * dn_b
    # 2a must remain inside the domain
    if cur_a + 2 * da > hi_a:
        da = (hi_a - cur_a) / 2.0
    if cur_a + 2 * da < lo_a:
        da = (lo_a - cur_a) / 2.0
    return ProbePlan(pair, float(da), float(db), cur_a, cur_b)


class FactorialProbeRunner:
    """Runs balanced four-arm probes against a clone factory.

    `clone_factory()` must return a FRESH simulator in the same state, so
    the four arms are counterfactuals of one another rather than a
    trajectory.  This is what makes the differencing valid.
    """

    def __init__(self, clone_factory, cfg: Dict, state_builder,
                 settle_slots: int = 12, measure_slots: int = 32,
                 base_seed: int = 4242):
        # settle 12 / measure 32, not 5 / 8.  The per-probe state jitter that
        # makes several regimes reachable also injects a transient; 12 slots
        # let it decay before measurement starts.  Averaging the response over
        # 32 slots instead of 8 halves the residual standard error, which is
        # what the dose effect has to clear.
        self.clone_factory = clone_factory
        self.cfg = cfg
        self.state_builder = state_builder
        self.settle = settle_slots
        self.measure = measure_slots
        self.base_seed = int(base_seed)

    def _run_arm(self, plan: ProbePlan, arm: str, intents, tenants,
                 margin_fn, probe_id: int = 0
                 ) -> Tuple[str, Dict[str, float], np.ndarray,
                            Dict[str, float]]:
        # *** BUG FIX. ***  The clone factory rebuilt the RAN from the same
        # config every time, so every replicate of every probe was BYTE
        # IDENTICAL: measured dg = -0.0839 for all ten replicates, zero
        # variance.  Forty probes therefore supplied FIVE unique design rows
        # repeated forty times, against an 18-wide design -- rank deficient,
        # so ridge shrank every coefficient and the regression gamma bore no
        # relation to the model-free differenced gamma (r = 0.029).
        #
        # Two randomisations, applied per PROBE (never per arm, so the four
        # arms of one probe remain matched counterfactuals of each other):
        #   1. a distinct RNG stream, giving genuine replicate variance
        #   2. a jittered load scale and starting controls, so probes visit
        #      DIFFERENT REGIMES.  Without this every probe landed in "mid"
        #      and regime conditioning was inert by construction.
        ran = self.clone_factory()
        rng = np.random.default_rng(self.base_seed + 9176 * probe_id)
        if hasattr(ran, "rng"):
            ran.rng = np.random.default_rng(self.base_seed + 7717 * probe_id)
        # Moderate spread: enough to reach all three regimes, not so wide
        # that within-regime load variation becomes its own nuisance term.
        scale = float(rng.choice([0.7, 0.85, 1.0, 1.2, 1.4]))
        try:
            for tid in list(getattr(ran, "slices", {})):
                ran.slices[tid]["load_mbps_per_ue"] = (
                    self.cfg["ran"]["slices"][tid]["load_mbps_per_ue"] * scale)
            if hasattr(ran, "_rebuild_ue"):
                ran._rebuild_ue()
        except Exception:
            pass
        for p, (lo, hi) in ((plan.pair[0], (0, 0)), ):
            pass
        # jitter the two probed knobs' reference points inside their domain so
        # the baseline arm is not always the same operating point
        for p, ref in ((plan.pair[0], plan.ref_a), (plan.pair[1], plan.ref_b)):
            try:
                lo, hi = self.cfg["sweep_domains"][p]
                j = float(rng.normal(0.0, 0.03 * (hi - lo)))
                ran.apply(p, float(min(max(ref + j, lo), hi)))
            except Exception:
                pass
        for _ in range(6):
            ran.step(1)
        kpm0 = ran.step(1)
        regime, ctx = self.state_builder.build(kpm0, ran, list(tenants))
        g0 = {i: margin_fn(intents[i], kpm0) for i in intents}

        a, b = plan.pair
        mult_a = {"00": 0.0, "10": 1.0, "01": 0.0, "11": 1.0, "2a_b": 2.0}[arm]
        mult_b = {"00": 0.0, "10": 0.0, "01": 1.0, "11": 1.0, "2a_b": 1.0}[arm]
        dnu = {a: mult_a * plan.dose_a, b: mult_b * plan.dose_b}
        # dose is relative to THIS probe's jittered starting point, so the
        # applied change really is dnu regardless of the jitter
        cur = ran.current_controls()
        ran.apply(a, float(cur.get(a, plan.ref_a)) + dnu[a])
        ran.apply(b, float(cur.get(b, plan.ref_b)) + dnu[b])

        for _ in range(self.settle):
            ran.step(1)
        kpm1 = ran.step(self.measure)
        g1 = {i: margin_fn(intents[i], kpm1) for i in intents}
        dg = {i: g1[i] - g0[i] for i in intents}
        return regime, dnu, ctx, dg

    def run_pair(self, plan: ProbePlan, n_probes: int, intents, tenants,
                 margin_fn, probe_id_start: int = 0) -> List[ProbeTrial]:
        """Equal support in every arm, by construction rather than by chance."""
        out = []
        pid = probe_id_start
        for _ in range(n_probes):
            for arm in ARMS:
                regime, dnu, ctx, dg = self._run_arm(
                    plan, arm, intents, tenants, margin_fn, probe_id=pid)
                out.append(ProbeTrial(pid, plan.pair, arm, regime, dnu, ctx,
                                      dg, plan.dose_a, plan.dose_b))
            pid += 1
        return out


# ----------------------------------------------------------------------
def gamma_by_differencing(trials: Sequence[ProbeTrial], intent: str,
                          regime: Optional[str] = None) -> Dict[str, float]:
    """gamma from C - A - B, the estimate that needs no regression.

    Reported alongside the regression fit as an independent check: if the
    two disagree, the model form is wrong, not merely noisy.
    """
    sel = [t for t in trials if regime is None or t.regime == regime]
    by = {arm: [t for t in sel if t.arm == arm] for arm in ARMS}
    if not all(by[a] for a in ("00", "10", "01", "11")):
        return {"gamma": float("nan"), "n": 0, "reason": "missing arm"}

    def mean_dg(arm):
        return float(np.mean([t.dg.get(intent, 0.0) for t in by[arm]]))

    base = mean_dg("00")
    a_eff = mean_dg("10") - base
    b_eff = mean_dg("01") - base
    ab_eff = mean_dg("11") - base
    da = float(np.mean([t.dose_a for t in by["11"]]))
    db = float(np.mean([t.dose_b for t in by["11"]]))
    denom = da * db
    if abs(denom) < 1e-12:
        return {"gamma": float("nan"), "n": len(sel), "reason": "zero dose"}
    gamma = (ab_eff - a_eff - b_eff) / denom

    out = {"gamma": float(gamma), "n": len(sel), "reason": "ok",
           "interaction_effect": float(ab_eff - a_eff - b_eff),
           "dose_product": float(denom)}
    # over-identification check with the doubled arm
    if by["2a_b"]:
        d2 = mean_dg("2a_b") - base
        a2 = 2.0 * a_eff
        g2 = (d2 - a2 - b_eff) / (2.0 * denom)
        out["gamma_double_arm"] = float(g2)
        out["bilinear_consistent"] = bool(
            abs(g2 - gamma) <= 0.5 * max(abs(gamma), 1e-9) + 1e-6)
    return out


def split_by_probe(trials: Sequence[ProbeTrial], train_frac: float = 0.6,
                   val_frac: float = 0.2, seed: int = 11
                   ) -> Tuple[List[ProbeTrial], List[ProbeTrial], List[ProbeTrial]]:
    """Split by WHOLE PROBE so matched arms never straddle the split."""
    ids = sorted({t.probe_id for t in trials})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    c1, c2 = int(n * train_frac), int(n * (train_frac + val_frac))
    tr, va, te = set(ids[:c1]), set(ids[c1:c2]), set(ids[c2:])
    return ([t for t in trials if t.probe_id in tr],
            [t for t in trials if t.probe_id in va],
            [t for t in trials if t.probe_id in te])


def arm_balance_report(trials: Sequence[ProbeTrial]) -> List[Dict]:
    """Prove the arms really are balanced; a design that drifted is a bug."""
    out = []
    pairs = sorted({t.pair for t in trials})
    for pr in pairs:
        sel = [t for t in trials if t.pair == pr]
        counts = {a: sum(1 for t in sel if t.arm == a) for a in ARMS}
        vals = [counts[a] for a in ("00", "10", "01", "11")]
        out.append({"pair": f"{pr[0]}|{pr[1]}", **counts,
                    "balanced": bool(max(vals) - min(vals) == 0),
                    "n_probes": len({t.probe_id for t in sel})})
    return out


def difference_against_baseline(trials: Sequence[ProbeTrial]
                                ) -> List[ProbeTrial]:
    """Subtract each probe's own 00-arm from all of its arms.

    *** This is the whole point of a MATCHED factorial design, and the first
    version failed to use it. ***

    Randomising the starting state per probe is what makes several regimes
    reachable and gives the replicates genuine variance -- without it every
    replicate was byte-identical and nothing was identifiable.  But that same
    randomisation injects per-probe STATE noise into every arm, and measured
    on this scenario it is as large as the effect being estimated:

        baseline (00-arm) sd   0.337
        treatment effect  sd   0.395      -> noise/signal 0.9x

    Fitting raw dg therefore asks the regression to find a dose effect
    underneath an equal-sized nuisance term, and held-out R^2 for s collapsed
    from 0.964 to 0.042.

    All five arms of one probe share the SAME seed, load scale and starting
    controls by construction -- verified: every arm of a probe reports the
    same regime -- so the nuisance term is IDENTICAL across the arms and
    cancels exactly on subtraction.  What remains is the causal contrast the
    design was built to measure.

    The 00 arms become identically zero and are dropped: they carry no
    treatment information once they are the reference.  Fit ALPHA on the RAW
    trials instead, since the baseline drift is precisely what alpha is for.
    """
    base: Dict[Tuple[int, Tuple[str, str]], Dict[str, float]] = {}
    for t in trials:
        if t.arm == "00":
            base[(t.probe_id, tuple(t.pair))] = dict(t.dg)
    out: List[ProbeTrial] = []
    for t in trials:
        if t.arm == "00":
            continue
        b = base.get((t.probe_id, tuple(t.pair)))
        if b is None:
            continue
        out.append(ProbeTrial(
            probe_id=t.probe_id, pair=t.pair, arm=t.arm, regime=t.regime,
            dnu=dict(t.dnu), ctx=np.asarray(t.ctx).copy(),
            dg={i: float(v) - float(b.get(i, 0.0)) for i, v in t.dg.items()},
            dose_a=t.dose_a, dose_b=t.dose_b))
    return out
