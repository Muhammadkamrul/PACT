"""
intact/v2/outer_v2.py
=====================
Outer-loop fixes.  Three defects measured in v1, three mechanisms here.

DEFECT 1 -- WEIGHT WAS TENANT-AGGREGATE, NOT CLAIM-SPECIFIC
v1: `out[jid] = mass[tenant]`, where mass[n] = sum over ALL of n's intents.
Every claim of a tenant carried the identical weight, even a claim that
cannot move the intent that is failing.  MILD sharpened w_i per intent and
then the aggregation summed the sharpness straight back out, which is why
p_hat never reached a decision.  `leverage_claim_weights` replaces the
aggregate with each claim's SHARE OF THE LEVERAGE on each intent, taken
from s -- the one estimand that validated.  No new model is introduced.

DEFECT 2 -- DEFICIT DISCHARGED ON OPPORTUNITY, NOT ACTUATION
v1 drained D_j when j entered S, whether or not a write executed.  A
converged xApp with nothing to say occupied a C1 slot and paid off its
debt with silence.  The contract said "actuation rate" and the code
enforced "seat at the table".  `DeficitLedgerV2` records THREE distinct
events -- opportunity, proposal, actuation -- and discharges only on
actuation.

DEFECT 3 -- q_hat HARDCODED TO 1.0
v1: `q_hat = {p: 1.0 for p in self.pairs}`.  H(j,k) was therefore computed
as though every configured pair co-fires every epoch.  `CoFireTracker`
estimates it from the operation log, as the formulation always intended.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ======================================================================
# DEFECT 1: leverage-weighted claim weights
# ======================================================================
def leverage_shares(params: Sequence[str], intents: Sequence[str],
                    s_lookup, regime: str, sigma_min: float = 1e-4
                    ) -> Dict[str, Dict[str, float]]:
    """share[p][i] = |s_{p,i}| / sum_{p'} |s_{p',i}|   -- normalised ACROSS
    PARAMETERS for a fixed intent.

    *** The normalisation direction is load-bearing. ***  Normalising across
    INTENTS instead would make every claim's weight sum to the same
    constant and would reproduce the v1 problem in a new disguise: every
    claim again ends up with the same total.  Across parameters, the
    question asked is "of everything that can move intent i, how much of
    that leverage does this parameter hold?", which is the question the
    ranking needs.
    """
    share: Dict[str, Dict[str, float]] = {p: {} for p in params}
    for i in intents:
        mags = {p: abs(float(s_lookup(regime, p, i))) for p in params}
        tot = sum(mags.values())
        if tot < sigma_min:
            # nothing measurably moves this intent in this regime; spread
            # evenly rather than concentrating on numerical noise
            for p in params:
                share[p][i] = 1.0 / max(len(params), 1)
        else:
            for p in params:
                share[p][i] = mags[p] / tot
    return share


def leverage_claim_weights(claims, intents, w_intent: Dict[str, float],
                           params: Sequence[str], s_lookup, regime: str
                           ) -> Dict[str, float]:
    """c_j = sum_i w_i * share[p(j)][i].

    A claim that cannot move intent i draws ~0 weight from it; a claim that
    is the dominant lever draws ~1.  MILD's urgency now flows only to the
    claims that can actually act on the at-risk intent.
    """
    share = leverage_shares(params, list(intents), s_lookup, regime)
    out = {}
    for jid, c in claims.items():
        p = c.param
        tot = 0.0
        for i in intents:
            tot += float(w_intent.get(i, 0.0)) * share.get(p, {}).get(i, 0.0)
        out[jid] = tot
    return out


# ======================================================================
# DEFECT 2: three-event deficit ledger
# ======================================================================
@dataclass
class ClaimEvents:
    opportunity: int = 0      # j was in S
    proposal: int = 0         # j submitted nu_req != nu_old
    qualified: int = 0        # proposal has a nonzero safe/feasible projection
    actuation: int = 0        # a nonzero write actually executed
    silent_opportunity: int = 0   # in S but proposed nothing


class DeficitLedgerV2:
    """D_j discharges on ACTUATION, and all three events are recorded.

    The distinction matters for the paper's own claim: C4 is described as
    an actuation floor, but v1 reduced the debt on selection.  That is an
    OPPORTUNITY guarantee.  Both are now reported so the difference is
    visible rather than asserted.
    """

    def __init__(self, claims, cap: float = 500.0,
                 discharge_on: str = "actuation",
                 accrual_on: str = "qualified"):
        self.r = {j: float(getattr(c, "r_j", 0.0)) for j, c in claims.items()}
        self.cap = float(cap)
        self.discharge_on = discharge_on
        if accrual_on not in {"epoch", "proposal", "qualified"}:
            raise ValueError("c4_accrual_on must be epoch, proposal, or qualified")
        self.accrual_on = accrual_on
        self.D = {j: 0.0 for j in claims}
        self.events = {j: ClaimEvents() for j in claims}
        self.epochs = 0

    def note_opportunity(self, jid: str) -> None:
        self.events[jid].opportunity += 1

    def note_proposal(self, jid: str) -> None:
        self.events[jid].proposal += 1

    def note_qualified(self, jid: str) -> None:
        self.events[jid].qualified += 1

    def note_actuation(self, jid: str) -> None:
        self.events[jid].actuation += 1

    def note_silent(self, jid: str) -> None:
        self.events[jid].silent_opportunity += 1

    def update(self, served: Iterable[str], eligible: Optional[Iterable[str]] = None) -> None:
        """Advance the virtual C4 queues.

        ``epoch`` reproduces the old unconditional contract.  ``proposal``
        accrues only when the xApp requests a nonzero change.  ``qualified``
        (the v2 default) additionally requires that the request has a
        nonzero C1/C2/safety-feasible projection when examined alone.  A
        scheduler cannot honestly guarantee actuation when there is no
        request or when every permitted value is unsafe.
        """
        served = set(served)
        eligible = set(self.D) if eligible is None else set(eligible)
        self.epochs += 1
        for j in self.D:
            accrue = 1.0 if (self.accrual_on == "epoch" or j in eligible) else 0.0
            self.D[j] = float(np.clip(
                self.D[j] + self.r[j] * accrue - (1.0 if j in served else 0.0),
                0.0, self.cap))

    def rates(self) -> List[Dict]:
        n = max(self.epochs, 1)
        rows = []
        for j, e in self.events.items():
            proposal_rate = e.proposal / n
            qualified_rate = e.qualified / n
            actuation_rate = e.actuation / n
            act_given_proposal = e.actuation / max(e.proposal, 1)
            act_given_qualified = e.actuation / max(e.qualified, 1)
            if self.accrual_on == "epoch":
                c4_rate, c4_exposure = actuation_rate, n
            elif self.accrual_on == "proposal":
                c4_rate, c4_exposure = act_given_proposal, e.proposal
            else:
                c4_rate, c4_exposure = act_given_qualified, e.qualified
            rows.append({
                "claim": j, "r_j": self.r[j],
                "opportunity_rate": e.opportunity / n,
                "proposal_rate": proposal_rate,
                "qualified_rate": qualified_rate,
                "actuation_rate": actuation_rate,
                "actuation_given_proposal": act_given_proposal,
                "actuation_given_qualified": act_given_qualified,
                "c4_target_basis": self.accrual_on,
                "c4_exposures": int(c4_exposure),
                "c4_achieved_rate": c4_rate,
                "c4_shortfall": (max(0.0, self.r[j] - c4_rate)
                                 if c4_exposure > 0 else 0.0),
                "silent_rate": e.silent_opportunity / max(e.opportunity, 1),
                # the two guarantees, reported separately and never conflated
                "opportunity_shortfall": max(0.0, self.r[j] - e.opportunity / n),
                "actuation_shortfall": max(0.0, self.r[j] - e.actuation / n),
            })
        return rows


# ======================================================================
# DEFECT 3: q_hat estimated from the operation log
# ======================================================================
class CoFireTracker:
    """Empirical probability that both claims of a pair actually WRITE.

    v1 hardcoded 1.0 "conservatively", which made H(j,k) fire at full
    strength for pairs that never co-fire and gave the penalty no
    discriminating power at all.  A Beta(1,1) prior keeps the estimate
    sane for pairs with little history rather than collapsing to 0.
    """

    def __init__(self, pairs: Sequence[Tuple[str, str]], window: int = 200,
                 prior_a: float = 1.0, prior_b: float = 1.0):
        self.pairs = [tuple(p) for p in pairs]
        self.window = int(window)
        self.pa, self.pb = float(prior_a), float(prior_b)
        self.hist: Dict[Tuple[str, str], deque] = {
            p: deque(maxlen=self.window) for p in self.pairs}

    def observe(self, actuated: Iterable[str]) -> None:
        a = set(actuated)
        for p in self.pairs:
            self.hist[p].append(1.0 if (p[0] in a and p[1] in a) else 0.0)

    def q_hat(self) -> Dict[Tuple[str, str], float]:
        out = {}
        for p in self.pairs:
            h = self.hist[p]
            k, n = float(sum(h)), float(len(h))
            out[p] = (k + self.pa) / (n + self.pa + self.pb)
        return out

    def report(self) -> List[Dict]:
        q = self.q_hat()
        return [{"pair": f"{p[0]}|{p[1]}", "q_hat": q[p],
                 "n_observed": len(self.hist[p]),
                 "n_cofire": int(sum(self.hist[p]))} for p in self.pairs]


# ======================================================================
# dose-aware pair harm
# ======================================================================
def pair_harm_v2(pairs, intents, w_intent: Dict[str, float],
                 gamma_lookup, regime: str, q_hat: Dict[Tuple[str, str], float],
                 expected_dose: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    """H(j,k) = q_hat * sum_i w_i * max(-gamma_i(r) * dnu_j * dnu_k, 0).

    Dose-scaled, because a bilinear interaction has no magnitude until the
    doses are named, and regime-conditioned, because the interaction gain
    itself depends on the operating point.  Only HARM counts: a pair whose
    interaction happens to help an intent should not be penalised for it.
    """
    H = {}
    for pr in pairs:
        pr = tuple(pr)
        da = float(expected_dose.get(pr[0], 0.0))
        db = float(expected_dose.get(pr[1], 0.0))
        dmg = 0.0
        for i in intents:
            g = float(gamma_lookup(regime, pr, i))
            dmg += float(w_intent.get(i, 0.0)) * max(-g * da * db, 0.0)
        H[pr] = float(q_hat.get(pr, 1.0)) * dmg
    return H


# ======================================================================
# implicit-conflict detection (structural + behavioural)
# ======================================================================
class ImplicitConflictDetector:
    """Two detectors for the conflict type nothing in v1 detected.

    STRUCTURAL (O-RAN WG3 implicit, graph variant): claims j and k write
    parameters joined by a declared RCP->RCP edge.  No estimation needed --
    the conflict exists by construction, so it is checked exactly like C1.
    Call this C1' (transitive authority).

    BEHAVIOURAL (WG3 implicit, feedback variant): two closed-loop xApps
    couple through the environment.  x1 moves a knob, the KPM shifts, x2
    observes the shift and reacts, and x2's own KPM degrades.  The
    signature is SUSTAINED COUNTER-ACTION: persistent negative correlation
    between the two claims' write directions with no shared parameter and
    no shared KPM.  This is also the mechanism behind v1's beta instability
    (state-stability 0.204 with sign flips), so detecting it turns a
    negative estimation result into a positive measurement.
    """

    def __init__(self, dependencies, claims, window: int = 120,
                 threshold: float = -0.35):
        self.edges = [(e.source, e.target) for e in dependencies]
        self.claims = claims
        self.window = int(window)
        self.threshold = float(threshold)
        self.dirs: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window))

    def structural_conflicts(self, S: Iterable[str]) -> List[Tuple[str, str]]:
        S = list(S)
        out = []
        for a in S:
            for b in S:
                if a >= b:
                    continue
                pa = self.claims[a].param
                pb = self.claims[b].param
                if (pa, pb) in self.edges or (pb, pa) in self.edges:
                    out.append((a, b))
        return out

    def observe_writes(self, dnu: Dict[str, float]) -> None:
        for j in self.claims:
            self.dirs[j].append(float(np.sign(dnu.get(j, 0.0))))

    def behavioural_conflicts(self) -> List[Dict]:
        out = []
        js = list(self.claims)
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                ja, jb = js[a], js[b]
                if self.claims[ja].param == self.claims[jb].param:
                    continue          # that is DIRECT conflict, not implicit
                x = np.asarray(self.dirs[ja], dtype=float)
                y = np.asarray(self.dirs[jb], dtype=float)
                n = min(len(x), len(y))
                if n < 30:
                    continue
                x, y = x[-n:], y[-n:]
                if x.std() < 1e-9 or y.std() < 1e-9:
                    continue
                r = float(np.corrcoef(x, y)[0, 1])
                if r <= self.threshold:
                    out.append({"claim_a": ja, "claim_b": jb,
                                "corr": r, "n": n,
                                "kind": "behavioural_counteraction"})
        return out
