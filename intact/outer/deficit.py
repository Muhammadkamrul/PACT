"""
intact/outer/deficit.py   --  INTACT v11 §8.4, §9
=================================================
TWO counters, and the whole fairness story is those two counters.

CLAIM DEFICIT  D_j    (v11 §8.4)
    D_j <- max(0,  D_j + r_j - 1{j in S})

    BOTH BRANCHES MATTER:
        not scheduled  ->  D_j += r_j          debt grows
        scheduled      ->  D_j -= (1 - r_j)    DEBT DRAINS
    Writing only the first branch -- easy to do on a slide -- leaves a
    counter that grows forever and never releases, so the claim would
    monopolise the schedule after its first turn.  max(0,.) stops credit
    accumulating for over-served claims.

    WHY IT CANNOT STARVE ANYONE:  suppose a claim were starved forever.  Its
    debt would grow without bound.  But debt enters the score LINEARLY, so
    eventually its score exceeds every competitor's and it IS scheduled --
    contradiction.  (Formally a Lyapunov drift argument, identical in
    structure to MaxWeight scheduling.)

    THE GUARANTEE IS CONDITIONAL (v11 §8.4):  it holds for any rate vector r
    lying in the scheduler's FEASIBLE RATE REGION.  We define that region;
    we do not supply a general membership algorithm.  For deployments this
    size the admissible sets are enumerable, so it is a small LP run once at
    onboarding.

TENANT DEFICIT  Lambda_n    (v11 §9)
    Lambda_n <- max(0,  Lambda_n + (rho_min - rho_n))
    Enters the WEIGHT, not the score.  Enforces C3 without a constrained
    solver.

WHAT THE CONTRACT ACTUALLY PROMISES (v11 §8.4)
    D_j is defined over  1{j in S}  -- MEMBERSHIP IN THE ELIGIBILITY SET.
    It drains when the claim is scheduled, whether or not its subsequent
    write survives the inner loop.  So INTACT guarantees contracted
    OPPORTUNITY TO ACT, not contracted EFFECT.  The gap between the two is
    exactly what the AWE metric measures.  Say this plainly in the paper; a
    reviewer who spots it unaided will read it as a hole.
"""
from __future__ import annotations
from typing import Dict, Set
from ..types import Claim, Tenant


class DeficitBook:
    def __init__(self, claims: Dict[str, Claim], tenants: Dict[str, Tenant],
                 d_cap: float = 500.0, lam_cap: float = 3.0):
        # d_cap is a SAFETY STOP, not a tuning knob: set it far above any
        # deficit reachable under a feasible rate vector.  A tight cap makes
        # two starved claims indistinguishable and freezes the shortfall.
        # lam_cap IS tight, because Lambda multiplies the weights.
        self.d_cap, self.lam_cap = d_cap, lam_cap
        self.D: Dict[str, float] = {j: 0.0 for j in claims}
        self.Lam: Dict[str, float] = {t: 0.0 for t in tenants}
        self.claims = claims
        self.tenants = tenants
        self.granted: Dict[str, int] = {j: 0 for j in claims}
        self.epochs = 0

    def update_claims(self, S: Set[str]) -> None:
        """Apply the deficit recursion after an epoch's selection."""
        self.epochs += 1
        for j, c in self.claims.items():
            served = 1.0 if j in S else 0.0
            self.granted[j] += int(served)
            self.D[j] = min(self.d_cap, max(0.0, self.D[j] + c.r_j - served))

    def update_tenants(self, rho: Dict[str, float]) -> None:
        """rho: realised priority-weighted fulfilment per tenant."""
        for t, ten in self.tenants.items():
            short = ten.rho_min - rho.get(t, 1.0)
            self.Lam[t] = min(self.lam_cap, max(0.0, self.Lam[t] + short))

    def realised_rate(self, j: str) -> float:
        return self.granted[j] / max(self.epochs, 1)
