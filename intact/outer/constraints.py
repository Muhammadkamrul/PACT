"""
intact/outer/constraints.py   --  INTACT v11 §8.3, §8.3.1
=========================================================
The hard constraints, and the PRECOMPUTED ADMISSIBILITY MASK.

C1  ONE WRITER PER KNOB.  No two claims in S write the same parameter.
    This is what delivers the guarantee advertised for the outer loop.  An
    objective written without it is not the same optimisation problem.

C2  RESOURCE FEASIBILITY -- and it is nearly vacuous.
    THE PREMISE THAT GENERATES THE PROBLEM IS FALSE.  Most treatments assume
    INTACT is handed one undifferentiated pool and must divide it.  In a
    multi-tenant O-RAN it is not: per-tenant envelopes ALREADY EXIST,
    because that is what RAN slicing is.  E_{n,r}(tau) comes from the
    non-RT RIC over A1.  INTACT DOES NOT COMPUTE IT.

    Consequence: two claims belonging to DIFFERENT tenants draw on DISJOINT
    envelopes, so no combination of values they can write will over-commit
    the pool.  CROSS-TENANT C2 IS VACUOUS BY CONSTRUCTION.  What remains:

        sum_{j in S_alloc(n,r)}  d_bar_{j,r}  <=  E_{n,r}(tau)   for all n,r

    a STATIC check over declared domains, evaluated at ONBOARDING and on any
    A1 envelope update -- NOT once per epoch.  At runtime it is a
    precomputed admissibility mask.

    d_bar_{j,r} is the MAXIMUM RESOURCE DEMAND the claim can place:
        d_bar_{j,r} = sup over nu in V_p and c in C_j of  d_r(nu, c)
    NOT "the top of the knob's domain" -- those coincide only when the
    mapping is the identity.  For the direct controls used here they do.

C3  TENANT FULFILMENT FLOOR.  Enforced through Lambda_n in the weight
    (deficit.py), not as a solver constraint.

C2 APPLIES TO ALLOCATIVE PARAMETERS ONLY.  For regulative ones it is
vacuous: two caps of 82 do not reserve 164 PRBs, they each say "no more
than this".  The test is one question: does the write RESERVE capacity, or
merely BOUND it?
"""
from __future__ import annotations
from itertools import combinations
from typing import Dict, FrozenSet, List, Set, Tuple
from ..types import Claim, Kind, Tenant


class AdmissibilityMask:
    """
    Precomputed at onboarding.  Stores the FORBIDDEN combinations so the
    selector can reject a candidate set in O(number of forbidden pairs).
    """

    def __init__(self, claims: Dict[str, Claim], tenants: Dict[str, Tenant], log):
        self.claims = claims
        self.tenants = tenants

        # ---- C1: group claims by the parameter they write ---------------
        self.knob_groups: Dict[str, List[str]] = {}
        for j, c in claims.items():
            self.knob_groups.setdefault(c.param, []).append(j)
        self.c1_pairs = {frozenset(p) for g in self.knob_groups.values()
                         for p in combinations(g, 2)}

        # ---- C2: within-tenant, per-resource domain sums ----------------
        self.c2_forbidden: Set[FrozenSet[str]] = set()
        by_tr: Dict[Tuple[str, str], List[str]] = {}
        for j, c in claims.items():
            if c.kind == Kind.ALLOCATIVE and c.resource:
                by_tr.setdefault((c.tenant, c.resource), []).append(j)

        for (tid, res), js in by_tr.items():
            env = tenants[tid].envelope.get(res, float("inf"))
            # any SUBSET whose d_bar sum exceeds the envelope is inadmissible.
            # We record the MINIMAL such subsets (adding claims only makes it
            # worse, so supersets are implied).
            for k in range(2, len(js) + 1):
                for combo in combinations(js, k):
                    tot = sum(claims[j].d_bar for j in combo)
                    if tot > env + 1e-9:
                        fs = frozenset(combo)
                        # skip if a subset is already forbidden
                        if not any(f < fs for f in self.c2_forbidden):
                            self.c2_forbidden.add(fs)

        log.info("admissibility mask: %d C1 same-knob pairs, %d C2 forbidden sets",
                 len(self.c1_pairs), len(self.c2_forbidden))
        for fs in sorted(self.c2_forbidden, key=lambda s: sorted(s)):
            js = sorted(fs)
            tid = claims[js[0]].tenant
            res = claims[js[0]].resource
            log.info("  C2 BINDS: %s cannot coexist -- sum d_bar=%.1f > E_{%s,%s}=%.1f",
                     js, sum(claims[j].d_bar for j in js), tid, res,
                     tenants[tid].envelope.get(res, float('inf')))
        # cross-tenant sanity, for the record
        tot = sum(t.envelope.get("PRB", 0.0) for t in tenants.values())
        log.info("  cross-tenant: sum of envelopes = %.1f PRB -> "
                 "cross-tenant over-commitment impossible by construction", tot)

    # ------------------------------------------------------------------
    def is_admissible(self, S: Set[str]) -> bool:
        """C1 and C2 in one call.  Called for every candidate set."""
        for pair in self.c1_pairs:                    # C1
            if pair <= S:
                return False
        for fs in self.c2_forbidden:                  # C2
            if fs <= S:
                return False
        return True
