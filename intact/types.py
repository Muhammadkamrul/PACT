"""
intact/types.py
===============
The five objects of the framework (INTACT v11 §3.1) as plain dataclasses.

Everything downstream -- estimation, the outer loop, the inner loop -- is
built from ONLY these types.  If you understand this file you can read the
rest of the codebase.

Mapping to the theory document (v11):
    Tenant       -> §3.1  n
    Intent       -> §3.1  i
    Claim        -> §3.1  j = (x, p, sigma)
    Write        -> §3.5  (x, p, nu_req)
    Disposition  -> §10.4 admit / override / reject  (+ escalation EMISSION)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Scope of a control parameter.  v11 §2.1 "Scoped authorisation".
# The scope decides WHO can be affected by a write, and therefore whether a
# cross-tenant effect is even structurally possible.
# ---------------------------------------------------------------------------
class _CaseInsensitive(str, Enum):
    """Accept "CELL", "cell", "Cell" from YAML without fuss."""
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for m in cls:
                if m.value.lower() == value.lower():
                    return m
        return None


class Scope(_CaseInsensitive):
    UE = "UE"         # affects only the UEs of one tenant
    SLICE = "slice"   # affects one slice (still may leak via shared PHY)
    CELL = "cell"     # affects the whole cell -> every tenant.  HOST ONLY.


# ---------------------------------------------------------------------------
# Kind of a control parameter.  v11 §8.3.1.
# The test is one question: does the write RESERVE capacity, or BOUND it?
#   REGULATIVE  -> a ceiling.  Two caps of 82 do not reserve 164 PRBs.
#                  C2 is VACUOUS for these.
#   ALLOCATIVE  -> a reservation.  The value IS the commitment.
#                  C2 applies:  sum of d_bar over co-eligible claims <= E_{n,r}
# ---------------------------------------------------------------------------
class Kind(_CaseInsensitive):
    REGULATIVE = "regulative"
    ALLOCATIVE = "allocative"


class Direction(_CaseInsensitive):
    """Is a higher KPI value better (+1) or worse (-1)?  v11 §4."""
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


class Outcome(str, Enum):
    """The inner loop's disposition.  ALWAYS exactly one of these (v11 §10.4)."""
    ADMIT = "admit"
    OVERRIDE = "override"
    REJECT = "reject"


@dataclass
class Tenant:
    """
    A customer of the neutral host.  v11 §3.1.

    Attributes
    ----------
    tid            short id, e.g. "T1"
    omega          commercial weight  omega_n   (contract term, v11 §5b)
    rho_min        contracted fulfilment floor  rho_n^min  (C3, v11 §9)
    envelope       per-resource envelope  E_{n,r}(tau)  in PRBs.
                   *** Supplied by the non-RT RIC.  INTACT does NOT compute it. ***
                   v11 §8.3.1.
    B_n            sovereignty budget: max host OVERRIDES per window (count)
    Bbar_n         sovereignty budget: max cumulative override MAGNITUDE
    """
    tid: str
    omega: float
    rho_min: float
    envelope: Dict[str, float] = field(default_factory=dict)   # {"PRB": 30.0}
    B_n: int = 20
    Bbar_n: float = 200.0
    is_host: bool = False


@dataclass
class Intent:
    """
    One tenant's declared performance requirement.  v11 §3.1, §4.

    The margin g_i is computed from (kpi_name, target, direction); see
    intact/estimation/margins.py.  Everything in the framework operates on
    the *dimensionless* margin, never the raw KPI -- that is what makes
    "which intent is in more trouble" a well-posed question across tenants.
    """
    iid: str
    tenant: str
    kpi: str                       # key into the RAN's KPM dict, e.g. "throughput_mbps"
    target: float                  # theta_i
    direction: Direction
    pi_class: float                # priority class  pi^class  (contract term)
    eta: float = 0.95              # required fulfilment fraction over the window
    epsilon: float = 0.02          # safety floor eps_i used by the inner loop (v11 §10.3)
    clip: float = float('inf')     # |g_i| bound.  'twice as bad as promised' is not
                                   # meaningfully better than 'three times' -- both are a
                                   # total breach.  Unclipped, one bad epoch dominates
                                   # every weighted sum in the framework (v11 §4).


@dataclass
class Claim:
    """
    "xApp x's permission to write parameter p at scope sigma."
    THE UNIT WE SCHEDULE.  v11 §3.1.

    Why the claim and not the xApp:  an xApp writing three parameters, only
    one of which clashes, would lose two harmless knobs under xApp-level
    blocking.  When ALL of an xApp's parameters clash, claim-level blocking
    gives the identical answer -- so it never does worse.
    """
    jid: str
    xapp: str
    tenant: str
    param: str                     # key into the RAN's control dict
    scope: Scope
    kind: Kind
    domain: Tuple[float, float]    # V_p  = admissible value range (inclusive)
    step: float                    # granularity of the domain grid
    r_j: float                     # contracted actuation rate (v11 §8.4)
    resource: Optional[str] = None # for ALLOCATIVE claims: which resource, e.g. "PRB"
    d_bar: float = 0.0             # max resource demand  d_bar_{j,r}   (v11 §8.3.2)
                                   # = sup over value AND context of d_r(nu,c).
                                   # For direct controls d_r(nu)=nu so d_bar = max(domain).

    max_step_frac: float = 1.0     # SLEW LIMIT (trust region): the largest
                                   # single-epoch move, as a fraction of the
                                   # knob's domain width.  1.0 = unconstrained.
                                   #
                                   # WHY IT EXISTS.  s_{p,i} is a LOCAL
                                   # derivative and the inner loop uses it as a
                                   # FIRST-ORDER extrapolation,
                                   #   g_hat(nu) = g(nu_old) + s*(nu - nu_old),
                                   # whose error grows with |nu - nu_old|^2.
                                   # Without a limit, the largest writes -- the
                                   # ones that matter most -- are judged by a
                                   # linear model far outside the region it was
                                   # fitted in.  Confining the move keeps the
                                   # linearisation self-consistent.

    def grid(self) -> List[float]:
        """The discrete admissible value set V_p that the projector searches."""
        lo, hi = self.domain
        n = int(round((hi - lo) / self.step))
        return [round(lo + k * self.step, 6) for k in range(n + 1)]


@dataclass
class Write:
    """A pending control message from an eligible claim.  v11 §3.5."""
    jid: str
    param: str
    nu_req: float                  # the value the xApp proposed
    nu_old: float                  # the value currently applied
    epoch: int
    slot: int


@dataclass
class Decision:
    """
    The full record of one inner-loop decision.  Everything needed to audit
    the mediator after the fact -- this is what makes the framework
    defensible to a tenant who disputes an override.
    """
    write: Write
    outcome: Outcome
    nu_star: float
    implicated: List[str] = field(default_factory=list)   # the set A (v11 §10.1)
    escalated: bool = False
    escalation_reason: str = ""     # "range-exhausted" | "safety-infeasible"
    latency_ms: float = 0.0
    magnitude: float = 0.0          # |nu* - nu_req|, charged against Bbar_n
    slew_capped: bool = False       # the request exceeded the trust region
                                    # and was clamped before analysis
