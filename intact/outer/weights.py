"""
intact/outer/weights.py   --  INTACT v11 §5, §9
===============================================
w_i = pi^class  x  omega_tenant  x  u(p_hat_i)  x  (1 + theta_Lambda * Lambda_n)

    pi^class : what kind of service.  Contract term, fixed at onboarding.
    omega    : what the tenant pays.  Contract term.
    u(p_hat) : URGENCY MULTIPLIER, recomputed every epoch.
    Lambda_n : tenant fulfilment deficit -- Layer 3 of the fairness stack.

*** u IS A MULTIPLIER, NEVER THE RAW RISK ***
    u(p) = 1 + lambda_u * p / (1 - p + eps0)     ->  u >= 1 ALWAYS.
At p=0 it equals 1, so a HEALTHY intent still competes on contract value.
As p->1 it grows without bound, so an intent seconds from breaching outranks
a comfortable premium one.
Substituting raw risk (w = pi*omega*p_hat) is a LIVE BUG: a healthy intent
gets p~0, hence w~0, hence V_j~0 for every claim, and THE SCHEDULER GOES
BLIND whenever the network is healthy -- which is most of the time.

lambda_u IS NOT COSMETIC.  It decides whether urgency can override contract
rank, and the ranking FLIPS across its range:
    lambda_u = 1.0 :  w(T2, risk .85) = 4.545  >  w(T3, risk .72) = 4.179
                      -> urgency wins; the at-risk cheaper tenant leads
    lambda_u = 0.1 :  w(T2) = 1.103            <  w(T3) = 1.498
                      -> contract wins
Report the value used in every experiment, and sweep it (v11 §12).
"""
from __future__ import annotations
from typing import Dict
from ..types import Claim, Intent, Tenant


def urgency(p_hat: float, lambda_u: float, eps0: float = 0.01) -> float:
    """u(p_hat).  Bounded below by 1 by construction."""
    p = min(max(p_hat, 0.0), 0.999)
    return 1.0 + lambda_u * p / (1.0 - p + eps0)


def relative_risk(p_hat: Dict[str, float]) -> Dict[str, float]:
    """Within-epoch rank of p_hat, mapped to [0, 1].

    WHY A RANK AND NOT THE PROBABILITY ITSELF
    -----------------------------------------
    Eligibility is a RANKING over w_i, so any factor common to every intent
    cancels and changes no decision.  The urgency term can therefore only act
    through its DISPERSION ACROSS INTENTS AT ONE EPOCH -- "who is worst right
    now" -- which is a comparative question, not an absolute one.

    A well-trained risk model answers the ABSOLUTE question instead: it is
    fitted and calibrated per intent against that intent's own base rate.
    When every intent's base rate is high, every calibrated p_hat is high,
    and the cross-intent contrast the outer loop needs disappears even though
    the model is individually accurate.  Measured on s001 with a MILD model
    scoring macro PR-AUC 0.956 and mean lift 1.62x:

        predictor    mean p_hat   cross-intent sd   CV(u)/CV(base)
        analytical      0.412          0.386             2.27
        MILD            0.883          0.089             0.78

    MILD is 4.3x LESS spread across intents than the analytical stand-in, and
    with lambda_u = 1.0 the 1/(1-p) term then explodes (u spans 2.3-74),
    so urgency stops modulating the contract weight and starts REPLACING it.

    Ranking is invariant to that saturation.  It is also invariant to any
    monotone recalibration of the predictor, so the outer loop no longer
    depends on p_hat being calibrated at all -- only on it being correctly
    ORDERED, which is exactly what lift measures and what MILD is good at.

    Ties share the mid-rank, EXCEPT when every intent ties: then the rank is
    0 for all of them, not the mid-rank.

    *** That exception is load-bearing, not a detail. ***  Mid-ranking a
    fully-tied epoch returns 0.5 everywhere, and u(0.5) = 1.98 at
    lambda_u = 1.0 -- so a system with NO risk information at all would
    silently carry a ~2x urgency multiplier.  It cancels from the ranking, so
    no decision changes, but it breaks two things that matter:

      * the invariant u(no risk) = 1, which rung L19 checks exactly, and
      * the graceful-degradation property (P6): a constant p_hat must reduce
        B6++ EXACTLY to B6++-U, not to B6++ with a constant boost.

    Same reasoning for the single-intent case: one intent carries no
    comparative information, so urgency is inert by construction.
    """
    iids = list(p_hat)
    k = len(iids)
    if k <= 1:
        return {i: 0.0 for i in iids}
    vals = [p_hat.get(i, 0.0) for i in iids]
    if max(vals) - min(vals) < 1e-12:
        return {i: 0.0 for i in iids}
    order = sorted(iids, key=lambda i: p_hat.get(i, 0.0))
    ranks: Dict[str, float] = {}
    pos = 0
    while pos < k:
        end = pos
        while (end + 1 < k
               and abs(p_hat.get(order[end + 1], 0.0)
                       - p_hat.get(order[pos], 0.0)) < 1e-12):
            end += 1
        mid = 0.5 * (pos + end)
        for t in range(pos, end + 1):
            ranks[order[t]] = mid / (k - 1)
        pos = end + 1
    return ranks


def compute_weights(intents: Dict[str, Intent], tenants: Dict[str, Tenant],
                    p_hat: Dict[str, float], lam: Dict[str, float],
                    cfg: Dict) -> Dict[str, float]:
    """
    Returns w_i for every intent.

    lam : the TENANT deficit Lambda_n (v11 §9).  The longer we under-serve a
          tenant relative to its contracted floor, the more its intents are
          worth in the next decision -- until the system is forced to serve
          it.  Same mechanism as the claim deficit, one level up.  Two
          counters, and the whole fairness story is those two counters.

    outer.urgency_mode selects what u(.) reads:
        "absolute"  u(p_hat_i)           -- DEFAULT, v18.2 behaviour
        "relative"  u(rank of p_hat_i)   -- opt-in, for saturated predictors

    *** absolute is the default deliberately. ***  Making "relative" the
    default silently changed B6++ everywhere -- including the known-answer
    ladder rungs and every previously-reported ensemble number -- which is
    exactly the kind of unannounced behaviour change that invalidates a
    frozen result set.  Opt in per experiment with
    --set outer.urgency_mode=relative.
    """
    lu = cfg["outer"]["lambda_u"]
    e0 = cfg["outer"]["eps0"]
    tl = cfg["outer"]["theta_Lambda"]
    mode = cfg["outer"].get("urgency_mode", "absolute")
    # v10 registered weight-factor ablations.  Each factor of
    #     w_i = pi_i * omega_n * u(p_i) * (1 + theta_Lambda * Lambda_n)
    # can be switched off EXACTLY (factor := 1).  All defaults are true and
    # reproduce the frozen weights bit-for-bit.  theta_Lambda = 0 already
    # removes the tenant-deficit factor exactly, so it needs no flag.
    use_pi = bool(cfg["outer"].get("weight_use_pi", True))
    use_omega = bool(cfg["outer"].get("weight_use_omega", True))
    use_u = bool(cfg["outer"].get("weight_use_urgency", True))
    signal = (relative_risk(p_hat) if mode == "relative"
              else {i: p_hat.get(i, 0.0) for i in intents})
    w = {}
    for iid, it in intents.items():
        base = ((it.pi_class if use_pi else 1.0)
                * (tenants[it.tenant].omega if use_omega else 1.0))
        u = urgency(signal.get(iid, 0.0), lu, e0) if use_u else 1.0
        floor_boost = 1.0 + tl * lam.get(it.tenant, 0.0)
        w[iid] = base * u * floor_boost
    return w


def claim_weights_from_intents(claims: Dict[str, Claim],
                               intents: Dict[str, Intent],
                               tenants: Dict[str, Tenant],
                               intent_weights: Dict[str, float]) -> Dict[str, float]:
    """Aggregate per-intent weights onto every claim owned by that tenant.

    This deliberately does *not* use a claim-effect estimate.  It answers the
    tenant-level question "whose overdue claim receives authority next?"; the
    proposal-aware, signed value question remains with the inner ``s``-based
    mediator.  Consequently beta, gamma and eta never enter B6++ eligibility.
    """
    mass = {t: 0.0 for t in tenants}
    for iid, intent in intents.items():
        mass[intent.tenant] = mass.get(intent.tenant, 0.0) + float(
            intent_weights.get(iid, 0.0))

    out = {}
    for jid, claim in claims.items():
        tenant = claim.tenant
        if tenant not in tenants:
            raise ValueError(f"claim {jid} references unknown tenant {tenant}")
        if mass.get(tenant, 0.0) <= 0.0:
            raise ValueError(
                f"B6++ requires claim-owning tenant {tenant} to have positive "
                "aggregate intent weight")
        out[jid] = mass[tenant]
    return out


def contractual_claim_weights(claims: Dict[str, Claim],
                              intents: Dict[str, Intent],
                              tenants: Dict[str, Tenant],
                              lam: Dict[str, float],
                              cfg: Dict) -> Dict[str, float]:
    """B6++-U ablation: INTACT weights with urgency fixed to ``u(0)=1``.

    For a claim owned by tenant ``n``::

        w_j^{-U} = omega_n * (sum_{i in I_n} pi_i)
                   * (1 + theta_Lambda * Lambda_n)

    The terms have deliberately narrow provenance:

    * ``pi_i`` and ``omega_n`` are fixed contractual terms;
    * ``Lambda_n`` is updated only from the observed tenant fulfilment and
      its contracted floor ``rho_min``;
    * there is no urgency multiplier, beta, gamma, eta, or learned value model.

    B6++ multiplies the ordinary claim deficit by this value before passing
    it to the selector.  Since the selector contributes ``theta_D * D_j``,
    this produces exactly ``theta_D * w_j^contract * D_j``.

    A claim-owning tenant is expected to declare at least one intent.  Raise
    early instead of silently assigning weight zero if a malformed scenario
    violates that contract-model invariant.
    """
    zero_risk = {iid: 0.0 for iid in intents}
    wi = compute_weights(intents, tenants, zero_risk, lam, cfg)
    return claim_weights_from_intents(claims, intents, tenants, wi)


def risk_aware_claim_weights(claims: Dict[str, Claim],
                             intents: Dict[str, Intent],
                             tenants: Dict[str, Tenant],
                             p_hat: Dict[str, float],
                             lam: Dict[str, float],
                             cfg: Dict) -> Dict[str, float]:
    """B6++ claim weights using exactly INTACT's dynamic intent weight.

    For claim ``j`` owned by tenant ``n``::

        w_j^{++} = sum_{i in I_n} [pi_i * omega_n * u(p_hat_i)
                                   * (1 + theta_Lambda * Lambda_n)]

    Risk raises the *tenant's authority urgency*.  It does not assert that a
    particular claim helps an at-risk intent; signed proposal-level protection
    remains the job of the inner sensitivity mediator.
    """
    wi = compute_weights(intents, tenants, p_hat, lam, cfg)
    return claim_weights_from_intents(claims, intents, tenants, wi)
