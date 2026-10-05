"""
intact/v2/experiment_v2.py
==========================
INTACTv2 as a SUBCLASS of the v1 Experiment.

WHY A SUBCLASS AND NOT AN EDIT
------------------------------
`intact/experiment.py` carries every v1 baseline and passes a 24-rung
known-answer ladder.  Editing it risks silently changing a baseline and
invalidating results you have already reported.  `ExperimentV2` overrides
exactly three things and inherits the rest, so:

  * every v1 mode behaves bit-identically to before, and
  * `--modes intactv2` is purely additive.

WHAT IS OVERRIDDEN

  _select()      leverage-weighted claim values, dose-scaled pair harm with
                 an ESTIMATED q_hat, proposal-gated admission
  _write_stage() a running margin ledger with deterministic ordering and a
                 risk-modulated safety buffer
  run()          records opportunity / proposal / actuation as three
                 distinct events, and the implicit-conflict detectors

THE FIVE v1 DEFECTS THIS CLOSES

 1. weight was tenant-aggregate  -> leverage_claim_weights, so MILD's p_hat
    reaches only the claims that can move the at-risk intent
 2. deficit discharged on OPPORTUNITY -> DeficitLedgerV2 discharges on
    ACTUATION, and a silent claim no longer pays its debt with silence
 3. q_hat hardcoded 1.0          -> CoFireTracker estimates it from the log
 4. no running margin ledger     -> the second claim now sees the first
    claim's predicted effect; v1 was sequential in name and parallel in fact
 5. MILD only gated              -> eps_eff = eps * (1 + lambda_risk * p_hat)
    puts the risk score where the decisions actually bind
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..experiment import Experiment
from ..types import Decision, Kind, Outcome, Write
from .arbiter_v2 import ArbiterV2, InnerDecision, effective_epsilon
from .outer_v2 import (CoFireTracker, DeficitLedgerV2, ImplicitConflictDetector,
                       leverage_claim_weights, pair_harm_v2)
from .ran_v2 import load_dependencies
from .state import StateBuilder
from .wif_objective import (individual_utilities,
                            portfolio_linear_margin_utility,
                            portfolio_wif_utility)

V2_MODES = ("intactv2", "intactv3", "intactv3_wif")

# INTACTv3 = INTACTv2 with the CLAIM-DEFICIT term removed.
#
# The deficit D_j enforces C4, a contracted minimum ACTUATION RATE per claim.
# It is a fairness guarantee to the xApp, not to the tenant, and it is worth
# asking whether it earns its place: what a tenant buys is INTENT FULFILMENT,
# and a claim that is starved while its tenant's intents are met has lost
# nothing the tenant cares about.  v3 replaces D_j with 1, so eligibility is
# ranked on leverage-weighted contract priority alone:
#       v2:  S = argmax theta_D * sum_j c_j * D_j     (priority x overdue-ness)
#       v3:  S = argmax theta_D * sum_j c_j           (priority only)
# Everything else -- C1, C2, the inner loop, MILD -- is identical, so the
# pair is a clean one-term ablation of the actuation floor.


class ExperimentV2(Experiment):
    """Adds mode 'intactv2'.  All v1 modes are inherited untouched."""

    def __init__(self, cfg: Dict, rundir, mode: str = "intactv2"):
        super().__init__(cfg, rundir, mode)
        self.is_v2 = mode in V2_MODES
        if not self.is_v2:
            return

        v2 = cfg.get("v2", {}) or {}
        self.state_builder = StateBuilder(cfg)
        self.deficit_v2 = DeficitLedgerV2(
            self.claims, cap=float(cfg["outer"].get("deficit_cap", 500.0)),
            discharge_on=v2.get("deficit_discharge_on", "actuation"),
            accrual_on=v2.get("c4_accrual_on", "qualified"))
        # Preserve the established deterministic inner ordering as a separate
        # mechanism.  Reusing the corrected C4 queue here changed INTACTv3
        # even though its outer C4 term was disabled, invalidating the
        # ablation.  This legacy epoch queue is never reported as C4 and never
        # enters outer selection.
        self.order_deficit_v2 = DeficitLedgerV2(
            self.claims, cap=float(cfg["outer"].get("deficit_cap", 500.0)),
            discharge_on="actuation", accrual_on="epoch")
        self.cofire = CoFireTracker(self.pairs)
        self.deps = load_dependencies(cfg)
        self.implicit = ImplicitConflictDetector(self.deps, self.claims)
        self.proposal_gated = bool(v2.get("proposal_gated_admission", True))
        self.qualified_gated = bool(v2.get("qualified_gated_admission", False))
        self.outer_score_dose = str(v2.get("outer_score_dose", "requested"))
        self.inner_order = str(v2.get("inner_order", "deficit"))
        self.c4_min_primary_gain = float(v2.get("c4_min_primary_gain", 1e-9))
        self.c4_selection_mode = str(
            v2.get("c4_selection_mode", "performance_frontier"))
        self.c4_performance_slack = float(
            v2.get("c4_performance_slack", 0.0))
        self.require_exact_frontier = bool(
            v2.get("require_exact_frontier", True))
        # INTACTv3-WIF is a development candidate, kept separate from the
        # frozen v3 replication label.  Its primary outer objective matches
        # the reported wIF definition (pi_class-weighted threshold crossing),
        # while the optional policy score may break only near-ties inside an
        # explicit and auditable primary slack.
        self.wif_temperature = float(v2.get("wif_temperature", 0.04))
        self.wif_action_cost = float(v2.get("wif_action_cost", 0.0))
        self.wif_confidence_z = float(v2.get("wif_confidence_z", 0.0))
        self.wif_performance_slack = float(
            v2.get("wif_performance_slack", 0.0))
        self.wif_primary_objective = str(
            v2.get("wif_primary_objective", "linear"))
        self.wif_linear_use_pi = bool(v2.get("wif_linear_use_pi", False))
        # Revised INTACTv2/v3 can optimise the metric it is evaluated on:
        # rescue high-priority intents near g=0 instead of accumulating
        # surplus margin on already-safe intents.  This is intentionally
        # opt-in so old result folders remain exactly reproducible.
        self.use_threshold_primary = bool(
            v2.get("use_threshold_primary", False))
        # Predictive reserve moves the smooth fulfilment boundary *before* a
        # forecast failure.  At p=0 the ordinary current-margin objective is
        # recovered exactly; at high p, a corrective write has value before
        # the current margin crosses zero.  This is the mechanism B3 lacks.
        self.risk_margin_reserve = float(
            v2.get("risk_margin_reserve", 0.0))
        # The earlier predictive-linear path computed a risk reserve but then
        # called a plain linear utility that never read it.  With similar
        # current/future slopes, MILD was therefore almost inert.  Keep signed
        # margin as the anchor and add only bounded primary-risk leverage.
        self.predictive_risk_gain = float(
            v2.get("predictive_risk_gain", 1.0))
        self.predictive_rescue_gain = float(
            v2.get("predictive_rescue_gain", 0.25))
        # The v1 gamma table estimates a binary eligibility effect.  v2/v3
        # score concrete write doses, so silently feeding that scalar into
        # their pair penalty mixes estimands.  It was also contrary to the
        # documented "v3 uses only validated s" design.  Keep the legacy
        # behaviour only as an explicit ablation.
        self.use_v1_eligibility_gamma = bool(
            v2.get("use_v1_eligibility_gamma", False))
        # ---- v10 registered ablation switches -------------------------
        # Every default reproduces the frozen INTACTv3 decisions exactly
        # (checked by tests/test_benchmark_additions.py).  They exist so the
        # component ablation is a configuration switch, never a code edit:
        #   predictive_use_pi=false   drop the contractual pi_class factor
        #                             from the predictive primary weights;
        #   joint_protection=false    drop the risk-gated joint safety check
        #                             from outer portfolio scoring (needed
        #                             because a MILD model refuses a changed
        #                             inner.tau_risk);
        #   threshold_protection_policy
        #                             inner-loop veto policy used with the
        #                             threshold/predictive objective.  The
        #                             frozen method uses portfolio_authorized
        #                             (no per-write veto); no_new_victims
        #                             restores the per-write safety veto.
        # v10: the offered-load forecast every consumer sees (xApps, the
        # sensitivity blend, MILD features, INTACTv3-F).  Default "oracle"
        # keeps the exact schedule value, i.e. previous behaviour.
        from ..estimation.forecast import ForecastProvider
        self.forecaster = ForecastProvider(cfg, self.log)
        self.planning_regime_source = str(
            v2.get("planning_regime_source", "tenant")).lower()
        if self.planning_regime_source not in ("tenant", "tenant_now",
                                               "cell", "cell_now"):
            raise ValueError("v2.planning_regime_source must be tenant, "
                             "tenant_now, cell or cell_now")
        self.predictive_use_pi = bool(v2.get("predictive_use_pi", True))
        self.joint_protection = bool(v2.get("joint_protection", True))
        self.threshold_protection_policy = str(
            v2.get("threshold_protection_policy", "portfolio_authorized"))
        if self.threshold_protection_policy not in (
                "portfolio_authorized", "no_new_victims", "strict"):
            raise ValueError(
                "v2.threshold_protection_policy must be portfolio_authorized,"
                " no_new_victims or strict")

        self.arbiter_v2 = ArbiterV2(
            cfg, self.intents, self.claims,
            s_lookup=lambda r, p, i: self.sens.get(r, p, i),
            feasible_fn=self._feasible_grid)
        if self.mode == "intactv3_wif" or self.use_threshold_primary:
            self.arbiter_v2.protection_policy = self.threshold_protection_policy

        self.params = sorted({c.param for c in self.claims.values()})
        self.v2_log: List[Dict] = []
        self.structural_conflicts = 0
        self.binding_edges = 0
        self.selection_diagnostics: List[Dict] = []
        self._last_claim_priorities: Dict[str, float] = {}
        self._last_intent_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _preview_proposals(self, g, p_hat, regime, proposals):
        """Project every proposal independently before outer scoring.

        The old outer loop scored the requested dose even when the inner
        safety projection later clipped most of it.  The preview uses the
        same grid, C2 envelope and safety logic as execution, so the outer
        score is based on the dose that can actually land.  It is deliberately
        individual (not a combinatorial rollout); the running ledger remains
        the final authority during execution.
        """
        controls = dict(self.ran.current_controls())
        qualified, projected = set(), {}
        for jid in sorted(self.claims):
            if jid not in proposals:
                continue
            nu_req, nu_reported_old = proposals[jid]
            nu_old = float(controls.get(self.claims[jid].param, nu_reported_old))
            if abs(float(nu_req) - nu_old) <= 1e-12:
                continue
            d = self.arbiter_v2._decide_one(
                jid, self.claims[jid], nu_old, float(nu_req), dict(g), p_hat,
                regime, dict(controls), 0)
            dose = float(d.nu_star - d.nu_old)
            projected[jid] = dose
            if d.outcome != "reject" and abs(dose) > 1e-12:
                qualified.add(jid)
        return qualified, projected

    # ------------------------------------------------------------------
    def _feasible_grid(self, jid, c, grid, controls):
        """Unconditional C2 clamp on the EXECUTED value (physics, not policy).

        *** Reads the RUNNING `controls` ledger, not ran.headroom(). ***
        The inner loop is sequential: decide_all updates `controls` after each
        decision but only pushes to the RAN at the end of the epoch, so
        ran.headroom() is PRE-EPOCH state.  Checking against it let the second
        allocative write of an epoch see an envelope that did not yet include
        the first, which produced 0.044 C2 violations per epoch for INTACTv2
        while B6++ and INTACTv1 were exactly 0.  Summing the tenant's own
        allocative parameters out of the live ledger is the same quantity the
        RAN would report, one step earlier.
        """
        if c.kind != Kind.ALLOCATIVE or not getattr(c, "resource", None):
            return grid
        try:
            env, _ = self.ran.headroom(c.tenant, c.resource)
        except Exception:
            return grid
        used_by_others = 0.0
        for k, other in self.claims.items():
            if k == jid or other.tenant != c.tenant:
                continue
            if other.kind != Kind.ALLOCATIVE or getattr(other, "resource", None) != c.resource:
                continue
            if other.param == c.param:
                continue
            used_by_others += float(controls.get(other.param, 0.0))
        return grid[grid <= (env - used_by_others) + 1e-9]

    # ------------------------------------------------------------------
    def _select_v2(self, g, p_hat, regime, proposals,
                   qualified=None, projected_dose=None) -> Set[str]:
        """Outer loop with leverage weights and an estimated q_hat."""
        from ..outer.weights import compute_weights
        w_policy = compute_weights(self.intents, self.tenants, p_hat,
                                   self.deficits.Lam, self.cfg)
        # Only relative weights matter to dose selection.  Normalising avoids
        # the urgency transform near p=1 changing the numerical scale of the
        # local objective or its minimum-gain tolerance.
        wmax = max((abs(float(v)) for v in w_policy.values()), default=1.0)
        wmax = wmax if wmax > 1e-12 else 1.0
        self._last_intent_weights = {
            i: float(w_policy.get(i, 0.0)) / wmax for i in self.intents}

        # DEFECT 1: claim-specific weight via the validated s, not the
        # tenant aggregate.  Normalised across PARAMETERS for a fixed
        # intent -- the other direction makes every claim's weight sum to
        # the same constant and reproduces the v1 problem in disguise.
        plan_s = self.arbiter_v2.planning_s
        cw = leverage_claim_weights(self.claims, self.intents, w_policy, self.params,
                                    plan_s,
                                    regime)

        # DEFECT 3: q_hat from the operation log, not a hardcoded 1.0.
        q_hat = self.cofire.q_hat()
        qualified = set(qualified or ())
        projected_dose = projected_dose or {}
        exp_dose = {
            j: abs(float(projected_dose.get(j, proposals[j][0] - proposals[j][1])))
            if self.outer_score_dose == "projected"
            else abs(float(proposals[j][0] - proposals[j][1]))
            for j in proposals}
        H = (pair_harm_v2(
            self.pairs, self.intents, w_policy,
            lambda r, pr, i: self.effects.gamma.get(i, {}).get(pr, 0.0),
            regime, q_hat, exp_dose)
             if self.use_v1_eligibility_gamma else {})

        # DEFECT 2 (part): a claim with nothing to write should not occupy
        # a C1 slot.  Proposal-gated admission frees the slot for a claim
        # that has something to say.
        cands = list(self.claims)
        if self.qualified_gated:
            cands = [j for j in cands if j in qualified]
        elif self.proposal_gated:
            # Assign even when empty.  The previous ``if live`` fallback
            # accidentally re-enabled every silent claim when no live
            # proposal existed in an epoch.
            cands = [j for j in cands
                     if j in proposals
                     and abs(proposals[j][0] - proposals[j][1]) > 1e-12]

        if self.mode == "intactv3_wif" or self.use_threshold_primary:
            # Outer scoring must preview PHYSICAL feasibility, not run the
            # complete per-claim safety veto in isolation.  The latter hides
            # portfolios where one claim creates headroom that makes another
            # claim safe.  C1/C2 are enforced here; the deterministic running
            # inner ledger remains final authority for safety and may still
            # project or reject a selected write.
            controls = dict(self.ran.current_controls())
            wif_dose = {}
            for j in cands:
                c = self.claims[j]
                nu_req, nu_reported_old = proposals[j]
                nu_old = float(controls.get(c.param, nu_reported_old))
                if self.arbiter_v2.local_optimize:
                    d = self.arbiter_v2._decide_one(
                        j, c, nu_old, float(nu_req), dict(g), p_hat,
                        regime, dict(controls), 0, self._last_intent_weights)
                    dose = float(d.nu_star) - nu_old
                else:
                    grid = self.arbiter_v2._grid(c, nu_old)
                    grid = self._feasible_grid(j, c, grid, controls)
                    if not len(grid):
                        continue
                    nu_eff = float(grid[np.argmin(
                        np.abs(grid - float(nu_req)))])
                    dose = nu_eff - nu_old
                if abs(dose) > 1e-12:
                    wif_dose[j] = dose
            allowed = set(wif_dose)

            def se_lookup(r, p, i):
                row = self.sens.tab.get((r, p, i))
                return float(row[1]) if row is not None else 0.0

            objective_args = dict(
                claims=self.claims, intents=self.intents, margins=g,
                regime=regime, projected_dose=wif_dose,
                sensitivity=plan_s,
                sensitivity_se=se_lookup,
                # Keep the primary metric's registered pi_class weighting;
                # tenant priority/C3/MILD remain in the secondary and inner
                # dose objectives.  Risk changes *when* an intent becomes
                # valuable through this pre-failure reserve.
                margin_reserve={
                    i: self.risk_margin_reserve * float(p_hat.get(i, 0.0))
                    for i in self.intents},
                temperature=self.wif_temperature,
                confidence_z=self.wif_confidence_z,
                action_cost=self.wif_action_cost)

            # Secondary policy value contains commercial priority, MILD and
            # C3 tenant recovery, but can never displace the best predicted
            # wIF set unless the declared primary slack permits it.  C4 claim
            # deficit is intentionally absent.
            def secondary(S):
                policy = sum(
                    float(w_policy.get(i, 0.0)) * sum(
                        plan_s(regime, self.claims[j].param, i)
                        * float(wif_dose.get(j, 0.0)) for j in S)
                    for i in self.intents)
                if self.mode != "intactv2":
                    return policy
                # C4 remains subordinate to predicted wIF.  It can select an
                # overdue, beneficial claim only inside the explicitly
                # declared primary slack; it can never buy an unbounded loss.
                d_cap = max(float(self.cfg["outer"].get(
                    "deficit_cap", 500.0)), 1e-9)
                c_scale = max((abs(float(v)) for v in cw.values()),
                              default=1.0) or 1.0
                # A threshold objective is deliberately flat for writes that
                # only add surplus to an already-fulfilled intent.  Treating
                # every such numerical tie as permission for C4 caused v2 to
                # execute 4--7 times as many writes as v3 and lose wIF through
                # accumulated model error.  C4 may break a tie only for a
                # claim with independently positive predicted primary value;
                # debt is not evidence that a write is useful.
                c4 = sum((float(cw.get(j, 0.0)) / c_scale)
                         * (float(self.deficit_v2.D.get(j, 0.0)) / d_cap)
                         for j in S
                         if base_primary({j}) > self.c4_min_primary_gain)
                return policy + float(self.cfg["outer"]["theta_D"]) * c4

            if self.wif_primary_objective == "threshold":
                base_primary = lambda S: portfolio_wif_utility(
                    S, **objective_args)
                priority_args = objective_args
            elif self.wif_primary_objective == "linear":
                linear_args = dict(
                    claims=self.claims, intents=self.intents, regime=regime,
                    projected_dose=wif_dose,
                    sensitivity=lambda r, p, i: self.sens.get(r, p, i),
                    use_pi_class=self.wif_linear_use_pi,
                    action_cost=self.wif_action_cost)
                base_primary = lambda S: portfolio_linear_margin_utility(
                    S, **linear_args)
                priority_args = None
            elif self.wif_primary_objective == "predictive_linear":
                # Conservative predictive generalisation of B3.  The signed
                # margin term remains the anchor.  Risk has three bounded
                # levers: an intent weight, the future-regime sensitivity
                # blend in plan_s, and a small smooth rescue term around the
                # risk-dependent reserve.  This avoids both computed-but-
                # inert risk and the flat ties of a threshold-only objective.
                predictive_weights = {
                    i: (float(self.intents[i].pi_class)
                        if self.predictive_use_pi else 1.0) * (
                        1.0 + self.predictive_risk_gain
                        * float(np.clip(p_hat.get(i, 0.0), 0.0, 1.0)))
                    for i in self.intents}
                linear_args = dict(
                    claims=self.claims, intents=self.intents, regime=regime,
                    projected_dose=wif_dose, sensitivity=plan_s,
                    use_pi_class=False, intent_weights=predictive_weights,
                    action_cost=self.wif_action_cost)
                rescue_args = dict(objective_args)
                rescue_args["intent_weights"] = predictive_weights

                def base_primary(S):
                    signed = portfolio_linear_margin_utility(S, **linear_args)
                    rescue = portfolio_wif_utility(S, **rescue_args)
                    return signed + self.predictive_rescue_gain * rescue
                priority_args = None
            else:
                raise ValueError(
                    "v2.wif_primary_objective must be 'linear', "
                    "'predictive_linear' or 'threshold'")

            def primary(S):
                # Joint safety check: permit offsetting claims, but never let
                # a portfolio move a currently protected intent below its
                # risk-adjusted floor.  The former per-claim preview rejected
                # such portfolios before their combined effect was visible.
                # v10: v2.joint_protection=false is the registered ablation.
                if not self.joint_protection:
                    return base_primary(S)
                for i, intent in self.intents.items():
                    if float(p_hat.get(i, 0.0)) < self.arbiter_v2.tau_risk:
                        continue
                    eps = effective_epsilon(
                        float(getattr(intent, "epsilon", 0.02)),
                        float(p_hat.get(i, 0.0)),
                        self.arbiter_v2.lambda_risk,
                        self.arbiter_v2.risk_mode)
                    g0 = float(g.get(i, 0.0))
                    after = g0 + sum(
                        plan_s(regime, self.claims[j].param, i)
                        * float(wif_dose.get(j, 0.0)) for j in S)
                    if g0 >= eps and after < eps - 1e-12:
                        return -1e6
                return base_primary(S)
            S, diag = self.selector.select_utility_frontier(
                primary, allowed=allowed, secondary_utility=secondary,
                performance_slack=self.wif_performance_slack,
                require_exact=self.require_exact_frontier)
            if priority_args is None:
                self._last_claim_priorities = {
                    j: primary({j}) for j in allowed}
            else:
                self._last_claim_priorities = individual_utilities(
                    allowed, **priority_args)
            self.selection_diagnostics.append(diag)
            return set(S)

        # Selector maximises sum(V_j + theta_D * D_j) over admissible sets.
        # Claims outside `cands` (silent this epoch) are given a large
        # negative value so the exhaustive search never picks them, which is
        # how proposal gating is expressed without touching the selector.
        # *** BUG FIX. ***  selector.score computes  sum(V_j + theta_D * D_j).
        # The first version passed V_j = c_j and D_j = deficit, which makes
        # the weight an ADDITIVE term:
        #       sum(c_j + theta_D * D_j)
        # With theta_D = 8 and c_j of order 1, the weight contributed ~10% of
        # the score and raw deficit decided everything.  That is why the
        # weights did not matter and why INTACTv2 was flat across the
        # over-subscription sweep: the score never saw the priorities.
        #
        # The declared objective is  S = argmax theta_D * sum_j c_j D_j  --
        # the weight MULTIPLIES the deficit.  Express that through the
        # existing scorer by folding the whole product into V and zeroing D.
        # *** DESIGN FIX: the outer value is now DIRECTIONAL. ***
        #
        # leverage_claim_weights builds c_j from |s|, i.e. HOW MUCH a claim
        # can move an intent -- with the sign thrown away.  The outer loop
        # therefore could not tell a write that HELPS from one that HURTS,
        # and admitted any overdue high-leverage claim regardless.  B3, which
        # scores the SIGNED predicted effect sum_i s_{p(j),i} * dnu_j and
        # admits only the subset that maximises it, beat INTACTv2 for exactly
        # that reason: it declined harmful writes and INTACTv2 did not.
        #
        # The declared formulation always had the directional term
        # (METHODOLOGY B+, V_j = sum_i w_i s_{p(j),i} (nu_req - nu_old)); the
        # implementation had dropped it.  Restored here and multiplied by the
        # contract urgency, so the objective is
        #
        #     V_j = [ sum_i w_i s_{p(j),i}(r) dnu_j^req ] * (1 + theta_D D_j)
        #
        # benefit x overdue-ness.  A claim whose write is predicted to harm
        # its own tenant's intents now scores NEGATIVE and is declined no
        # matter how overdue it is, while an overdue helpful claim is
        # promoted.  It uses only s -- the one validated model.
        # *** SCALE FIX -- the defect that made INTACTv2 collapse in long runs.
        #
        # The two score terms were combined RAW, and they are not
        # commensurable.  Measured over 300 epochs on a calibrated scenario:
        #
        #     |benefit|        median 0.0000    p95   5,384
        #     |theta_D c D|    median   700     p95  77,337
        #
        # The deficit term is ~14x larger at p95 and unboundedly larger at the
        # median, because D_j grows to its cap (500) while benefit stays near
        # zero for most claims.  Both are unbounded: benefit inherits u(p_hat),
        # which explodes as p_hat -> 1, and the deficit simply accumulates.
        #
        # Consequence: in a SHORT run deficits are small and the directional
        # benefit decides, but in a long run (2000 train + 800 eval) the
        # deficit saturates and INTACTv2 degenerates into "serve whoever is
        # most overdue", ignoring whether the write helps at all.  It then
        # issued 1727 writes and scored 0.314 while INTACTv3 -- which has no
        # deficit term and issued 525 writes -- scored 0.350.
        #
        # Both terms are now normalised to O(1) WITHIN THE EPOCH, so theta_D
        # becomes a genuine, scale-free trade-off dial:
        #     0    -> pure directional benefit (equivalent to INTACTv3)
        #     ~1   -> benefit and contract urgency carry equal weight
        #     >>1  -> contract dominates
        drop = -1e9
        theta_D = float(self.cfg["outer"]["theta_D"])
        use_deficit = self.mode == "intactv2"
        d_cap = max(float(self.cfg["outer"].get("deficit_cap", 500.0)), 1e-9)

        raw_benefit, raw_cw = {}, {}
        for j in cands:
            c = self.claims[j]
            nu_req, nu_old = proposals.get(j, (0.0, 0.0))
            dnu_requested = float(nu_req) - float(nu_old)
            dnu = (float(projected_dose.get(j, dnu_requested))
                   if self.outer_score_dose == "projected"
                   else dnu_requested)
            raw_benefit[j] = sum(
                w_policy.get(i, 0.0) * plan_s(regime, c.param, i)
                for i in self.intents) * dnu
            raw_cw[j] = cw.get(j, 0.0)
        b_scale = max((abs(v) for v in raw_benefit.values()), default=0.0)
        b_scale = b_scale if b_scale > 1e-12 else 1.0
        c_scale = max((abs(v) for v in raw_cw.values()), default=0.0)
        c_scale = c_scale if c_scale > 1e-12 else 1.0

        V = {}
        for j in self.claims:
            if j not in cands:
                V[j] = drop
                continue
            V[j] = raw_benefit[j] / b_scale          # in [-1, 1]
        # ADDITIVE, not multiplicative.  Multiplying benefit by the deficit
        # makes an overdue HARMFUL claim score worse the longer it waits, so
        # the contract guarantee inverts: measured wIF collapsed to 0.240,
        # i.e. the scheduler declined almost everything and behaved like the
        # do-nothing baseline.  Additively, the signed benefit decides in the
        # normal case and the contract-weighted deficit can OVERRIDE it once a
        # claim is badly starved -- which is exactly what a minimum actuation
        # rate is supposed to buy.  Same structure as the v1 selector,
        # score = sum_j (V_j + theta_D D_j), with the leverage weight applied
        # to the deficit rather than to a sign-free magnitude.
        # Preserve INTACTv3's proven risk/contract/C3-weighted primary score.
        # INTACTv2 adds only C4 as a bounded secondary criterion, making the
        # pair a clean one-term ablation.
        D = {}
        for j in self.claims:
            if j not in cands:
                D[j] = 0.0
                continue
            # A zero estimate is absence of evidence, not evidence that the
            # write is harmless.  Do not let any secondary weight activate a
            # claim unless the primary model predicts a strictly positive
            # wIF contribution.
            if V[j] <= self.c4_min_primary_gain:
                D[j] = 0.0
            else:
                c4_term = ((raw_cw[j] / c_scale)
                           * (self.deficit_v2.D.get(j, 0.0) / d_cap))
                D[j] = theta_D * c4_term if use_deficit else 0.0
        self._last_claim_priorities = dict(V)
        if self.c4_selection_mode == "performance_frontier":
            S, diag = self.selector.select_performance_frontier(
                V, D, H, performance_slack=self.c4_performance_slack,
                require_exact=self.require_exact_frontier,
                secondary_weight=1.0)
        elif use_deficit and self.c4_selection_mode == "legacy_additive":
            S, sc = self.selector.select(V, D, H)
            diag = {"candidate_count": float(self.selector.candidate_count()),
                    "frontier_size": 0.0, "best_primary": float("nan"),
                    "selected_primary": float(sc), "primary_loss": float("nan"),
                    "selected_deficit": float(sum(D.get(j, 0.0) for j in S)),
                    "exact": float(self.selector.candidate_count()
                                   <= self.selector.exhaustive_limit),
                    "c4_changed_choice": float("nan")}
        else:
            zero = {j: 0.0 for j in self.claims}
            S, sc = self.selector.select(V, zero, H)
            diag = {"candidate_count": float(self.selector.candidate_count()),
                    "frontier_size": 1.0, "best_primary": float(sc),
                    "selected_primary": float(sc), "primary_loss": 0.0,
                    "selected_deficit": 0.0,
                    "exact": float(self.selector.candidate_count()
                                   <= self.selector.exhaustive_limit),
                    "c4_changed_choice": 0.0}
        self.selection_diagnostics.append(diag)
        return {j for j in S if j in cands}

    # ------------------------------------------------------------------
    def run(self, n_epochs: int) -> Dict:
        if not self.is_v2:
            return super().run(n_epochs)
        return self._run_v2(n_epochs)

    def _run_v2(self, n_epochs: int) -> Dict:
        from ..estimation.sensitivity import regime_key
        spe = self.cfg["ran"]["slots_per_epoch"]
        pre = self.cfg["ran"].get("pre_slots", spe)
        post = self.cfg["ran"].get("post_slots", spe)

        while self.epoch < n_epochs:
            self.epoch += 1
            self._sync_xapp_clocks()   # identical workload for every method
            # _proposal_calls_epoch is a PER-EPOCH double-query detector; not
            # clearing it made every claim after epoch 1 look like a repeat
            # query (2388 spurious flags over 200 epochs).
            self._proposal_calls_epoch = {}

            kpm = self.forecaster.apply(self.ran.step(pre))
            g = self.tracker.update(kpm)
            if hasattr(self.risk, "observe"):
                self.risk.observe(self.epoch, g,
                                  {i: self.tracker.fulfilment(i) for i in self.intents},
                                  kpm, self.ran.current_controls())
            p_hat = self.risk.predict(self.tracker)
            self._last_kpm = kpm
            regime = regime_key(kpm, self.cfg)
            # Map the exogenous forecast to the same three Phase-A table
            # names used by the sensitivity sweep.  The current-regime table
            # remains the only authority for safety; this future table is for
            # value/dose planning.
            ledges = list((self.cfg.get("v2", {}) or {}).get(
                "regime_load_edges", [0.9, 1.25]))
            future_regimes = {}
            # v10.3: which load signal labels the PLANNING regime.
            #   "tenant" (default, previous behaviour) -- each intent is
            #       planned with its OWN tenant's load, so a tenant entering
            #       congestion is planned with the congested-regime table
            #       even while the cell average is still normal;
            #   "cell"   -- every intent uses the cell-wide load, which
            #       isolates pure look-ahead from per-tenant resolution.
            cell_row = kpm.get("_cell", {})
            src = self.planning_regime_source
            for iid, intent in self.intents.items():
                row = (cell_row if src.startswith("cell")
                       else kpm.get(intent.tenant, cell_row))
                if src.endswith("_now"):
                    # measured load only: no forecast of any kind is read
                    future_load = float(row.get("offered_input_ratio", 1.0))
                else:
                    future_load = float(row.get(
                        "offered_input_ratio_forecast",
                        row.get("offered_input_ratio", 1.0)))
                fidx = sum(future_load >= float(edge) for edge in ledges)
                future_regimes[iid] = f"load{min(fidx, len(ledges))}"
            self.arbiter_v2.set_predictive_context(
                p_hat, future_regimes)
            v2_regime, ctx = self.state_builder.build(kpm, self.ran,
                                                      list(self.tenants))

            # ---- proposals FIRST, so admission can be proposal-gated ----
            ctl_now = self.ran.current_controls()
            proposals = {}
            for j in sorted(self.claims):
                pr = self._ask_xapp(j, kpm, ctl_now)
                if pr is not None:
                    proposals[j] = pr
                    if abs(pr[0] - pr[1]) > 1e-12:
                        self.deficit_v2.note_proposal(j)
            self._proposals = proposals

            qualified, projected_dose = self._preview_proposals(
                g, p_hat, regime, proposals)
            for j in qualified:
                self.deficit_v2.note_qualified(j)

            S = self._select_v2(g, p_hat, regime, proposals,
                                qualified=qualified,
                                projected_dose=projected_dose)
            for j in S:
                self.deficit_v2.note_opportunity(j)
                if j not in proposals or abs(proposals[j][0] - proposals[j][1]) <= 1e-12:
                    self.deficit_v2.note_silent(j)

            # ---- IMPLICIT CONFLICT, structural variant (C1') ------------
            sc = self.implicit.structural_conflicts(S)
            self.structural_conflicts += len(sc)

            # ---- INNER LOOP with a RUNNING LEDGER -----------------------
            controls = dict(self.ran.current_controls())
            prop_map = {j: proposals[j][0] for j in S if j in proposals}
            decisions, ledger = self.arbiter_v2.decide_all(
                sorted(S), g, p_hat, regime, prop_map, controls,
                ({j: 0.0 for j in self.claims}
                 if self.mode in ("intactv3", "intactv3_wif")
                 else self.order_deficit_v2.D),
                priorities=(self._last_claim_priorities
                            if (self.inner_order == "benefit"
                                or self.mode == "intactv3_wif") else None),
                intent_weights=self._last_intent_weights)

            actuated, written_params, dnu_map = [], {}, {}
            for d in decisions:
                c = self.claims[d.claim]
                dnu = d.nu_star - d.nu_old
                dnu_map[d.claim] = dnu
                if d.outcome == "reject" or abs(dnu) <= 1e-12:
                    continue
                actuated.append(d.claim)
                self.deficit_v2.note_actuation(d.claim)
                if c.param in written_params:
                    self.metrics.c1_violations += 1
                written_params[c.param] = d.claim
                self.ran.apply(c.param, d.nu_star)
                if c.kind == Kind.ALLOCATIVE and getattr(c, "resource", None):
                    env, used = self.ran.headroom(c.tenant, c.resource)
                    if used > env + 1e-9:
                        self.metrics.c2_violations += 1

            # MetricBook expects v1 Decision objects, so translate.  Keeping
            # ONE metrics implementation is what makes INTACTv2 comparable
            # to every baseline on identical definitions.
            v1_decisions = []
            for d in decisions:
                oc = {"admit": Outcome.ADMIT, "override": Outcome.OVERRIDE,
                      "reject": Outcome.REJECT}[d.outcome]
                v1_decisions.append(Decision(
                    write=Write(jid=d.claim, param=d.param, nu_req=d.nu_req,
                                nu_old=d.nu_old, epoch=self.epoch, slot=0),
                    outcome=oc, nu_star=d.nu_star,
                    escalated=d.escalated, implicated=list(d.implicated)))
                # Uniform predicted-safety metric for v2 modes.  Previously
                # this counter was never updated in ExperimentV2, so the
                # reported all-zero value was tautological rather than a
                # checked guarantee.
                if oc != Outcome.REJECT:
                    self.metrics.safety_violations += sum(
                        d.g_before.get(i, 0.0) >= d.eps_eff.get(i, 0.0)
                        and d.g_after.get(i, d.g_before.get(i, 0.0))
                        < d.eps_eff.get(i, 0.0) - 1e-12
                        for i in d.g_before)
                self.metrics.n_applied += int(oc != Outcome.REJECT)
                self.rd.decision({
                    "epoch": self.epoch, "jid": d.claim, "param": d.param,
                    "nu_req": d.nu_req, "nu_old": d.nu_old,
                    "nu_eff": d.nu_eff, "nu_star": d.nu_star,
                    "outcome": d.outcome, "reason": d.reason,
                    "order_index": d.order_index,
                    "g_before": d.g_before, "g_after": d.g_after,
                    "eps_eff": d.eps_eff,
                })

            self.cofire.observe(actuated)
            self.implicit.observe_writes(dnu_map)
            # C4 is conditional on an event the scheduler could actually
            # serve.  The default is a nonzero proposal with a nonzero safe
            # and feasible projection; epoch/proposal bases remain available
            # as explicit ablations.
            if self.deficit_v2.accrual_on == "proposal":
                eligible = {j for j, (req, old) in proposals.items()
                            if abs(float(req) - float(old)) > 1e-12}
            elif self.deficit_v2.accrual_on == "qualified":
                eligible = qualified
            else:
                eligible = set(self.claims)
            self.deficit_v2.update(actuated, eligible=eligible)
            self.order_deficit_v2.update(actuated, eligible=set(self.claims))

            # ---- learn --------------------------------------------------
            kpm2 = self.ran.step(post)
            g2 = self.tracker.update(kpm2)
            dg = {i: g2[i] - g[i] for i in self.intents}
            if hasattr(self, "effects_v2") and self.effects_v2 is not None:
                self.effects_v2.observe(v2_regime, {
                    self.claims[j].param: dnu_map.get(j, 0.0) for j in dnu_map},
                    ctx, dg)

            # C3 TENANT-FLOOR LEDGER.  This update was present in the v1
            # experiment but omitted from the rewritten v2 loop, leaving
            # every Lambda_n identically zero.  Consequently theta_Lambda
            # could not alter a single v2/v3 decision.  Update it from the
            # same contractual, pi-weighted tenant fulfilment definition used
            # by MetricBook; dynamic risk is deliberately excluded.
            rho_t = {}
            for tenant in self.tenants:
                mine = [(self.intents[i].pi_class,
                         self.tracker.fulfilment(i))
                        for i in self.intents
                        if self.intents[i].tenant == tenant]
                if mine:
                    den = sum(p for p, _ in mine)
                    rho_t[tenant] = (sum(p * f for p, f in mine) / den
                                     if den > 0 else 1.0)
                else:
                    rho_t[tenant] = 1.0
            self.deficits.update_tenants(rho_t)

            self.metrics.record_epoch(g2, p_hat, S, self.deficit_v2.D,
                                      v1_decisions, g_before=g)
            self.v2_log.append({
                "epoch": self.epoch, "regime": v2_regime,
                "n_selected": len(S), "n_actuated": len(actuated),
                "n_qualified": len(qualified),
                "structural_conflicts": len(sc),
                "selection": dict(self.selection_diagnostics[-1]),
                "decisions": [d.as_row() for d in decisions]})
            self.rd.event({
                "epoch": self.epoch, "mode": self.mode,
                "s_regime": regime, "state_regime": v2_regime,
                "S": sorted(S), "qualified": sorted(qualified),
                "actuated": sorted(actuated), "g_before": g, "g": g2,
                "p_hat": p_hat, "Lambda": dict(self.deficits.Lam),
                "claim_deficit": dict(self.deficit_v2.D),
                "projected_dose": projected_dose,
                "actual_dose": dnu_map,
                "claim_priority": dict(self._last_claim_priorities),
                "intent_weight": dict(self._last_intent_weights),
                "forecast_regime": dict(future_regimes),
                "selection": dict(self.selection_diagnostics[-1]),
                "kpm_cell": kpm2.get("_cell", {}),
            })
            if hasattr(self.ran, "binding_edge_count"):
                self.binding_edges += int(self.ran.binding_edge_count())

            if self.epoch % self.cfg["run"]["checkpoint_every"] == 0:
                self.log.info("epoch %d | regime %s | |S|=%d actuated=%d",
                              self.epoch, v2_regime, len(S), len(actuated))

        return self._finish_v2()

    # ------------------------------------------------------------------
    def _finish_v2(self) -> Dict:
        out = self.metrics.summary(self.deficits, self.inner)
        labels = {"intactv2": "INTACTv2", "intactv3": "INTACTv3",
                  "intactv3_wif": "INTACTv3-WIF"}
        out["_label"] = labels[self.mode]
        out["uses_claim_deficit"] = self.mode == "intactv2"
        out["uses_v1_eligibility_gamma"] = self.use_v1_eligibility_gamma
        out["uses_threshold_primary"] = self.use_threshold_primary
        out["risk_source"] = getattr(self.risk, "source", "unknown")
        out["risk_margin_reserve"] = self.risk_margin_reserve
        out["predictive_risk_gain"] = self.predictive_risk_gain
        out["predictive_rescue_gain"] = self.predictive_rescue_gain
        out["inner_local_optimize"] = self.arbiter_v2.local_optimize
        out.update(self.forecaster.stats())
        out["predictive_use_pi"] = self.predictive_use_pi
        out["planning_regime_source"] = self.planning_regime_source
        out["joint_protection"] = self.joint_protection
        out["inner_protection_policy"] = self.arbiter_v2.protection_policy
        out["interaction_physics_terms"] = len(
            getattr(self.ran, "interactions", []))
        out["binding_dependency_edge_epochs"] = self.binding_edges
        out["v2"] = {
            "deficit_events": self.deficit_v2.rates(),
            "cofire": self.cofire.report(),
            "structural_conflicts_total": self.structural_conflicts,
            "behavioural_conflicts": self.implicit.behavioural_conflicts(),
        }
        if self.selection_diagnostics:
            keys = ("candidate_count", "frontier_size", "primary_loss",
                    "selected_deficit", "exact", "c4_changed_choice")
            out["v2"]["selection"] = {
                "mode": (self.c4_selection_mode if self.mode == "intactv2"
                         else ("wif_aligned" if self.mode == "intactv3_wif"
                               else "deficit_free")),
                "performance_slack": (self.c4_performance_slack
                                      if self.mode == "intactv2" else 0.0),
                **{f"mean_{k}": float(np.nanmean([
                    d.get(k, float("nan")) for d in self.selection_diagnostics]))
                   for k in keys},
                "max_primary_loss": float(np.nanmax([
                    d.get("primary_loss", float("nan"))
                    for d in self.selection_diagnostics])),
            }
            # Promote the essential audit quantities so run_v2 can aggregate
            # them across seeds and paper_figures_v2 can plot them without
            # special-casing nested JSON.
            sel = out["v2"]["selection"]
            out["c4_frontier_mean_primary_loss"] = sel["mean_primary_loss"]
            out["c4_frontier_max_primary_loss"] = sel["max_primary_loss"]
            out["c4_changed_choice_rate"] = sel["mean_c4_changed_choice"]
            out["c4_frontier_exact_rate"] = sel["mean_exact"]
            out["c4_frontier_mean_size"] = sel["mean_frontier_size"]
        # The two guarantees, reported SEPARATELY.  v1 called C4 an actuation
        # floor while discharging on selection, which is an OPPORTUNITY
        # guarantee.  Both are now visible instead of conflated.
        rates = self.deficit_v2.rates()
        # Replace the inherited v1 claim-rate fields.  ExperimentV2 uses its
        # own actuation-aware ledger; leaving MetricBook's untouched v1
        # ledger in the top-level summary produced the impossible combination
        # “thousands of applied writes” and “zero realised rate for every
        # claim” in the supplied v7 JSON.
        out["realised_actuation_rate"] = {
            r["claim"]: r["actuation_rate"] for r in rates}
        out["contracted_actuation_rate"] = {
            r["claim"]: r["r_j"] for r in rates}
        out["actuation_shortfall"] = {
            r["claim"]: r["actuation_shortfall"] for r in rates}
        raw_shortfall = [r["actuation_shortfall"] for r in rates]
        out["max_positive_actuation_shortfall"] = max(raw_shortfall or [0.0])
        out["mean_positive_actuation_shortfall"] = float(
            np.mean([v for v in raw_shortfall if v > 0.0] or [0.0]))
        out["n_undercontract_claims"] = int(sum(v > 0.0 for v in raw_shortfall))
        out["n_starved_claims_005"] = int(sum(v > 0.05 for v in raw_shortfall))
        out["claim_starvation_rate_005"] = (
            out["n_starved_claims_005"] / max(len(rates), 1))
        out["max_opportunity_shortfall"] = max(
            [r["opportunity_shortfall"] for r in rates] or [0.0])
        out["max_actuation_shortfall_v2"] = max(
            [r["actuation_shortfall"] for r in rates] or [0.0])
        out["max_c4_shortfall"] = max(
            [r["c4_shortfall"] for r in rates if r["c4_exposures"] > 0]
            or [0.0])
        out["mean_c4_shortfall"] = float(np.mean(
            [r["c4_shortfall"] for r in rates if r["c4_exposures"] > 0]
            or [0.0]))
        out["mean_silent_rate"] = float(np.mean(
            [r["silent_rate"] for r in rates] or [0.0]))

        # ---- FEASIBLE-NORMALISED SHORTFALL --------------------------------
        # When a C1 group is over-subscribed, sum(r_j) > 1 and at most 1.0 of
        # actuation can be handed out, so EVERY C1-respecting method is short
        # by construction.  Raw shortfall then measures the CONTRACT's
        # infeasibility, not the scheduler's quality, and saturates: INTACTv2
        # scored starvation 1.000 purely because the contracts cannot all be
        # met.  The fair target is each claim's PROPORTIONAL share of what is
        # actually available, r_j / max(1, sum of r over its C1 group).
        from collections import defaultdict
        grp = defaultdict(list)
        for j, c in self.claims.items():
            grp[c.param].append(j)
        feasible_r = {}
        for p, js in grp.items():
            tot = sum(self.deficit_v2.r[j] for j in js)
            scale = 1.0 / tot if tot > 1.0 else 1.0
            for j in js:
                feasible_r[j] = self.deficit_v2.r[j] * scale
        by_claim = {r["claim"]: r for r in rates}
        fs = []
        for j, tgt in feasible_r.items():
            got = by_claim[j]["actuation_rate"] if j in by_claim else 0.0
            fs.append(max(0.0, tgt - got))
        out["max_feasible_shortfall"] = max(fs) if fs else 0.0
        out["mean_feasible_shortfall"] = float(np.mean(fs)) if fs else 0.0
        out["feasible_starvation_rate"] = (
            float(np.mean([f > 0.05 for f in fs])) if fs else 0.0)
        out["oversubscription_ratio"] = float(max(
            [sum(self.deficit_v2.r[j] for j in js) for js in grp.values()] or [0.0]))
        return out
