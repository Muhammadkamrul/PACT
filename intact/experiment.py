"""
intact/experiment.py
====================
The orchestrator.  Ties the RAN, the estimators, the two loops, and the
metrics into one resumable run.

THE EPOCH CYCLE  (matches the sequence diagram, phases 2-4)
    1. read KPMs from the RAN                     -> margins g_i, fulfilment rho_i
    2. risk predictor                             -> p_hat_i
    3. weights                                    -> w_i
    4. claim values V_j, pair harm H(j,k)
    5. OUTER LOOP: choose S(tau) under C1/C2/C3
    6. forced exploration (safety-vetoed)         <- the identification step
    7. for each eligible claim, its xApp proposes a write
    8. INNER LOOP: gate -> project -> disposition -> apply nu*
    9. advance the RAN one epoch
   10. observe Delta g_i, update beta/gamma, update deficits
   11. checkpoint

BASELINES (v11 §13) are implemented as MODES of the same loop, so the
comparison is apples-to-apples:
    B0  none        no mediation at all
    B1  static      fixed xApp priority (CMF-style)
    B2  predeploy   one conflict-free portfolio fixed before deployment
    B3  value_only  collect proposals; immediate global value arbitration
    B4  scheduler   RL-free stand-in for Cinemre et al.: pick the single
                    claim that maximises ONE GLOBAL objective (total
                    throughput).  Has NO way to express a tenant trade-off.
    B5  outer_only  our outer loop; inner loop admits everything
    B6   inner_only  no eligibility control; inner loop arbitrates every write
    B6+  b6plus      C1/C2 + unweighted claim deficit; then inner mediation
    B6++-U b6pp_no_urgency  B6++ with the outer urgency multiplier fixed to 1
    B6++ b6plusplus  B6+ with INTACT's risk/contract/C3 weight; no beta/gamma
    B7   single_loop proposal-aware s selection; no inner mediator
    B7+  b7plus      B7 + write-level eta and envelope feasibility
    INTACT full
"""
from __future__ import annotations
import random
from typing import Dict, List, Set
import numpy as np

from .types import Write, Outcome, Kind, Decision
from .config import build_tenants, build_intents, build_claims
from .ran.analytic import AnalyticRAN
from .estimation.margins import MarginTracker
from .estimation.risk import RiskPredictor, make_risk_predictor
from .estimation.effects import EffectEstimator
from .estimation.sensitivity import SensitivityTable, sweep, regime_key
from .outer.weights import (compute_weights, contractual_claim_weights,
                            risk_aware_claim_weights)
from .outer.deficit import DeficitBook
from .outer.constraints import AdmissibilityMask
from .outer.selector import Selector, claim_values, pair_harm
from .inner.arbiter import InnerLoop
from .xapps import build_xapps
from .metrics import MetricBook


class Experiment:
    def __init__(self, cfg: Dict, rundir, mode: str = "intact"):
        self.cfg, self.rd, self.mode = cfg, rundir, mode
        self.log = rundir.log

        self.tenants = build_tenants(cfg)
        self.intents = build_intents(cfg)
        self.claims = build_claims(cfg)
        self.xapps = build_xapps(cfg)

        self.ran = AnalyticRAN(cfg, [t for t in self.tenants if t in cfg["ran"]["slices"]])
        self.tracker = MarginTracker(self.intents, cfg["margins"]["window_slots"])
        self.risk = make_risk_predictor(cfg, self.intents, self.tenants,
                                        self.claims, self.log)
        # v10: offered-load forecast source (see intact/estimation/forecast.py).
        # Default "oracle" = the exact schedule value, i.e. earlier behaviour.
        from .estimation.forecast import ForecastProvider
        self.forecaster = ForecastProvider(cfg, self.log)
        self.mask = AdmissibilityMask(self.claims, self.tenants, self.log)

        self.pairs = [tuple(p) for p in cfg["outer"]["coupled_pairs"]]
        self.effects = EffectEstimator(list(self.claims), list(self.intents),
                                       self.pairs, cfg["estimation"]["n_context"], cfg)
        from .estimation.effects import WriteEffectEstimator
        # write-level interaction, used ONLY by B7+ (see its docstring)
        self.weffects = WriteEffectEstimator(list(self.claims), list(self.intents),
                                             self.pairs, cfg)
        self.selector = Selector(self.claims, self.mask, cfg, self.log)
        # B2 is deliberately decided ONCE, before any runtime observation or
        # proposal.  It models pre-deployment exclusion: a conflicting claim
        # that loses this portfolio decision never becomes available later,
        # even in epochs where the winner is silent.  B1 is different: its
        # same fixed ranking is applied at runtime to proposals that actually
        # exist in that epoch.
        pre_order = sorted(
            self.claims,
            key=lambda j: (-self.tenants[self.claims[j].tenant].omega
                           * self.claims[j].r_j, j))
        self.predeploy_portfolio = set()
        for j in pre_order:
            if self.mask.is_admissible(self.predeploy_portfolio | {j}):
                self.predeploy_portfolio.add(j)
        self.deficits = DeficitBook(self.claims, self.tenants,
                                    d_cap=cfg["outer"]["deficit_cap"],
                                    lam_cap=cfg["outer"]["lambda_cap"])
        self.sens = SensitivityTable(cfg)
        self.inner = InnerLoop(self.claims, self.intents, self.tenants,
                               self.sens, cfg, self.log)
        self.metrics = MetricBook(self.intents, self.tenants, self.claims)

        # checkpoint identity = (mode, seed).  Tagging by mode alone made
        # every seed after the first resume the previous seed's finished run.
        self.ck_tag = f"{mode}_s{cfg['seed']}"
        self.rng = random.Random(cfg["seed"])
        self.epoch = 0
        self.prev_g: Dict[str, float] = {}
        # --- exploration accounting (v11 §7.3) ---------------------------
        # "8% exploration" can be 0.5% in practice once the safety veto
        # bites.  Count attempts, acceptances and vetoes so the identifying
        # variation is auditable rather than assumed.
        self._last_kpm, self._w_now, self._regime_now = {}, {}, "load1"
        self._proposals = {}          # j -> (nu_req, nu_old), one per epoch
        self._planned_values = {}     # j -> the exact value scored by B7/B7+
        self.scored_vs_executed = []  # regression evidence: they must match
        self._proposal_calls_epoch = {}
        self.proposal_call_violations = []
        self.expl = {"attempt": 0, "accept": 0, "veto": 0,
                     "forced_on": {j: 0 for j in self.claims},
                     "forced_off": {j: 0 for j in self.claims}}
        self.freeze_effects = False        # set during the EVALUATION phase
        self.beta_trace = []               # for the convergence figure
        # The estimator audit must inspect the data that the runtime estimator
        # ACTUALLY saw.  Reimplementing a second, almost-equivalent training
        # loop in scripts/estimator_audit.py was the source of several false
        # diagnoses (no inner arbitration, no pair/context terms, post-state
        # covariates).  Keep a compact trace only for randomised training.
        self.record_effect_rows = (mode == "train_random")
        self.effect_rows = []

    # ------------------------------------------------------------------
    def calibrate(self) -> None:
        """Phase 1: the offline sensitivity sweep (v11 §7.4.2, build Phase 3)."""
        params = sorted({c.param for c in self.claims.values()})
        params = [p for p in params if p in self.cfg["sweep_domains"]]
        self.log.info("PHASE 1: offline sensitivity sweep over %d parameters", len(params))
        tab = sweep(self.ran, self.cfg, self.intents, params, self.log)
        self.sens.tab = tab.tab
        self.inner.sens = self.sens
        self.rd.save_json("sensitivity_table.json",
                          {f"{k[0]}|{k[1]}|{k[2]}": {"slope": v[0], "se": v[1], "n": v[2]}
                           for k, v in self.sens.tab.items()})

    # ------------------------------------------------------------------
    def _claims_touching(self) -> Dict[str, Set[str]]:
        """Which intents each claim measurably affects -- used by the safety
        veto so exploration never perturbs something an at-risk intent
        depends on."""
        out = {}
        for j, c in self.claims.items():
            out[j] = {i for i in self.intents
                      if abs(self.effects.beta[i].get(j, 0.0)) > 1e-6}
        return out

    # ------------------------------------------------------------------
    def draw_admissible_set(self, rng=None, exclude=(), require=(),
                            max_draws: int = 100000) -> Set[str]:
        """Draw Bernoulli(0.5) eligibility conditional on C1/C2.

        The previous runtime gave up after 50 rejection draws.  In the large
        scenario only about one percent of unconstrained draws are admissible,
        so that shortcut silently produced many empty treatment rows.  This
        method is also used by the counterfactual validator, ensuring training
        and ground truth draw from the same assignment distribution.
        """
        rng = rng or self.rng
        excluded, forced = set(exclude), set(require)
        if excluded & forced:
            raise ValueError("a claim cannot be both excluded and required")
        if not self.mask.is_admissible(forced):
            raise ValueError("required claims are not jointly admissible")
        optional = [j for j in sorted(self.claims)
                    if j not in excluded and j not in forced]
        for _ in range(max_draws):
            S = forced | {j for j in optional if rng.random() < 0.5}
            if self.mask.is_admissible(S):
                return S
        raise RuntimeError(
            f"failed to draw an admissible eligibility set in {max_draws} tries")

    # ------------------------------------------------------------------
    def _sync_xapp_clocks(self) -> None:
        """Lock every duty-cycled xApp's phase to the EPOCH, not the poll count.

        *** This is the defect that invalidated every cross-method comparison.
        ***

        EnergyXApp.propose() advances an internal counter with `self._t += 1`,
        and that counter drives its duty cycle:
            if (self._t // self.period) % 2 == 1:  use target_alt

        But architectures poll a DIFFERENT NUMBER OF xApps.  v1 modes query
        only the claims they selected; proposal-aware modes (B3/B7/B7+, and
        INTACTv2/v3, which need every proposal for proposal-gated admission)
        query all of them.  Measured over 200 epochs on s000:

            value_only  _t = 200, 200, 200, 200, 200
            b6plus      _t = 100, 100,  55,  55,  55
            intactv2    _t = 200, 200, 200, 200, 200

        So the energy xApps were in COMPLETELY DIFFERENT DUTY-CYCLE PHASES
        depending on which scheduler was being evaluated.  The methods were
        not facing the same workload, and the sign of the resulting bias
        depends on the phase -- which is exactly why INTACTv2 looked fine on
        s000 (0.515) and poor across 20 scenarios (0.330).

        An energy-saving duty cycle is a function of TIME ("reduce power at
        night"), never of how often the RIC happened to ask.  Locking `_t` to
        the epoch makes the workload identical for all fifteen methods, which
        is a precondition for the benchmark to mean anything.
        """
        for xa in self.xapps.values():
            if hasattr(xa, "_t"):
                xa._t = self.epoch

    def _ask_xapp(self, j, kpm, controls):
        """Query one claim once and return ``(requested, old)`` or ``None``.

        Proposal-aware B7/B7+ must inspect pending writes before selection.
        Every other architecture selects first and queries only the eligible
        xApps, preserving the asynchronous semantics in the architecture doc.
        All calls go through this function so L17 can detect a double query.
        """
        n = self._proposal_calls_epoch.get(j, 0) + 1
        self._proposal_calls_epoch[j] = n
        if n > 1:
            self.proposal_call_violations.append(
                {"epoch": self.epoch, "jid": j, "calls": n, "mode": self.mode})
        c = self.claims[j]
        xa = self.xapps.get(c.xapp)
        if xa is None:
            return None
        nu = xa.propose(kpm, controls, self.rng)
        if nu is None:
            return None
        return float(nu), float(controls.get(c.param, nu))

    # ------------------------------------------------------------------
    def _single_loop_plan(self, j: str, nu: float, nu_old: float) -> float:
        """The exact grid/slew/feasibility value B7 or B7+ will execute."""
        c = self.claims[j]
        lo_d, hi_d = c.domain
        if c.step > 0:
            nu = round(round((nu - lo_d) / c.step) * c.step + lo_d, 9)
        cap = c.max_step_frac * max(hi_d - lo_d, 1e-12)
        if abs(nu - nu_old) > cap:
            nu = nu_old + (cap if nu > nu_old else -cap)
        if self.mode == "b7plus" and c.kind == Kind.ALLOCATIVE and c.resource:
            env, used = self.ran.headroom(c.tenant, c.resource)
            nu = min(nu, env - (used - nu_old))
        return float(min(max(nu, lo_d), hi_d))

    # ------------------------------------------------------------------
    def _select(self, V, H, g, p_hat) -> Set[str]:
        """Dispatch on baseline mode.  Same loop, different eligibility rule."""
        if self.mode == "train_random":
            # ---- BETA-TRAINING PHASE ---------------------------------
            # Eligibility is drawn UNIFORMLY AT RANDOM subject only to the
            # hard constraints.  This is the cleanest possible instrument:
            # Z is independent of the outcome by construction, so beta and
            # gamma are unbiased WITHOUT relying on the eps_exp perturbation
            # alone.  It is the analogue of a randomised A/B assignment.
            return self.draw_admissible_set(self.rng)
        if self.mode in ("none", "inner_only"):
            return set(self.claims)                    # everybody writes
        if self.mode == "predeploy":
            # ---- B2  PRE-DEPLOYMENT EXCLUSION (PACIFISTA-style) ------
            # The portfolio was frozen in __init__.  No runtime state,
            # proposal, risk, deficit, beta/gamma/eta or sensitivity can
            # restore an excluded capability.
            return set(self.predeploy_portfolio)
        if self.mode == "value_only":
            # ---- B3  ALWAYS VALUE-LEVEL ARBITRATION (QACM-style) ------
            # Collect current proposals, score their immediate GLOBAL margin
            # effect with s, and choose an admissible subset.  There is no
            # ownership/commercial weighting, claim deficit, tenant Lambda,
            # beta/gamma/eta, or second safety/sovereignty stage.
            Vq = {}
            for j, c in self.claims.items():
                if j not in self._proposals:
                    Vq[j] = -1e15
                    continue
                nu, nu_old = self._proposals[j]
                nu = self._single_loop_plan(j, nu, nu_old)
                self._planned_values[j] = nu
                dnu = nu - nu_old
                Vq[j] = sum(self.sens.get(self._regime_now, c.param, i) * dnu
                            for i in self.intents)
            zero_debt = {j: 0.0 for j in self.claims}
            S, _ = self.selector.select(Vq, zero_debt, {})
            return S
        if self.mode == "b6plus":
            # ---- B6+  STRUCTURED ELIGIBILITY + INNER MEDIATION -------
            # B6 (arbitrate every write) PLUS the outer loop's HARD
            # STRUCTURE -- C1, C2 and the actuation deficit -- but with NO
            # value scoring at selection time.  Eligibility is therefore
            # value-FREE: it decides only WHO MAY WRITE, by contract and by
            # constraint.  All value judgement happens later, at write time,
            # in the inner arbiter.
            #
            # This is not a pure one-stage implementation: eligibility is
            # selected first and actual writes are mediated second.  What is
            # removed relative to INTACT is the beta/gamma VALUE-SELECTION
            # model at eligibility.  B6+ judges value only on the actual write
            # at mediation time; B7 judges it only at proposal selection.
            S, _ = self.selector.select({j: 0.0 for j in self.claims},
                                        self.deficits.D, {})
            return S
        if self.mode in ("b6pp_no_urgency", "b6plusplus"):
            # ---- B6++  RISK/CONTRACT/C3-WEIGHTED mediation -----------
            # Every per-intent weight is exactly the one INTACT computes:
            #
            #   w_i = pi_i * omega_n * u(p_hat_i)
            #         * (1 + theta_Lambda * Lambda_n)
            #   score_j = theta_D * (sum_{i in I_n} w_i) * D_j.
            #
            # Unlike INTACT, no beta/gamma value model consumes the weights.
            # They determine only which tenant's overdue claim receives an
            # opportunity; the inner s-based mediator judges the actual write.
            # B6++-U is the pre-declared ablation with u(p_hat)=1, allowing the
            # ensemble to test whether outer risk urgency helps in practice.
            if self.mode == "b6plusplus":
                cw = risk_aware_claim_weights(
                    self.claims, self.intents, self.tenants, p_hat,
                    self.deficits.Lam, self.cfg)
            else:
                cw = contractual_claim_weights(
                    self.claims, self.intents, self.tenants,
                    self.deficits.Lam, self.cfg)
            weighted_D = {j: cw[j] * self.deficits.D[j] for j in self.claims}
            S, _ = self.selector.select({j: 0.0 for j in self.claims},
                                        weighted_D, {})
            return S
        if self.mode in ("single_loop", "b7plus"):
            # ---- B7: ONE loop, value-aware, no inner arbiter -----------
            # Tests directly whether direct value-aware scheduling from the
            # OFFLINE sensitivity table can replace the two-loop split.
            #   V_j^write = sum_i w_i * s_{p,i} * (nu_req - nu_old)
            # i.e. score each claim by the effect of the write it is ACTUALLY
            # proposing, rather than by a learned claim-level beta.
            # Selection keeps C1/C2, the actuation deficit and the tenant
            # deficit.  Selected writes execute VERBATIM -- no gate, no
            # projection, no escalation.
            # LIMITATION, stated rather than hidden: effects are assumed
            # ADDITIVE.  No learned gamma / H is used, because reusing the
            # outer regression would smuggle the very thing this baseline is
            # meant to do without.
            Vw = {}
            for j, c in self.claims.items():
                # USE THE CACHED PROPOSAL.  Calling propose() again here is
                # what allowed B7 to score one value and execute another --
                # xApps are stateful, so a second call can return a different
                # number for identical inputs.
                if j not in self._proposals:
                    # A proposal-aware scheduler can schedule only writes that
                    # actually exist.  A large negative sentinel keeps silent
                    # claims out even when their deficit is high; the empty set
                    # remains available when every xApp is silent.
                    Vw[j] = -1e15
                    continue
                nu, nu_old = self._proposals[j]
                nu = self._single_loop_plan(j, nu, nu_old)
                self._planned_values[j] = nu
                d = nu - nu_old
                Vw[j] = sum(self._w_now.get(i, 0.0)
                            * self.sens.get(self._regime_now, c.param, i) * d
                            for i in self.intents)
            # B7  : additive only, no pair term  (documented limitation)
            # B7+ : adds the learned WRITE-DOSE interaction eta.  Marginal
            #       sensitivities cannot express what two knob movements do
            #       together beyond the sum of their parts.  Claim-level
            #       eligibility gamma is deliberately not substituted here.
            Hw = {}
            if self.mode == "b7plus":
                # WRITE-LEVEL interaction, not the claim-level gamma.  For a
                # candidate pair both proposing this epoch, the extra effect
                # is eta * dnu_j * dnu_k, weighted and counted only where it
                # HARMS -- the same shape as H(j,k) but the right quantity.
                for (ja, jb) in self.pairs:
                    if ja not in self._proposals or jb not in self._proposals:
                        Hw[(ja, jb)] = 0.0; continue
                    da = self._planned_values[ja] - self._proposals[ja][1]
                    db = self._planned_values[jb] - self._proposals[jb][1]
                    Hw[(ja, jb)] = sum(
                        self._w_now.get(i, 0.0)
                        * max(-self.weffects.eta[i][(ja, jb)] * da * db, 0.0)
                        for i in self.intents)
            S, _ = self.selector.select(Vw, self.deficits.D, Hw)
            return S
        if self.mode == "static":
            # B1: fixed runtime priority (CMF-style).  Rank only claims that
            # ACTUALLY proposed in this epoch.  A silent high-priority xApp
            # does not suppress a lower-priority proposal; that permanent
            # capability loss is B2, not B1.
            order = sorted(self._proposals,
                           key=lambda j: -(self.tenants[self.claims[j].tenant].omega
                                           * self.claims[j].r_j))
            S = set()
            for j in order:
                if self.mask.is_admissible(S | {j}):
                    S.add(j)
            return S
        if self.mode == "scheduler_global":
            # B4: Cinemre-style.  ONE global objective (total throughput
            # margin), no tenant weighting, no fairness credit, no floors.
            V_glob = {j: sum(self.effects.beta[i].get(j, 0.0) for i in self.intents)
                      for j in self.claims}
            # Cinemre-style schedulers always ACTIVATE something; the
            # degenerate "activate nobody" set is excluded so the baseline
            # is a fair representative of prior art.
            best, bs = None, float("-inf")
            for combo in self.selector._candidates():
                if not combo or not self.mask.is_admissible(combo):
                    continue
                s = sum(V_glob[j] for j in combo)
                if s > bs:
                    best, bs = set(combo), s
            return best or set()
        # outer_only and intact both use the real selector
        D = self.deficits.D
        S, _ = self.selector.select(V, D, H)
        at_risk = {i for i in self.intents
                   if p_hat.get(i, 0) >= self.cfg["inner"]["tau_risk"]}
        S2 = self.selector.explore(S, self.cfg["estimation"]["eps_exp"],
                                   at_risk, self._claims_touching(), self.rng,
                                   counters=self.expl)
        return S2

    # ------------------------------------------------------------------
    def run(self, n_epochs: int) -> Dict:
        """Main loop.  Resumable: call again after a crash and it continues."""
        st = self.rd.load_checkpoint(self.ck_tag)
        if st:
            self.__dict__.update(st["obj"])
            self.epoch = st["epoch"]
            self.log.info("resumed at epoch %d", self.epoch)

        # An epoch has TWO halves and therefore spans 2 x slots_per_epoch:
        #   PRE  : observe the state the decision is made from   -> g_before
        #   POST : observe the consequence of the writes         -> g_after
        # Delta g = g_after - g_before is the outcome the effect estimator
        # regresses on Z, so both halves are load-bearing.
        spe = self.cfg["ran"]["slots_per_epoch"]
        pre = self.cfg["ran"].get("pre_slots", spe)
        post = self.cfg["ran"].get("post_slots", spe)
        if self.epoch == 0:
            ms = self.cfg["ran"]["slot_ms"]
            self.log.info("epoch = %d pre-slots + %d post-slots = %d slots = %.2f s "
                          "(slots_per_epoch=%d names ONE HALF)",
                          pre, post, pre + post, (pre + post) * ms / 1000.0, spe)
        ck = self.cfg["run"]["checkpoint_every"]

        while self.epoch < n_epochs:
            self.epoch += 1
            self._sync_xapp_clocks()   # identical workload for every method

            # -- 1-2: observe, margins, risk ---------------------------
            kpm = self.forecaster.apply(self.ran.step(pre))   # PRE half
            g = self.tracker.update(kpm)
            # MILD needs the raw trajectory, not just the tracker, so feed it
            # the same observation the rest of the loop sees.
            if hasattr(self.risk, "observe"):
                self.risk.observe(self.epoch, g,
                                  {i: self.tracker.fulfilment(i) for i in self.intents},
                                  kpm, self.ran.current_controls())
            p_hat = self.risk.predict(self.tracker)
            self._last_kpm = kpm
            regime = regime_key(kpm, self.cfg)

            # -- 3: weights (contract x urgency x tenant deficit) -------
            w = compute_weights(self.intents, self.tenants, p_hat,
                                self.deficits.Lam, self.cfg)

            # -- 4: claim values and pair harm -------------------------
            V = claim_values(self.claims, self.intents, w, self.effects.beta)
            q_hat = {p: 1.0 for p in self.pairs}   # conservative default
            H = pair_harm(self.pairs, self.intents, w, self.effects.gamma, q_hat)

            # -- 5-6: OUTER LOOP + forced exploration ------------------
            self._w_now, self._regime_now = w, regime

            # Proposal semantics are architecture-specific.  B7/B7+ are the
            # only proposal-aware modes: they must inspect all pending writes
            # before selecting, so they cache each proposal once and execute
            # the exact value they scored.  Every other mode selects first and
            # queries only eligible xApps, as in the original architecture.
            ctl_now = self.ran.current_controls()
            self._proposals = {}
            self._planned_values = {}
            self._proposal_calls_epoch = {}
            if self.mode in ("static", "value_only", "single_loop", "b7plus"):
                for j in sorted(self.claims):
                    proposal = self._ask_xapp(j, kpm, ctl_now)
                    if proposal is not None:
                        self._proposals[j] = proposal

            S = self._select(V, H, g, p_hat)

            # -- 7-8: xApps propose; INNER LOOP disposes ---------------
            controls = self.ran.current_controls()
            decisions = []
            written_params = {}          # for the C1 violation count
            for j in sorted(S):
                c = self.claims[j]
                if self.mode in ("static", "value_only", "single_loop", "b7plus"):
                    proposal = self._proposals.get(j)
                else:
                    proposal = self._ask_xapp(j, kpm, controls)
                if proposal is None:
                    continue                      # xApp chose not to write
                nu_req, nu_old = proposal
                wr = Write(jid=j, param=c.param, nu_req=nu_req,
                           nu_old=nu_old, epoch=self.epoch, slot=0)

                if self.mode in ("none", "static", "predeploy", "value_only",
                                 "scheduler_global", "outer_only",
                                 "single_loop", "b7plus"):
                    # No value-level ARBITRATION in these modes.  But the
                    # single-loop modes must still APPLY the same grid, slew
                    # and envelope clamp they SCORED with, or the value that
                    # was judged is not the value that runs.
                    nu_ap = (self._planned_values[j]
                             if self.mode in ("value_only", "single_loop", "b7plus")
                             else nu_req)
                    d = Decision(write=wr, outcome=Outcome.ADMIT, nu_star=nu_ap)
                    if self.mode in ("value_only", "single_loop", "b7plus"):
                        self.scored_vs_executed.append(
                            {"epoch": self.epoch, "jid": j,
                             "requested": self._proposals[j][0],
                             "scored": self._planned_values[j],
                             "executed": nu_ap})
                else:
                    d = self.inner.decide(wr, regime, g, p_hat, self.ran)

                if d.outcome != Outcome.REJECT:
                    # ---- C1: did somebody already write this knob? --------
                    if c.param in written_params:
                        self.metrics.c1_violations += 1
                    written_params[c.param] = j
                    # ---- C2: does the tenant now exceed its envelope? -----
                    if c.kind == Kind.ALLOCATIVE and c.resource:
                        env, used = self.ran.headroom(c.tenant, c.resource)
                        mine = controls.get(c.param, 0.0)
                        if (used - mine + d.nu_star) > env + 1e-9:
                            self.metrics.c2_violations += 1
                    # ---- safety: predicted to breach an at-risk floor? ----
                    # Computed the SAME way for every method, including those
                    # with no gate, so B6/B7 are judged on the same basis.
                    reg = regime
                    for i in self.sens.affected_intents(reg, c.param, list(self.intents)):
                        if p_hat.get(i, 0.0) < self.cfg["inner"]["tau_risk"]:
                            continue
                        sp = self.sens.get(reg, c.param, i)
                        g_old = g[i]
                        g_new = g_old + sp * (d.nu_star - wr.nu_old)
                        # a violation is a write that CAUSES the crossing:
                        # safe at the status quo, unsafe after.  A write that
                        # leaves an ALREADY-breached intent unharmed is not a
                        # violation, and counting it as one made every method
                        # look equally unsafe.
                        if g_old >= self.intents[i].epsilon > g_new:
                            self.metrics.safety_violations += 1
                            break
                    self.metrics.n_applied += 1
                    self.ran.apply(c.param, d.nu_star)
                    # Later writes in an unconstrained baseline see the state
                    # produced by earlier writes in the same epoch.  This is
                    # relevant only when a baseline violates C1; admissible
                    # architectures have at most one writer per parameter.
                    controls[c.param] = float(d.nu_star)
                decisions.append(d)
                self.rd.decision({
                    "epoch": self.epoch, "jid": j, "param": c.param,
                    "nu_req": wr.nu_req, "nu_old": wr.nu_old, "nu_star": d.nu_star,
                    "outcome": d.outcome.value, "implicated": d.implicated,
                    "escalated": d.escalated, "reason": d.escalation_reason,
                    "latency_ms": d.latency_ms, "magnitude": d.magnitude})

            # -- 9-10: learn ------------------------------------------
            kpm2 = self.ran.step(post)          # POST half
            g2 = self.tracker.update(kpm2)
            dg = {i: g2[i] - g[i] for i in self.intents}
            # Context must be measured BEFORE treatment.  kpm2 is a descendant
            # of the writes and conditioning on it creates post-treatment bias.
            ctx = np.array([kpm["_cell"]["prb_util_pct"] / 100.0,
                            kpm["_cell"]["retx_prb"] / max(self.ran.n_prb, 1)]
                           [:self.cfg["estimation"]["n_context"]])
            # During EVALUATION the estimates are FROZEN, so every policy is
            # judged on the same information.  Otherwise we would be mixing
            # "learning the scheduler" with "evaluating the scheduler", and a
            # policy that happens to explore more would look better for the
            # wrong reason.
            if not self.freeze_effects:
                self.effects.observe(S, ctx, dg)
                # write-level design: the ACTUAL applied deltas, so eta is
                # estimated on the quantity B7+ actually optimises
                dnu_applied = {d.write.jid: (d.nu_star - d.write.nu_old)
                               for d in decisions if d.outcome != Outcome.REJECT}
                self.weffects.observe(dnu_applied, dg)
                if self.record_effect_rows:
                    D_raw = {j: float(dnu_applied.get(j, 0.0)) for j in self.claims}
                    D = {j: D_raw[j] / max(
                            self.claims[j].domain[1] - self.claims[j].domain[0], 1e-12)
                         for j in self.claims}
                    self.effect_rows.append({
                        "Z": {j: (1.0 if j in S else 0.0) for j in self.claims},
                        "W": {j: (1.0 if abs(D_raw[j]) > 1e-12 else 0.0)
                              for j in self.claims},
                        "D": D, "D_raw": D_raw,
                        "dg": {i: float(dg[i]) for i in self.intents},
                        "context": ctx.tolist(), "regime": regime})
                if self.epoch % self.cfg["estimation"]["refit_every"] == 0:
                    self.effects.fit()
                    self.weffects.fit()
                    if self.epoch % self.cfg["estimation"]["trace_every"] == 0:
                        import copy as _c
                        self.beta_trace.append(
                            {"epoch": self.epoch,
                             "beta": _c.deepcopy(self.effects.beta)})

            self.deficits.update_claims(S)
            # F_n = sum_i pi_i Phi_i / sum_i pi_i  -- the SAME definition the
            # metrics use, so the algorithm and the evaluation agree.  Uses the
            # CONTRACTUAL pi_class, never the dynamic w_i: the floor is a
            # contractual test and must not move with runtime urgency.
            rho_t = {}
            for t in self.tenants:
                mine = [(self.intents[i].pi_class, self.tracker.fulfilment(i))
                        for i in self.intents if self.intents[i].tenant == t]
                if mine:
                    den = sum(p for p, _ in mine)
                    rho_t[t] = (sum(p * f for p, f in mine) / den) if den > 0 else 1.0
                else:
                    rho_t[t] = 1.0
            self.deficits.update_tenants(rho_t)

            self.metrics.record_epoch(g2, p_hat, S, self.deficits.D, decisions,
                                      g_before=g)
            self.rd.event({"epoch": self.epoch, "mode": self.mode, "regime": regime,
                           "S": sorted(S), "g": g2, "p_hat": p_hat, "w": w,
                           "V": V, "D": dict(self.deficits.D),
                           "Lambda": dict(self.deficits.Lam),
                           "n_writes": len(decisions),
                           "kpm_cell": kpm2["_cell"]})

            if self.epoch % ck == 0:
                self._checkpoint()
            if self.epoch % self.cfg["run"]["log_every"] == 0:
                self.log.info("[%s] epoch %4d | S=%s | wIF=%.3f | util=%.0f%%",
                              self.mode, self.epoch, sorted(S),
                              np.mean([np.mean(self.metrics.fulfilled[i])
                                       for i in self.intents]),
                              kpm2["_cell"]["prb_util_pct"])

        self._checkpoint()
        out = self.metrics.summary(self.deficits, self.inner)
        out.update(self.forecaster.stats())
        return out

    # ------------------------------------------------------------------
    def _checkpoint(self):
        """Pickle everything needed to resume.  The RAN's RNG state is part
        of it, so a resumed run is bit-identical to an uninterrupted one."""
        keep = {k: v for k, v in self.__dict__.items()
                if k not in ("rd", "log", "cfg", "ck_tag")}
        self.rd.save_checkpoint({"epoch": self.epoch, "obj": keep}, tag=self.ck_tag)
