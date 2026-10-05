"""
intact/inner/arbiter.py   --  INTACT v11 §10
============================================
THE INNER LOOP.  One projection, three dispositions, one emission.

DEFAULT IS ADMIT.  Every branch that leaves the trunk is an exception that
must be justified by a MEASUREMENT.

STEP 1  RELEVANCE-AND-RISK GATE                                   (v11 §10.1)
    I_p = { i : |s_{p,i}| >= sigma_min }        <- table lookup, 2-5 intents
    A   = { i in I_p : p_hat_i >= tau_risk  AND  Delta g_hat_i < -delta }

    Mediate ONLY if some intent is at risk AND THIS WRITE measurably pushes
    it further in.  Four cases, one of which is mediation:

      at risk?  implicated?  ->  action
        no          -           ADMIT   (stops over-intervention)
        YES         no          ADMIT   (stops punishing an UNRELATED claim)
        YES         YES         mediate
        no          YES         ADMIT   (harm exists, nothing near breaking)

    Row 2 is the row that answers "why should the host care about ANY
    at-risk intent?"  It shouldn't.  It should care about intents this write
    is measurably damaging.

    RELEVANCE IS MEASURED CAUSAL INFLUENCE, NOT CO-OWNERSHIP.  INTACT exists
    precisely because the host's energy xApp legitimately damages Telco-2's
    stream intent across a tenant boundary.

STEP 2  IS THE REQUESTED VALUE ALREADY SAFE?                      (v11 §10.2)
    g_hat_i(nu_req) >= eps_i for all i in A  ->  ADMIT UNCHANGED.
    A write can move an intent in the harmful direction by more than delta
    and still leave it comfortably safe.  HARMFUL DIRECTION != UNSAFE OUTCOME.

STEP 3  PROJECT ONTO THE ADMISSIBLE SET                           (v11 §10.3)
    nu* = argmin |nu - nu_req|   s.t.  nu in F_safe INTERSECT F_feasible

    "Project the requested action onto the admissible set, changing it as
    little as possible."  MINIMAL SUFFICIENT INTERVENTION -- nearest to the
    request, NOT best for the network.  The argmax alternative has no term
    pulling toward the tenant's request, so the host drifts the knob to
    whatever maximises weighted margin, i.e. becomes the tenant's
    controller.  This rule bounds the sovereignty MAGNITUDE by construction.

    F_feasible IS CONDITIONAL ON PARAMETER KIND:
        allocative -> { nu : d_r(nu,c) <= E_{n,r}(tau) }
        regulative -> V_p   (VACUOUS -- only safety can bind)
    Writing an MCS ceiling of 20 does not consume 20 PRBs.

STEP 4  READ OFF THE DISPOSITION                                  (v11 §10.4)
        nu* = nu_req   ->  ADMIT
        nu* = nu_old   ->  REJECT
        otherwise      ->  OVERRIDE
    REJECT IS NOT A SEPARATE BRANCH.  It is the case where the minimal
    sufficient value happens to be the status quo.  One search, three
    readings -- that is what removes the branch contradiction.

    If F_safe INTERSECT F_feasible = EMPTY  ->  EMIT AN ESCALATION
    CERTIFICATE and fall back to the NON-WORSENING tier:
        Omega_nw = { nu : g_hat_i(nu) >= g_hat_i(nu_old) }   NEVER EMPTY,
        because nu_old is always in it.
    ESCALATION IS AN EMISSION, NOT A DISPOSITION.  And under the local-linear
    model, Omega_nw collapses to {nu_old}, so escalation ALWAYS coincides
    with rejection: "reject + escalate".

PROPERTY -- NO NEW VICTIMS                                        (v11 §10.6)
    nu* always lies between nu_req and nu_old, and g_hat is affine hence
    monotone, so for EVERY intent i:
        g_hat_i(nu*) >= min( g_hat_i(nu_req), g_hat_i(nu_old) )
    The compromise can never leave an intent worse off than BOTH applying
    the write and rejecting it.  Asserted in code below.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple
from ..types import Claim, Kind, Write, Decision, Outcome


class InnerLoop:
    def __init__(self, claims, intents, tenants, sens, cfg, log):
        self.claims, self.intents, self.tenants = claims, intents, tenants
        self.sens = sens
        self.cfg = cfg
        self.log = log
        c = cfg["inner"]
        self.tau_risk = c["tau_risk"]
        self.delta = c["delta"]
        self.check_no_new_victims = c["assert_no_new_victims"]
        self.reject_instead_of_attenuate = bool(
            c.get("reject_instead_of_attenuate", False))
        # Runtime evidence counter.  Earlier revisions logged violations but
        # never propagated them into MetricBook, making the ensemble column
        # identically zero regardless of runtime behaviour.
        self.no_new_victim_violations = 0
        # sovereignty budget usage, per tenant, per window
        self.override_count: Dict[str, int] = {t: 0 for t in tenants}
        self.override_mag: Dict[str, float] = {t: 0.0 for t in tenants}

    # ------------------------------------------------------------------
    def _g_hat(self, iid, regime, param, nu, nu_old, g_now) -> float:
        """g_hat_i(nu) = g_i(now) + s_{p,i} * (nu - nu_old).   v11 §10.2."""
        s = self.sens.get(regime, param, iid)
        return g_now[iid] + s * (nu - nu_old)

    # ------------------------------------------------------------------
    def decide(self, write: Write, regime: str, g_now: Dict[str, float],
               p_hat: Dict[str, float], ran) -> Decision:
        t0 = time.perf_counter()
        c = self.claims[write.jid]
        nu_req, nu_old = write.nu_req, write.nu_old

        # ---------- STEP -1: QUANTISE + SLEW LIMIT (trust region) -------
        # Applied BEFORE anything else, so the gate, the safety test and the
        # projection all reason about the write that can ACTUALLY be applied.
        #   * quantise to the declared grid  (no continuous values)
        #   * clamp the move to max_step_frac of the domain width
        # This keeps every write inside the region where the local linear
        # model g_hat(nu) = g(nu_old) + s*(nu - nu_old) was fitted, which is
        # what makes the safety prediction trustworthy.
        lo_d, hi_d = c.domain
        span = max(hi_d - lo_d, 1e-12)
        nu_req = min(max(nu_req, lo_d), hi_d)
        if c.step > 0:
            nu_req = round(round((nu_req - lo_d) / c.step) * c.step + lo_d, 9)
        slew_capped = False
        if c.max_step_frac < 1.0:
            cap = c.max_step_frac * span
            if abs(nu_req - nu_old) > cap + 1e-12:
                nu_req = nu_old + (cap if nu_req > nu_old else -cap)
                nu_req = min(max(nu_req, lo_d), hi_d)
                if c.step > 0:
                    nu_req = round(round((nu_req - lo_d) / c.step) * c.step + lo_d, 9)
                slew_capped = True

        # ---------- STEP 0: FEASIBILITY IS UNCONDITIONAL ---------------
        # *** BUG FIX. ***  The relevance-and-risk gate governs SAFETY
        # mediation.  It must NOT gate FEASIBILITY.  A write that
        # over-commits the tenant's envelope is inadmissible whether or not
        # any intent happens to be at risk -- it is a physical impossibility,
        # not a trade-off.
        #
        # Previously the "A is empty -> ADMIT" fast path returned before
        # F_feasible was ever built, so an allocative write could exceed
        # E_{n,r} freely.  Measured consequence: j1 writing quota_T1 = 30
        # while quota_T1b still held 6 put tenant T1 at 36 PRB against an
        # envelope of 30, and the cell utilisation metric then saturated at
        # its 150% clamp.
        #
        # C2 (outer, static, over DOMAINS) is structural prevention;
        # this check is runtime enforcement (v11 s8.3.2).  The runtime half
        # had a hole in it whenever the network was healthy -- which is most
        # of the time.
        nu_eff, feas_clamped = nu_req, False
        if c.kind == Kind.ALLOCATIVE:
            feas0 = self._feasible(c, c.grid(), ran)
            if not feas0:
                return self._finish(write, Outcome.REJECT, nu_old, [], True,
                                    "resource-exhausted", t0)
            if not any(abs(v - nu_req) < 1e-9 for v in feas0):
                nu_eff = min(feas0, key=lambda v: abs(v - nu_req))
                feas_clamped = True
                if self.reject_instead_of_attenuate:
                    return self._finish(write, Outcome.REJECT, nu_old, [],
                                        False, "feasibility-veto", t0,
                                        slew_capped)

        # ---------- STEP 1: relevance-and-risk gate --------------------
        I_p = self.sens.affected_intents(regime, c.param, list(self.intents))
        A = []
        for i in I_p:
            dg = self.sens.get(regime, c.param, i) * (nu_eff - nu_old)
            g_pred = g_now[i] + dg
            # The ordinary harm deadband suppresses insignificant mediation,
            # but it must never suppress an actual predicted safety-buffer
            # crossing.  The old mismatch allowed |dg| <= delta writes to be
            # admitted while the uniform safety metric correctly counted the
            # crossing, making the evidence gate impossible to pass.
            crosses_floor = (g_now[i] >= self.intents[i].epsilon
                             > g_pred)
            materially_harmful = dg < -self.delta
            if (p_hat.get(i, 0.0) >= self.tau_risk
                    and (materially_harmful or crosses_floor)):
                A.append(i)

        if not A:
            # nothing at risk -- but still return the FEASIBLE value, which
            # may differ from what was asked for
            out0 = Outcome.OVERRIDE if feas_clamped else Outcome.ADMIT
            return self._finish(write, out0, nu_eff, A, False,
                                "feasibility-clamp" if feas_clamped else "", t0,
                                slew_capped)

        # ---------- STEP 2: is the request already safe? ---------------
        if all(self._g_hat(i, regime, c.param, nu_eff, nu_old, g_now)
               >= self.intents[i].epsilon for i in A):
            out0 = Outcome.OVERRIDE if feas_clamped else Outcome.ADMIT
            return self._finish(write, out0, nu_eff, A, False,
                                "feasibility-clamp" if feas_clamped else "", t0,
                                slew_capped)

        # Registered WCNC ablation: keep the relevance/risk test and every
        # outer-loop component, but replace minimal sufficient attenuation
        # by a binary reject.  This is deliberately after the requested-value
        # safety test, so harmless writes are still admitted unchanged.
        if self.reject_instead_of_attenuate:
            return self._finish(write, Outcome.REJECT, nu_old, A, False,
                                "safety-veto", t0, slew_capped)

        # ---------- degraded mode: no trustworthy value model ----------
        # Fall back to admit/reject only.  Rejection needs no value model at
        # all -- it holds nu_old, whose consequence is directly observed.
        if self.sens.is_degraded(c.param, regime):
            self.log.warning("value model degraded for %s in %s -> reject only",
                             c.param, regime)
            return self._finish(write, Outcome.REJECT, nu_old, A, False,
                                "degraded-value-model", t0, slew_capped)

        # ---------- restrict the search to [nu_req, nu_old] -------------
        # *** THIS GUARD IS WHAT MAKES no-new-victims TRUE. ***
        # v11 §10.6 argues that nu* always lies BETWEEN nu_req and nu_old,
        # so by monotonicity of g_hat no intent ends up worse than both
        # endpoints.  That argument silently assumed the search is confined
        # to that interval -- it is not, unless we confine it.
        #
        # Concretely: the host proposes LOWERING power (helps its own energy
        # intent, hurts T3's latency intent).  T3 needs power RAISED.  An
        # unconstrained F_safe then contains values ABOVE nu_old, and the
        # projector would push power beyond the status quo -- helping T3 at
        # the host's expense by MORE than simply rejecting the write.  That
        # is the host optimising, not mediating, and it creates a new victim.
        #
        # Confining the search to the closed interval between the request
        # and the status quo is the correct reading of MINIMAL SUFFICIENT
        # INTERVENTION: never move the knob further than doing nothing would.
        # If the interval contains no safe value, that is precisely the
        # escalation case, and Tier B returns nu_old.
        lo_i, hi_i = min(nu_eff, nu_old), max(nu_eff, nu_old)
        grid = [v for v in c.grid() if lo_i - 1e-9 <= v <= hi_i + 1e-9]
        if nu_old not in grid:
            grid.append(nu_old)          # the status quo is always a candidate

        # ---------- STEP 3: build F_safe and F_feasible ----------------
        F_safe = [nu for nu in grid
                  if all(self._g_hat(i, regime, c.param, nu, nu_old, g_now)
                         >= self.intents[i].epsilon for i in A)]
        F_feas = self._feasible(c, grid, ran)
        F = [nu for nu in F_safe if nu in set(F_feas)]

        if F:
            nu_star = min(F, key=lambda v: abs(v - nu_eff))
            esc, reason = False, ""
        else:
            # ---------- Tier B: escalate + non-worsening ---------------
            esc = True
            reason = ("range-exhausted" if not F_safe else "safety-infeasible")
            Omega_nw = [nu for nu in F_feas
                        if all(self._g_hat(i, regime, c.param, nu, nu_old, g_now)
                               >= self._g_hat(i, regime, c.param, nu_old, nu_old, g_now)
                               for i in A)]
            if nu_old not in Omega_nw:
                Omega_nw.append(nu_old)         # nu_old is ALWAYS in Omega_nw
            nu_star = min(Omega_nw, key=lambda v: abs(v - nu_eff))

        # ---------- STEP 4: read off the disposition -------------------
        if abs(nu_star - nu_req) < 1e-9:
            out = Outcome.ADMIT
        elif abs(nu_star - nu_old) < 1e-9:
            out = Outcome.REJECT
        else:
            out = Outcome.OVERRIDE

        # ---------- no-new-victims assertion (v11 §10.6) ---------------
        if self.check_no_new_victims and out == Outcome.OVERRIDE:
            for i in I_p:
                lo = min(self._g_hat(i, regime, c.param, nu_req, nu_old, g_now),
                         self._g_hat(i, regime, c.param, nu_old, nu_old, g_now))
                got = self._g_hat(i, regime, c.param, nu_star, nu_old, g_now)
                if got < lo - 1e-6:
                    self.no_new_victim_violations += 1
                    self.log.error("NO-NEW-VICTIMS VIOLATED for %s: %.4f < %.4f "
                                   "-- the local-linear model is falsified here",
                                   i, got, lo)
        return self._finish(write, out, nu_star, A, esc, reason, t0, slew_capped)

    # ------------------------------------------------------------------
    def _feasible(self, c: Claim, grid, ran) -> List[float]:
        """
        F_feasible.  VACUOUS for regulative parameters (v11 §10.3).

        For allocative ones: d_r(nu,c) must fit what remains of the tenant's
        envelope after its OTHER allocative claims.
        """
        if c.kind == Kind.REGULATIVE:
            return list(grid)
        env, used_all = ran.headroom(c.tenant, c.resource)
        mine = ran.current_controls().get(c.param, 0.0)
        used_by_others = used_all - mine
        return [nu for nu in grid if ran.demand(c.param, nu) <= env - used_by_others + 1e-9]

    # ------------------------------------------------------------------
    def _finish(self, write, out, nu_star, A, esc, reason, t0,
                slew_capped: bool = False) -> Decision:
        mag = abs(nu_star - write.nu_req)
        d = Decision(write=write, outcome=out, nu_star=nu_star,
                     implicated=list(A), escalated=esc, escalation_reason=reason,
                     latency_ms=(time.perf_counter() - t0) * 1000.0,
                     magnitude=mag)
        d.slew_capped = slew_capped
        if out == Outcome.OVERRIDE:
            tid = self.claims[write.jid].tenant
            self.override_count[tid] += 1
            self.override_mag[tid] += mag
        return d
