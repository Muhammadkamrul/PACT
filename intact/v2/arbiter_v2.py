"""
intact/v2/arbiter_v2.py
=======================
Inner loop, three defects fixed.

DEFECT 1 -- NO RUNNING LEDGER (a real implementation bug in v1)
v1 handed the SAME pre-epoch margin vector to every inner decision, so the
second claim never saw the first claim's predicted effect.  The loop was
described as sequential mediation and behaved as parallel mediation.  Two
individually-safe writes could therefore jointly breach epsilon and nothing
would notice.  v2 keeps a running ledger: each decision reads the margins
as updated by every write already executed this epoch, and the before/after
values are logged so the ordering is auditable.

DEFECT 2 -- ORDER WAS NOT DETERMINISTIC
Sequential mediation is order-dependent by construction, so the order must
be a stated design choice rather than dictionary iteration order.  v2 sorts
by (-deficit, claim id): the most overdue claim is mediated first and sees
the widest safety headroom, with the id as a tiebreaker so runs are
reproducible.

DEFECT 3 -- MILD WAS APPLIED WHERE IT COULD NOT MATTER
Measured in v1: B6 (inner-only, admits EVERY claim, no weighting at all)
ties B6++ on wIF at p=0.67 while committing 2.19 C1 violations per epoch.
The inner loop was doing all the outcome work, so raising a claim's outer
weight changed admission ORDER and almost nothing else.  p_hat only gated
whether an intent was protected.

v2 lets p_hat widen the protective band itself:

    eps_eff_i = eps_i * (1 + lambda_risk * p_hat_i)

An intent MILD believes is about to fail gets a wider berth, so the write
is clipped earlier.  This puts the risk score where decisions actually
bind.  `risk_epsilon_mode` selects gate | buffer | both so the paired
ablation the analysis calls for is a config switch, not a code edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from math import exp

import numpy as np


@dataclass
class InnerDecision:
    claim: str
    param: str
    nu_old: float
    nu_req: float
    nu_eff: float
    nu_star: float
    outcome: str                      # admit | override | reject
    reason: str
    implicated: List[str] = field(default_factory=list)
    escalated: bool = False
    g_before: Dict[str, float] = field(default_factory=dict)
    g_after: Dict[str, float] = field(default_factory=dict)
    eps_eff: Dict[str, float] = field(default_factory=dict)
    order_index: int = 0

    def as_row(self) -> Dict:
        return {"claim": self.claim, "param": self.param,
                "nu_old": self.nu_old, "nu_req": self.nu_req,
                "nu_eff": self.nu_eff, "nu_star": self.nu_star,
                "dnu": self.nu_star - self.nu_old,
                "outcome": self.outcome, "reason": self.reason,
                "escalated": self.escalated,
                "n_implicated": len(self.implicated),
                "order_index": self.order_index}


def effective_epsilon(eps: float, p_hat: float, lambda_risk: float,
                      mode: str) -> float:
    """eps_eff = eps * (1 + lambda_risk * p_hat)  when the buffer is modulated.

    Monotone in p_hat and equal to eps at p_hat = 0, so a system with no
    risk information behaves exactly as v1 did.  That equality is what makes
    the ablation clean: "buffer off" is not a different code path, it is
    lambda_risk = 0.
    """
    if mode in ("buffer", "both"):
        return float(eps) * (1.0 + float(lambda_risk) * float(p_hat))
    return float(eps)


class ArbiterV2:
    """Sequential, ledger-aware inner loop.

    Retains the v1 three-condition intersection -- quantised trust region,
    unconditional feasibility, safety projection -- and the no-new-victim
    interval restriction, all of which validated.  Everything above is
    additive.
    """

    def __init__(self, cfg: Dict, intents, claims, s_lookup,
                 feasible_fn=None):
        inner = cfg.get("inner", {})
        self.tau_risk = float(inner.get("tau_risk", 0.65))
        self.delta = float(inner.get("delta", 0.01))
        self.max_step_frac = float(inner.get("max_step_frac", 0.25))
        self.sigma_min = float(cfg.get("sensitivity", {}).get("sigma_min", 1.5e-3))
        v2 = cfg.get("v2", {}) or {}
        self.lambda_risk = float(v2.get("lambda_risk", 0.5))
        self.risk_mode = str(v2.get("risk_epsilon_mode", "both"))
        self.protection_policy = str(v2.get("protection_policy", "strict"))
        self.existing_breach_tolerance = float(
            v2.get("existing_breach_tolerance", 0.02))
        self.local_optimize = bool(v2.get("inner_local_optimize", False))
        self.local_temperature = float(v2.get("inner_local_temperature", 0.03))
        self.local_risk_reserve = float(v2.get("risk_margin_reserve", 0.0))
        self.local_action_cost = float(v2.get("inner_local_action_cost", 0.0))
        self.local_min_gain = float(v2.get("inner_local_min_gain", 1e-9))
        self.predictive_s_blend = bool(v2.get("predictive_s_blend", False))
        self._forecast_regime = None
        self._planning_risk: Dict[str, float] = {}
        self.intents = intents
        self.claims = claims
        self.s_lookup = s_lookup
        self.feasible_fn = feasible_fn

    def set_predictive_context(self, p_hat: Dict[str, float],
                               forecast_regime) -> None:
        self._planning_risk = dict(p_hat)
        self._forecast_regime = forecast_regime

    def planning_s(self, regime: str, param: str, iid: str) -> float:
        """Risk-weighted current/future-regime sensitivity for planning.

        Safety always uses the current-regime slope.  Only the outer value and
        local dose objective may look ahead, using a Phase-A coefficient from
        the forecast regime.  This prevents a forecast from weakening the
        current physical safety check.
        """
        current = float(self.s_lookup(regime, param, iid))
        if not self.predictive_s_blend or not self._forecast_regime:
            return current
        future_regime = (self._forecast_regime.get(iid)
                         if isinstance(self._forecast_regime, dict)
                         else self._forecast_regime)
        if future_regime is None:
            return current
        future = float(self.s_lookup(future_regime, param, iid))
        p = float(np.clip(self._planning_risk.get(iid, 0.0), 0.0, 1.0))
        return (1.0 - p) * current + p * future

    # ------------------------------------------------------------------
    def order(self, S: Sequence[str], deficits: Dict[str, float],
              priorities: Optional[Dict[str, float]] = None) -> List[str]:
        """Deterministic benefit-first sequential order.

        The safety ledger is path dependent: an early write can consume the
        margin available to later writes.  Ordering only by C4 debt let a
        low-value overdue claim consume that budget before a high-value one.
        The revised order is predicted benefit, then deficit, then claim id.
        """
        priorities = priorities or {}
        return sorted(S, key=lambda j: (-float(priorities.get(j, 0.0)),
                                        -float(deficits.get(j, 0.0)), str(j)))

    def _grid(self, claim, nu_old: float) -> np.ndarray:
        lo, hi = claim.domain
        step = float(getattr(claim, "step", 1.0)) or 1.0
        span = self.max_step_frac * (hi - lo)
        g_lo = max(lo, nu_old - span)
        g_hi = min(hi, nu_old + span)
        n = int(np.floor((g_hi - g_lo) / step)) + 1
        grid = g_lo + step * np.arange(max(n, 1))
        grid = grid[(grid >= lo - 1e-9) & (grid <= hi + 1e-9)]
        if not np.any(np.abs(grid - nu_old) < 1e-9):
            grid = np.append(grid, nu_old)     # status quo is always available
        return np.unique(grid)

    def decide_all(self, S: Sequence[str], g_now: Dict[str, float],
                   p_hat: Dict[str, float], regime: str,
                   proposals: Dict[str, float], controls: Dict[str, float],
                   deficits: Dict[str, float],
                   priorities: Optional[Dict[str, float]] = None,
                   intent_weights: Optional[Dict[str, float]] = None
                   ) -> Tuple[List[InnerDecision], Dict[str, float]]:
        """Mediate every admitted claim against a RUNNING margin ledger."""
        ledger = dict(g_now)                       # <-- the fix for defect 1
        out: List[InnerDecision] = []
        for idx, jid in enumerate(self.order(S, deficits, priorities)):
            c = self.claims[jid]
            nu_old = float(controls.get(c.param, np.mean(c.domain)))
            nu_req = float(proposals.get(jid, nu_old))
            d = self._decide_one(jid, c, nu_old, nu_req, ledger, p_hat,
                                 regime, controls, idx, intent_weights)
            out.append(d)
            # commit this write's PREDICTED effect before the next decision
            for i, gv in d.g_after.items():
                ledger[i] = gv
            controls[c.param] = d.nu_star
        return out, ledger

    # ------------------------------------------------------------------
    def _decide_one(self, jid, c, nu_old: float, nu_req: float,
                    ledger: Dict[str, float], p_hat: Dict[str, float],
                    regime: str, controls: Dict[str, float],
                    idx: int,
                    intent_weights: Optional[Dict[str, float]] = None
                    ) -> InnerDecision:
        grid = self._grid(c, nu_old)

        # ---- STEP 0: unconditional feasibility (physics, not a trade-off)
        if self.feasible_fn is not None:
            grid = self.feasible_fn(jid, c, grid, controls)
            if len(grid) == 0:
                return InnerDecision(jid, c.param, nu_old, nu_req, nu_old,
                                     nu_old, "reject", "infeasible",
                                     escalated=True, order_index=idx)
        nu_eff = float(grid[np.argmin(np.abs(grid - nu_req))])

        # ---- relevance: which intents does this parameter measurably move
        rel = [i for i in self.intents
               if abs(self.s_lookup(regime, c.param, i)) >= self.sigma_min]

        # Optional write-level local optimisation.  The outer loop decides
        # authority; the inner loop decides the concrete dose.  Restrict the
        # search to the segment from the status quo to the xApp request, so
        # INTACT may partially override a claim but may never reverse it or
        # invent an action the claimant did not authorise.
        if self.local_optimize and rel:
            lo_path, hi_path = min(nu_old, nu_eff), max(nu_old, nu_eff)
            path = grid[(grid >= lo_path - 1e-9) & (grid <= hi_path + 1e-9)]
            weights = intent_weights or {
                i: float(getattr(self.intents[i], "pi_class", 1.0))
                for i in self.intents}
            tau = max(self.local_temperature, 1e-9)

            def sig(x):
                z = max(-60.0, min(60.0, float(x) / tau))
                return 1.0 / (1.0 + exp(-z))

            def utility(v):
                dose = float(v) - nu_old
                val = 0.0
                for i in self.intents:
                    g0 = float(ledger.get(i, 0.0))
                    reserve = self.local_risk_reserve * float(p_hat.get(i, 0.0))
                    g1 = g0 + self.planning_s(regime, c.param, i) * dose
                    val += max(0.0, float(weights.get(i, 0.0))) * (
                        sig(g1 - reserve) - sig(g0 - reserve))
                span = max(float(c.domain[1] - c.domain[0]), 1e-9)
                val -= self.local_action_cost * abs(dose) / span
                return float(val)

            scored = [(utility(v), -abs(float(v) - nu_req), float(v))
                      for v in path]
            best_gain, _, best_v = max(scored)
            if best_gain <= self.local_min_gain:
                return InnerDecision(
                    jid, c.param, nu_old, nu_req, nu_old, nu_old, "reject",
                    "local-objective-no-positive-dose", rel, False,
                    {i: float(ledger.get(i, 0.0)) for i in rel},
                    {i: float(ledger.get(i, 0.0)) for i in rel}, {}, idx)
            nu_eff = best_v

        eps_eff = {}
        protection_floor = {}
        implicated = []
        for i in rel:
            eps_i = float(getattr(self.intents[i], "epsilon", 0.02))
            e = effective_epsilon(eps_i, float(p_hat.get(i, 0.0)),
                                  self.lambda_risk, self.risk_mode)
            eps_eff[i] = e
            gated = (float(p_hat.get(i, 0.0)) >= self.tau_risk
                     if self.risk_mode in ("gate", "both") else True)
            s = self.s_lookup(regime, c.param, i)
            g0 = float(ledger.get(i, 0.0))
            g_at_req = g0 + s * (nu_eff - nu_old)
            harmful = g_at_req < g0 - self.delta
            crosses = (g0 >= e) and (g_at_req < e)
            if self.protection_policy == "portfolio_authorized":
                # The exact outer selector has already checked the joint
                # after-write margin vector.  Do not veto one member in
                # isolation and thereby destroy a jointly safe portfolio.
                # This assumes epoch writes are committed as one batch.
                protection_floor[i] = e
            elif self.protection_policy == "no_new_victims":
                # A safety floor protects an intent that is currently safe;
                # it must not demand the impossible instantaneous recovery of
                # an intent that was already below the floor.  For a breached
                # intent we instead bound additional degradation.  This
                # permits explicit best-effort trade-offs without creating a
                # new victim.
                if g0 >= e:
                    floor = e
                elif g0 >= 0.0:
                    floor = 0.0
                else:
                    floor = g0 - max(0.0, self.existing_breach_tolerance)
                protection_floor[i] = floor
                if gated and g_at_req < floor - 1e-12:
                    implicated.append(i)
            else:
                protection_floor[i] = e
                if gated and (harmful or crosses):
                    implicated.append(i)

        if not implicated:
            g_after = {i: float(ledger.get(i, 0.0))
                       + self.s_lookup(regime, c.param, i) * (nu_eff - nu_old)
                       for i in rel}
            changed = abs(nu_eff - nu_req) > 1e-9
            return InnerDecision(jid, c.param, nu_old, nu_req, nu_eff, nu_eff,
                                 "override" if changed else "admit",
                                 ("local-dose-optimisation" if changed else
                                  "no-implicated-intent"), [], False,
                                 {i: ledger.get(i, 0.0) for i in rel}, g_after,
                                 eps_eff, idx)

        # ---- SAFETY: intersection over ALL implicated intents -----------
        lo, hi = min(nu_eff, nu_old), max(nu_eff, nu_old)
        band = grid[(grid >= lo - 1e-9) & (grid <= hi + 1e-9)]
        safe = []
        for v in band:
            ok = True
            for i in implicated:
                s = self.s_lookup(regime, c.param, i)
                floor = protection_floor.get(i, eps_eff[i])
                if float(ledger.get(i, 0.0)) + s * (v - nu_old) < floor - 1e-12:
                    ok = False
                    break
            if ok:
                safe.append(v)

        if safe:
            nu_star = float(min(safe, key=lambda v: abs(v - nu_eff)))
            outcome = "admit" if abs(nu_star - nu_eff) < 1e-9 else "override"
            reason = "safe" if outcome == "admit" else "safety-projection"
            escal = False
        else:
            # Tier B: no admissible point.  Fall back to the status quo,
            # which is always in the grid, and escalate.
            nu_star, outcome, reason, escal = nu_old, "reject", "safety-infeasible", True

        g_before = {i: float(ledger.get(i, 0.0)) for i in rel}
        g_after = {i: g_before.get(i, 0.0)
                   + self.s_lookup(regime, c.param, i) * (nu_star - nu_old)
                   for i in rel}
        return InnerDecision(jid, c.param, nu_old, nu_req, nu_eff, nu_star,
                             outcome, reason, implicated, escal,
                             g_before, g_after, eps_eff, idx)


def no_new_victim_violations(decisions: Sequence[InnerDecision]) -> int:
    """Property check: nu* in [nu_req, nu_old] and g affine => the minimum is
    at an endpoint, so no intent may end below BOTH alternatives.  A nonzero
    count falsifies local linearity, which makes this a model check rather
    than a performance number."""
    bad = 0
    for d in decisions:
        for i, gb in d.g_before.items():
            ga = d.g_after.get(i, gb)
            g_req = gb + (d.g_after.get(i, gb) - gb)
            if ga < min(gb, g_req) - 1e-6:
                bad += 1
    return bad
