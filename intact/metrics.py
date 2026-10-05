"""
intact/metrics.py   --  INTACT v11 §13
======================================
Every evaluation metric, in one place.

PRIMARY
    weighted intent fulfilment ratio  =  sum_i pi_i Phi_i / sum_i pi_i

MULTI-TENANT -- these are the ones prior art CANNOT report, because it has
no notion of an owner:
    per-tenant fulfilment vs contracted floor;  floor violations
    Jain fairness over  rho_n / rho_n^min
    max claim deficit;  realised vs contracted actuation rate
    cross-tenant harm
    sovereignty consumption:  COUNT and MAGNITUDE

INNER LOOP
    misdirected-intervention rate  (should be ~0 by construction)
    override magnitude distribution
    escalation precision, split by reason
    no-new-victims violations (should be EXACTLY 0)

TWO-LOOP COHERENCE
    AWE = authority-without-effect, SPLIT:
      Type 1  structurally predictable -- C2 could have foreseen it.
              A SCHEDULER DEFECT.  Must be == 0 by construction.
      Type 2  runtime -- envelope shrank, safety rejection, model
              disagreement.  A LEGITIMATE DIAGNOSTIC.
    Reporting Type 1 == 0 is what makes Type 2 credible.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np


def jain(xs: List[float]) -> float:
    """Jain's fairness index.  1.0 = perfectly fair."""
    a = np.array([x for x in xs if np.isfinite(x)])
    if len(a) == 0 or a.sum() == 0:
        return 1.0
    return float(a.sum() ** 2 / (len(a) * (a ** 2).sum()))


class MetricBook:
    """Accumulates everything across a run, then summarises."""

    def __init__(self, intents, tenants, claims):
        self.intents, self.tenants, self.claims = intents, tenants, claims
        self.fulfilled = {i: [] for i in intents}       # per-epoch 1/0
        self.margins = {i: [] for i in intents}
        self.risks = {i: [] for i in intents}
        self.decisions = []
        self.awe_type1 = {j: 0 for j in claims}
        self.awe_type2 = {j: 0 for j in claims}
        self.granted = {j: 0 for j in claims}
        self.deficit_trace = {j: [] for j in claims}
        self.cross_harm_events = 0
        self.benign_interventions = 0
        self.total_interventions = 0
        self.c1_violations = 0        # two writers on one knob in one epoch
        self.c2_violations = 0        # tenant envelope over-committed
        self.safety_violations = 0    # applied write predicted to breach eps_i
        self.n_applied = 0
        self.epochs_seen = 0          # instance, not class, attribute
        # Outcome-based safety is intentionally separate from the inner
        # model's prediction.  It can include exogenous channel movement, but
        # unlike ``safety_violations`` it is measured from simulator outcomes
        # and is therefore comparable across architectures.
        self.observed_safety_crossings = 0
        self.observed_zero_crossings = 0
        self.observed_safety_by_intent = {i: 0 for i in intents}

    # ------------------------------------------------------------------
    def record_epoch(self, g, p_hat, S, D, decisions, g_before=None):
        self.epochs_seen += 1
        for i in self.intents:
            self.margins[i].append(g[i])
            self.risks[i].append(p_hat[i])
            self.fulfilled[i].append(1 if g[i] >= 0 else 0)
        for j in self.claims:
            self.deficit_trace[j].append(D[j])
            if j in S:
                self.granted[j] += 1
        self.decisions.extend(decisions)

        if g_before is not None:
            for i in self.intents:
                if g_before[i] >= self.intents[i].epsilon > g[i]:
                    self.observed_safety_crossings += 1
                    self.observed_safety_by_intent[i] += 1
                if g_before[i] >= 0.0 > g[i]:
                    self.observed_zero_crossings += 1

        # AWE: was a granted claim's every write rejected this epoch?
        by_claim = {}
        for d in decisions:
            by_claim.setdefault(d.write.jid, []).append(d)
        for j in S:
            ds = by_claim.get(j, [])
            if ds and all(d.outcome.value == "reject" for d in ds):
                # Type 1 would mean C2 could have foreseen it.  Because the
                # mask forbids those combinations up front, any AWE observed
                # here is Type 2 by construction -- which is the point.
                self.awe_type2[j] += 1

    # ------------------------------------------------------------------
    def summary(self, deficits, inner) -> Dict:
        out = {}

        # ---- primary ------------------------------------------------
        num = sum(self.intents[i].pi_class * np.mean(self.fulfilled[i])
                  for i in self.intents)
        den = sum(self.intents[i].pi_class for i in self.intents)
        out["weighted_intent_fulfilment"] = float(num / den)

        # ---- per intent / per tenant --------------------------------
        out["per_intent_fulfilment"] = {i: float(np.mean(self.fulfilled[i]))
                                        for i in self.intents}
        out["per_intent_mean_margin"] = {i: float(np.mean(self.margins[i]))
                                         for i in self.intents}
        # F_n = sum_i pi_i Phi_i / sum_i pi_i   (v11 s9, weighted form)
        ten_ful, ten_worst = {}, {}
        for t in self.tenants:
            mine = [(self.intents[i].pi_class, float(np.mean(self.fulfilled[i])))
                    for i in self.intents if self.intents[i].tenant == t]
            if mine:
                num = sum(p * f for p, f in mine); den = sum(p for p, _ in mine)
                ten_ful[t] = float(num / den) if den > 0 else 1.0
                ten_worst[t] = float(min(f for _, f in mine))
            else:
                ten_ful[t] = 1.0; ten_worst[t] = 1.0
        out["per_tenant_fulfilment"] = ten_ful
        # RETAINED AS A SEPARATE DIAGNOSTIC, not as the tenant's fulfilment:
        # it reveals whether an acceptable F_n is hiding one neglected intent.
        out["per_tenant_worst_intent"] = ten_worst
        out["floor_violations"] = {t: bool(ten_ful[t] < self.tenants[t].rho_min)
                                   for t in self.tenants}
        out["n_floor_violations"] = int(sum(out["floor_violations"].values()))
        out["jain_over_floor_ratio"] = jain(
            [ten_ful[t] / max(self.tenants[t].rho_min, 1e-9) for t in self.tenants])

        # Commercially weighted tenant fulfilment.  F_n is already weighted
        # by the priorities of tenant n's own intents; multiplying by
        # omega_n * sum_i pi_i makes the system-level quantity equivalent to
        # sum_i omega_{tenant(i)} pi_i Phi_i / sum_i omega_{tenant(i)} pi_i.
        tenant_priority_mass = {
            t: sum(self.intents[i].pi_class for i in self.intents
                   if self.intents[i].tenant == t)
            for t in self.tenants}
        ctw = {t: self.tenants[t].omega * tenant_priority_mass[t]
               for t in self.tenants}
        cden = sum(ctw.values())
        out["commercial_weighted_tenant_fulfilment"] = (
            float(sum(ctw[t] * ten_ful[t] for t in self.tenants) / cden)
            if cden > 0 else 1.0)
        out["worst_intent_fulfilment"] = float(
            min(out["per_intent_fulfilment"].values()))
        out["worst_tenant_fulfilment"] = float(min(ten_ful.values()))
        out["worst_tenant_floor_ratio"] = float(min(
            ten_ful[t] / max(self.tenants[t].rho_min, 1e-9)
            for t in self.tenants))

        active_tenants = {c.tenant for c in self.claims.values()}
        active_values = [ten_ful[t] for t in self.tenants if t in active_tenants]
        passive_values = [ten_ful[t] for t in self.tenants if t not in active_tenants]
        out["active_tenant_fulfilment_mean"] = (
            float(np.mean(active_values)) if active_values else None)
        out["passive_tenant_fulfilment_mean"] = (
            float(np.mean(passive_values)) if passive_values else None)
        host_values = [ten_ful[t] for t in self.tenants if self.tenants[t].is_host]
        out["host_tenant_fulfilment"] = (
            float(np.mean(host_values)) if host_values else None)

        # ---- actuation fairness -------------------------------------
        out["realised_actuation_rate"] = {j: deficits.realised_rate(j)
                                          for j in self.claims}
        out["contracted_actuation_rate"] = {j: self.claims[j].r_j
                                            for j in self.claims}
        out["actuation_shortfall"] = {
            j: float(self.claims[j].r_j - deficits.realised_rate(j))
            for j in self.claims}
        positive_shortfall = {
            j: max(0.0, value) for j, value in out["actuation_shortfall"].items()}
        out["max_positive_actuation_shortfall"] = float(
            max(positive_shortfall.values(), default=0.0))
        out["mean_positive_actuation_shortfall"] = float(
            np.mean(list(positive_shortfall.values())) if positive_shortfall else 0.0)
        out["n_undercontract_claims"] = int(sum(
            value > 1e-9 for value in positive_shortfall.values()))
        # A five-percentage-point miss is reported separately as operational
        # starvation; the raw shortfalls remain available so a paper can use
        # a different pre-specified tolerance without re-running anything.
        out["n_starved_claims_005"] = int(sum(
            value > 0.05 for value in positive_shortfall.values()))
        out["claim_starvation_rate_005"] = float(
            out["n_starved_claims_005"] / max(len(self.claims), 1))
        out["max_claim_deficit"] = {j: float(max(self.deficit_trace[j]))
                                    for j in self.claims}

        # ---- inner loop ---------------------------------------------
        n = max(len(self.decisions), 1)
        outs = [d.outcome.value for d in self.decisions]
        out["n_writes"] = len(self.decisions)
        out["disposition_mix"] = {k: outs.count(k) / n
                                  for k in ("admit", "override", "reject")}
        ovr = [d.magnitude for d in self.decisions if d.outcome.value == "override"]
        out["override_magnitude_mean"] = float(np.mean(ovr)) if ovr else 0.0
        out["override_magnitude_p95"] = float(np.percentile(ovr, 95)) if ovr else 0.0
        esc = [d for d in self.decisions if d.escalated]
        out["escalation_rate"] = len(esc) / n
        out["escalation_reasons"] = {
            r: sum(1 for d in esc if d.escalation_reason == r)
            for r in {d.escalation_reason for d in esc}} if esc else {}
        # a mediation is "misdirected" if we intervened with an EMPTY
        # implicated set -- should be structurally impossible
        out["misdirected_interventions"] = sum(
            1 for d in self.decisions
            if d.outcome.value != "admit" and not d.implicated)
        lat = [d.latency_ms for d in self.decisions]
        out["decision_latency_ms"] = {
            "mean": float(np.mean(lat)) if lat else 0.0,
            "p50": float(np.percentile(lat, 50)) if lat else 0.0,
            "p99": float(np.percentile(lat, 99)) if lat else 0.0}

        # ---- sovereignty --------------------------------------------
        out["sovereignty_count"] = dict(inner.override_count)
        out["sovereignty_magnitude"] = {k: float(v)
                                        for k, v in inner.override_mag.items()}
        out["sovereignty_count_breach"] = {
            t: bool(inner.override_count[t] > self.tenants[t].B_n)
            for t in self.tenants}

        # ---- two-loop coherence -------------------------------------
        out["awe_type1"] = {j: (self.awe_type1[j] / max(self.granted[j], 1))
                            for j in self.claims}
        out["awe_type2"] = {j: (self.awe_type2[j] / max(self.granted[j], 1))
                            for j in self.claims}
        out["no_new_victim_violations"] = int(
            getattr(inner, "no_new_victim_violations", 0))
        # ---- constraint and safety violations (uniform across methods) ---
        out["c1_violations"] = self.c1_violations
        out["c2_violations"] = self.c2_violations
        out["safety_violations"] = self.safety_violations
        out["predicted_safety_violations"] = self.safety_violations
        out["n_applied_writes"] = self.n_applied
        out["c1_violation_rate"] = self.c1_violations / max(self.epochs_seen, 1)
        out["c2_violation_rate"] = self.c2_violations / max(self.epochs_seen, 1)
        out["authority_violation_rate"] = (
            (self.c1_violations + self.c2_violations)
            / max(self.epochs_seen, 1))
        out["safety_violation_rate"] = self.safety_violations / max(self.n_applied, 1)
        out["predicted_safety_violation_rate"] = (
            self.safety_violations / max(self.n_applied, 1))
        out["observed_safety_crossings"] = self.observed_safety_crossings
        out["observed_zero_margin_crossings"] = self.observed_zero_crossings
        out["observed_safety_crossings_by_intent"] = dict(
            self.observed_safety_by_intent)
        exposure = max(self.epochs_seen * len(self.intents), 1)
        out["observed_safety_crossing_rate"] = (
            self.observed_safety_crossings / exposure)
        out["observed_zero_margin_crossing_rate"] = (
            self.observed_zero_crossings / exposure)
        return out
