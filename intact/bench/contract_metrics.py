"""Contract-structure metrics: priority respect, tenant classes and C3.

All quantities are computed from one finished run: the scenario
configuration, the run summary produced by MetricBook, and the per-epoch
fulfilment bits recorded by ``LeanRunDir`` (1 when an intent's margin is
non-negative after the epoch -- the same definition as weighted intent
fulfilment).

Intent priority (pi_class)
    intent_priority_inversion_pct   % of (epoch, intent-pair) cases where the
        HIGHER-priority intent fails while the LOWER-priority intent is met.
        Pairs of risk-equivalent intents (same tenant, KPI, target and
        direction) are excluded because no method can separate them.
        Lower = priority respected better.
    intent_priority_kendall_tau     rank correlation between pi_class and
        run-level fulfilment across intents.  Higher = better.
    intent_priority_lift_pp         100 * (pi-weighted - unweighted
        fulfilment).  Positive = high-priority intents are served better.
    ful_pi_<value>                  mean fulfilment of intents in each class.

Tenant priority (omega), non-host tenants that hold intents
    tenant_priority_inversion_pct   % of (epoch, tenant-pair) cases where the
        higher-omega tenant's epoch fulfilment is strictly below the
        lower-omega tenant's.  Lower = better.
    tenant_priority_kendall_tau, tenant_priority_lift_pp,
    omega_weighted_tenant_fulfilment, ful_tenant_omega_rank_<r> (r=1 is the
    highest-paying tenant of the scenario).

Tenant classes (tenants that hold at least one intent)
    tenant_fulfilment_active_mean   tenants that own at least one claim
    tenant_fulfilment_passive_mean  tenants with intents but no claim (NaN
                                    when a scenario has none)
    tenant_fulfilment_host          the host (cell owner)

C3 tenant floor (contracted minimum fulfilment rho_min)
    c3_floor_violation_pct   % of intent-holding tenants whose run-level
                             fulfilment is below rho_min
    c3_mean_shortfall_pp     mean of 100 * max(0, rho_min - F_n)
    c3_max_shortfall_pp      worst tenant's shortfall

Intent targets
    unweighted_intent_fulfilment, intent_target_miss_pct = 100 * (1 - mean)
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def _kendall(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    try:
        from scipy.stats import kendalltau
        v = kendalltau(a, b).statistic
        return float(v) if v is not None and math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _eq_key(it: Dict) -> tuple:
    return (it.get("tenant"), it.get("kpi"),
            round(float(it.get("target", 0.0)), 9), it.get("direction"))


def contract_metrics(cfg: Dict, summary: Dict, trace: List[Dict],
                     intent_order: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    intents = {i["iid"]: i for i in cfg.get("intents", [])}
    tenants = {t["tid"]: t for t in cfg.get("tenants", [])}
    owners = {c["tenant"] for c in cfg.get("claims", [])}
    per_int = summary.get("per_intent_fulfilment") or {}
    per_ten = summary.get("per_tenant_fulfilment") or {}
    order = [i for i in (intent_order or sorted(per_int)) if i in intents]
    if not order:
        return out
    phi = np.array([float(per_int.get(i, np.nan)) for i in order])
    pis = np.array([float(intents[i].get("pi_class", 1.0)) for i in order])
    wif = float(summary.get("weighted_intent_fulfilment", np.nan))
    uif = float(np.nanmean(phi))
    out["unweighted_intent_fulfilment"] = uif
    out["intent_target_miss_pct"] = 100.0 * (1.0 - uif)
    out["intent_priority_lift_pp"] = 100.0 * (wif - uif)
    out["intent_priority_kendall_tau"] = _kendall(pis, phi)
    for pv in sorted(set(pis.tolist())):
        out[f"ful_pi_{pv:g}"] = float(np.nanmean(phi[pis == pv]))

    F = np.array([r.get("f", []) for r in trace], dtype=float) \
        if trace else np.zeros((0, len(order)))
    has_bits = F.ndim == 2 and F.shape[0] > 0 and F.shape[1] == len(order)
    keys = [_eq_key(intents[i]) for i in order]
    K = len(order)
    pairs = [(a, b) for a in range(K) for b in range(K)
             if pis[a] > pis[b] + 1e-12 and keys[a] != keys[b]]
    if has_bits and pairs:
        A = np.array([p[0] for p in pairs])
        B = np.array([p[1] for p in pairs])
        inv = (F[:, A] == 0) & (F[:, B] == 1)
        out["intent_priority_inversion_pct"] = 100.0 * float(inv.mean())
    else:
        out["intent_priority_inversion_pct"] = float("nan")

    # ---- tenants ---------------------------------------------------------
    t_int = {t: [k for k, i in enumerate(order)
                 if intents[i].get("tenant") == t] for t in tenants}
    with_int = [t for t in tenants if t_int[t]]
    is_host = {t: bool(tenants[t].get("is_host", False)) for t in tenants}
    active = [t for t in with_int if t in owners and not is_host[t]]
    passive = [t for t in with_int if t not in owners and not is_host[t]]
    host = [t for t in with_int if is_host[t]]
    mean_of = lambda ts: (float(np.mean([float(per_ten.get(t, np.nan))
                                         for t in ts])) if ts else float("nan"))
    out["tenant_fulfilment_active_mean"] = mean_of(active)
    out["tenant_fulfilment_passive_mean"] = mean_of(passive)
    out["tenant_fulfilment_host"] = mean_of(host)
    out["n_active_tenants_with_intents"] = float(len(active))
    out["n_passive_tenants_with_intents"] = float(len(passive))

    rho = {t: float(tenants[t].get("rho_min", 0.0)) for t in with_int}
    Fn = {t: float(per_ten.get(t, np.nan)) for t in with_int}
    if with_int:
        short = [max(0.0, rho[t] - Fn[t]) for t in with_int]
        out["c3_floor_violation_pct"] = 100.0 * float(np.mean(
            [Fn[t] < rho[t] - 1e-12 for t in with_int]))
        out["c3_mean_shortfall_pp"] = 100.0 * float(np.mean(short))
        out["c3_max_shortfall_pp"] = 100.0 * float(np.max(short))

    nh = [t for t in with_int if not is_host[t]]
    if nh:
        om = np.array([float(tenants[t].get("omega", 1.0)) for t in nh])
        fv = np.array([Fn[t] for t in nh])
        out["omega_weighted_tenant_fulfilment"] = float(
            np.sum(om * fv) / max(np.sum(om), 1e-12))
        out["tenant_priority_lift_pp"] = 100.0 * (
            out["omega_weighted_tenant_fulfilment"] - float(np.mean(fv)))
        out["tenant_priority_kendall_tau"] = _kendall(om, fv)
        ranked = sorted(nh, key=lambda t: (-float(tenants[t].get("omega", 1.0)),
                                           str(t)))
        for r, t in enumerate(ranked, 1):
            out[f"ful_tenant_omega_rank_{r}"] = Fn[t]
        if has_bits and len(nh) > 1:
            cols = []
            for t in nh:
                idx = t_int[t]
                w = pis[idx]
                cols.append((F[:, idx] * w).sum(axis=1) / max(w.sum(), 1e-12))
            Ft = np.stack(cols, axis=1)
            tp = [(a, b) for a in range(len(nh)) for b in range(len(nh))
                  if om[a] > om[b] + 1e-12]
            if tp:
                A = np.array([p[0] for p in tp])
                B = np.array([p[1] for p in tp])
                out["tenant_priority_inversion_pct"] = 100.0 * float(
                    (Ft[:, A] < Ft[:, B] - 1e-12).mean())
    out.setdefault("tenant_priority_inversion_pct", float("nan"))
    return out
