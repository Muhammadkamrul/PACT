"""Threshold-aware outer objective for the INTACTv3-WIF candidate.

The reported primary metric is a priority-weighted fraction of epochs in
which intent margin is non-negative.  A raw sum of predicted margin changes
does not represent that metric: it rewards extra headroom on an intent that
is already fulfilled exactly as much as rescuing an intent just below zero.

This module supplies a smooth, deterministic proxy for the *change* in that
metric.  It uses only pre-decision margins, Phase-A sensitivity estimates and
the independently projected dose of each live claim.  No realised post-write
outcome is used, so it is deployable rather than an oracle.
"""
from __future__ import annotations

from math import exp, sqrt
from typing import Callable, Dict, Iterable, Mapping, Optional, Set


def smooth_fulfilment(margin: float, temperature: float) -> float:
    """Stable logistic approximation to ``1[margin >= 0]``.

    A small positive temperature concentrates value around the fulfilment
    boundary.  It must not be tuned on the final test scenarios.
    """
    tau = max(float(temperature), 1e-9)
    x = max(-60.0, min(60.0, float(margin) / tau))
    return 1.0 / (1.0 + exp(-x))


def portfolio_wif_utility(
        selected: Set[str],
        claims: Mapping,
        intents: Mapping,
        margins: Mapping[str, float],
        regime: str,
        projected_dose: Mapping[str, float],
        sensitivity: Callable[[str, str, str], float],
        sensitivity_se: Optional[Callable[[str, str, str], float]] = None,
        intent_weights: Optional[Mapping[str, float]] = None,
        margin_reserve: Optional[Mapping[str, float]] = None,
        temperature: float = 0.04,
        confidence_z: float = 0.0,
        action_cost: float = 0.0) -> float:
    """Predict the priority-weighted fulfilment improvement of a portfolio.

    ``confidence_z`` subtracts a simple independent-error bound from the
    predicted margin.  ``action_cost`` discourages writes that consume
    authority but have negligible predicted benefit; each dose is normalised
    by the claim parameter's configured range.
    """
    weights = {iid: (float(intent_weights[iid])
                     if intent_weights is not None and iid in intent_weights
                     else float(intent.pi_class))
               for iid, intent in intents.items()}
    reserve = margin_reserve or {}
    den = sum(max(0.0, w) for w in weights.values())
    if den <= 0.0:
        return 0.0

    gain = 0.0
    for iid, intent in intents.items():
        delta = 0.0
        variance = 0.0
        for jid in selected:
            dose = float(projected_dose.get(jid, 0.0))
            param = claims[jid].param
            delta += float(sensitivity(regime, param, iid)) * dose
            if sensitivity_se is not None:
                se = max(0.0, float(sensitivity_se(regime, param, iid)))
                variance += (se * dose) ** 2
        before_margin = float(margins[iid]) - float(reserve.get(iid, 0.0))
        robust_after = before_margin + delta
        robust_after -= max(0.0, float(confidence_z)) * sqrt(variance)
        before = smooth_fulfilment(before_margin, temperature)
        after = smooth_fulfilment(robust_after, temperature)
        gain += max(0.0, weights[iid]) * (after - before)

    normalised_cost = 0.0
    for jid in selected:
        claim = claims[jid]
        lo, hi = claim.domain
        span = max(float(hi) - float(lo), 1e-9)
        normalised_cost += abs(float(projected_dose.get(jid, 0.0))) / span
    return gain / den - max(0.0, float(action_cost)) * normalised_cost


def portfolio_linear_margin_utility(
        selected: Set[str], claims: Mapping, intents: Mapping,
        regime: str, projected_dose: Mapping[str, float],
        sensitivity: Callable[[str, str, str], float],
        use_pi_class: bool = False,
        intent_weights: Optional[Mapping[str, float]] = None,
        action_cost: float = 0.0) -> float:
    """B3-compatible signed margin utility, optionally priority weighted.

    This is included as a registered ablation because the supplied ensemble
    shows that B3's unweighted linear score is a strong empirical baseline.
    The implementation still scores the physically projected dose and passes
    the selected set through C1/C2 plus the running inner safety ledger.
    """
    weights = {i: (float(intent_weights[i])
                   if intent_weights is not None and i in intent_weights
                   else (float(intent.pi_class) if use_pi_class else 1.0))
               for i, intent in intents.items()}
    den = max(sum(weights.values()), 1e-9)
    gain = sum(
        weights[i] * sum(
            float(sensitivity(regime, claims[j].param, i))
            * float(projected_dose.get(j, 0.0)) for j in selected)
        for i in intents) / den
    cost = 0.0
    for j in selected:
        lo, hi = claims[j].domain
        cost += abs(float(projected_dose.get(j, 0.0))) / max(hi - lo, 1e-9)
    return gain - max(0.0, float(action_cost)) * cost


def individual_utilities(claim_ids: Iterable[str], **kwargs) -> Dict[str, float]:
    """Convenience values used only for deterministic inner-loop ordering."""
    return {jid: portfolio_wif_utility({jid}, **kwargs) for jid in claim_ids}
