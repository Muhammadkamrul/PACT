"""
intact/v2/ran_v2.py
===================
INTACTv2 RAN extensions.  Two mechanisms the v1 simulator did not have.

WHY THIS FILE EXISTS
--------------------
v1 measured gamma == 0 on every well-identified coupled pair.  That was not
cancellation and not an estimator defect: the v1 AnalyticRAN contains no
term in which two control parameters MULTIPLY.  quota affects one slice's
allocation, txpower affects the RF environment, and the two compose almost
additively.  There was no interaction to find, so H(j,k) penalised a
phantom.

This module adds the two things a conflict-arbitration paper actually needs
to demonstrate:

 1. DOSE-BASED PAIR INTERACTION  (the "indirect conflict" of O-RAN WG3)
    A declared cross-parameter term whose magnitude depends on the PRODUCT
    of the two applied doses, not on mere co-eligibility.  This is what
    makes gamma identifiable at all.

 2. RCP -> RCP DEPENDENCY EDGES  (the "implicit conflict" of O-RAN WG3,
    structural variant)
    One control parameter structurally overrides another.  v1 already had
    exactly one such edge by accident -- prbcap clips quota via
    min(share*quota, cap) -- but never declared or detected it.  Here the
    graph is explicit, so C1' (transitive authority) can be checked.

NOISE FLOOR REQUIREMENT
-----------------------
A planted interaction is only worth measuring if it clears the measurement
noise.  Per-epoch margin noise in this simulator is sigma ~= 0.015, so we
require

    |gamma * dnu_j * dnu_k|  >=  3 * sigma  =  0.045

at typical write sizes.  `check_noise_floor` verifies this BEFORE any
estimation runs, and `make_interaction_spec` sizes gamma to satisfy it.
Without that check a "gamma = 0" result is uninterpretable: you cannot tell
a good estimator on a flat world from a bad estimator on a rich one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Per-epoch margin noise measured on the v1 simulator.  Used to size planted
# interactions so they are detectable in principle.
SIGMA_MARGIN = 0.015
NOISE_FLOOR_MULTIPLE = 3.0


@dataclass
class InteractionTerm:
    """One declared dose-product interaction.

    effect on `kpi` of tenant `tenant` is

        gamma * (nu_a - ref_a) * (nu_b - ref_b) * regime_gain[r]

    `regime_gain` is what makes the interaction STATE DEPENDENT: the same
    two knobs interact strongly under congestion and weakly when the cell is
    empty.  A single scalar gamma cannot represent that, which is precisely
    the finding v1 produced and v2 must be able to reproduce deliberately.
    """
    param_a: str
    param_b: str
    tenant: str
    kpi: str                      # throughput_mbps | delay_ms | buffer_kb | delivery_pct
    gamma: float
    ref_a: float
    ref_b: float
    regime_gain: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.3, "mid": 1.0, "high": 1.8})
    note: str = ""

    def value(self, controls: Dict[str, float], regime: str) -> float:
        da = float(controls.get(self.param_a, self.ref_a)) - self.ref_a
        db = float(controls.get(self.param_b, self.ref_b)) - self.ref_b
        return self.gamma * da * db * self.regime_gain.get(regime, 1.0)

    def typical_magnitude(self, dose_a: float, dose_b: float,
                          regime: str = "mid") -> float:
        return abs(self.gamma * dose_a * dose_b * self.regime_gain.get(regime, 1.0))


@dataclass
class DependencyEdge:
    """A declared RCP -> RCP edge: `source` structurally constrains `target`.

    kind:
      "clip"  target's effective value is min(target, source)   [prbcap->quota]
      "scale" target's effective value is target * (source/ref)
      "offset" target's effective value is target + k*(source-ref)

    Detection of implicit conflict does NOT require estimation.  If claim j
    writes `source` and claim k writes `target`, the conflict exists by
    construction and the arbiter can test it exactly like C1.
    """
    source: str
    target: str
    kind: str = "clip"
    ref: float = 0.0
    k: float = 1.0
    note: str = ""

    def apply(self, controls: Dict[str, float]) -> Tuple[float, bool]:
        """Return (effective target value, binding?)."""
        tgt = float(controls.get(self.target, 0.0))
        src = float(controls.get(self.source, 0.0))
        if self.kind == "clip":
            eff = min(tgt, src)
        elif self.kind == "scale":
            eff = tgt * (src / self.ref) if self.ref else tgt
        elif self.kind == "offset":
            eff = tgt + self.k * (src - self.ref)
        else:
            eff = tgt
        return eff, abs(eff - tgt) > 1e-9


# ----------------------------------------------------------------------
def load_interactions(cfg: Dict) -> List[InteractionTerm]:
    out = []
    for d in cfg.get("interactions", []) or []:
        out.append(InteractionTerm(
            param_a=d["param_a"], param_b=d["param_b"], tenant=d["tenant"],
            kpi=d["kpi"], gamma=float(d["gamma"]),
            ref_a=float(d["ref_a"]), ref_b=float(d["ref_b"]),
            regime_gain=d.get("regime_gain",
                              {"low": 0.3, "mid": 1.0, "high": 1.8}),
            note=d.get("note", "")))
    return out


def load_dependencies(cfg: Dict) -> List[DependencyEdge]:
    out = []
    for d in cfg.get("rcp_dependencies", []) or []:
        out.append(DependencyEdge(
            source=d["source"], target=d["target"],
            kind=d.get("kind", "clip"), ref=float(d.get("ref", 0.0)),
            k=float(d.get("k", 1.0)), note=d.get("note", "")))
    return out


def check_noise_floor(interactions: Sequence[InteractionTerm],
                      doses: Dict[str, float],
                      sigma: float = SIGMA_MARGIN,
                      multiple: float = NOISE_FLOOR_MULTIPLE) -> List[Dict]:
    """Verify every planted interaction clears `multiple` * sigma.

    Run this BEFORE estimation.  A pair that fails is unmeasurable by
    construction and reporting "gamma not recovered" for it would be
    meaningless.
    """
    need = multiple * sigma
    rows = []
    for it in interactions:
        da = doses.get(it.param_a, 1.0)
        db = doses.get(it.param_b, 1.0)
        mag = it.typical_magnitude(da, db, "mid")
        rows.append({
            "param_a": it.param_a, "param_b": it.param_b,
            "tenant": it.tenant, "kpi": it.kpi, "gamma": it.gamma,
            "dose_a": da, "dose_b": db,
            "typical_effect": mag, "required": need,
            "detectable": bool(mag >= need * (1.0 - 1e-9)),
            "margin_multiple": mag / max(sigma, 1e-12)})
    return rows


def size_gamma_for_detectability(dose_a: float, dose_b: float,
                                 regime_gain: float = 1.0,
                                 sigma: float = SIGMA_MARGIN,
                                 multiple: float = NOISE_FLOOR_MULTIPLE
                                 ) -> float:
    """Smallest |gamma| whose typical effect clears the noise floor."""
    denom = max(abs(dose_a * dose_b * regime_gain), 1e-12)
    return (multiple * sigma) / denom


# ----------------------------------------------------------------------
class InteractionMixin:
    """Mixed into AnalyticRANv2.  Applies interactions and dependency edges.

    Kept as a mixin so the v1 physics is untouched and any v1 result remains
    reproducible by simply not declaring interactions.
    """

    def _init_v2(self, cfg: Dict) -> None:
        self.interactions: List[InteractionTerm] = load_interactions(cfg)
        self.dependencies: List[DependencyEdge] = load_dependencies(cfg)
        self._binding_edges: List[Tuple[str, str]] = []

    # -- regime label used by the interaction gain ---------------------
    def _regime_label(self, util_pct: float) -> str:
        edges = self.cfg.get("regime_edges", [70.0, 88.0])
        if util_pct < edges[0]:
            return "low"
        if util_pct < edges[1]:
            return "mid"
        return "high"

    def effective_controls(self) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
        """Apply RCP->RCP edges; report which edges are BINDING.

        A binding edge means one claim's write is being structurally
        overridden by another's -- implicit conflict, observable exactly.
        """
        eff = dict(self._controls)
        binding = []
        for e in self.dependencies:
            if e.source in eff and e.target in eff:
                v, is_binding = e.apply(eff)
                eff[e.target] = v
                if is_binding:
                    binding.append((e.source, e.target))
        self._binding_edges = binding
        return eff, binding

    def interaction_delta(self, tenant: str, kpi: str, util_pct: float
                          ) -> float:
        """Total dose-product interaction on one tenant's KPI this epoch."""
        if not self.interactions:
            return 0.0
        r = self._regime_label(util_pct)
        tot = 0.0
        for it in self.interactions:
            if it.tenant == tenant and it.kpi == kpi:
                tot += it.value(self._controls, r)
        return tot

    def binding_edge_count(self) -> int:
        return len(self._binding_edges)
