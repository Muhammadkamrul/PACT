"""
intact/v2/state.py
==================
Operating state, split into TWO ROLES that v1 conflated.

v1 had one flat 2-vector context (pre-treatment PRB utilisation and retx
fraction) and used prb_util_pct alone to define the regime.  Two numbers
cannot represent an operating point, which is why beta remained
state-dependent even after CUPED adjustment.

THE SPLIT
---------
REGIME LABEL r   selects WHICH coefficient table to use.
    Coarse and discrete (2-3 bands), so per-(regime, pair) cells stay
    populated.  Handles the NONLINEAR, sign-flipping state dependence as a
    lookup, with no functional-form assumption.

CUPED CONTEXT c  linearly adjusts WITHIN a regime.
    Continuous, one parameter per variable regardless of range.  Handles
    residual drift cheaply.

Load legitimately appears in both -- coarse in the regime, fine in c.

THREE ADMISSION RULES FOR A CONTEXT VARIABLE
--------------------------------------------
A covariate that violates any of these does harm, not good:

 1. PRE-TREATMENT.  Measured before this epoch's writes land, or
    conditioning on it creates post-treatment bias.
 2. EXOGENOUS TO THE SCORED CLAIM.  Never condition on a parameter the
    claim itself writes.  txpower, tilt and other control values belong in
    the REGIME LABEL, never in c: putting them in c absorbs the claim's own
    effect into theta.c and biases beta toward zero.
 3. VARIANCE REDUCING.  In small per-regime cells a covariate that does not
    correlate with dg makes estimates worse, not better.

BUG FIXED HERE
--------------
The v1 alpha validator read kpm.get("CELL", {}).get("prb_util_pct") and
.get("retx"), while the RAN emits kpm["_cell"]["prb_util_pct"] and
kpm["_cell"]["retx_prb"].  Every read fell through to its 0.0 default, so
BOTH context columns were silently constant zero and the CUPED adjustment
never happened.  `cell_context` is now the single place those keys are
read, so the experiment and every validator share one implementation and
one feature scaling.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

REGIME_NAMES = ("low", "mid", "high")


# ----------------------------------------------------------------------
def cell_context(kpm: Dict, n_prb: int) -> Tuple[float, float]:
    """(utilisation fraction, retx fraction) from a PRE-TREATMENT KPM dict.

    ONE implementation, used by the experiment, every probe and every
    validator, so feature scaling can never drift between them again.
    """
    cell = kpm.get("_cell", {}) or kpm.get("CELL", {}) or {}
    util = float(cell.get("prb_util_pct", 0.0)) / 100.0
    retx = float(cell.get("retx_prb", cell.get("retx", 0.0))) / max(n_prb, 1)
    return util, retx


def regime_label(util_pct: float, offered_load: float,
                 util_edges: Sequence[float] = (70.0, 88.0),
                 load_edges: Sequence[float] = (0.9, 1.25)) -> str:
    """Coarse regime from BOTH utilisation and offered load.

    Offered load is set by the traffic model and moved by no xApp, so it is
    the cleanest available regime signal.  Utilisation is a consequence of
    load AND of control writes, so alone it partially reflects the very
    actions we are trying to attribute -- which is why v1's utilisation-only
    regime was not enough.

    The two are combined by taking the more severe band, so a lightly
    utilised cell under heavy offered load (a starved cell) is correctly
    labelled "high" rather than "low".
    """
    u = 0 if util_pct < util_edges[0] else (1 if util_pct < util_edges[1] else 2)
    l = 0 if offered_load < load_edges[0] else (1 if offered_load < load_edges[1] else 2)
    return REGIME_NAMES[max(u, l)]


# ----------------------------------------------------------------------
class StateBuilder:
    """Builds (regime label, CUPED context vector) from a pre-treatment KPM.

    Context columns, all pre-treatment and all exogenous to any single
    claim:
        0  cell utilisation fraction
        1  cell retx fraction
        2  mean offered load across slices   (traffic model, never written)
        3  mean normalised buffer occupancy  (queue state before writes)

    NOTE what is deliberately ABSENT: txpower, tilt, quota, prbcap, mcs,
    schedw, schedpol, cio.  Those are exactly the parameters claims write.
    They inform the REGIME LABEL only.
    """

    COLUMNS = ("util", "retx", "offered_load", "buffer_occ")

    def __init__(self, cfg: Dict):
        self.n_prb = int(cfg["ran"]["n_prb"])
        v2 = cfg.get("v2", {}) or {}
        self.n_context = int(v2.get("n_context", 4))
        self.util_edges = tuple(v2.get("regime_util_edges", (70.0, 88.0)))
        self.load_edges = tuple(v2.get("regime_load_edges", (0.9, 1.25)))
        self.max_queue_kb = float(cfg["ran"].get("max_queue_kb", 6.0))
        self._base_load = None

    def _offered_load(self, ran) -> float:
        """Mean per-UE offered load, normalised by the scenario baseline."""
        # Dynamic profiles keep the immutable per-UE baseline in
        # ``st['load_mbps']`` and apply a deterministic multiplier at read
        # time.  Reading the array directly therefore returned 1.0 forever
        # and silently collapsed the advertised two-signal regime label back
        # to utilisation-only.  Ask the RAN for the current exogenous input
        # when that interface exists.
        if (hasattr(ran, "current_offered_load_mbps")
                and hasattr(ran, "baseline_offered_load_mbps")):
            base = max(float(ran.baseline_offered_load_mbps()), 1e-9)
            return float(ran.current_offered_load_mbps()) / base
        try:
            loads = [st["load_mbps"] for st in ran.ue.values()]
        except Exception:
            return 1.0
        if not loads:
            return 1.0
        cur = float(np.mean(loads))
        if self._base_load is None:
            self._base_load = max(cur, 1e-9)
        return cur / self._base_load

    def _buffer_occ(self, kpm: Dict, tenants: Sequence[str]) -> float:
        vals = []
        for t in tenants:
            d = kpm.get(t)
            if isinstance(d, dict) and "buffer_kb" in d:
                vals.append(float(d["buffer_kb"]) / max(self.max_queue_kb, 1e-9))
        return float(np.mean(vals)) if vals else 0.0

    def build(self, kpm: Dict, ran, tenants: Sequence[str]
              ) -> Tuple[str, np.ndarray]:
        util, retx = cell_context(kpm, self.n_prb)
        load = self._offered_load(ran)
        buf = self._buffer_occ(kpm, tenants)
        c = np.array([util, retx, load, buf], dtype=float)[:self.n_context]
        r = regime_label(util * 100.0, load, self.util_edges, self.load_edges)
        return r, c

    def column_names(self) -> List[str]:
        return list(self.COLUMNS[:self.n_context])
