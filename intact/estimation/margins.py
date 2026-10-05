"""
intact/estimation/margins.py   --  INTACT v11 §4
================================================
Turn raw, incomparable KPIs into a DIMENSIONLESS MARGIN.

You cannot add "9.2 Mb/s" to "12 ms".  Every intent is first converted to

        g_i  =  d_i * (measured - promised) / |promised|

where d_i = +1 if higher is better, -1 if lower is better.

    g = 0     exactly on target
    g = +0.15 fifteen percent better than promised
    g = -0.05 five percent short

THIS IS THE WHOLE TRICK.  After this step, "which intent is in more
trouble" is a well-posed question, and everything downstream -- weights,
claim values, the safety floor, the projection -- is arithmetic on
comparable numbers.
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Deque
from ..types import Intent, Direction


def margin(intent: Intent, kpm: Dict[str, Dict[str, float]]) -> float:
    """Compute g_i from the current KPM report."""
    src = kpm["_cell"] if intent.tenant == "_cell" or intent.kpi in kpm["_cell"] else kpm[intent.tenant]
    measured = src[intent.kpi]
    d = 1.0 if intent.direction == Direction.HIGHER_BETTER else -1.0
    g = d * (measured - intent.target) / abs(intent.target)
    c = getattr(intent, "clip", float("inf"))
    return max(-c, min(c, g))       # v11 s4, clipped form


class MarginTracker:
    """
    Keeps a rolling window of margins per intent so we can compute
      - g_i(t)      the instantaneous margin
      - rho_i(t)    FULFILMENT: fraction of the last W slots with g_i >= 0
                    (v11 §3.3).  This is what the tenant floor C3 is on.
      - the trajectory that the risk predictor reads (v11 §10.2).
    """

    def __init__(self, intents: Dict[str, Intent], window: int):
        self.intents = intents
        self.window = window
        self.hist: Dict[str, Deque[float]] = {i: deque(maxlen=window) for i in intents}

    def update(self, kpm) -> Dict[str, float]:
        g = {}
        for iid, it in self.intents.items():
            gi = margin(it, kpm)
            self.hist[iid].append(gi)
            g[iid] = gi
        return g

    def fulfilment(self, iid: str) -> float:
        """rho_i: fraction of the window in which the intent was satisfied."""
        h = self.hist[iid]
        return sum(1 for x in h if x >= 0) / max(len(h), 1)

    def trend(self, iid: str, k: int = 8) -> float:
        """Mean per-slot change over the last k samples.  Negative = eroding.
        This is what lets the risk predictor see the RATCHET (v11 §10.2)."""
        h = list(self.hist[iid])
        if len(h) < 3:
            return 0.0
        k = min(k, len(h) - 1)
        return (h[-1] - h[-1 - k]) / k
