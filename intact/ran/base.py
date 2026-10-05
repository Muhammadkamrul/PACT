"""
intact/ran/base.py
==================
The RAN backend interface.  *** THIS IS THE SWAP POINT. ***

The whole point of this abstraction: the INTACT mediation logic (the two
loops, the estimators, the deficit counters) NEVER touches a RAN directly.
It only ever calls the four methods below.  That means when you move from
the analytical simulator to srsRAN/OAI you write ONE new file implementing
this interface -- and the mediation code, which is the actual contribution
of the paper, does not change by a single line.

That is also the sentence you put in the paper:
    "The mediation plane is backend-agnostic; we evaluate it against a
     calibrated analytical RAN model and against srsRAN via E2."
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class RANBackend(ABC):
    """Anything that can accept control writes and return KPMs."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Re-initialise to a deterministic starting state."""

    @abstractmethod
    def current_controls(self) -> Dict[str, float]:
        """The values currently APPLIED to every knob.  This is nu_old."""

    @abstractmethod
    def apply(self, param: str, value: float) -> None:
        """Commit one control value.  Called only after the inner loop has
        decided the disposition -- so this is nu_star, never nu_req."""

    @abstractmethod
    def step(self, n_slots: int) -> Dict[str, Dict[str, float]]:
        """
        Advance the RAN by n_slots and return KPMs.

        Returns
        -------
        {tenant_id: {kpi_name: value, ...}, ..., "_cell": {...}}
        The "_cell" entry carries cell-wide measurements (e.g. PRB
        utilisation) that host intents are written against.
        """

    # ---- optional hooks, with safe defaults ---------------------------
    def headroom(self, tenant: str, resource: str) -> float:
        """Resource currently available to this tenant, for F_feasible
        (v11 §10.3).  Default: unbounded, i.e. feasibility never binds."""
        return float("inf")

    def demand(self, param: str, value: float) -> float:
        """
        d_r(nu, c): resource demand of a candidate value.  v11 §8.3.2.

        For the DIRECT allocative controls used here, d_r(nu, c) = nu after
        unit normalisation, so this default is exact.  Context-dependent
        mappings would override it -- and would then need a demand model
        supplied by the controlled RAN function (v11 §8.3.2, out of scope).
        """
        return value
