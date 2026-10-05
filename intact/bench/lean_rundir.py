"""Compact run folder for large paired benchmarks.

``intact.runlog.RunDir`` writes a full JSON event per epoch, one JSON record
per inner decision and a pickled checkpoint.  That is ideal for debugging a
single run and far too large for a benchmark with thousands of runs (about
7 MB per 800-epoch run).  ``LeanRunDir`` exposes the same interface that the
experiments use (``log``, ``event``, ``decision``, ``save_checkpoint``,
``load_checkpoint``, ``save_json``, ``close``) but stores only what the
benchmark needs:

* a compact per-epoch trace (gzip JSON lines)::

      {"e": epoch, "S": [selected claims], "W": {claim: executed value},
       "p": [p_hat per intent, sorted intent order]}

  This is enough to decide whether two methods took *identical decisions*
  (the automated "is this ablation knob inert?" check) and to summarise how
  active the risk signal was;
* ``run.log`` containing warnings and errors only.

It never writes checkpoints: benchmark resumption happens at the level of a
complete (scenario, seed, method) result, which is cached with a
fingerprint by ``scripts/run_benchmark.py``.
"""
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class LeanRunDir:
    """Drop-in replacement for ``intact.runlog.RunDir`` (benchmark use)."""

    def __init__(self, path: str | Path, keep_trace: bool = True,
                 file_level: int = logging.WARNING,
                 console_level: int = logging.ERROR,
                 spans: Optional[Dict[str, float]] = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.root = self.path.parent
        self.resumed = False
        self.keep_trace = bool(keep_trace)
        self.trace: List[Dict[str, Any]] = []
        self._pending: Dict[str, float] = {}
        self._intent_order: Optional[List[str]] = None
        # claim id -> parameter domain width, to normalise |v* - v_req|
        self.spans = {str(k): max(float(v), 1e-12)
                      for k, v in (spans or {}).items()}
        self.n_decisions = 0
        self.disp = {"admit": 0, "override_safety": 0, "override_clamp": 0,
                     "reject_safety": 0, "reject_other": 0}
        self.dev_all: List[float] = []
        self.dev_override: List[float] = []
        self.dev_reject: List[float] = []
        self._setup_logging(file_level, console_level)
        self.log = logging.getLogger("intact")

    # ------------------------------------------------------------------
    def _setup_logging(self, file_level: int, console_level: int) -> None:
        logger = logging.getLogger("intact")
        for h in list(logger.handlers):
            try:
                h.close()
            finally:
                logger.removeHandler(h)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(self.path / "run.log")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False

    # ------------------------------------------------------------------
    def decision(self, obj: Dict[str, Any]) -> None:
        """Buffer executed writes of the current epoch.

        Both the v1 and the v2 experiment loops emit every decision of an
        epoch *before* the epoch event, so the buffer is flushed by event().
        A write counts as executed when it was not rejected and it changed
        the parameter value.
        """
        try:
            jid = str(obj.get("jid"))
            outcome = str(obj.get("outcome", ""))
            reason = str(obj.get("reason") or "")
            implicated = obj.get("implicated") or []
            nu_star = float(obj.get("nu_star"))
            nu_old = float(obj.get("nu_old", nu_star))
            nu_req = float(obj.get("nu_req", nu_star))
            span = self.spans.get(jid, 1.0)
            dev = abs(nu_star - nu_req) / span
            self.n_decisions += 1
            self.dev_all.append(dev)
            safety = ("safety" in reason or "range-exhausted" in reason
                      or bool(implicated))
            if outcome == "admit":
                self.disp["admit"] += 1
            elif outcome == "override":
                self.disp["override_safety" if safety
                          else "override_clamp"] += 1
                self.dev_override.append(dev)
            elif outcome == "reject":
                self.disp["reject_safety" if safety else "reject_other"] += 1
                self.dev_reject.append(dev)
            if outcome == "reject" or abs(nu_star - nu_old) <= 1e-12:
                return
            self._pending[jid] = round(nu_star, 9)
        except Exception:
            # A malformed diagnostic record must never break a simulation.
            return

    def event(self, obj: Dict[str, Any]) -> None:
        p_hat = obj.get("p_hat") or {}
        if self._intent_order is None and p_hat:
            self._intent_order = sorted(p_hat)
        g = obj.get("g") or {}
        rec = {
            "e": int(obj.get("epoch", len(self.trace) + 1)),
            "S": sorted(str(x) for x in (obj.get("S") or [])),
            "W": dict(sorted(self._pending.items())),
            "p": [round(float(p_hat.get(i, 0.0)), 4)
                  for i in (self._intent_order or [])],
            # 1 when the intent is fulfilled after the epoch (g >= 0), the
            # same definition MetricBook uses for weighted fulfilment
            "f": [1 if float(g.get(i, 0.0)) >= 0.0 else 0
                  for i in (self._intent_order or [])],
        }
        self.trace.append(rec)
        self._pending = {}

    # ------------------------------------------------------------------
    def save_checkpoint(self, state: Dict[str, Any], tag: str = "main") -> None:
        return None

    def load_checkpoint(self, tag: str = "main") -> Optional[Dict[str, Any]]:
        return None

    def save_json(self, name: str, obj: Any) -> None:
        (self.path / name).write_text(json.dumps(obj, default=float))

    # ------------------------------------------------------------------
    def risk_stats(self, tau: float) -> Dict[str, float]:
        """How active was the risk signal during the evaluated run?"""
        vals = [v for rec in self.trace for v in rec.get("p", [])]
        if not vals:
            return {"risk_mean_p": 0.0, "risk_frac_at_or_above_tau": 0.0}
        n = float(len(vals))
        return {
            "risk_mean_p": float(sum(vals) / n),
            "risk_frac_at_or_above_tau": float(
                sum(1 for v in vals if v >= tau) / n),
        }

    def decision_stats(self, eval_epochs: int) -> Dict[str, float]:
        """Arbitration states and mediation magnitude for one run.

        Dispositions are counted per epoch.  ``override_clamp`` is a value
        change forced by the grid, the trust region or the C2 envelope;
        ``override_safety`` is a value change forced by safety mediation.
        Mediation magnitude is |v* - v_req| normalised by the claim's
        parameter range (0 = executed exactly as requested, 1 = moved by the
        whole range); for a rejection v* is the unchanged status quo.
        """
        e = max(int(eval_epochs), 1)
        mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
        sel = [len(r.get("S", [])) for r in self.trace]
        out = {f"{k}_per_epoch": v / e for k, v in self.disp.items()}
        out.update({
            "decisions_per_epoch": self.n_decisions / e,
            "override_per_epoch": (self.disp["override_safety"]
                                   + self.disp["override_clamp"]) / e,
            "reject_per_epoch": (self.disp["reject_safety"]
                                 + self.disp["reject_other"]) / e,
            "selected_claims_per_epoch": mean(sel),
            "mediation_abs_dev_norm_all": mean(self.dev_all),
            "mediation_abs_dev_norm_overrides": mean(self.dev_override),
            "mediation_abs_dev_norm_rejects": mean(self.dev_reject),
        })
        return out

    def write_trace(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt") as fh:
            fh.write(json.dumps({"intents": self._intent_order or []}) + "\n")
            for rec in self.trace:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        tmp.replace(path)

    def close(self) -> None:
        logger = logging.getLogger("intact")
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            finally:
                logger.removeHandler(h)


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    """Read a trace written by :meth:`LeanRunDir.write_trace`."""
    out: List[Dict[str, Any]] = []
    with gzip.open(path, "rt") as fh:
        first = True
        for line in fh:
            if first:
                first = False
                continue
            out.append(json.loads(line))
    return out


def trace_divergence(a: List[Dict[str, Any]],
                     b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare two decision traces of the same scenario and seed.

    Returns the fraction of epochs whose selected set differs, the fraction
    whose executed writes differ, and the first epoch at which either
    differs (``None`` when the two runs took bit-identical decisions).
    """
    n = min(len(a), len(b))
    if n == 0:
        return {"epochs_compared": 0, "frac_epochs_selected_set_differs": float("nan"),
                "frac_epochs_writes_differ": float("nan"),
                "first_divergence_epoch": None, "identical": False}
    s_diff = w_diff = 0
    first = None
    for k in range(n):
        ds = a[k]["S"] != b[k]["S"]
        wa, wb = a[k]["W"], b[k]["W"]
        dw = (set(wa) != set(wb)
              or any(abs(float(wa[j]) - float(wb[j])) > 1e-9 for j in wa))
        s_diff += int(ds)
        w_diff += int(dw)
        if first is None and (ds or dw):
            first = int(a[k]["e"])
    identical = first is None and len(a) == len(b)
    return {"epochs_compared": n,
            "frac_epochs_selected_set_differs": s_diff / n,
            "frac_epochs_writes_differ": w_diff / n,
            "first_divergence_epoch": first,
            "identical": bool(identical)}
