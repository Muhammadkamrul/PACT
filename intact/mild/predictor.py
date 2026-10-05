"""
intact/mild/predictor.py
========================
The bridge from MILD into INTACT.

*** THIS IS THE ONLY FILE INTACT TOUCHES. ***
It exposes the SAME `predict(tracker) -> {iid: p_hat}` interface as the
analytical stand-in in `estimation/risk.py`, so switching between them is a
one-line config change and nothing else in the framework changes.

WHERE p_hat IS USED (both loops -- v11)
    OUTER LOOP  §5   w_i = pi x omega x u(p_hat_i) x (1 + theta_L Lambda_n)
                     -> which claims are worth scheduling this epoch
    INNER LOOP  §10.1  the relevance-and-risk gate:
                     mediate only if  p_hat_i >= tau_risk  AND this write
                     measurably pushes intent i further in
So a mis-calibrated p_hat does not merely mispredict: it changes WHO GETS
CONTROL AUTHORITY.  That is why `neg_margin` in the loss matters so much --
a predictor idling at p=0.3 inflates every weight, permanently.

DROP-IN REPLACEMENT
Any object with `.predict(X) -> (N, K)` works here, including your existing
Keras model.  Set `mild.backend: keras` and give the path.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import hashlib
import json
import copy
import numpy as np

from .features import raw_frame, engineer, Standardiser
from .model import MildMoE
from .teacher import OvRLogisticTeacher
from .ran_teacher import RANNonlinearTeacher


def mild_scenario_signature(cfg) -> str:
    """Hash the topology/contract schema on which a MILD model is valid.

    Random ensemble scenarios reuse identifiers such as ``i1`` while changing
    their owner, KPI, target and control topology.  Matching names alone is
    therefore not evidence of transportability.  Seeds and run lengths are
    intentionally excluded so paired evaluation seeds may share one model.
    """
    ran = copy.deepcopy(cfg.get("ran", {}))
    # Paired evaluation seeds must share one topology-specific model.  Only
    # the stochastic seed is non-semantic; load, coupling, envelopes, control
    # domains and initial state remain part of the signature.
    ran.pop("seed", None)
    payload = {
        "tenants": cfg.get("tenants", []),
        "intents": cfg.get("intents", []),
        "xapps": cfg.get("xapps", []),
        "claims": cfg.get("claims", []),
        "ran": ran,
        "sweep_domains": cfg.get("sweep_domains", {}),
        "margins": cfg.get("margins", {}),
        "risk": cfg.get("risk", {}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MildRiskPredictor:
    """Interface-compatible with estimation.risk.RiskPredictor."""

    def __init__(self, model_dir: str, cfg, intents, tenants, claims, log):
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text())
        self.feats: List[str] = meta["features"]
        self.iids: List[str] = meta["intents"]
        self.horizon = meta["horizon"]
        self.horizon_unit = meta.get("horizon_unit", "legacy_slots")
        self.sample_slots = int(meta.get("sample_slots", 1))
        self.controls = meta["controls"]
        self.intents, self.tenants = intents, tenants
        self.log = log
        self.source = "mild_ran"

        trained_signature = meta.get("scenario_signature")
        current_signature = mild_scenario_signature(cfg)
        if trained_signature is None:
            raise ValueError(
                "MILD model metadata predates scenario-signature validation; "
                "retrain it with scripts/train_mild.py before enabling it")
        if trained_signature != current_signature:
            raise ValueError(
                "MILD model/scenario mismatch: the trained tenant, intent, "
                "xApp, claim, RAN, margin, risk or control schema differs "
                "from this scenario. Retrain MILD for this exact scenario; "
                "silent cross-topology reuse is not valid.")
        if set(self.iids) != set(intents):
            raise ValueError(
                f"MILD intent mismatch: model={sorted(self.iids)}, "
                f"scenario={sorted(intents)}")
        runtime_sample_slots = (
            int(cfg["ran"].get("pre_slots", cfg["ran"]["slots_per_epoch"]))
            + int(cfg["ran"].get("post_slots", cfg["ran"]["slots_per_epoch"])))
        if (self.horizon_unit != "scheduler_epochs"
                or self.sample_slots != runtime_sample_slots):
            raise ValueError(
                "MILD cadence mismatch: model metadata says "
                f"unit={self.horizon_unit}, sample_slots={self.sample_slots}, "
                f"but runtime observes once per {runtime_sample_slots} slots. "
                "Retrain with the current train_mild.py; using slot-trained "
                "rolling features at epoch inference cadence is invalid.")

        # ---- evidence gate ------------------------------------------
        # A predictor with poor recall does not merely mispredict: p_hat
        # decides WHO GETS CONTROL AUTHORITY.  Shipping an unvalidated one
        # silently is worse than falling back to the analytical stand-in,
        # so refuse unless the config explicitly opts in.
        if not meta.get("usable", True):
            if not cfg.get("mild", {}).get("allow_unusable", False):
                raise ValueError(
                    f"MILD model at {d} failed its evidence gate "
                    f"(gates={meta.get('evidence_gates')}, safe recall="
                    f"{meta.get('macro_recall_currently_safe')}, safe false "
                    f"alarm={meta.get('macro_false_alarm_currently_safe')}, "
                    f"safe PR lift="
                    f"{meta.get('macro_pr_auc_lift_currently_safe')}). Retrain "
                    f"with more slots, or set mild.allow_unusable: true to "
                    f"use it deliberately.")
            log.warning("MILD model failed its evidence gate but "
                        "mild.allow_unusable is set; proceeding.")

        # ---- threshold consistency ----------------------------------
        # Operating points were selected so that tau_risk means the same
        # thing for every intent.  If the runtime tau differs from the one
        # calibration targeted, that alignment is void.
        trained_tau = meta.get("tau_risk")
        runtime_tau = float(cfg["inner"]["tau_risk"])
        if trained_tau is not None and abs(float(trained_tau) - runtime_tau) > 1e-9:
            raise ValueError(
                f"MILD tau_risk mismatch: calibrated for {trained_tau}, "
                f"runtime inner.tau_risk is {runtime_tau}. Recalibrate or "
                f"align the config; a shifted threshold invalidates every "
                f"reported precision/recall figure.")

        st = np.load(d / "scaler.npz")
        self.std = Standardiser(); self.std.mu, self.std.sd = st["mu"], st["sd"]
        self.model = MildMoE.load(str(d / "model.npz"), len(self.feats))
        self.teacher = None
        if bool(meta.get("teacher_augmented", False)):
            teacher_type = str(meta.get("teacher_type", "logistic"))
            teacher_path = d / ("teacher.joblib"
                                if teacher_type == "ran_nonlinear"
                                else "teacher.npz")
            if not teacher_path.exists():
                raise FileNotFoundError(
                    f"teacher-augmented MILD metadata requires {teacher_path}")
            self.teacher = (RANNonlinearTeacher.load(teacher_path)
                            if teacher_type == "ran_nonlinear"
                            else OvRLogisticTeacher.load(teacher_path))
            if self.teacher.intents != self.iids:
                raise ValueError("MILD teacher intent order does not match model")
        self.student_blend = float(meta.get("student_blend", 1.0))
        self.ewma_span = int(meta.get("ewma_span", 1))
        self.equivalence_groups = meta.get(
            "intent_equivalence_groups", [[i] for i in self.iids])
        self._ewma = None

        # ---- calibration + operating-point alignment ------------------
        cal_p = d / "calibration.json"
        self.cal = None
        if cal_p.exists():
            from .calibrate import IntentCalibrator
            self.cal = IntentCalibrator.from_dict(json.loads(cal_p.read_text()))
            log.info("MILD calibration loaded (per-intent Platt + operating "
                     "point aligned to tau=%.2f)", self.cal.tau)
        else:
            log.warning("no calibration.json beside the MILD model; p_hat is "
                        "an uncalibrated network score and u(p_hat) will not "
                        "read as a probability")

        # rolling buffer of raw records, long enough for the widest window
        self.buf: List[Dict] = []
        self.buf_len = 40
        log.info("MILD loaded: %d features, %d intents, horizon=%d epochs",
                 len(self.feats), len(self.iids), self.horizon)

    # ------------------------------------------------------------------
    def observe(self, t: int, g: Dict[str, float], rho: Dict[str, float],
                kpm: Dict, controls: Dict) -> None:
        """Feed one timestep.  Called by the experiment every epoch."""
        self.buf.append({"t": t, "g": dict(g), "rho": dict(rho),
                         "kpm": {k: dict(v) for k, v in kpm.items()},
                         "controls": dict(controls)})
        if len(self.buf) > self.buf_len:
            self.buf.pop(0)

    # ------------------------------------------------------------------
    def predict(self, tracker=None) -> Dict[str, float]:
        """
        Returns {intent_id: p_hat}.  `tracker` is accepted (and ignored) so
        this is a drop-in for the analytical predictor's signature.
        """
        if len(self.buf) < 3:
            return {i: 0.0 for i in self.iids}
        df = raw_frame(self.buf, self.intents, self.tenants, self.controls)
        X, feats = engineer(df)
        # align columns to the training feature order; anything missing is 0
        idx = {f: k for k, f in enumerate(feats)}
        Xa = np.zeros((X.shape[0], len(self.feats)), dtype="float32")
        for k, f in enumerate(self.feats):
            if f in idx:
                Xa[:, k] = X[:, idx[f]]
        Xa = self.std.transform(Xa)
        latest = Xa[-1:]
        if self.teacher is not None:
            teacher_p = self.teacher.predict(latest)
            teacher_dist = self.teacher.distribution(latest)
            student_p = self.model.predict(latest, teacher_dist)
            p = (self.student_blend * student_p
                 + (1.0 - self.student_blend) * teacher_p)
        else:
            p = self.model.predict(latest)
        index = {iid: k for k, iid in enumerate(self.iids)}
        for group in self.equivalence_groups:
            cols = [index[i] for i in group]
            if len(cols) > 1:
                p[:, cols] = p[:, cols].mean(axis=1, keepdims=True)
        if self.ewma_span > 1:
            alpha = 2.0 / (self.ewma_span + 1.0)
            self._ewma = (p.copy() if self._ewma is None else
                          alpha * p + (1.0 - alpha) * self._ewma)
            p = self._ewma
        if self.cal is not None:
            p = self.cal.transform(p)
        return {i: float(p[0, k]) for k, i in enumerate(self.iids)}
