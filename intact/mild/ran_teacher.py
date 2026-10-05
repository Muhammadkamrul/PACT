"""Nonlinear, local-feature teacher for the RAN adaptation of MILD.

The published MILD teacher is logistic regression.  That is retained as an
option, but RAN failure boundaries are strongly nonlinear in load, controls
and margin.  This teacher uses one small histogram-gradient model per intent
and, crucially, prevents a 14-intent head from fitting hundreds of unrelated
tenant columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


class RANNonlinearTeacher:
    """Per-intent nonlinear teacher with an auditable feature mask."""

    def __init__(self, intent_ids: List[str], feature_names: List[str],
                 intent_tenants: Dict[str, str], temperature: float = 2.0,
                 equivalence_groups: List[List[str]] | None = None,
                 algorithm: str = "extra_trees"):
        self.intents = list(intent_ids)
        self.feature_names = list(feature_names)
        self.intent_tenants = dict(intent_tenants)
        self.temperature = float(max(temperature, 1e-6))
        self.masks = [self._mask(i) for i in self.intents]
        self.equivalence_groups = equivalence_groups or [[i] for i in self.intents]
        self.algorithm = str(algorithm)
        self.models = []
        self.constants = []

    def _mask(self, iid: str) -> np.ndarray:
        tenant = self.intent_tenants[iid]
        local_prefixes = (
            f"g_{iid}", f"rho_{iid}",
            f"tput_{tenant}", f"delay_{tenant}", f"buf_{tenant}",
            f"load_{tenant}", f"load_forecast_{tenant}")
        keep = []
        for q, name in enumerate(self.feature_names):
            if (name.startswith(local_prefixes)
                    or name.startswith("cell_")
                    or name.startswith("ctl_")):
                keep.append(q)
        if not keep:
            keep = list(range(len(self.feature_names)))
        return np.asarray(keep, dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 160,
            learning_rate: float = 0.05, max_leaf_nodes: int = 15,
            min_samples_leaf: int = 12) -> "RANNonlinearTeacher":
        from sklearn.ensemble import (ExtraTreesClassifier,
                                      HistGradientBoostingClassifier)

        self.models = [None] * len(self.intents)
        self.constants = [None] * len(self.intents)
        index = {iid: k for k, iid in enumerate(self.intents)}
        for group_no, group in enumerate(self.equivalence_groups):
            members = [index[i] for i in group]
            k = members[0]
            mask = self.masks[k]
            target = np.asarray(y[:, k], dtype=int)
            for q in members[1:]:
                if not np.array_equal(target, np.asarray(y[:, q], dtype=int)):
                    raise ValueError(
                        "equivalent intent heads do not have identical labels: "
                        f"{group}")
            rate = float(target.mean())
            if target.min() == target.max():
                for q in members:
                    self.models[q] = None
                    self.constants[q] = min(max(rate, 1e-5), 1.0 - 1e-5)
                    self.masks[q] = mask
                continue
            n_pos = max(int(target.sum()), 1)
            n_neg = max(len(target) - n_pos, 1)
            sample_weight = np.where(target > 0,
                                     len(target) / (2.0 * n_pos),
                                     len(target) / (2.0 * n_neg))
            if self.algorithm == "extra_trees":
                model = ExtraTreesClassifier(
                    n_estimators=max(100, int(max_iter)),
                    min_samples_leaf=max(2, int(min_samples_leaf) // 2),
                    max_features="sqrt", class_weight="balanced_subsample",
                    # Runtime predicts one epoch at a time.  Parallel workers
                    # add overhead and can emit one warning per tree call.
                    n_jobs=1, random_state=1000 + group_no)
                sample_weight = None
            elif self.algorithm == "hist_gradient_boosting":
                model = HistGradientBoostingClassifier(
                    learning_rate=float(learning_rate), max_iter=int(max_iter),
                    max_leaf_nodes=int(max_leaf_nodes),
                    min_samples_leaf=int(min_samples_leaf),
                    l2_regularization=1.0, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=12,
                    random_state=1000 + group_no)
            else:
                raise ValueError("RAN teacher algorithm must be extra_trees "
                                 "or hist_gradient_boosting")
            model.fit(X[:, mask], target, sample_weight=sample_weight)
            for q in members:
                self.models[q] = model
                self.constants[q] = None
                self.masks[q] = mask
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((len(X), len(self.intents)), dtype=float)
        index = {iid: k for k, iid in enumerate(self.intents)}
        for group in self.equivalence_groups:
            cols = [index[i] for i in group]
            k = cols[0]
            model, constant, mask = (
                self.models[k], self.constants[k], self.masks[k])
            values = (np.full(len(X), float(constant)) if model is None else
                      model.predict_proba(X[:, mask])[:, 1])
            out[:, cols] = values[:, None]
        return out

    def distribution(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict(X), 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p)) / self.temperature
        return _softmax(logits)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "RANNonlinearTeacher":
        obj = joblib.load(path)
        if not isinstance(obj, RANNonlinearTeacher):
            raise TypeError("invalid RAN nonlinear teacher artifact")
        for model in obj.models:
            if model is not None and hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except ValueError:
                    pass
        return obj
