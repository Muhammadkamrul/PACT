"""One-vs-rest logistic teacher used by MILD-RAN.

The published MILD architecture is teacher augmented.  Earlier INTACT
ports removed that teacher, which made the implementation a plain gated
multi-head MLP rather than the method described in the MILD paper.  This
module restores the teacher without serialising an opaque pickle: fitted
coefficients are stored in a small, auditable NumPy archive.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


class OvRLogisticTeacher:
    """Balanced one-vs-rest logistic regression for every intent."""

    def __init__(self, intent_ids: List[str], temperature: float = 2.0):
        self.intents = list(intent_ids)
        self.temperature = float(max(temperature, 1e-6))
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, c: float = 1.0,
            max_iter: int = 600) -> "OvRLogisticTeacher":
        """Fit the same balanced OvR teacher used conceptually by MILD.

        scikit-learn is used only during training.  Runtime inference is the
        explicit matrix multiplication in :meth:`predict`, so deployment
        needs only NumPy and the saved coefficients.
        """
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features, n_intents = X.shape[1], y.shape[1]
        self.W = np.zeros((n_intents, n_features), dtype="float32")
        self.b = np.zeros(n_intents, dtype="float32")
        for k in range(n_intents):
            target = y[:, k].astype(int)
            rate = float(target.mean())
            if target.min() == target.max():
                # A constant head has no identifiable slopes.  Preserve its
                # smoothed base rate without inventing feature dependence.
                rate = min(max(rate, 1e-5), 1.0 - 1e-5)
                self.b[k] = np.log(rate / (1.0 - rate))
                continue
            model = LogisticRegression(
                C=float(c), class_weight="balanced", solver="lbfgs",
                max_iter=int(max_iter), random_state=0)
            model.fit(X, target)
            self.W[k] = model.coef_[0].astype("float32")
            self.b[k] = np.float32(model.intercept_[0])
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("teacher is not fitted")
        return np.asarray(X, dtype="float32") @ self.W.T + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(X))

    def distribution(self, X: np.ndarray) -> np.ndarray:
        """Temperature-scaled distribution consumed by the MoE gate."""
        return _softmax(self.logits(X) / self.temperature)

    def save(self, path: str | Path) -> None:
        if self.W is None or self.b is None:
            raise RuntimeError("teacher is not fitted")
        np.savez(path, intents=np.asarray(self.intents), W=self.W, b=self.b,
                 temperature=np.float32(self.temperature))

    @staticmethod
    def load(path: str | Path) -> "OvRLogisticTeacher":
        z = np.load(path, allow_pickle=True)
        teacher = OvRLogisticTeacher(
            [str(x) for x in z["intents"]],
            float(z["temperature"]) if "temperature" in z else 2.0)
        teacher.W = np.asarray(z["W"], dtype="float32")
        teacher.b = np.asarray(z["b"], dtype="float32")
        return teacher

