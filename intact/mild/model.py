"""
intact/mild/model.py
====================
MILD mixture-of-experts, ported to NUMPY.

WHY NUMPY AND NOT KERAS
-----------------------
The architecture is faithful to `hybrid_mild_model.py`; only the backend
differs.  Reasons: (1) this environment has no TensorFlow, and a framework
that cannot run because an optional dependency is missing is not much use;
(2) INTACT calls the predictor once per epoch on ONE row, so a 200-line
numpy forward pass is faster than a Keras graph; (3) it keeps the whole
simulator dependency-free beyond numpy.

*** YOUR EXISTING KERAS MODEL DROPS STRAIGHT IN. ***
`predictor.py` only requires an object with `.predict(X) -> (N, K)`.
Point it at your trained Keras model and nothing else changes.

ARCHITECTURE  (identical in structure to create_mild_moe_with_teacher)
    x -> [Dense(96) relu, Dense(96) relu]            shared encoder
      -> gate:   Dense(48) relu -> Dense(K) softmax
      -> per intent k:  Dense(64) relu -> Dense(32) relu
                        -> MULTIPLY by gate_k        <- the MoE gating
                        -> Dense(16) relu -> Dense(1) sigmoid

WHAT I DROPPED FROM THE ORIGINAL, AND WHY
    * the TEACHER input and the distillation loss.  Those exist to transfer
      knowledge from a Logistic one-vs-rest teacher.  INTACT has no such
      teacher, and adding one would mean maintaining a second model whose
      only job is to supervise the first.
    * head-decorrelation.  Useful when heads collapse; re-add if you observe
      it (the hook is left in place).

WHAT I KEPT, BECAUSE IT IS THE VALUABLE PART
    * the gated mixture of per-intent experts;
    * the LEAD-TIME FOCAL loss (losses.py).  Weighting by how EARLY the
      warning is, is precisely what INTACT needs: a prediction that fires
      one slot before breach is nearly worthless to a scheduler that decides
      once per epoch.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _relu(x):  return np.maximum(x, 0.0)
def _drelu(x): return (x > 0).astype(x.dtype)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


class Dense:
    """He-initialised affine layer with Adam state."""

    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)).astype("float32")
        self.b = np.zeros(n_out, dtype="float32")
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def __call__(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.gW = self.x.T @ g
        self.gb = g.sum(axis=0)
        return g @ self.W.T

    def step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, l2=1e-5):
        for P, G, M, V in ((self.W, self.gW + l2 * self.W, self.mW, self.vW),
                           (self.b, self.gb, self.mb, self.vb)):
            M *= b1; M += (1 - b1) * G
            V *= b2; V += (1 - b2) * G * G
            P -= lr * (M / (1 - b1 ** t)) / (np.sqrt(V / (1 - b2 ** t)) + eps)


class MildMoE:
    """Mixture-of-experts risk model: one gated expert per intent."""

    def __init__(self, n_features: int, intent_ids: List[str],
                 enc_units=(96, 96), gate_units=48, seed=0,
                 teacher_dim: int = 0):
        rng = np.random.default_rng(seed)
        self.intents = list(intent_ids)
        self.teacher_dim = int(teacher_dim)
        K = len(self.intents)
        self.enc = []
        prev = n_features
        for u in enc_units:
            self.enc.append(Dense(prev, u, rng)); prev = u
        self.shared_dim = prev
        self.g1 = Dense(prev + self.teacher_dim, gate_units, rng)
        self.g2 = Dense(gate_units, K, rng)
        self.e1 = [Dense(prev, 64, rng) for _ in range(K)]
        self.e2 = [Dense(64, 32, rng) for _ in range(K)]
        self.h = [Dense(32, 16, rng) for _ in range(K)]
        self.o = [Dense(16, 1, rng) for _ in range(K)]
        self._t = 0

    # ------------------------------------------------------------------
    def forward(self, X, teacher_dist=None):
        a = X
        self._enc_pre = []
        for L in self.enc:
            z = L(a); self._enc_pre.append(z); a = _relu(z)
        shared = a

        if self.teacher_dim:
            if teacher_dist is None:
                raise ValueError("teacher-augmented MILD requires teacher_dist")
            teacher_dist = np.asarray(teacher_dist, dtype="float32")
            if teacher_dist.shape != (len(X), self.teacher_dim):
                raise ValueError(
                    f"teacher_dist shape {teacher_dist.shape}, expected "
                    f"{(len(X), self.teacher_dim)}")
            gate_input = np.concatenate([shared, teacher_dist], axis=1)
        else:
            gate_input = shared
        zg1 = self.g1(gate_input); ag1 = _relu(zg1)
        zg2 = self.g2(ag1)
        gate = _softmax(zg2)
        self._zg1, self._gate = zg1, gate

        self._cache = []
        outs = []
        for k in range(len(self.intents)):
            z1 = self.e1[k](shared); a1 = _relu(z1)
            z2 = self.e2[k](a1);     a2 = _relu(z2)
            gk = gate[:, k:k + 1]
            a2g = a2 * gk                       # <- the gating multiply
            z3 = self.h[k](a2g);     a3 = _relu(z3)
            z4 = self.o[k](a3)
            p = _sigmoid(z4)
            self._cache.append((z1, z2, a2, gk, z3, z4, p))
            outs.append(p)
        self._shared = shared
        return np.concatenate(outs, axis=1), gate

    def predict(self, X, teacher_dist=None) -> np.ndarray:
        """P(intent k breaches within the horizon).  Shape (N, K)."""
        p, _ = self.forward(X, teacher_dist)
        return p

    # ------------------------------------------------------------------
    def backward(self, dP, dGate, lr):
        """dP: (N,K) gradient wrt the sigmoid OUTPUTS. dGate: wrt gate probs."""
        self._t += 1
        N = dP.shape[0]
        dshared = np.zeros_like(self._shared)
        dgate_from_experts = np.zeros_like(self._gate)

        for k in range(len(self.intents)):
            z1, z2, a2, gk, z3, z4, p = self._cache[k]
            dz4 = dP[:, k:k + 1] * p * (1 - p)          # sigmoid'
            da3 = self.o[k].backward(dz4)
            dz3 = da3 * _drelu(z3)
            da2g = self.h[k].backward(dz3)
            dgate_from_experts[:, k:k + 1] += (da2g * a2).sum(axis=1, keepdims=True)
            da2 = da2g * gk
            dz2 = da2 * _drelu(z2)
            da1 = self.e2[k].backward(dz2)
            dz1 = da1 * _drelu(z1)
            dshared += self.e1[k].backward(dz1)

        # gate path: softmax jacobian
        dg_total = dGate + dgate_from_experts
        g = self._gate
        dzg2 = g * (dg_total - (dg_total * g).sum(axis=1, keepdims=True))
        dag1 = self.g2.backward(dzg2)
        dzg1 = dag1 * _drelu(self._zg1)
        dgate_input = self.g1.backward(dzg1)
        dshared += dgate_input[:, :self.shared_dim]

        a = dshared
        for L, z in zip(reversed(self.enc), reversed(self._enc_pre)):
            a = L.backward(a * _drelu(z))

        for L in (self.enc + [self.g1, self.g2] + self.e1 + self.e2
                  + self.h + self.o):
            L.step(lr, self._t)

    # ------------------------------------------------------------------
    def get_weights(self):
        """Snapshot every parameter.  Needed for early stopping: without a
        restore the saved model is whatever the LAST epoch produced, which
        is exactly the overfitted one."""
        return [(L.W.copy(), L.b.copy()) for _, L in self._named()]

    def set_weights(self, weights):
        for (W, b), (_, L) in zip(weights, self._named()):
            L.W, L.b = W.copy(), b.copy()

    # ------------------------------------------------------------------
    def save(self, path):
        d = {}
        for name, L in self._named():
            d[f"{name}.W"] = L.W; d[f"{name}.b"] = L.b
        np.savez(path, intents=np.array(self.intents),
                 teacher_dim=np.int32(self.teacher_dim), **d)

    def _named(self):
        out = [(f"enc{i}", L) for i, L in enumerate(self.enc)]
        out += [("g1", self.g1), ("g2", self.g2)]
        for k in range(len(self.intents)):
            out += [(f"e1_{k}", self.e1[k]), (f"e2_{k}", self.e2[k]),
                    (f"h_{k}", self.h[k]), (f"o_{k}", self.o[k])]
        return out

    @staticmethod
    def load(path, n_features):
        z = np.load(path, allow_pickle=True)
        intents = [str(x) for x in z["intents"]]
        teacher_dim = int(z["teacher_dim"]) if "teacher_dim" in z else 0
        m = MildMoE(n_features, intents, teacher_dim=teacher_dim)
        for name, L in m._named():
            L.W = z[f"{name}.W"]; L.b = z[f"{name}.b"]
        return m
