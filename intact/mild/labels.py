"""
intact/mild/labels.py
=====================
Forward-looking labels.  ADAPTED FROM the original `build_forward_labels`.

THE LABEL THAT MATTERS  (v11 §7.1, problem L1)
----------------------------------------------
For every timestep t and intent i:
    y_ttf[t,i] = time remaining until intent i BREACHES, if a breach occurs
                 within the horizon H;  0 otherwise
    y_bin[t,i] = 1 if a breach occurs within H
This is LABELABLE from a trace: replay it, find where fulfilment dropped
below the requirement, and stamp the window before it.  That is the whole
reason INTACT uses supervised learning HERE and randomised estimation for
attribution -- "which xApp caused it" is a counterfactual and has no label.

WHAT I DROPPED FROM THE ORIGINAL, AND WHY
-----------------------------------------
`y_cause` -- the cause-aware gating mask that distinguished
"conflicting_*_cause" from "conflicting_victim".  In INTACT that job is done
by beta_{j,i}, which is MEASURED and CAUSAL rather than annotated, and is
naturally multi-cause and multi-victim.  Keeping a second, weaker cause
signal would create two disagreeing answers to the same question.

WHAT I KEPT
-----------
The gate is still supervised, but with a DIFFERENT and simpler target:
which intents are in a pre-failure window right now.  That keeps the
mixture-of-experts specialisation the original architecture depends on,
without importing an attribution mechanism INTACT has already replaced.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def find_breach_events(rho: np.ndarray, eta: float, min_gap: int = 20) -> List[int]:
    """
    A BREACH is the first timestep at which the rolling fulfilment fraction
    falls below the intent's requirement eta.  Consecutive breaches within
    `min_gap` are treated as one event so a single incident does not get
    counted many times.
    """
    below = rho < eta
    events, last = [], -10 ** 9
    for t in range(1, len(below)):
        if below[t] and not below[t - 1] and (t - last) > min_gap:
            events.append(t)
            last = t
    return events


def build_labels(rho_mat: np.ndarray, etas: np.ndarray, horizon: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rho_mat : (N, K) rolling fulfilment per intent
    etas    : (K,)   required fulfilment per intent
    horizon : H, in timesteps

    Returns y_ttf (N,K) float, y_bin (N,K) float, y_gate (N,K) float.
    """
    N, K = rho_mat.shape
    y_ttf = np.zeros((N, K), dtype="float32")
    y_bin = np.zeros((N, K), dtype="float32")

    for k in range(K):
        for fail_t in find_breach_events(rho_mat[:, k], etas[k]):
            start = max(fail_t - horizon, 0)
            idx = np.arange(start, fail_t)
            if len(idx) == 0:
                continue
            y_ttf[idx, k] = (fail_t - idx).astype("float32")
            y_bin[idx, k] = 1.0

    # gate target: normalise the pre-failure indicator into a distribution
    # over intents, so the mixture specialises on whichever intent is
    # currently in trouble.  Rows with no impending failure are left at zero
    # and are masked out of the gate loss.
    s = y_bin.sum(axis=1, keepdims=True)
    y_gate = np.divide(y_bin, s, out=np.zeros_like(y_bin), where=s > 0)
    return y_ttf, y_bin, y_gate


def horizon_from_erosion(rho_mat: np.ndarray, etas: np.ndarray,
                         slots_per_epoch: int) -> int:
    """
    *** THE v11 §10.2 CONSTRAINT, MADE MEASURABLE. ***

    H must exceed the FASTEST observed erosion time from "at risk" to breach.
    Otherwise a sequence of individually-safe writes walks an intent into
    failure before the gate ever fires, and the ratchet is real.

    We measure it: for each breach, count back to the last time the intent
    was comfortably above its requirement, and take the FASTEST such descent
    across all intents.  Returns a recommended H with a safety factor.
    """
    fastest = 10 ** 9
    N, K = rho_mat.shape
    for k in range(K):
        for fail_t in find_breach_events(rho_mat[:, k], etas[k]):
            comfort = etas[k] + 0.5 * (1.0 - etas[k])
            t = fail_t
            while t > 0 and rho_mat[t, k] < comfort:
                t -= 1
            fastest = min(fastest, fail_t - t)
    if fastest >= 10 ** 9:
        return 4 * slots_per_epoch          # no breaches observed; use a default
    return int(max(2 * fastest, slots_per_epoch + 1))


# ======================================================================
# MARGIN-CROSSING LABELS  (v18.3 -- the default from here on)
# ======================================================================
# WHY THE FULFILMENT-BREACH LABEL ABOVE WAS THE WRONG TARGET
# ----------------------------------------------------------
# `build_labels` marks the H slots before the FIRST time rolling fulfilment
# rho_i crosses below eta_i.  Three things go wrong with that in INTACT:
#
#  1. eta defaults to 0.95 while make_scenario auto-calibrates intent targets
#     so each intent is met roughly HALF the time.  Measured on this
#     scenario: mean rho 0.34-0.65 against eta 0.95, so rho < eta in
#     53-99.8% of slots.  "Breached" is the normal state, the FIRST crossing
#     is rare, and one intent produced 4 events in 20,000 slots.
#
#  2. Every slot in which the intent is ALREADY breached is labelled
#     NEGATIVE.  The analytical predictor this model replaces does the exact
#     opposite -- risk.py projects the margin and returns p > 0.5 whenever
#     g < 0 -- so the two predictors were trained on contradictory targets
#     and could not be substituted for one another.
#
#  3. Nothing downstream consumes rho-vs-eta.  The inner gate tests
#     g_i >= epsilon_i, and observed_safety_crossing_rate counts
#     g(before) >= epsilon > g(after).  The predictor was aimed at a
#     quantity the framework never uses.
#
# THE TARGET THAT MATCHES CONSUMPTION
#     y[t,i] = 1  iff  g_i falls below epsilon_i at some slot in (t, t+H]
#
# This is the same crossing the inner loop mediates and the same one the
# safety metric counts, so p_hat, the gate and the reported metric finally
# refer to one event.  It also yields a workable positive rate instead of
# 0.3%.

def build_margin_labels(g_mat: np.ndarray, eps: np.ndarray, horizon: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    g_mat   : (N, K) per-slot intent margins
    eps     : (K,)   per-intent safety buffer  (Intent.epsilon)
    horizon : H, in slots

    Returns y_ttf, y_bin, y_gate, currently_safe.

    ``currently_safe`` is the mask g_i(t) >= eps_i.  Evaluation must be
    reported BOTH overall and restricted to this subset: predicting that an
    already-unsafe intent stays unsafe is trivial, whereas predicting a
    crossing while the intent still looks healthy is the ratchet case the
    architecture exists to catch.
    """
    g_mat = np.asarray(g_mat, dtype="float32")
    N, K = g_mat.shape
    eps = np.asarray(eps, dtype="float32").reshape(1, K)

    unsafe = g_mat < eps
    y_ttf = np.zeros((N, K), dtype="float32")
    y_bin = np.zeros((N, K), dtype="float32")

    # Walk h downwards so the SMALLEST qualifying lead time wins, giving the
    # true time-to-crossing rather than any crossing in the window.
    for h in range(horizon, 0, -1):
        shifted = np.zeros_like(unsafe)
        if h < N:
            shifted[:N - h] = unsafe[h:]
            shifted[N - h:] = unsafe[-1]
        else:
            shifted[:] = unsafe[-1]
        y_ttf = np.where(shifted, np.float32(h), y_ttf)
        y_bin = np.maximum(y_bin, shifted.astype("float32"))

    s = y_bin.sum(axis=1, keepdims=True)
    y_gate = np.divide(y_bin, s, out=np.zeros_like(y_bin), where=s > 0)
    currently_safe = ~unsafe
    return y_ttf, y_bin, y_gate, currently_safe


def resolve_eta(intents, tenants, log=None) -> np.ndarray:
    """Per-intent required fulfilment, made consistent with the contract.

    `Intent.eta` defaults to 0.95, which no scenario overrides, while tenant
    contracts carry a fulfilment floor `rho_min` of roughly 0.3-0.6.  Scoring
    an intent against 0.95 while its owner is contractually entitled to 0.45
    is not a stricter test, it is a different one, and it is what made the
    legacy breach label degenerate.  Resolution order:

        explicit intent eta  ->  owning tenant's rho_min  ->  0.95
    """
    iids = list(intents)
    out = []
    for i in iids:
        it = intents[i]
        raw = getattr(it, "eta", None)
        explicit = raw is not None and abs(float(raw) - 0.95) > 1e-9
        if explicit:
            out.append(float(raw))
            continue
        ten = tenants.get(getattr(it, "tenant", None))
        if ten is not None and getattr(ten, "rho_min", None) is not None:
            out.append(float(ten.rho_min))
        else:
            out.append(float(raw) if raw is not None else 0.95)
    arr = np.asarray(out, dtype="float32")
    if log is not None:
        log.info("resolved eta per intent: %s",
                 {i: round(float(v), 3) for i, v in zip(iids, arr)})
    return arr
