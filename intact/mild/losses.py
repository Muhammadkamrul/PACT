"""
intact/mild/losses.py
=====================
Numpy port of the lead-time focal loss from the original `custom_loss.py`.

WHY THIS LOSS AND NOT PLAIN BCE  -- this is the part worth keeping
------------------------------------------------------------------
Three ideas, all of which matter more in INTACT than they did originally:

  LEAD-TIME WEIGHTING.   time_w = min_w + (1-min_w) * clip(ttf/H, 0, 1)
      A warning issued ONE SLOT before breach is nearly worthless to a
      scheduler that redecides once per epoch.  Weighting by remaining
      time-to-failure makes the model prefer EARLY warnings, which is
      exactly the v11 §10.2 ratchet requirement.

  CLASS BALANCE.         cls_w = 1 + (pos_weight - 1) * is_pos
      Breaches are rare.  Without this the model predicts "never fails" and
      scores well on accuracy while being useless.

  FOCAL MODULATION.      (1 - p_t)^gamma
      Down-weights the easy, obviously-safe samples so the gradient budget
      goes to the ambiguous ones near the decision boundary -- which is
      precisely the region where tau_risk sits.

  NEGATIVE MARGIN PENALTY.  relu(p - neg_margin) on negatives
      Keeps p_hat genuinely LOW when nothing is wrong.  This matters more in
      INTACT than in the original, because p_hat drives the urgency
      multiplier u(p_hat): a predictor that idles at p=0.3 inflates every
      weight all the time and destroys the contract ranking.
"""
from __future__ import annotations
import numpy as np


def lead_time_focal_bce(y_ttf, p, horizon, pos_weight=3.0, min_time_weight=0.3,
                        gamma=2.0, neg_margin=0.2, lambda_neg=0.2, eps=1e-6):
    """
    Returns (loss_scalar, dL/dp) -- both needed because the numpy model does
    its own backward pass.

    y_ttf : (N,K) remaining time to failure, 0 for negatives
    p     : (N,K) predicted probability
    """
    p = np.clip(p, eps, 1 - eps)
    is_pos = (y_ttf > 0).astype("float32")

    time_w = min_time_weight + (1 - min_time_weight) * np.clip(
        y_ttf / float(horizon), 0.0, 1.0)
    time_w = np.where(is_pos > 0, time_w, 1.0)      # negatives: weight 1
    # pos_weight may be a SCALAR or a per-intent vector of shape (K,).  A
    # single global weight under-serves rare intents: at a 0.3% positive rate
    # the gradient from that intent is invisible next to one at 6%.
    pw = np.asarray(pos_weight, dtype="float32").reshape(1, -1) \
        if np.ndim(pos_weight) else float(pos_weight)
    cls_w = 1.0 + (pw - 1.0) * is_pos
    w = cls_w * time_w

    bce = -(is_pos * np.log(p) + (1 - is_pos) * np.log(1 - p))
    dbce = (p - is_pos) / (p * (1 - p))

    p_t = p * is_pos + (1 - p) * (1 - is_pos)
    mod = (1 - p_t) ** gamma
    dp_t = is_pos - (1 - is_pos)                    # d p_t / d p
    dmod = -gamma * (1 - p_t) ** (gamma - 1) * dp_t

    focal = mod * bce
    dfocal = dmod * bce + mod * dbce

    over = np.maximum(p - neg_margin, 0.0) * (1 - is_pos)
    dover = ((p > neg_margin) & (is_pos == 0)).astype("float32")

    N = p.size
    loss = float((w * focal).sum() / N + lambda_neg * over.sum() / N)
    grad = (w * dfocal) / N + lambda_neg * dover / N
    return loss, grad


def gate_kl_with_sparsity(y_gate, gate, supervise_weight=0.3,
                          sparsity_weight=0.005, eps=1e-8):
    """
    Numpy port of `gate_kldiv_with_sparsity`.

    KL pulls the gate toward whichever intents are actually in a pre-failure
    window (rows with no impending failure are MASKED OUT, exactly as in the
    original).  The sparsity term p(1-p) discourages a mushy uniform gate, so
    the experts genuinely specialise instead of all learning the same thing.
    """
    gate = np.clip(gate, eps, 1.0)
    mask = (y_gate.sum(axis=1, keepdims=True) > 0).astype("float32")
    kl = (y_gate * (np.log(np.clip(y_gate, eps, 1)) - np.log(gate))).sum(
        axis=1, keepdims=True)
    loss_kl = float((kl * mask).mean())
    dkl = -mask * y_gate / gate

    sp = (gate * (1 - gate)).sum(axis=1, keepdims=True)
    loss_sp = float(sp.mean())
    dsp = (1 - 2 * gate)

    N = gate.shape[0]
    loss = supervise_weight * loss_kl + sparsity_weight * loss_sp
    grad = (supervise_weight * dkl + sparsity_weight * dsp) / N
    return loss, grad


def soft_target_bce(target, p, eps=1e-6):
    """Binary cross entropy used to distil the OvR teacher into each head."""
    target = np.asarray(target, dtype="float32")
    p = np.clip(np.asarray(p, dtype="float32"), eps, 1.0 - eps)
    n = p.size
    loss = -float((target * np.log(p) + (1.0 - target) * np.log(1.0 - p)).sum() / n)
    grad = ((p - target) / (p * (1.0 - p))) / n
    return loss, grad


def teacher_gate_kl(teacher_dist, gate, weight=0.7, eps=1e-8):
    """KL(teacher distribution || gate), as in the published MILD loss."""
    teacher_dist = np.asarray(teacher_dist, dtype="float32")
    gate = np.clip(np.asarray(gate, dtype="float32"), eps, 1.0)
    loss = float(weight * np.mean(np.sum(
        teacher_dist * (np.log(np.clip(teacher_dist, eps, 1.0))
                        - np.log(gate)), axis=1)))
    grad = -float(weight) * teacher_dist / gate / max(len(gate), 1)
    return loss, grad
