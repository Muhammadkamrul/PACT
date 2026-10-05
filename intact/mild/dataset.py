"""
intact/mild/dataset.py
======================
Collect a training trace from the RAN and turn it into (X, y).

THIS REPLACES `generate_hard_dataset.py`.  The original synthesised
datacentre KPI traces with hand-built failure archetypes (XOR events,
co-drift, benign mimics).  We do not need to synthesise anything: the
analytical RAN GENERATES the failures for us, and they are the failures
INTACT will actually face.

But the original had one genuinely good idea we must keep --
    *** BENIGN MIMICS ***
episodes that LOOK like the run-up to a failure but do not fail.  Without
them the model learns "margin dipped => breach" and fires constantly.  Since
p_hat drives the urgency multiplier, a trigger-happy predictor inflates
every weight all the time and destroys the contract ranking.  So we inject
benign load transients on purpose (`benign_mimic_rate`).

We also keep the original's principle of REGIME VARIATION: the same control
action must have different consequences under different load, or the model
learns a single global rule that is wrong half the time.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import copy
import numpy as np

from ..estimation.margins import margin, MarginTracker
from .features import raw_frame, engineer, Standardiser
from .labels import (build_labels, horizon_from_erosion,
                     build_margin_labels, resolve_eta)


def collect_trace(ran, cfg, intents, tenants, xapps, claims, n_slots: int,
                  rng: np.random.Generator, log,
                  label_mode: str | None = None) -> List[Dict]:
    """
    Drive the RAN with RANDOMISED control activity and record everything.

    Why randomised: the training trace must contain the full range of
    situations, including ones a good scheduler would avoid.  A trace
    collected under the deployed policy would only show states that policy
    visits, and the predictor would be blind exactly where it is most needed.
    """
    tracker = MarginTracker(intents, cfg["margins"]["window_slots"])
    controls = sorted({c.param for c in claims.values()})
    recs = []
    mimic_rate = cfg["mild"]["benign_mimic_rate"]
    load_scales = cfg["mild"]["load_scales"]
    base_load = {t: np.array(ran.ue[t]["load_mbps"], copy=True) for t in ran.slices}

    # Runtime calls MILD once per scheduling epoch, not once per radio slot.
    # Training at slot cadence and inference at epoch cadence made a 15-sample
    # rolling window mean 1.9 s in training but 30 s at runtime.  Sample the
    # training trace at the identical physical cadence.
    sample_slots = int(cfg.get("mild", {}).get(
        "sample_slots", int(cfg["ran"].get("pre_slots", cfg["ran"]["slots_per_epoch"]))
        + int(cfg["ran"].get("post_slots", cfg["ran"]["slots_per_epoch"]))))
    sample_slots = max(sample_slots, 1)
    label_mode = label_mode or cfg.get("mild", {}).get(
        "label_mode", "margin_crossing")
    horizon = int(cfg.get("risk", {}).get(
        "horizon_epochs",
        max(1, int(np.ceil(cfg["risk"]["horizon_slots"] / sample_slots)))))
    persistence = max(1, int(cfg.get("mild", {}).get(
        "failure_persistence_epochs", 3)))
    episode_epochs = max(
        horizon + persistence + 1,
        int(cfg.get("mild", {}).get("status_quo_episode_epochs", 96)))
    n_samples = max(int(n_slots) // sample_slots, 1)
    use_runtime_profile = bool(cfg.get("mild", {}).get(
        "use_runtime_load_profile", bool(cfg["ran"].get("load_profile"))))
    period_samples = max(1, int(cfg["mild"]["regime_period_slots"]) // sample_slots)

    mimic_until, scale = -1, 1.0
    for t in range(n_samples):
        # ---- regime variation: slow drift through load scales -----------
        if not use_runtime_profile and t % period_samples == 0:
            scale = float(rng.choice(load_scales))
            for tid in ran.slices:
                ran.ue[tid]["load_mbps"] = base_load[tid] * scale

        # ---- benign mimic: a transient that looks bad but recovers ------
        if t > mimic_until and rng.random() < mimic_rate:
            dur = int(rng.integers(6, 18))
            mimic_until = t + dur
            for tid in ran.slices:
                ran.ue[tid]["load_mbps"] = base_load[tid] * scale * 1.9
        elif t == mimic_until:
            for tid in ran.slices:
                ran.ue[tid]["load_mbps"] = base_load[tid] * scale

        # ---- randomised control activity --------------------------------
        # *** BUG FIX (v18.3). ***  Drawing UNIFORMLY over each knob's whole
        # domain every ~7 slots is far more destructive than any real xApp,
        # and combined with load scales up to 1.7 it drove the cell into
        # permanent saturation.  Measured on configs/base.yaml: intent
        # margins sat below epsilon in 50-99% of slots, so BOTH label
        # definitions degenerated -- the rho-vs-eta breach almost never
        # fired, and the margin crossing almost always did.  A predictor
        # trained there scores PR-AUC 0.99 by answering "yes", a lift of
        # 1.01 over the base rate.
        #
        # A BOUNDED RANDOM WALK explores the same domain but reaches extreme
        # values gradually, so the trace contains healthy, marginal and
        # saturated states instead of only the last.  Set
        # mild.write_mode: uniform to restore the old behaviour.
        write_mode = cfg["mild"].get("write_mode", "walk")
        walk_frac = float(cfg["mild"].get("walk_frac", 0.12))
        randomise_now = (label_mode != "status_quo_episode"
                         or t % episode_epochs == 0)
        if randomise_now:
            for j, c in claims.items():
                write_probability = (
                    float(cfg["mild"].get("status_quo_episode_write_prob", 1.0))
                    if label_mode == "status_quo_episode"
                    else float(cfg["mild"]["write_prob"]))
                if rng.random() < write_probability:
                    lo, hi = c.domain
                    if write_mode == "uniform":
                        ran.apply(c.param, float(rng.uniform(lo, hi)))
                    else:
                        cur = float(ran.current_controls().get(
                            c.param, 0.5 * (lo + hi)))
                        episode_frac = float(cfg["mild"].get(
                            "status_quo_episode_walk_frac", 0.20))
                        step = ((episode_frac if label_mode == "status_quo_episode"
                                 else walk_frac) * (hi - lo))
                        nxt = cur + float(rng.normal(0.0, step))
                        # reflect at the boundaries so the walk does not park on
                        # a rail and stay there for thousands of slots
                        if nxt < lo:
                            nxt = lo + (lo - nxt)
                        if nxt > hi:
                            nxt = hi - (nxt - hi)
                        ran.apply(c.param, float(min(max(nxt, lo), hi)))

        kpm = ran.step(sample_slots)
        g = tracker.update(kpm)
        rho = {i: tracker.fulfilment(i) for i in intents}
        record = {"t": t, "g": dict(g), "rho": rho,
                  "kpm": {k: dict(v) for k, v in kpm.items()},
                  "controls": dict(ran.current_controls())}
        if label_mode == "status_quo_episode":
            record["status_quo_episode"] = t // episode_epochs

        if label_mode == "status_quo_crossing":
            # The old label followed whatever RANDOM actions happened later
            # in the training walk.  Those actions are unknowable at time t,
            # so even a perfect predictor faced irreducible label noise.
            #
            # The deployable estimand is instead:
            #   P(g_i crosses epsilon_i within H | current state,
            #     current controls held fixed, exogenous RAN evolution).
            # It answers the scheduler's actual question: "what fails if I
            # do not intervene now?"  The clone is used only to make TRAINING
            # labels; runtime receives no future KPI or margin.
            probe = ran.clone()
            for param, value in probe.current_controls().items():
                probe.apply(param, value)  # freeze a slow actuator in place
            probe_tracker = copy.deepcopy(tracker)
            ttf = {iid: 0 for iid in intents}
            run_below = {iid: 0 for iid in intents}
            currently_safe = {
                iid: float(g[iid]) >= float(intents[iid].epsilon)
                for iid in intents}
            for lead in range(1, horizon + 1):
                future_kpm = probe.step(sample_slots)
                future_g = probe_tracker.update(future_kpm)
                for iid in intents:
                    if not currently_safe[iid] or ttf[iid] > 0:
                        continue
                    if float(future_g[iid]) < float(intents[iid].epsilon):
                        run_below[iid] += 1
                    else:
                        run_below[iid] = 0
                    if run_below[iid] >= persistence:
                        # Record the onset, not the confirmation time.
                        ttf[iid] = lead - persistence + 1
            record["status_quo_ttf"] = ttf
            record["status_quo_safe"] = currently_safe

        recs.append(record)

        if (t + 1) % 1000 == 0:
            log.info("  MILD trace: %d / %d epoch-cadence samples "
                     "(%d physical slots/sample)",
                     t + 1, n_samples, sample_slots)
    return recs


def build_dataset(recs, cfg, intents, tenants, claims, log, label_mode=None):
    """recs -> (X, y_ttf, y_bin, y_gate, feats, H, iids, safe_mask)

    label_mode:
        "margin_crossing"    g_i falls below epsilon_i within H   (DEFAULT)
        "fulfilment_breach"  legacy rho_i crosses eta_i           (v18.2)
        "status_quo_crossing" cloned no-new-write crossing within H
        "status_quo_episode"  fast held-control episodes; boundary rows excluded
    """
    label_mode = label_mode or cfg.get("mild", {}).get(
        "label_mode", "margin_crossing")
    controls = sorted({c.param for c in claims.values()})
    df = raw_frame(recs, intents, tenants, controls)
    X, feats = engineer(df)

    iids = list(intents)
    rho = np.column_stack([df[f"rho_{i}"].values for i in iids]).astype("float32")
    g = np.column_stack([df[f"g_{i}"].values for i in iids]).astype("float32")
    etas = resolve_eta(intents, tenants, log)
    eps = np.array([intents[i].epsilon for i in iids], dtype="float32")

    # ---- CHOOSE H BY MEASUREMENT, NOT BY GUESS (v11 s10.2) -------------
    # H is in OBSERVATIONS.  collect_trace now has the same one-observation-
    # per-epoch cadence as runtime, so a configured 20 means 20 scheduler
    # decisions in both places.
    sample_slots = int(cfg.get("mild", {}).get(
        "sample_slots",
        int(cfg["ran"].get("pre_slots", cfg["ran"]["slots_per_epoch"]))
        + int(cfg["ran"].get("post_slots", cfg["ran"]["slots_per_epoch"]))))
    H_cfg = int(cfg.get("risk", {}).get(
        "horizon_epochs",
        max(1, int(np.ceil(cfg["risk"]["horizon_slots"] / max(sample_slots, 1))))))
    H = H_cfg
    if label_mode == "status_quo_episode":
        persistence = max(1, int(cfg.get("mild", {}).get(
            "failure_persistence_epochs", 3)))
        episode = np.asarray([
            int(r.get("status_quo_episode", -1)) for r in recs])
        y_ttf = np.zeros_like(g, dtype="float32")
        valid = np.zeros(len(recs), dtype=bool)
        for ep in np.unique(episode):
            idx = np.where(episode == ep)[0]
            if len(idx) <= H + persistence:
                continue
            first, last = int(idx[0]), int(idx[-1])
            # Drop the rolling-feature warm-up after a control reset, and
            # drop rows whose H-step label would cross the episode boundary.
            warmup = max(15, persistence)
            valid[first + warmup:last - H - persistence + 2] = True
            for k, iid in enumerate(iids):
                below = g[idx, k] < eps[k]
                onsets = []
                run = 0
                for q, is_below in enumerate(below):
                    run = run + 1 if is_below else 0
                    if run == persistence:
                        onsets.append(first + q - persistence + 1)
                for t in idx:
                    if not valid[t] or g[t, k] < eps[k]:
                        continue
                    future = next((u for u in onsets if t < u <= t + H), None)
                    if future is not None:
                        y_ttf[t, k] = float(future - t)
        y_bin = (y_ttf > 0).astype("float32")
        safe = g >= eps.reshape(1, -1)
        total = y_bin.sum(axis=1, keepdims=True)
        y_gate = np.divide(y_bin, total, out=np.zeros_like(y_bin),
                           where=total > 0)
        for q, r in enumerate(recs):
            r["mild_valid"] = bool(valid[q])
        X, y_ttf, y_bin, y_gate, safe = (
            a[valid] for a in (X, y_ttf, y_bin, y_gate, safe))
    elif label_mode == "status_quo_crossing":
        H = H_cfg
    else:
        H_meas = horizon_from_erosion(rho, etas, 1)
        H = max(H_cfg, H_meas)
        if H > H_cfg:
            log.warning("RATCHET GUARD: measured fastest erosion needs H >= %d, "
                        "config had %d.  Using %d.", H_meas, H_cfg, H)

    # ---- CONTRACT-CONSISTENCY CHECK ------------------------------------
    # The legacy label is only meaningful when rho actually spends most of
    # its time ABOVE eta.  Say so loudly rather than silently producing a
    # dataset with four positive events in twenty thousand slots.
    frac_below = (rho < etas.reshape(1, -1)).mean(axis=0)
    for k, i in enumerate(iids):
        if frac_below[k] > 0.5:
            log.warning("intent %s sits below its required fulfilment in "
                        "%.1f%% of slots (mean rho %.3f vs eta %.3f). The "
                        "legacy breach label is degenerate here.",
                        i, 100 * frac_below[k], rho[:, k].mean(), etas[k])

    if label_mode == "status_quo_episode":
        pass  # labels were constructed and boundary-filtered above
    elif label_mode == "status_quo_crossing":
        if not recs or "status_quo_ttf" not in recs[0]:
            raise ValueError(
                "status_quo_crossing requires collect_trace(..., "
                "label_mode='status_quo_crossing')")
        y_ttf = np.asarray([
            [float(r["status_quo_ttf"].get(i, 0)) for i in iids]
            for r in recs], dtype="float32")
        y_bin = (y_ttf > 0).astype("float32")
        safe = np.asarray([
            [bool(r["status_quo_safe"].get(i, False)) for i in iids]
            for r in recs], dtype=bool)
        total = y_bin.sum(axis=1, keepdims=True)
        y_gate = np.divide(y_bin, total, out=np.zeros_like(y_bin),
                           where=total > 0)
    elif label_mode == "fulfilment_breach":
        y_ttf, y_bin, y_gate = build_labels(rho, etas, H)
        safe = rho >= etas.reshape(1, -1)
    else:
        y_ttf, y_bin, y_gate, safe = build_margin_labels(g, eps, H)

    pos = y_bin.mean(axis=0)
    log.info("MILD dataset: %d samples, %d features, H=%d epochs, mode=%s",
             len(X), X.shape[1], H, label_mode)
    for k, i in enumerate(iids):
        n_safe = int(safe[:, k].sum())
        pos_safe = (float(y_bin[safe[:, k], k].mean())
                    if n_safe else float("nan"))
        log.info("   %-4s positive rate %.3f  (among currently-safe rows "
                 "%.3f, n=%d)", i, pos[k], pos_safe, n_safe)
    if pos.max() < 0.01:
        log.warning("VERY FEW POSITIVES -- raise load or tighten targets "
                    "before trusting p_hat.")
    return X, y_ttf, y_bin, y_gate, feats, H, iids, safe
