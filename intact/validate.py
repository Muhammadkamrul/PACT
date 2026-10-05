"""
intact/validate.py
==================
GROUND-TRUTH VALIDATION.  Only possible in simulation -- exploit it now,
because you lose it the moment you move to srsRAN.

The analytical RAN lets us MEASURE the true effects directly, by brute force:

    beta_true[j][i]  =  E[ dg_i | claim j eligible ]  -  E[ dg_i | not eligible ]

with everything else held fixed and the SAME channel/traffic trace on both
sides (common random numbers).  That is the estimand the online regression
is trying to recover.  Then we can ask the only question that matters:

        does  beta_hat  converge to  beta_true ?

WHY SIGN ACCURACY MATTERS MORE THAN MAE
For the scheduler, confusing beta = +0.08 with beta_hat = -0.03 is FAR more
consequential than estimating 0.08 as 0.07: the first flips V_j and changes
who gets authority; the second barely moves the ranking.  So we report sign
accuracy as a first-class metric, not just MAE/RMSE.
"""
from __future__ import annotations
import copy
from typing import Dict, List
import numpy as np

from .estimation.margins import margin


def measure_ground_truth_beta(ran_factory, cfg, intents, claims, xapps,
                              n_reps: int, log) -> Dict[str, Dict[str, float]]:
    """
    Brute-force beta_true by direct A/B measurement, one claim at a time.

    For each claim j and each replicate:
      * build TWO identical RANs from the SAME seed (common random numbers)
      * in world A, claim j's xApp writes;  in world B it does not
      * everything else is identical
      * beta_true[i][j] = mean over reps of ( dg_i^A - dg_i^B )

    Because the seeds match, the channel and traffic realisations are
    IDENTICAL across the two worlds, so the difference isolates the claim.
    """
    spe = cfg["ran"]["slots_per_epoch"]
    out = {i: {j: 0.0 for j in claims} for i in intents}
    n_settle = cfg["validation"]["settle_epochs"]
    n_meas = cfg["validation"]["measure_epochs"]

    for j, c in claims.items():
        diffs = {i: [] for i in intents}
        for rep in range(n_reps):
            seed = cfg["validation"]["base_seed"] + rep * 37
            worlds = {}
            for world in ("on", "off"):
                ran = ran_factory(seed)
                # settle from an identical starting point
                for _ in range(n_settle):
                    ran.step(spe)
                g_before = None
                acc = {i: [] for i in intents}
                for _ in range(n_meas):
                    kpm = ran.step(spe)
                    g0 = {i: margin(intents[i], kpm) for i in intents}
                    if world == "on":
                        xa = xapps.get(c.xapp)
                        if xa is not None:
                            nu = xa.propose(kpm, ran.current_controls(), None)
                            if nu is not None:
                                ran.apply(c.param, nu)
                    kpm2 = ran.step(spe)
                    g1 = {i: margin(intents[i], kpm2) for i in intents}
                    for i in intents:
                        acc[i].append(g1[i] - g0[i])
                worlds[world] = {i: float(np.mean(acc[i])) for i in intents}
            for i in intents:
                diffs[i].append(worlds["on"][i] - worlds["off"][i])
        for i in intents:
            out[i][j] = float(np.mean(diffs[i]))
        log.info("ground-truth beta for %s: %s", j,
                 {i: round(out[i][j], 4) for i in intents})
    return out


def recovery_metrics(beta_hat: Dict, beta_true: Dict, se: Dict | None = None,
                     tol: float = 1e-3) -> Dict:
    """
    Compare the ONLINE estimate against ground truth.

    sign_accuracy counts a pair as correct when both are within `tol` of zero
    (both "no effect") or both have the same non-zero sign.  Near-zero true
    effects would otherwise be scored on a coin flip.
    """
    errs, signs, covered, n_cov = [], [], 0, 0
    per_pair = []
    for i in beta_true:
        for j in beta_true[i]:
            bt, bh = beta_true[i][j], beta_hat.get(i, {}).get(j, 0.0)
            e = abs(bh - bt); errs.append(e)
            both_zero = abs(bt) < tol and abs(bh) < tol
            same_sign = (bt > tol and bh > tol) or (bt < -tol and bh < -tol)
            ok = both_zero or same_sign
            signs.append(ok)
            row = {"intent": i, "claim": j, "beta_true": bt,
                   "beta_hat": bh, "abs_err": e, "sign_ok": ok}
            if se is not None:
                s = se.get(i, {}).get(j, float("inf"))
                if np.isfinite(s) and s > 0:
                    n_cov += 1
                    inside = abs(bh - bt) <= 1.96 * s
                    covered += int(inside)
                    row["ci_covers_truth"] = bool(inside)
            per_pair.append(row)
    return {
        "MAE_beta": float(np.mean(errs)),
        "RMSE_beta": float(np.sqrt(np.mean(np.square(errs)))),
        "max_abs_err": float(np.max(errs)),
        "sign_accuracy": float(np.mean(signs)),
        "ci_coverage_95": (covered / n_cov) if n_cov else None,
        "n_pairs": len(errs),
        "per_pair": per_pair,
    }


def convergence_curve(beta_trace: List[Dict], beta_true: Dict) -> List[Dict]:
    """MAE and sign accuracy against epoch -- answers 'is 300 epochs enough?'"""
    out = []
    for snap in beta_trace:
        m = recovery_metrics(snap["beta"], beta_true)
        out.append({"epoch": snap["epoch"], "MAE": m["MAE_beta"],
                    "sign_accuracy": m["sign_accuracy"]})
    return out


def response_lag_probe(ran_factory, cfg, intents, param, lo, hi, log,
                       n_slots: int = 40) -> Dict:
    """
    How many slots does a control change take to show up in the KPMs?

    THIS MATTERS FOR beta: if the RAN's response lag exceeds one epoch, part
    of claim j's effect lands in the NEXT epoch's row and is attributed to
    whatever was eligible then.  The observation window must match the lag.
    """
    spe = cfg["ran"]["slots_per_epoch"]
    ran = ran_factory(cfg["validation"]["base_seed"])
    ran.apply(param, lo)
    for _ in range(6):
        ran.step(spe)
    base = {i: margin(intents[i], ran.step(spe)) for i in intents}
    ran.apply(param, hi)                              # step change
    traj = {i: [] for i in intents}
    for _ in range(n_slots):
        kpm = ran.step(1)
        for i in intents:
            traj[i].append(margin(intents[i], kpm) - base[i])
    lag = {}
    for i in intents:
        a = np.array(traj[i]); final = a[-5:].mean()
        if abs(final) < 1e-6:
            lag[i] = 0
            continue
        # first slot at which 90% of the final response is reached
        reached = np.where(np.abs(a) >= 0.9 * abs(final))[0]
        lag[i] = int(reached[0]) if len(reached) else n_slots
    log.info("response lag for %s (slots to 90%% of final): %s", param, lag)
    return {"param": param, "lag_slots": lag,
            "epoch_slots": spe, "trajectory": {i: traj[i] for i in traj}}


def measure_ground_truth_gamma(ran_factory, cfg, intents, claims, xapps,
                               pairs, n_reps, log):
    """
    gamma_true by a matched 2x2 DIFFERENCE-IN-DIFFERENCES.

        gamma_{jk,i} = (dg|11 - dg|10) - (dg|01 - dg|00)

    All four arms run from the SAME seed, so the channel and traffic
    realisations are identical and the only difference is which claims acted.
    The interaction is therefore what a pair does BEYOND the sum of its parts,
    with the individual effects differenced away.

    Kept completely separate from the regression training data.
    """
    spe = cfg["ran"]["slots_per_epoch"]
    n_settle = cfg["validation"]["settle_epochs"]
    n_meas = cfg["validation"]["measure_epochs"]
    out = {i: {} for i in intents}

    for (ja, jb) in pairs:
        if ja not in claims or jb not in claims:
            continue
        acc = {i: [] for i in intents}
        for rep in range(n_reps):
            seed = cfg["validation"]["base_seed"] + rep * 53
            arm = {}
            for za in (0, 1):
                for zb in (0, 1):
                    ran = ran_factory(seed)          # COMMON RANDOM NUMBERS
                    for _ in range(n_settle):
                        ran.step(spe)
                    d = {i: [] for i in intents}
                    for _ in range(n_meas):
                        kpm = ran.step(spe)
                        g0 = {i: margin(intents[i], kpm) for i in intents}
                        for z, j in ((za, ja), (zb, jb)):
                            if not z:
                                continue
                            c = claims[j]; xa = xapps.get(c.xapp)
                            if xa is None:
                                continue
                            nu = xa.propose(kpm, ran.current_controls(), None)
                            if nu is not None:
                                ran.apply(c.param, nu)
                        kpm2 = ran.step(spe)
                        for i in intents:
                            d[i].append(margin(intents[i], kpm2) - g0[i])
                    arm[(za, zb)] = {i: float(np.mean(d[i])) for i in intents}
            for i in intents:
                acc[i].append((arm[(1, 1)][i] - arm[(1, 0)][i])
                              - (arm[(0, 1)][i] - arm[(0, 0)][i]))
        for i in intents:
            out[i][(ja, jb)] = float(np.mean(acc[i]))
        log.info("ground-truth gamma for %s: %s", (ja, jb),
                 {i: round(out[i][(ja, jb)], 4) for i in intents})
    return out


def validate_sensitivity_heldout(ran_factory, cfg, intents, params, log,
                                 train_frac=0.6):
    """
    HELD-OUT INTERVENTION VALIDATION for s_{p,i}.

    The controlled sweep fits a slope and reports its standard error, but an
    SE only says how well the line fits the points it was FITTED ON.  Here we

        1. split the swept knob values into TRAIN and HELD-OUT,
        2. fit s on the TRAIN points only,
        3. PREDICT the margin at the held-out values,
        4. actually APPLY those values to a cloned RAN state and observe,
        5. compare prediction with observation.

    Held-out points never enter the fit, so this measures extrapolation
    accuracy rather than goodness of fit.
    """
    sc = cfg["sensitivity"]
    rows = []
    for prm in params:
        if prm not in cfg["sweep_domains"]:
            continue
        lo, hi = cfg["sweep_domains"][prm]
        current = cfg["ran"].get("initial_controls", {}).get(prm, 0.5 * (lo + hi))
        if sc.get("local_fit", False):
            # validate over the trust region, because that is the only range
            # the model is ever asked to predict on.  Centre it on the actual
            # operating point, not the midpoint of the declared domain.
            half = sc.get("local_frac",
                          cfg["inner"].get("max_step_frac", 0.25)) * (hi - lo)
            lo, hi = max(lo, current - half), min(hi, current + half)
        grid = list(np.linspace(lo, hi, max(sc["sweep_points"] + 4, 8)))
        # Stable per-parameter split: Python's hash is process-randomised.
        prm_offset = sum((k + 1) * b for k, b in enumerate(prm.encode("utf-8")))
        rng = np.random.default_rng(cfg["validation"]["base_seed"] + prm_offset)
        idx = rng.permutation(len(grid))
        n_tr = max(3, int(train_frac * len(grid)))
        tr = sorted(grid[k] for k in idx[:n_tr])
        te = sorted(grid[k] for k in idx[n_tr:])
        if len(te) < 2:
            continue
        for scale in sc["sweep_load_scales"]:
            def sample(vals):
                """One fresh common-random-number world per knob value.

                The old validator applied all values sequentially to one RAN.
                Queue and controller history then made 'knob value' perfectly
                confounded with time/order.  Resetting to the same seed and
                advancing the same number of slots isolates the intervention.
                """
                obs = {}
                for v in vals:
                    ran = ran_factory(cfg["validation"]["base_seed"])
                    for tid in ran.slices:
                        ran.ue[tid]["load_mbps"] = np.full(
                            len(ran.ue[tid]["load_mbps"]),
                            ran.slices[tid]["load_mbps_per_ue"] * scale)
                    for p0, v0 in cfg["ran"].get("initial_controls", {}).items():
                        ran.apply(p0, float(v0))
                    ran.step(sc["settle_slots"])
                    ran.apply(prm, float(v))
                    ran.step(sc["settle_slots"])
                    kpm = ran.step(sc["measure_slots"])
                    obs[v] = {i: margin(intents[i], kpm) for i in intents}
                return obs
            o_tr, o_te = sample(tr), sample(te)
            for iid in intents:
                X = np.array(tr); Y = np.array([o_tr[v][iid] for v in tr])
                if np.std(X) < 1e-9:
                    continue
                A = np.column_stack([np.ones_like(X), X])
                coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
                pred = coef[0] + coef[1] * np.array(te)
                obsv = np.array([o_te[v][iid] for v in te])
                At = np.column_stack([np.ones(len(te)), np.array(te)])
                coef_te, *_ = np.linalg.lstsq(At, obsv, rcond=None)
                slope_te = float(coef_te[1])
                err = pred - obsv
                ss = float(((obsv - obsv.mean()) ** 2).sum())
                material = bool(abs(slope_te) >= sc["sigma_min"])
                rows.append({
                    "param": prm, "intent": iid, "load_scale": scale,
                    "current_value": float(current),
                    "slope_train": float(coef[1]),
                    "slope_heldout": slope_te,
                    "material_effect": material,
                    "slope_sign_ok": (bool(np.sign(coef[1]) == np.sign(slope_te))
                                      if material else None),
                    "n_train": len(tr),
                    "n_heldout": len(te),
                    "MAE": float(np.mean(np.abs(err))),
                    "RMSE": float(np.sqrt(np.mean(err ** 2))),
                    # R2 is UNDEFINED when the held-out observations are
                    # constant (SST = 0): the knob does not move this intent,
                    # so there is no variance to explain.  Flagged and
                    # excluded rather than reported as nan.
                    "R2_heldout": (float(1 - (err ** 2).sum() / ss)
                                   if ss > 1e-12 and np.isfinite(err).all() else None),
                    "degenerate_no_variation": bool(ss <= 1e-12
                                                    or not np.isfinite(err).all()),
                    # Retained as a compatibility alias; unlike the old
                    # pointwise-centred sign score, this compares fitted slopes.
                    "sign_ok": (bool(np.sign(coef[1]) == np.sign(slope_te))
                                if material else None),
                    "pred": pred.tolist(), "obs": obsv.tolist(),
                    "train_values": tr, "heldout_values": te,
                })
        log.info("held-out sensitivity for %s: %d (intent, regime) fits", prm,
                 sum(1 for r in rows if r["param"] == prm))
    return rows


# ═══════════════════════════════════════════════════════════════════════
# PAIRED-CLONE GROUND TRUTH  --  replaces measure_ground_truth_beta
#
# WHY THE OLD HARNESS COULD NOT VALIDATE ANYTHING
# It ran an always-ON world against an always-OFF world from a FRESH state,
# and averaged (g1 - g0) over n_meas epochs.  Because g1 of one epoch is g0
# of the next, that sum TELESCOPES:
#       mean_k (g1_k - g0_k)  ==  (g_final - g_initial) / n_meas
# So the "ground truth" was
#   (a) a TRANSIENT quantity -- the controller's approach to its setpoint --
#       while the regression estimates a STEADY-STATE marginal effect, and
#   (b) scaled by 1/n_meas, an arbitrary window length.
# Two different estimands.  Comparing them produced 11-15% sign accuracy,
# large estimates where truth was ~0, and 41% CI coverage: exactly the
# signature of a mis-specified comparison rather than a bad estimator.
#
# WHAT THIS DOES INSTEAD
# Settle to STEADY STATE with every xApp free, then for each measurement
# epoch CLONE the entire simulator (RAN + RNG + xApp state) twice and run one
# epoch with claim j eligible and one without.  The difference in Delta g
# between the two clones IS the regression's estimand -- the marginal effect
# of eligibility in THIS epoch, at THIS operating point -- measured
# counterfactually with common random numbers.
# ═══════════════════════════════════════════════════════════════════════
def measure_ground_truth_beta_paired(ran_factory, cfg, intents, claims, xapps,
                                     n_reps, n_epochs, log, claim_subset=None):
    import copy
    spe = cfg["ran"]["slots_per_epoch"]
    settle = max(cfg["validation"]["settle_epochs"], 20)
    base = cfg["validation"]["base_seed"]
    js = list(claim_subset or claims)
    acc = {i: {j: [] for j in js} for i in intents}

    def free_epoch(ran, xa_map):
        """one epoch with every xApp acting -- the B0 dynamics"""
        kpm = ran.step(spe)
        for j2, c2 in claims.items():
            x2 = xa_map.get(c2.xapp)
            if x2 is None:
                continue
            nu = x2.propose(kpm, ran.current_controls(), None)
            if nu is not None:
                ran.apply(c2.param, float(nu))
        ran.step(spe)

    for rep in range(n_reps):
        ran = ran_factory(base + rep * 97)
        xa_map = copy.deepcopy(xapps)
        for _ in range(settle):                       # reach STEADY STATE
            free_epoch(ran, xa_map)

        for _ in range(n_epochs):
            for j in js:
                c = claims[j]
                res = {}
                for z in (0, 1):
                    # CLONE everything: RAN, its RNG, and the xApp state.
                    # Without cloning the xApp, the ON arm advances its duty
                    # cycle and the OFF arm does not, so the two arms would
                    # differ by more than the treatment.
                    r2 = copy.deepcopy(ran)
                    x2 = copy.deepcopy(xa_map)
                    kpm = r2.step(spe)
                    g0 = {i: margin(intents[i], kpm) for i in intents}
                    if z:
                        xa = x2.get(c.xapp)
                        if xa is not None:
                            nu = xa.propose(kpm, r2.current_controls(), None)
                            if nu is not None:
                                r2.apply(c.param, float(nu))
                    kpm2 = r2.step(spe)
                    res[z] = {i: margin(intents[i], kpm2) - g0[i] for i in intents}
                for i in intents:
                    acc[i][j].append(res[1][i] - res[0][i])
            free_epoch(ran, xa_map)                   # advance the shared state

    out = {i: {j: float(np.mean(acc[i][j])) if acc[i][j] else 0.0 for j in js}
           for i in intents}
    log.info("paired-clone ground truth: %d claims x %d reps x %d epochs",
             len(js), n_reps, n_epochs)
    return out


def stratified_recovery(beta_hat, beta_true, se=None, delta=None):
    """
    TASK-APPROPRIATE VALIDATION METRICS.

    Sign accuracy over ALL claim-intent pairs is close to meaningless when
    most true effects are ~0: asking whether the sign of +0.0001 was recovered
    is not a real question.  Split the pairs instead:

      NULL     |beta_true| <  delta   -> report a FALSE-POSITIVE rate
      MATERIAL |beta_true| >= delta   -> report sign accuracy, MAE, RMSE,
                                         correlation and CI coverage

    delta defaults to the 75th percentile of |beta_true|, so "material" means
    the effects large enough that a scheduler's ranking could turn on them.
    """
    pairs = [(i, j, beta_true[i][j], beta_hat.get(i, {}).get(j, 0.0))
             for i in beta_true for j in beta_true[i]]
    mags = np.array([abs(t) for _, _, t, _ in pairs])
    if delta is None:
        delta = float(np.percentile(mags, 75)) if len(mags) else 0.0
        delta = max(delta, 1e-6)
    null = [(i, j, t, h) for i, j, t, h in pairs if abs(t) < delta]
    matl = [(i, j, t, h) for i, j, t, h in pairs if abs(t) >= delta]

    def blk(rows, tag):
        if not rows:
            return {}
        t = np.array([r[2] for r in rows]); h = np.array([r[3] for r in rows])
        e = np.abs(h - t)
        d = {f"{tag}_n": len(rows), f"{tag}_MAE": float(e.mean()),
             f"{tag}_RMSE": float(np.sqrt((e ** 2).mean()))}
        if tag == "material":
            d["material_sign_accuracy"] = float(np.mean(np.sign(h) == np.sign(t)))
            if np.std(t) > 1e-12 and np.std(h) > 1e-12:
                d["material_correlation"] = float(np.corrcoef(t, h)[0, 1])
            if se is not None:
                cov = [abs(r[3] - r[2]) <= 1.96 * se.get(r[0], {}).get(r[1], np.inf)
                       for r in rows]
                d["material_CI95_coverage"] = float(np.mean(cov))
        else:
            # a false positive is a NULL effect the estimator calls material
            d["null_false_positive_rate"] = float(
                np.mean([abs(r[3]) >= delta for r in rows]))
        return d

    out = {"delta": delta, "n_pairs": len(pairs)}
    out.update(blk(matl, "material")); out.update(blk(null, "null"))
    out["per_pair"] = [{"intent": i, "claim": j, "beta_true": t, "beta_hat": h,
                        "material": bool(abs(t) >= delta)} for i, j, t, h in pairs]
    return out


def measure_ground_truth_paired(exp, cfg, n_probes, log, pairs=None,
                                return_details=False):
    """
    beta, eligibility-pair gamma, and write-dose eta ground truth as PAIRED
    ONE-EPOCH COUNTERFACTUALS drawn from the OPERATIONAL state distribution.

    At a randomly chosen live epoch:
        clone the RAN (RNG state included)
        in arm A  claim j is eligible;  in arm B it is not
        EVERY OTHER claim keeps the SAME eligibility in both arms
        advance ONE epoch in each and difference

    beta_{j,i} = E[ dg_i | j eligible ] - E[ dg_i | j not eligible ]

    versus the OLD harness, which ran one claim for 15 consecutive epochs
    from a pristine state with every other knob frozen.  That measured a
    cumulative trajectory from a state the system never visits, which is why
    it disagreed with the regression no matter how long the regression
    trained.
    """
    from .types import Write, Outcome
    from .estimation.sensitivity import regime_key

    pre = cfg["ran"].get("pre_slots", cfg["ran"]["slots_per_epoch"])
    post = cfg["ran"].get("post_slots", cfg["ran"]["slots_per_epoch"])
    rng = np.random.default_rng(cfg["validation"]["base_seed"])
    claims, intents = exp.claims, exp.intents
    acc = {i: {j: [] for j in claims} for i in intents}
    gacc = {i: {} for i in intents}
    eacc = {i: {} for i in intents}
    pairs = pairs or []
    gamma_supported = {}
    eta_support_counts = {tuple(pr): 0 for pr in pairs}
    eta_dose_products = {tuple(pr): [] for pr in pairs}
    draw_attempts = []

    def snapshot(source):
        """Only the mutable state required to execute a training epoch."""
        return {
            "ran": source.ran.clone(),
            "xapps": copy.deepcopy(source.xapps),
            "tracker": copy.deepcopy(source.tracker),
            "risk": copy.deepcopy(source.risk),
            "inner": copy.deepcopy(source.inner),
            "rng": copy.deepcopy(source.rng),
            "epoch": int(source.epoch),
        }

    def clone_state(state):
        return copy.deepcopy(state)

    def one_epoch(state, elig):
        """Execute exactly one ``train_random`` epoch with forced eligibility.

        This includes the margin tracker, risk calculation, xApp inactivity,
        grid/slew/envelope logic in the inner arbiter, and the PRE/POST timing.
        The former validator applied raw proposals directly and therefore did
        not measure the estimand used by the runtime EffectEstimator.
        """
        state["epoch"] += 1
        ran, tracker = state["ran"], state["tracker"]
        risk, inner = state["risk"], state["inner"]
        kpm = ran.step(pre)
        g0 = tracker.update(kpm)
        if hasattr(risk, "observe"):
            risk.observe(state["epoch"], g0,
                         {i: tracker.fulfilment(i) for i in intents},
                         kpm, ran.current_controls())
        p_hat = risk.predict(tracker)
        regime = regime_key(kpm, cfg)
        controls = ran.current_controls()
        applied = {}
        for j in sorted(elig):
            c = claims[j]
            xa = state["xapps"].get(c.xapp)
            if xa is None:
                continue
            nu = xa.propose(kpm, controls, state["rng"])
            if nu is None:
                continue
            old = controls.get(c.param, nu)
            wr = Write(jid=j, param=c.param, nu_req=float(nu),
                       nu_old=float(old), epoch=state["epoch"], slot=0)
            d = inner.decide(wr, regime, g0, p_hat, ran)
            if d.outcome != Outcome.REJECT:
                ran.apply(c.param, d.nu_star)
                controls[c.param] = float(d.nu_star)
                applied[j] = float(d.nu_star - old)
        kpm2 = ran.step(post)
        g1 = tracker.update(kpm2)
        return ({i: float(g1[i] - g0[i]) for i in intents}, applied)

    def background(exclude=(), additions=(set(),), max_draws=1000000):
        """Random background for which every requested counterfactual exists."""
        excluded = set(exclude)
        additions = [set(a) for a in additions]
        if any(not exp.mask.is_admissible(a) for a in additions):
            return None
        optional = [j for j in sorted(claims) if j not in excluded]
        for attempt in range(1, max_draws + 1):
            bg = {j for j in optional if rng.random() < 0.5}
            if all(exp.mask.is_admissible(bg | a) for a in additions):
                draw_attempts.append(attempt)
                return bg
        raise RuntimeError("counterfactual support draw failed; check C1/C2 mask")

    live = snapshot(exp)

    for probe in range(n_probes):
        for j in sorted(claims):
            base = background(exclude={j}, additions=(set(), {j}))
            if base is None:
                continue
            d_on, _ = one_epoch(clone_state(live), base | {j})
            d_off, _ = one_epoch(clone_state(live), base)
            for i in intents:
                acc[i][j].append(d_on[i] - d_off[i])

        for (ja, jb) in pairs:
            if ja not in claims or jb not in claims:
                continue
            pr = (ja, jb)
            arm_add = (set(), {ja}, {jb}, {ja, jb})
            base = background(exclude={ja, jb}, additions=arm_add)
            gamma_supported[pr] = base is not None
            if base is None:
                continue
            arms, doses = {}, {}
            for za in (0, 1):
                for zb in (0, 1):
                    e = set(base) - {ja, jb}
                    if za: e.add(ja)
                    if zb: e.add(jb)
                    arms[(za, zb)], doses[(za, zb)] = one_epoch(
                        clone_state(live), e)
            did = {}
            for i in intents:
                did[i] = ((arms[(1, 1)][i] - arms[(1, 0)][i])
                          - (arms[(0, 1)][i] - arms[(0, 0)][i]))
                gacc[i].setdefault((ja, jb), []).append(did[i])

            # Eta is not gamma with a different name.  It is the coefficient
            # of dnu_a*dnu_b.  A 2x2 eligibility DiD identifies that quantity
            # only when each claim supplies the SAME non-zero applied dose in
            # its singleton and joint arm.  When that condition holds, divide
            # the DiD by the dose product.  Otherwise the probe supports gamma
            # but not eta and is honestly omitted from eta accuracy.
            da10 = float(doses[(1, 0)].get(ja, 0.0))
            da11 = float(doses[(1, 1)].get(ja, 0.0))
            db01 = float(doses[(0, 1)].get(jb, 0.0))
            db11 = float(doses[(1, 1)].get(jb, 0.0))
            scale = max(abs(da10), abs(da11), abs(db01), abs(db11), 1.0)
            tol = 1e-7 * scale
            dose_invariant = (abs(da10 - da11) <= tol
                              and abs(db01 - db11) <= tol)
            da = 0.5 * (da10 + da11)
            db = 0.5 * (db01 + db11)
            product = da * db
            if dose_invariant and abs(da) > tol and abs(db) > tol:
                eta_support_counts[pr] = eta_support_counts.get(pr, 0) + 1
                eta_dose_products.setdefault(pr, []).append(float(product))
                for i in intents:
                    eacc[i].setdefault(pr, []).append(float(did[i] / product))

        # Move the shared system through the SAME admissible, xApp-driven
        # dynamics used for training.  Merely stepping the RAN with all knobs
        # frozen sampled a state distribution the policies never visit.
        for _ in range(int(rng.integers(1, 6))):
            e = background()
            one_epoch(live, e)
        if (probe + 1) % 10 == 0:
            log.info("  paired ground truth: %d / %d probes", probe + 1, n_probes)

    beta = {i: {j: float(np.mean(v)) if v else 0.0 for j, v in acc[i].items()}
            for i in intents}
    # IS BETA A CONSTANT?  across-probe dispersion of the per-epoch effect.
    # |mean| / sd  near 0 means the effect changes sign from state to state,
    # in which case a scalar beta is not a well-posed quantity.
    beta_stab = {i: {j: (abs(float(np.mean(v))) / float(np.std(v, ddof=1))
                         if len(v) > 1 and np.std(v, ddof=1) > 1e-12
                         else (float("inf") if v and abs(np.mean(v)) > 1e-12 else 0.0))
                     for j, v in acc[i].items()} for i in intents}
    beta_se = {i: {j: float(np.std(v, ddof=1) / np.sqrt(len(v)))
                   if len(v) > 1 else 0.0
                   for j, v in acc[i].items()} for i in intents}
    gamma = {i: {pr: float(np.mean(v)) for pr, v in gacc[i].items()} for i in intents}
    gamma_se = {i: {pr: (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                          if len(v) > 1 else 0.0)
                    for pr, v in gacc[i].items()} for i in intents}
    gamma_stab = {
        i: {pr: (abs(float(np.mean(v))) / float(np.std(v, ddof=1))
                 if len(v) > 1 and np.std(v, ddof=1) > 1e-12
                 else (float("inf") if v and abs(np.mean(v)) > 1e-12 else 0.0))
            for pr, v in gacc[i].items()}
        for i in intents}
    eta = {i: {pr: float(np.mean(v)) for pr, v in eacc[i].items()}
           for i in intents}
    eta_se = {i: {pr: (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                        if len(v) > 1 else 0.0)
                  for pr, v in eacc[i].items()} for i in intents}
    eta_stab = {
        i: {pr: (abs(float(np.mean(v))) / float(np.std(v, ddof=1))
                 if len(v) > 1 and np.std(v, ddof=1) > 1e-12
                 else (float("inf") if v and abs(np.mean(v)) > 1e-12 else 0.0))
            for pr, v in eacc[i].items()}
        for i in intents}
    opposite = {}
    for i in intents:
        opposite[i] = {}
        for j, v in acc[i].items():
            m = float(np.mean(v)) if v else 0.0
            opposite[i][j] = (float(np.mean(np.asarray(v) * m < 0))
                              if v and abs(m) > 1e-12 else None)
    details = {
        "n_probes": int(n_probes),
        "beta_opposite_sign_fraction": opposite,
        "gamma_se": gamma_se,
        "gamma_stability": gamma_stab,
        "gamma_supported": {f"{a}|{b}": bool(ok)
                            for (a, b), ok in gamma_supported.items()},
        "eta_true": eta,
        "eta_se": eta_se,
        "eta_stability": eta_stab,
        "eta_supported": {f"{a}|{b}": bool(eta_support_counts.get((a, b), 0))
                          for (a, b) in pairs},
        "eta_support_counts": {f"{a}|{b}": int(eta_support_counts.get((a, b), 0))
                               for (a, b) in pairs},
        "eta_dose_product_std": {
            f"{a}|{b}": (float(np.std(eta_dose_products.get((a, b), [])))
                           if eta_dose_products.get((a, b)) else None)
            for (a, b) in pairs},
        "admissible_draw_attempts": {
            "n": len(draw_attempts),
            "median": float(np.median(draw_attempts)) if draw_attempts else None,
            "mean": float(np.mean(draw_attempts)) if draw_attempts else None,
            "max": int(max(draw_attempts)) if draw_attempts else None,
        },
    }
    result = (beta, beta_se, gamma, beta_stab)
    return result + (details,) if return_details else result


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _trim(mat, frac=0.05):
    m = sorted(mat, key=lambda r: abs(r["beta_true"]))
    k = int(len(m) * frac)
    return m[k:len(m) - k] if len(m) - 2 * k >= 3 else m


def _trimmed_mae(mat):
    t = _trim(mat)
    return float(np.mean([r["abs_err"] for r in t])) if t else float("nan")


def _trimmed_sign(mat):
    t = _trim(mat)
    return float(np.mean([(r["beta_true"] > 0) == (r["beta_hat"] > 0)
                          for r in t])) if t else float("nan")


def recovery_metrics_material(beta_hat, beta_true, se=None, delta=None,
                              beta_true_se=None):
    """
    Split NULL from MATERIAL effects before scoring anything.

    Sign accuracy over all pairs is close to meaningless when most true
    effects are ~0: asking whether the sign of +0.0001 was recovered scores
    a coin flip and drags the headline number down.  What matters is

        MATERIAL  |beta_true| >= delta   -> did we get the DIRECTION right?
        NULL      |beta_true| <  delta   -> did we falsely claim an effect?

    With counterfactual standard errors, materiality is pair-specific:
    |beta_true| >= 2*SE_true (plus a tiny numerical floor).  A caller-supplied
    ``delta`` deliberately overrides that rule with one global threshold.
    """
    rows = []
    for i in beta_true:
        for j in beta_true[i]:
            bt = beta_true[i][j]; bh = beta_hat.get(i, {}).get(j, 0.0)
            rows.append({"intent": i, "claim": j, "beta_true": bt, "beta_hat": bh,
                         "abs_err": abs(bh - bt),
                         "se_hat": (se or {}).get(i, {}).get(j, float("nan")),
                         "se_true": (beta_true_se or {}).get(i, {}).get(j, 0.0)})
    # ---- choosing materiality ------------------------------------------
    # Use each pair's OWN counterfactual precision.  The former code used
    # 2*median(SE) as one global delta, so a noisy pair could be called
    # material while a precisely measured smaller effect was called null.
    #
    #   MATERIAL(i,j) <=> |truth(i,j)| >= max(2*SE_truth(i,j), numeric floor)
    #
    # The tiny scale-relative floor prevents deterministic floating-point
    # crumbs around zero from being labelled scientific effects.
    mags = sorted(abs(r["beta_true"]) for r in rows)
    numeric_floor = max(0.01 * (max(mags) if mags else 0.0), 1e-9)
    for r in rows:
        if delta is not None:
            threshold = float(delta)
        elif beta_true_se:
            threshold = max(2.0 * max(r["se_true"], 0.0), numeric_floor)
        else:
            threshold = max(0.05 * (max(mags) if mags else 0.0), 1e-9)
        r["material_threshold"] = float(threshold)
        r["material"] = bool(abs(r["beta_true"]) >= threshold)
    mat = [r for r in rows if r["material"]]
    nul = [r for r in rows if not r["material"]]
    # a NULL pair is a false positive only if the estimator claims an effect
    # BIGGER than the smallest one the ground truth could resolve

    def _sign_ok(r):
        return (r["beta_true"] > 0) == (r["beta_hat"] > 0)

    thresholds = [r["material_threshold"] for r in rows]
    display_delta = float(np.median(thresholds)) if thresholds else 0.0
    out = {"delta": display_delta,
           "material_rule": ("global_delta" if delta is not None
                             else "pair_specific_2se_truth" if beta_true_se
                             else "scale_relative"),
           "numeric_floor": numeric_floor,
           "n_pairs": len(rows),
           "n_material": len(mat), "n_null": len(nul)}
    if mat:
        e = np.array([r["abs_err"] for r in mat])
        bt = np.array([r["beta_true"] for r in mat])
        bh = np.array([r["beta_hat"] for r in mat])
        out.update({
            "material_sign_accuracy": float(np.mean([_sign_ok(r) for r in mat])),
            "material_MAE": float(e.mean()),
            "material_RMSE": float(np.sqrt((e ** 2).mean())),
            "material_correlation": (float(np.corrcoef(bt, bh)[0, 1])
                                     if np.std(bt) > 0 and np.std(bh) > 0 else float("nan")),
            # SPEARMAN: rank correlation, insensitive to the heavy tail.  If
            # Pearson is negative but Spearman is positive, a few extreme
            # pairs are driving Pearson and the bulk relationship is fine.
            "material_spearman": _spearman(bt, bh),
            # metrics with the top and bottom 5% of |beta_true| trimmed, to
            # show whether the outliers ARE the story
            "material_MAE_trim5": _trimmed_mae(mat),
            "material_sign_acc_trim5": _trimmed_sign(mat),
            "material_ci_coverage": float(np.mean(
                [abs(r["beta_hat"] - r["beta_true"]) <= 1.96 * r["se_hat"]
                 for r in mat if np.isfinite(r["se_hat"])])) if any(
                     np.isfinite(r["se_hat"]) for r in mat) else None,
        })
    if nul:
        # a FALSE POSITIVE: the estimator claims a material effect where the
        # counterfactual says there is none
        # SIGNIFICANT false positive: |beta_hat| > 2 SE(beta_hat) where the
        # counterfactual says null.  Threshold-free, so it does not drift as
        # the ground truth becomes more precise.
        sig = [r for r in nul if np.isfinite(r["se_hat"]) and r["se_hat"] > 0]
        out["null_false_positive_rate"] = float(np.mean(
            [abs(r["beta_hat"]) > 2 * r["se_hat"] for r in sig])) if sig else None
        out["null_fp_rate_vs_delta"] = float(np.mean(
            [abs(r["beta_hat"]) >= r["material_threshold"] for r in nul]))
        out["null_mean_abs_hat"] = float(np.mean([abs(r["beta_hat"]) for r in nul]))
    out["per_pair"] = rows
    return out
