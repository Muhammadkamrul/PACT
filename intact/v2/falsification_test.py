#!/usr/bin/env python3
"""
scripts/falsification_test.py
=============================
THE experiment that separates two hypotheses v1 could not distinguish.

v1 measured gamma == 0 on every well-identified coupled pair and could not
say which of these was true:

    H1  the ESTIMATOR cannot recover an interaction
    H2  the SIMULATOR contains no interaction to recover

Both produce the identical observable.  The only way to tell them apart is
to plant an interaction of KNOWN magnitude in a synthetic world and ask the
production estimator to find it.

    recovers planted gamma  ->  H1 is false; a null on the operational
                                scenario means the RAN really is separable
    fails on planted gamma  ->  H1 is true; the estimator is broken and no
                                operational result may be interpreted

Also verifies the NOISE FLOOR requirement before estimating anything: a
planted interaction must produce an effect of at least

    3 * sigma  =  3 * 0.015  =  0.045

at typical dose.  A pair below that is unmeasurable BY CONSTRUCTION, and
reporting "gamma not recovered" for it would be meaningless.

USAGE
    python scripts/falsification_test.py --out runs_falsify
    python scripts/falsification_test.py --out runs_falsify --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from intact.v2.effects_v2 import RegimeDoseEstimator
from intact.v2.ran_v2 import (SIGMA_MARGIN, NOISE_FLOOR_MULTIPLE,
                              size_gamma_for_detectability)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})

REGIMES = ("low", "mid", "high")


def synth_world(n_rows, gamma_by_regime, s_by_regime, doses,
                sigma=SIGMA_MARGIN, seed=7, n_ctx=4):
    """Balanced four-arm factorial in a world with KNOWN coefficients."""
    rng = np.random.default_rng(seed)
    pa, pb = "txpower", "quota"
    arms = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
    rows = []
    for k in range(n_rows):
        r = REGIMES[k % 3]
        ma, mb = arms[k % len(arms)]
        dnu = {pa: ma * doses[pa], pb: mb * doses[pb]}
        ctx = rng.normal(0.0, 1.0, n_ctx)
        dg = (s_by_regime[r][pa] * dnu[pa]
              + s_by_regime[r][pb] * dnu[pb]
              + gamma_by_regime[r] * dnu[pa] * dnu[pb]
              + 0.01 * ctx[0]
              + rng.normal(0.0, sigma))
        rows.append((r, dnu, ctx, {"i1": dg}))
    return rows


def fit_and_score(rows, gamma_by_regime, n_ctx=4):
    cfg = {"estimation": {"ridge": 1e-3, "forgetting_factor": 1.0},
           "v2": {"min_cell_obs": 20}}
    est = RegimeDoseEstimator(["txpower", "quota"], ["i1"],
                              [("txpower", "quota")], n_ctx, cfg)
    for r, dnu, ctx, dg in rows:
        est.observe(r, dnu, ctx, dg)
    est.fit()
    out = []
    for r in REGIMES:
        cell = est.gamma[r]["i1"][("txpower", "quota")]
        true = gamma_by_regime[r]
        out.append({"regime": r, "gamma_true": true,
                    "gamma_hat": cell.value, "se": cell.se,
                    "n_obs": cell.n_obs, "identified": cell.identified,
                    "abs_error": abs(cell.value - true),
                    "rel_error": abs(cell.value - true) / max(abs(true), 1e-12),
                    "sign_ok": bool(np.sign(cell.value) == np.sign(true)),
                    "covers": bool(abs(cell.value - true) <= 1.96 * cell.se)
                    if np.isfinite(cell.se) else False})
    return est, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_falsify")
    ap.add_argument("--n-rows", type=int, default=3000)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep gamma magnitude across the noise floor")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    doses = {"txpower": -6.0, "quota": 9.0}
    s_by_regime = {"low":  {"txpower": -0.008, "quota": 0.012},
                   "mid":  {"txpower": -0.013, "quota": 0.016},
                   "high": {"txpower": -0.021, "quota": 0.022}}

    # size gamma so the MID regime sits exactly at 3 sigma
    g_mid = size_gamma_for_detectability(doses["txpower"], doses["quota"], 1.0)
    gamma_by_regime = {"low": 0.3 * g_mid, "mid": g_mid, "high": 1.8 * g_mid}

    print("=" * 70)
    print("FALSIFICATION TEST -- can the estimator recover a KNOWN gamma?")
    print("=" * 70)
    print(f"  noise sigma            {SIGMA_MARGIN}")
    print(f"  required effect (3s)   {NOISE_FLOOR_MULTIPLE * SIGMA_MARGIN:.4f}")
    print(f"  doses                  {doses}")
    print()
    print(f"  {'regime':>7} {'gamma_true':>12} {'typical effect':>15} {'detectable':>11}")
    floor_rows = []
    for r in REGIMES:
        eff = abs(gamma_by_regime[r] * doses["txpower"] * doses["quota"])
        det = eff >= NOISE_FLOOR_MULTIPLE * SIGMA_MARGIN * (1.0 - 1e-9)
        floor_rows.append({"regime": r, "gamma_true": gamma_by_regime[r],
                           "typical_effect": eff, "detectable": bool(det)})
        print(f"  {r:>7} {gamma_by_regime[r]:>12.6f} {eff:>15.4f} {str(det):>11}")
    print()

    est, scores = fit_and_score(synth_world(args.n_rows, gamma_by_regime,
                                            s_by_regime, doses,
                                            seed=args.seed), gamma_by_regime)
    print(f"  {'regime':>7} {'true':>11} {'estimate':>11} {'rel err':>9} "
          f"{'sign':>6} {'covers':>7} {'n':>6}")
    for s in scores:
        print(f"  {s['regime']:>7} {s['gamma_true']:>11.6f} "
              f"{s['gamma_hat']:>11.6f} {s['rel_error']:>9.3f} "
              f"{str(s['sign_ok']):>6} {str(s['covers']):>7} {s['n_obs']:>6}")

    verdict_ok = all(s["sign_ok"] and s["rel_error"] < 0.35 for s in scores)
    print()
    print("  VERDICT: " + (
        "ESTIMATOR IS SOUND.  A gamma==0 result on the operational scenario\n"
        "           means the simulated RAN is genuinely separable (H2), not that\n"
        "           the estimator failed."
        if verdict_ok else
        "ESTIMATOR FAILED on a planted, above-noise-floor interaction.\n"
        "           No operational gamma result may be interpreted until fixed (H1)."))

    result = {"noise_floor": floor_rows, "recovery": scores,
              "estimator_sound": bool(verdict_ok),
              "sigma": SIGMA_MARGIN, "doses": doses}

    # ---------------- optional magnitude sweep ------------------------
    sweep = []
    if args.sweep:
        print("\n  sweeping gamma magnitude across the noise floor ...")
        for mult in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            gb = {r: mult * gamma_by_regime[r] for r in REGIMES}
            _, sc = fit_and_score(synth_world(args.n_rows, gb, s_by_regime,
                                              doses, seed=args.seed), gb)
            eff = abs(gb["mid"] * doses["txpower"] * doses["quota"])
            rec = float(np.mean([s["rel_error"] for s in sc]))
            sweep.append({"multiple_of_3sigma": mult, "effect": eff,
                          "mean_rel_error": rec,
                          "sign_ok_frac": float(np.mean([s["sign_ok"] for s in sc]))})
            print(f"    x{mult:<5} effect={eff:.4f}  mean rel err={rec:.3f}")
        result["sweep"] = sweep

    (out / "falsification.json").write_text(json.dumps(result, indent=2))

    # ---------------- figure ------------------------------------------
    fig, axs = plt.subplots(1, 3 if sweep else 2, figsize=(13 if sweep else 9, 3.4))
    t = [s["gamma_true"] for s in scores]
    h = [s["gamma_hat"] for s in scores]
    axs[0].scatter(t, h, s=60, color="#3B1F6B", zorder=3)
    lim = [min(t + h) * 1.15, max(t + h) * 1.15]
    axs[0].plot(lim, lim, "--", color="#333", lw=.9)
    for s in scores:
        axs[0].annotate(s["regime"], (s["gamma_true"], s["gamma_hat"]),
                        textcoords="offset points", xytext=(6, -3), fontsize=7)
    axs[0].set_xlabel("planted gamma"); axs[0].set_ylabel("recovered gamma")
    axs[0].set_title("recovery of a KNOWN interaction")

    r_ = [s["regime"] for s in scores]
    e_ = [s["rel_error"] for s in scores]
    axs[1].bar(r_, e_, color=["#2E7D6B" if v < .35 else "#B03A3A" for v in e_])
    axs[1].axhline(0.35, color="#B03A3A", ls="--", lw=.9)
    axs[1].set_ylabel("relative error"); axs[1].set_title("error by regime")

    if sweep:
        m = [s["multiple_of_3sigma"] for s in sweep]
        er = [s["mean_rel_error"] for s in sweep]
        axs[2].semilogx(m, er, "o-", color="#3B1F6B")
        axs[2].axvline(1.0, color="#B03A3A", ls="--", lw=.9)
        axs[2].set_xlabel("gamma as multiple of the 3-sigma floor")
        axs[2].set_ylabel("mean relative error")
        axs[2].set_title("detectability threshold")
    fig.tight_layout()
    fig.savefig(out / "fig_falsification.png", dpi=200)
    plt.close(fig)

    print(f"\n  wrote falsification.json, fig_falsification.png -> {out}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
