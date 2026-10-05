#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np

ROOT = Path("runs_w3v_confirmation")
PROPOSED = "W3R-V"

COMPARATORS = [
    "QACM-style",
    "QACM-contract",
    "B3-contract",
]


def signflip(d, n=99999, seed=7):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    s = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(d)))
    return float(
        (1 + np.sum(np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-15))
        / (n + 1)
    )


def boot(d, n=20000, seed=3):
    d = np.asarray(d, float)
    b = np.random.default_rng(seed).choice(
        d, (n, len(d))
    ).mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def holm(p):
    out = {}
    run = 0.0
    ordered = sorted(p, key=p.get)

    for r, x in enumerate(ordered):
        run = max(run, min(1.0, (len(ordered) - r) * p[x]))
        out[x] = run

    return out


# Load all result.json files
R = {}

for f in ROOT.glob("s*/*/result.json"):
    r = json.loads(f.read_text())
    R.setdefault(r["method"], {})[r["seed"]] = r

seeds = sorted(R[PROPOSED])

print(f"Confirmation seeds: {len(seeds)}")
print(f"Seed range: {seeds[0]}-{seeds[-1]}")
print()

# Discover tenants from the proposed method
first = R[PROPOSED][seeds[0]]
tenants = list(first["tenant_fulfillment"])

print("Tenants:", tenants)
print()

for tenant in tenants:

    print("=" * 78)
    print(f"TENANT: {tenant}")
    print("=" * 78)

    P = {}
    stats = {}

    proposed = np.array([
        R[PROPOSED][s]["tenant_fulfillment"][tenant]
        for s in seeds
    ], dtype=float)

    for comp in COMPARATORS:

        comparator = np.array([
            R[comp][s]["tenant_fulfillment"][tenant]
            for s in seeds
        ], dtype=float)

        # Percentage-point paired differences
        d = 100.0 * (proposed - comparator)

        p = signflip(d)
        lo, hi = boot(d)

        P[comp] = p
        stats[comp] = {
            "proposed": proposed.mean(),
            "comparator": comparator.mean(),
            "delta": d.mean(),
            "lo": lo,
            "hi": hi,
            "wins": int((d > 1e-12).sum()),
            "losses": int((d < -1e-12).sum()),
            "raw_p": p,
        }

    adjusted = holm(P)

    for comp in COMPARATORS:
        x = stats[comp]
        hp = adjusted[comp]

        print(f"\nvs {comp}")
        print(f"  PACT mean       : {x['proposed']:.4f}")
        print(f"  Comparator mean : {x['comparator']:.4f}")
        print(f"  Delta           : {x['delta']:+.3f} pp")
        print(f"  95% CI          : [{x['lo']:+.3f}, {x['hi']:+.3f}] pp")
        print(f"  Wins/Losses     : {x['wins']}/{x['losses']}")
        print(f"  raw p           : {x['raw_p']:.6f}")
        print(f"  Holm p          : {hp:.6f}")

        if hp < 0.05:
            direction = "higher" if x["delta"] > 0 else "lower"
            print(f"  RESULT          : SIGNIFICANT ({direction})")
        else:
            print("  RESULT          : NOT SIGNIFICANT")

    print()
