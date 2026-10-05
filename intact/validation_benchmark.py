"""Controlled positive/null benchmarks for the production effect estimators.

This module is deliberately separate from :mod:`intact.validate`.

``intact.validate`` asks an operational question: do claim eligibility and
joint writes have stable effects in the end-to-end RAN/controller workload?
This module asks an implementation question: when a data-generating process
contains a predeclared mixture of exact zeros, positive effects and negative
effects, can the estimators recover them on independent data?

Keeping those questions separate prevents a circular validation in which the
operational simulator is modified to contain the same equation being tested.
The controlled benchmark uses the real ``EffectEstimator`` and
``WriteEffectEstimator`` classes; it does not maintain a second estimator.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .estimation.effects import EffectEstimator, WriteEffectEstimator


Pair = Tuple[str, str]


@dataclass(frozen=True)
class BenchmarkDimensions:
    """Names and pair design shared by every controlled repetition."""

    claims: Tuple[str, ...]
    intents: Tuple[str, ...]
    pairs: Tuple[Pair, ...]
    params: Tuple[str, ...]
    regimes: Tuple[str, ...]


def make_dimensions(cfg: Mapping) -> BenchmarkDimensions:
    n_claims = int(cfg.get("n_claims", 8))
    n_intents = int(cfg.get("n_intents", 4))
    n_params = int(cfg.get("n_params", 6))
    n_regimes = int(cfg.get("n_regimes", 3))
    if n_claims < 4 or n_claims % 2:
        raise ValueError("controlled benchmark n_claims must be even and >= 4")
    claims = tuple(f"j{k + 1}" for k in range(n_claims))
    intents = tuple(f"i{k + 1}" for k in range(n_intents))
    params = tuple(f"p{k + 1}" for k in range(n_params))
    regimes = tuple(f"load{k}" for k in range(n_regimes))
    pairs = tuple((claims[k], claims[k + 1])
                  for k in range(0, n_claims, 2))
    return BenchmarkDimensions(claims, intents, pairs, params, regimes)


def _mixed_values(labels: Sequence, zero_fraction: float,
                  lo: float, hi: float, rng: np.random.Generator) -> Dict:
    """Return an exactly controlled zero/positive/negative coefficient mix."""
    labels = list(labels)
    if not 0.0 < zero_fraction < 1.0:
        raise ValueError("zero_fraction must lie strictly between zero and one")
    n_zero = int(round(len(labels) * zero_fraction))
    n_signal = len(labels) - n_zero
    if n_signal < 2:
        raise ValueError("benchmark needs at least two non-zero coefficients")
    order = list(rng.permutation(len(labels)))
    zero_idx = set(order[:n_zero])
    magnitudes = np.linspace(float(lo), float(hi), n_signal)
    rng.shuffle(magnitudes)
    out, q = {}, 0
    for idx, label in enumerate(labels):
        if idx in zero_idx:
            out[label] = 0.0
        else:
            sign = -1.0 if q % 2 else 1.0
            out[label] = float(sign * magnitudes[q])
            q += 1
    return out


def make_truth(cfg: Mapping, dims: BenchmarkDimensions) -> Dict:
    """Create one predeclared sparse truth table, fixed across repetitions."""
    rng = np.random.default_rng(int(cfg.get("truth_seed", 20260824)))
    zf = float(cfg.get("zero_fraction", 0.5))
    beta_labels = [(i, j) for i in dims.intents for j in dims.claims]
    gamma_labels = [(i, p) for i in dims.intents for p in dims.pairs]
    eta_labels = [(i, p) for i in dims.intents for p in dims.pairs]
    s_labels = [(r, p, i) for r in dims.regimes
                for p in dims.params for i in dims.intents]
    beta = _mixed_values(beta_labels, zf, *cfg.get("beta_abs_range", [0.05, 0.14]), rng)
    gamma = _mixed_values(gamma_labels, zf, *cfg.get("gamma_abs_range", [0.04, 0.10]), rng)
    eta = _mixed_values(eta_labels, zf, *cfg.get("eta_abs_range", [0.05, 0.14]), rng)
    slopes = _mixed_values(s_labels, zf, *cfg.get("s_abs_range", [0.08, 0.28]), rng)

    # State modifiers are not part of the stable-scalar positive control.
    # They are used only by the explicitly labelled transport stress test.
    # Some marginally-null effects become non-zero under a state shift, and
    # some stable signals change magnitude or direction.
    def modifiers(values: Mapping, scale: float) -> Dict:
        out = {}
        nonzero_scale = max((abs(v) for v in values.values()), default=1.0)
        for k, (label, value) in enumerate(values.items()):
            if value == 0.0:
                out[label] = (scale * nonzero_scale
                              * (-1.0 if k % 2 else 1.0)) if k % 4 == 0 else 0.0
            else:
                out[label] = float(scale * abs(value)
                                   * (-1.0 if k % 3 == 0 else 1.0))
        return out

    return {
        "beta": beta, "gamma": gamma, "eta": eta, "s": slopes,
        "beta_state_modifier": modifiers(beta, float(cfg.get("state_modifier_scale", 1.25))),
        "gamma_state_modifier": modifiers(gamma, float(cfg.get("state_modifier_scale", 1.25))),
        "eta_state_modifier": modifiers(eta, float(cfg.get("state_modifier_scale", 1.25))),
    }


def _state_mean(prob_high: float) -> float:
    """State is coded -1/+1, hence E[state] = 2p(high)-1."""
    return 2.0 * float(prob_high) - 1.0


def _truth_at_mix(base: Mapping, modifier: Mapping, prob_high: float) -> Dict:
    m = _state_mean(prob_high)
    return {k: float(v + m * modifier.get(k, 0.0)) for k, v in base.items()}


def _draw_state(rng: np.random.Generator, prob_high: float) -> float:
    return 1.0 if rng.random() < prob_high else -1.0


def _context_step(rng: np.random.Generator, previous: np.ndarray,
                  state: float) -> np.ndarray:
    innovation = rng.normal(size=len(previous))
    out = 0.55 * previous + np.sqrt(1.0 - 0.55 ** 2) * innovation
    if len(out):
        out[0] += 0.35 * state
    return out


def _effect_value(z: np.ndarray, dims: BenchmarkDimensions,
                  intent: str, beta: Mapping, gamma: Mapping) -> float:
    claim_index = {j: k for k, j in enumerate(dims.claims)}
    ans = sum(beta[(intent, j)] * z[claim_index[j]] for j in dims.claims)
    ans += sum(gamma[(intent, pr)]
               * z[claim_index[pr[0]]] * z[claim_index[pr[1]]]
               for pr in dims.pairs)
    return float(ans)


def _dose_effect(dose: np.ndarray, dims: BenchmarkDimensions,
                 intent: str, main: Mapping, eta: Mapping) -> float:
    claim_index = {j: k for k, j in enumerate(dims.claims)}
    ans = sum(main[(intent, j)] * dose[claim_index[j]] for j in dims.claims)
    ans += sum(eta[(intent, pr)]
               * dose[claim_index[pr[0]]] * dose[claim_index[pr[1]]]
               for pr in dims.pairs)
    return float(ans)


def _r2(observed: Iterable[float], predicted: Iterable[float]) -> float | None:
    y = np.asarray(list(observed), float)
    p = np.asarray(list(predicted), float)
    if len(y) < 2 or np.var(y) <= 1e-15:
        return None
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def _coefficient_record(world: str, repetition: int, model: str,
                        intent: str, treatment: str, truth_train: float,
                        truth_test: float, estimate: float, se: float,
                        modifier: float = 0.0) -> Dict:
    null = bool(abs(truth_test) <= 1e-12)
    finite_se = bool(np.isfinite(se) and se > 0)
    return {
        "world": world, "repetition": int(repetition), "model": model,
        "intent": intent, "treatment": treatment,
        "truth_train_marginal": float(truth_train),
        "truth_test": float(truth_test), "estimate": float(estimate),
        "se_hat": float(se), "state_modifier": float(modifier),
        "truth_class": ("null" if null else "positive" if truth_test > 0 else "negative"),
        "material": not null,
        "sign_ok": (None if null else bool(np.sign(estimate) == np.sign(truth_test))),
        "significant": (bool(abs(estimate) > 1.96 * se) if finite_se else None),
        "ci95_covers": (bool(abs(estimate - truth_test) <= 1.96 * se)
                         if finite_se else None),
        "abs_error": float(abs(estimate - truth_test)),
    }


def _estimator_cfg(base_cfg: Mapping, benchmark_cfg: Mapping) -> Dict:
    cfg = copy.deepcopy(dict(base_cfg))
    est = cfg.setdefault("estimation", {})
    overrides = benchmark_cfg.get("estimation_overrides", {})
    for key, value in overrides.items():
        est[key] = value
    return cfg


def _run_eligibility_world(base_cfg: Mapping, bcfg: Mapping,
                           dims: BenchmarkDimensions, truth: Mapping,
                           repetition: int, world: str) -> Tuple[List[Dict], List[Dict]]:
    seed = int(bcfg.get("data_seed", 73129)) + 10007 * repetition
    rng = np.random.default_rng(seed)
    n_context = int(bcfg.get("n_context", 2))
    ecfg = _estimator_cfg(base_cfg, bcfg)
    estimator = EffectEstimator(list(dims.claims), list(dims.intents),
                                list(dims.pairs), n_context, ecfg)
    noise_sd = float(bcfg.get("eligibility_noise_sd", 0.05))
    n_train = int(bcfg.get("train_rows", 3000))
    n_validation = int(bcfg.get("validation_rows", 1000))
    n_test = int(bcfg.get("test_rows", 2000))
    train_p = float(bcfg.get("train_high_state_probability", 0.5))
    test_p = (float(bcfg.get("shifted_test_high_state_probability", 0.8))
              if world == "state_shift" else train_p)
    modifiers_b = (truth["beta_state_modifier"] if world == "state_shift"
                   else {k: 0.0 for k in truth["beta"]})
    modifiers_g = (truth["gamma_state_modifier"] if world == "state_shift"
                   else {k: 0.0 for k in truth["gamma"]})
    theta = {i: np.asarray([0.025 * (k + 1), -0.018 * (k + 1)])[:n_context]
             for k, i in enumerate(dims.intents)}
    alpha = {i: 0.005 * (k - 1.5) for k, i in enumerate(dims.intents)}
    previous_context = np.zeros(n_context)
    previous_noise = {i: 0.0 for i in dims.intents}

    for _ in range(n_train):
        state = _draw_state(rng, train_p)
        previous_context = _context_step(rng, previous_context, state)
        z = rng.binomial(1, 0.5, len(dims.claims)).astype(float)
        S = {j for j, value in zip(dims.claims, z) if value > 0.5}
        beta_now = {k: truth["beta"][k] + state * modifiers_b[k]
                    for k in truth["beta"]}
        gamma_now = {k: truth["gamma"][k] + state * modifiers_g[k]
                     for k in truth["gamma"]}
        dg = {}
        for i in dims.intents:
            eps = 0.35 * previous_noise[i] + rng.normal(scale=noise_sd)
            previous_noise[i] = eps
            dg[i] = (alpha[i] + _effect_value(z, dims, i, beta_now, gamma_now)
                     + float(theta[i] @ previous_context) + eps)
        estimator.observe(S, previous_context, dg)
    estimator.fit()

    train_beta = _truth_at_mix(truth["beta"], modifiers_b, train_p)
    test_beta = _truth_at_mix(truth["beta"], modifiers_b, test_p)
    train_gamma = _truth_at_mix(truth["gamma"], modifiers_g, train_p)
    test_gamma = _truth_at_mix(truth["gamma"], modifiers_g, test_p)
    records = []
    for i in dims.intents:
        for j in dims.claims:
            key = (i, j)
            records.append(_coefficient_record(
                world, repetition, "beta", i, j, train_beta[key],
                test_beta[key], estimator.beta[i][j], estimator.se_beta[i][j],
                modifiers_b[key]))
        for pr in dims.pairs:
            key = (i, pr)
            records.append(_coefficient_record(
                world, repetition, "gamma", i, f"{pr[0]}|{pr[1]}",
                train_gamma[key], test_gamma[key], estimator.gamma[i][pr],
                estimator.se_gamma[i][pr], modifiers_g[key]))

    split_rows = []
    for split, n_rows, prob_high in (
            ("validation", n_validation, train_p), ("test", n_test, test_p)):
        true_beta = _truth_at_mix(truth["beta"], modifiers_b, prob_high)
        true_gamma = _truth_at_mix(truth["gamma"], modifiers_g, prob_high)
        observed, predicted = [], []
        for _ in range(n_rows):
            z = rng.binomial(1, 0.5, len(dims.claims)).astype(float)
            for i in dims.intents:
                observed.append(_effect_value(z, dims, i, true_beta, true_gamma))
                predicted.append(_effect_value(
                    z, dims, i,
                    {(ii, j): estimator.beta[ii][j]
                     for ii in dims.intents for j in dims.claims},
                    {(ii, pr): estimator.gamma[ii][pr]
                     for ii in dims.intents for pr in dims.pairs}))
        split_rows.append({
            "world": world, "repetition": repetition,
            "model": "beta+gamma", "split": split, "n_rows": n_rows,
            "effect_prediction_R2": _r2(observed, predicted),
        })
    return records, split_rows


def _draw_dose(rng: np.random.Generator, n_claims: int,
               zero_probability: float) -> np.ndarray:
    active = rng.random(n_claims) >= zero_probability
    magnitude = rng.choice([0.5, 1.0], size=n_claims)
    sign = rng.choice([-1.0, 1.0], size=n_claims)
    return active.astype(float) * magnitude * sign


def _run_dose_world(base_cfg: Mapping, bcfg: Mapping,
                    dims: BenchmarkDimensions, truth: Mapping,
                    repetition: int, world: str) -> Tuple[List[Dict], List[Dict]]:
    seed = int(bcfg.get("data_seed", 73129)) + 10007 * repetition + 3001
    rng = np.random.default_rng(seed)
    ecfg = _estimator_cfg(base_cfg, bcfg)
    estimator = WriteEffectEstimator(list(dims.claims), list(dims.intents),
                                     list(dims.pairs), ecfg)
    n_train = int(bcfg.get("train_rows", 3000))
    n_validation = int(bcfg.get("validation_rows", 1000))
    n_test = int(bcfg.get("test_rows", 2000))
    train_p = float(bcfg.get("train_high_state_probability", 0.5))
    test_p = (float(bcfg.get("shifted_test_high_state_probability", 0.8))
              if world == "state_shift" else train_p)
    zero_probability = float(bcfg.get("dose_zero_probability", 0.35))
    noise_sd = float(bcfg.get("dose_noise_sd", 0.05))
    modifiers = (truth["eta_state_modifier"] if world == "state_shift"
                 else {k: 0.0 for k in truth["eta"]})
    main = {(i, j): (0.08 + 0.015 * ((ii + jj) % 4))
            * (-1.0 if (ii + jj) % 2 else 1.0)
            for ii, i in enumerate(dims.intents)
            for jj, j in enumerate(dims.claims)}
    for _ in range(n_train):
        state = _draw_state(rng, train_p)
        dose = _draw_dose(rng, len(dims.claims), zero_probability)
        eta_now = {k: truth["eta"][k] + state * modifiers[k]
                   for k in truth["eta"]}
        dg = {i: (_dose_effect(dose, dims, i, main, eta_now)
                  + rng.normal(scale=noise_sd)) for i in dims.intents}
        estimator.observe(dict(zip(dims.claims, dose)), dg)
    estimator.fit()

    train_eta = _truth_at_mix(truth["eta"], modifiers, train_p)
    test_eta = _truth_at_mix(truth["eta"], modifiers, test_p)
    records = []
    for i in dims.intents:
        for pr in dims.pairs:
            key = (i, pr)
            records.append(_coefficient_record(
                world, repetition, "eta", i, f"{pr[0]}|{pr[1]}",
                train_eta[key], test_eta[key], estimator.eta[i][pr],
                estimator.se_eta[i][pr], modifiers[key]))

    fitted_main = {(i, j): estimator.s_hat[i][j]
                   for i in dims.intents for j in dims.claims}
    fitted_eta = {(i, pr): estimator.eta[i][pr]
                  for i in dims.intents for pr in dims.pairs}
    split_rows = []
    for split, n_rows, prob_high in (
            ("validation", n_validation, train_p), ("test", n_test, test_p)):
        eta_split = _truth_at_mix(truth["eta"], modifiers, prob_high)
        observed, predicted = [], []
        for _ in range(n_rows):
            dose = _draw_dose(rng, len(dims.claims), zero_probability)
            for i in dims.intents:
                observed.append(_dose_effect(dose, dims, i, main, eta_split))
                predicted.append(_dose_effect(dose, dims, i, fitted_main, fitted_eta))
        split_rows.append({
            "world": world, "repetition": repetition, "model": "eta",
            "split": split, "n_rows": n_rows,
            "effect_prediction_R2": _r2(observed, predicted),
        })
    return records, split_rows


def _ols_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(A.T @ A)
    return float(coef[0]), float(coef[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


def _run_s_benchmark(bcfg: Mapping, dims: BenchmarkDimensions,
                     truth: Mapping, repetition: int) -> Tuple[List[Dict], List[Dict]]:
    seed = int(bcfg.get("data_seed", 73129)) + 10007 * repetition + 6007
    rng = np.random.default_rng(seed)
    noise_sd = float(bcfg.get("s_noise_sd", 0.018))
    n_rep = int(bcfg.get("s_replicates_per_value", 5))
    train_values = np.asarray(bcfg.get("s_train_values", [-1.0, -0.5, 0.0, 0.5, 1.0]), float)
    validation_values = np.asarray(bcfg.get("s_validation_values", [-0.75, 0.25]), float)
    test_values = np.asarray(bcfg.get("s_test_values", [-0.25, 0.75]), float)
    if set(train_values) & set(validation_values) or set(train_values) & set(test_values):
        raise ValueError("s validation/test knob values must be held out from training")
    records, split_rows = [], []
    for ri, regime in enumerate(dims.regimes):
        for pi, param in enumerate(dims.params):
            for ii, intent in enumerate(dims.intents):
                key = (regime, param, intent)
                slope_true = float(truth["s"][key])
                intercept = 0.05 * (ri - 1) + 0.01 * pi - 0.005 * ii
                tx = np.repeat(train_values, n_rep)
                ty = intercept + slope_true * tx + rng.normal(scale=noise_sd, size=len(tx))
                intercept_hat, slope_hat, se_hat = _ols_slope(tx, ty)
                records.append(_coefficient_record(
                    "stable_mixed", repetition, "s", intent,
                    f"{regime}|{param}", slope_true, slope_true,
                    slope_hat, se_hat, 0.0))
                for split, values in (("validation", validation_values),
                                      ("test", test_values)):
                    x = np.repeat(values, n_rep)
                    y = intercept + slope_true * x + rng.normal(
                        scale=noise_sd, size=len(x))
                    pred = intercept_hat + slope_hat * x
                    split_rows.append({
                        "world": "stable_mixed", "repetition": repetition,
                        "model": "s", "split": split, "n_rows": len(x),
                        "effect_prediction_R2": _r2(y, pred),
                        "material": bool(abs(slope_true) > 1e-12),
                        "regime": regime, "param": param, "intent": intent,
                    })
    return records, split_rows


def run_controlled_benchmark(base_cfg: Mapping, benchmark_cfg: Mapping) -> Dict:
    """Run the mixed-signal benchmark and return serialisable record tables."""
    dims = make_dimensions(benchmark_cfg)
    truth = make_truth(benchmark_cfg, dims)
    repetitions = int(benchmark_cfg.get("repetitions", 20))
    worlds = list(benchmark_cfg.get("worlds", ["stable_mixed", "state_shift"]))
    unknown = set(worlds) - {"stable_mixed", "state_shift"}
    if unknown:
        raise ValueError(f"unknown controlled benchmark world(s): {sorted(unknown)}")
    records: List[Dict] = []
    split_rows: List[Dict] = []
    for repetition in range(repetitions):
        sr, sp = _run_s_benchmark(benchmark_cfg, dims, truth, repetition)
        records.extend(sr); split_rows.extend(sp)
        for world in worlds:
            er, ep = _run_eligibility_world(
                base_cfg, benchmark_cfg, dims, truth, repetition, world)
            dr, dp = _run_dose_world(
                base_cfg, benchmark_cfg, dims, truth, repetition, world)
            records.extend(er); records.extend(dr)
            split_rows.extend(ep); split_rows.extend(dp)
    truth_counts = {}
    for model in ("s", "beta", "gamma", "eta"):
        vals = list(truth[model].values())
        truth_counts[model] = {
            "n_total": len(vals),
            "n_exact_zero": int(sum(abs(v) <= 1e-12 for v in vals)),
            "n_nonzero": int(sum(abs(v) > 1e-12 for v in vals)),
            "n_positive": int(sum(v > 0 for v in vals)),
            "n_negative": int(sum(v < 0 for v in vals)),
        }
    return {
        "records": records, "split_performance": split_rows,
        "truth_counts": truth_counts,
        "design": {
            "repetitions": repetitions, "worlds": worlds,
            "claims": list(dims.claims), "intents": list(dims.intents),
            "pairs": [list(p) for p in dims.pairs],
            "params": list(dims.params), "regimes": list(dims.regimes),
            "train_rows": int(benchmark_cfg.get("train_rows", 3000)),
            "validation_rows": int(benchmark_cfg.get("validation_rows", 1000)),
            "test_rows": int(benchmark_cfg.get("test_rows", 2000)),
            "estimation_overrides": dict(
                benchmark_cfg.get("estimation_overrides", {})),
            "data_separation": (
                "coefficients fit on training rows only; validation rows are "
                "reported diagnostics; test rows are generated independently "
                "and used only after fitting; s uses disjoint knob values"),
        },
    }
