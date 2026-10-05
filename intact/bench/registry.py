"""Every method the v10 benchmark can run, with its exact configuration.

A method is fully described by
    mode       the experiment mode (``scripts/run_v2.py`` ALL_MODES, or
               ``all_reject`` for the S(t)=empty baseline);
    risk       which risk predictor feeds p_hat:
                 analytical  the analytical stand-in,
                 mild        the scenario-specific trained MILD model,
                 zero        p_hat = 0 (no risk information at all),
                 constant    p_hat = risk.constant_p (no discrimination),
                 forecast    the exogenous-forecast diagnostic (oracle-like);
    overrides  dotted configuration overrides applied on top of the scenario.

The risk selection keys (``mild.enabled``, ``mild.model_dir``,
``risk.predictor``) are added automatically from ``risk`` and may not appear
in ``overrides``.  ``validate_registry`` rejects typos and every override
that a MILD model would refuse (its scenario signature covers ``risk.*``,
``ran.*`` and the topology, and it checks ``inner.tau_risk``).

Suites
------
main         all baselines + INTACTv3 with no/analytical/forecast/MILD risk
core         B3, INTACTv3-A, INTACTv3-MILD (quick look)
ladder       cumulative steps from B3 to INTACTv3-MILD (where does the gain
             come from?)
ablation     leave-one-out components of the proposed method
sensitivity  one-at-a-time hyper-parameter sweeps
v1fullmild   full-component INTACTv1 development tuning with MILD risk
v1paper      frozen full-component WCNC method and its six ablations
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

RISK_SOURCES = ("analytical", "mild", "zero", "constant", "forecast")
EXTRA_MODES = ("all_reject",)

# Keys the benchmark is allowed to create although a scenario may not
# declare them (the simulator reads them with a default).
CREATABLE_KEYS = {"risk.predictor", "risk.constant_p", "mild.enabled",
                  "mild.model_dir"}

# Anything a MILD model's runtime checks depend on.
MILD_FORBIDDEN_PREFIXES = ("risk.", "ran.", "tenants", "intents", "xapps",
                           "claims", "sweep_domains", "margins", "mild.")
MILD_FORBIDDEN_KEYS = {"inner.tau_risk"}

PROPOSED = "INTACTv3-MILD"

# Cumulative ladder from the strongest prior baseline to the proposed
# method.  Each rung adds exactly one idea to the previous rung.
LADDER = [
    ("B3", "B3 (value arbitration)"),
    ("L1-pipeline", "+ INTACT selection/execution pipeline"),
    ("L2-pi", "+ intent priority pi"),
    ("INTACTv3-0", "+ rescue term (= INTACTv3, no risk)"),
    ("INTACTv3-C05", "+ risk levers on, uninformative p=0.5"),
    ("INTACTv3-A", "+ discriminating risk (analytical)"),
    ("INTACTv3-MILD", "MILD instead of analytical"),
]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    mode: str
    risk: str = "analytical"
    overrides: Tuple[Tuple[str, object], ...] = ()
    group: str = "baseline"          # baseline|intact|proposed|ladder|ablation|sensitivity
    suites: Tuple[str, ...] = ()
    reference: Optional[str] = None  # for Δ and decision-divergence checks
    component: str = ""              # ablation/sensitivity category
    param: str = ""                  # sensitivity: dotted parameter
    value: object = None             # sensitivity: swept value
    nominal_default: object = None   # sensitivity: registered default
    description: str = ""
    order: int = 0

    def override_dict(self) -> Dict[str, object]:
        return dict(self.overrides)

    def as_dict(self) -> Dict[str, object]:
        return {"key": self.key, "label": self.label, "mode": self.mode,
                "risk": self.risk, "overrides": self.override_dict(),
                "group": self.group, "suites": list(self.suites),
                "reference": self.reference, "component": self.component,
                "param": self.param, "value": self.value,
                "nominal_default": self.nominal_default,
                "description": self.description, "order": self.order}


def _ov(**kw) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(kw.items()))


def _ovd(d: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(d.items()))


# ---------------------------------------------------------------------------
BASELINES = [
    # key, label, mode, description
    ("AR", "All-reject", "all_reject",
     "No control: S(t) is empty at every epoch (zero writes)."),
    ("B0", "B0 all-admit", "none",
     "No mediation: every xApp writes; exposes direct C1 conflicts."),
    ("B1", "B1 static priority", "static",
     "Fixed runtime priority by omega*r_j (CMF-style)."),
    ("B2", "B2 pre-deploy", "predeploy",
     "One conflict-free portfolio frozen before deployment (PACIFISTA-style)."),
    ("B3", "B3 value arbitration", "value_only",
     "Proposal-aware unweighted signed-margin selection (QACM-style); "
     "strongest prior baseline."),
    ("B4", "B4 global scheduler", "scheduler_global",
     "Single global objective, no tenant trade-off (Cinemre-style)."),
    ("B5", "B5 outer only", "outer_only",
     "INTACTv1 outer loop, inner loop admits everything."),
    ("B6", "B6 inner only", "inner_only",
     "No eligibility control; inner loop arbitrates every write."),
    ("B6plus", "B6+", "b6plus",
     "C1/C2 + unweighted claim deficit, then inner mediation."),
    ("B6ppU", "B6++-U", "b6pp_no_urgency",
     "B6++ with the urgency multiplier fixed to 1."),
    ("B6pp", "B6++", "b6plusplus",
     "B6+ with INTACT's risk/contract/C3 claim weight."),
    ("B7", "B7 single loop", "single_loop",
     "Proposal-aware weighted s selection, no inner mediator."),
    ("B7plus", "B7+", "b7plus",
     "B7 + write-level interaction and envelope feasibility."),
    ("INTACTv1", "INTACTv1", "intact",
     "Original two-loop INTACT with beta/gamma value model."),
]


def build_registry(ablation_risk: str = "mild",
                   sensitivity_risk: str = "analytical") -> Dict[str, MethodSpec]:
    """Return the complete method registry.

    ``ablation_risk`` and ``sensitivity_risk`` choose the risk source of the
    leave-one-out and sensitivity variants (``mild`` or ``analytical``).  The
    variant keys contain the source (``_M_`` or ``_A_``) so both can coexist
    in one output directory without cache collisions.
    """
    if ablation_risk not in ("mild", "analytical"):
        raise ValueError("ablation_risk must be mild or analytical")
    if sensitivity_risk not in ("mild", "analytical"):
        raise ValueError("sensitivity_risk must be mild or analytical")
    reg: Dict[str, MethodSpec] = {}

    def add(spec: MethodSpec) -> None:
        if spec.key in reg:
            old = reg[spec.key]
            merged = tuple(dict.fromkeys(old.suites + spec.suites))
            reg[spec.key] = MethodSpec(**{**old.__dict__, "suites": merged})
            return
        reg[spec.key] = spec

    order = 0
    for key, label, mode, desc in BASELINES:
        order += 1
        suites = ("main",) + (("core", "ladder") if key == "B3" else ())
        add(MethodSpec(key, label, mode, "analytical", (), "baseline",
                       suites, None, "", description=desc, order=order))

    # ---- INTACTv3 risk-source family (shared by main/ladder/ablation) ----
    add(MethodSpec("INTACTv3-0", "INTACTv3 (no risk, p=0)", "intactv3",
                   "zero", (), "intact", ("main", "ladder", "ablation"),
                   "INTACTv3-A", "risk source",
                   description="INTACTv3 with p_hat=0: no urgency/risk "
                               "information of any kind.", order=100))
    add(MethodSpec("INTACTv3-C05", "INTACTv3 (constant p=0.5)", "intactv3",
                   "constant", _ov(**{"risk.constant_p": 0.5}), "intact",
                   ("ladder", "ablation"), "INTACTv3-0", "risk source",
                   description="All risk levers switched on with an "
                               "uninformative constant p=0.5.", order=101))
    add(MethodSpec("INTACTv3-A", "INTACTv3-A (analytical risk)", "intactv3",
                   "analytical", (), "intact",
                   ("main", "core", "ladder", "ablation"),
                   "INTACTv3-0", "risk source",
                   description="INTACTv3 with the analytical risk stand-in.",
                   order=102))
    add(MethodSpec("INTACTv3-F", "INTACTv3-F (forecast oracle)", "intactv3",
                   "forecast", (), "intact", ("main", "ablation"),
                   "INTACTv3-A", "risk source",
                   description="INTACTv3 with the exogenous-forecast "
                               "diagnostic (oracle-like reference, not a "
                               "deployable predictor).", order=103))
    add(MethodSpec(PROPOSED, "INTACTv3-MILD (proposed)", "intactv3", "mild",
                   (), "proposed",
                   ("main", "core", "ladder", "ablation"),
                   "INTACTv3-A", "risk source",
                   description="Proposed: INTACTv3 with the scenario-"
                               "specific gated MILD predictor.", order=104))

    # ---- ladder rungs that are not shared ------------------------------
    add(MethodSpec("L1-pipeline", "INTACT pipeline + B3 objective",
                   "intactv3", "zero",
                   _ov(**{"v2.wif_primary_objective": "linear",
                          "v2.wif_linear_use_pi": False}),
                   "ladder", ("ladder",), "B3", "ladder",
                   description="INTACTv3 selection/execution machinery "
                               "with B3's unweighted linear objective and "
                               "no risk.", order=201))
    add(MethodSpec("L2-pi", "+ intent priority pi", "intactv3", "zero",
                   _ov(**{"v2.wif_primary_objective": "linear",
                          "v2.wif_linear_use_pi": True}),
                   "ladder", ("ladder",), "L1-pipeline", "ladder",
                   description="L1 plus pi_class weighting of the linear "
                               "objective (still no risk).", order=202))
    # L3 = INTACTv3-0 (adds the rescue term), L4 = INTACTv3-C05,
    # L5 = INTACTv3-A, L6 = INTACTv3-MILD.

    # ---- ablations of the proposed method ------------------------------
    # Blocks (MethodSpec.component):
    #   A  objective form: threshold/predictive objective (frozen method)
    #      versus the classic urgency-weighted objective;
    #   WT weight factors inside the threshold objective (leave one out);
    #   WC the same weight factors inside the classic objective;
    #   R  non-weight risk levers;  O  other objective terms;
    #   E  execution;  S  risk source (shared INTACTv3-* keys above).
    base = PROPOSED if ablation_risk == "mild" else "INTACTv3-A"
    tag = "M" if ablation_risk == "mild" else "A"
    classic = f"abl_{tag}_C_full"
    no_pi_t = {"v2.predictive_use_pi": False, "outer.weight_use_pi": False}
    no_om = {"outer.weight_use_omega": False}
    no_u_t = {"v2.predictive_risk_gain": 0.0, "outer.weight_use_urgency": False}
    no_dl = {"outer.theta_Lambda": 0.0}
    cl = {"v2.use_threshold_primary": False}
    ablations = [
        # key suffix, label, block, overrides, mode, reference, description
        ("C_full", "use_threshold_primary: true -> false (classic)",
         "A objective form", dict(cl), "intactv3", base,
         "use_threshold_primary: false -> V_j = sum_i w_i s dnu with "
         "w_i = pi*omega*u(p)*(1+theta_Lambda*Lambda) in the PRIMARY "
         "objective, and the inner no-new-victims veto."),
        ("T_no_pi", "T: - intent class pi", "WT weights (threshold objective)",
         dict(no_pi_t), "intactv3", base,
         "pi_class removed from primary (1+g p) weights and secondary w_i."),
        ("T_no_omega", "T: - tenant priority omega",
         "WT weights (threshold objective)", dict(no_om), "intactv3", base,
         "omega removed from w_i (secondary tie-breaker only in this "
         "objective)."),
        ("T_no_urgency", "T: - risk urgency", "WT weights (threshold objective)",
         dict(no_u_t), "intactv3", base,
         "(1+g p) primary weight and u(p) secondary weight both removed."),
        ("T_no_deficit", "T: - tenant deficit Lambda (C3)",
         "WT weights (threshold objective)", dict(no_dl), "intactv3", base,
         "theta_Lambda = 0 (secondary tie-breaker only in this objective)."),
        ("T_no_weights", "T: - all four weight factors",
         "WT weights (threshold objective)",
         {**no_pi_t, **no_om, **no_u_t, **no_dl}, "intactv3", base,
         "Uniform intent weights everywhere."),
        ("C_no_pi", "C: - intent class pi", "WC weights (classic objective)",
         {**cl, "outer.weight_use_pi": False}, "intactv3", classic,
         "Classic objective without pi_class in w_i."),
        ("C_no_omega", "C: - tenant priority omega",
         "WC weights (classic objective)",
         {**cl, **no_om}, "intactv3", classic,
         "Classic objective without omega in w_i."),
        ("C_no_urgency", "C: - risk urgency u(p)",
         "WC weights (classic objective)",
         {**cl, "outer.weight_use_urgency": False}, "intactv3", classic,
         "Classic objective with u(p) = 1."),
        ("C_no_deficit", "C: - tenant deficit Lambda (C3)",
         "WC weights (classic objective)", {**cl, **no_dl}, "intactv3", classic,
         "Classic objective with theta_Lambda = 0."),
        ("C_no_weights", "C: - all four weight factors",
         "WC weights (classic objective)",
         {**cl, "outer.weight_use_pi": False, **no_om,
          "outer.weight_use_urgency": False, **no_dl}, "intactv3", classic,
         "Classic objective with uniform weights."),
        ("no_outer_risk", "- all outer risk levers", "R risk levers",
         {"v2.predictive_risk_gain": 0.0, "v2.risk_margin_reserve": 0.0,
          "v2.predictive_s_blend": False, "outer.weight_use_urgency": False},
         "intactv3", base,
         "No (1+g p) weight, no reserve, no forecast-sensitivity blend, "
         "u(p)=1; joint protection kept."),
        ("no_reserve", "- pre-failure reserve", "R risk levers",
         {"v2.risk_margin_reserve": 0.0}, "intactv3", base,
         "risk_margin_reserve = 0."),
        ("no_s_blend", "predictive_s_blend: true -> false", "D design switches",
         {"v2.predictive_s_blend": False}, "intactv3", base,
         "Predictive (forecast-regime) sensitivity blending disabled: plan "
         "with the current-regime slope only."),
        ("no_joint_protection", "- risk-gated joint protection",
         "R risk levers", {"v2.joint_protection": False}, "intactv3", base,
         "Outer portfolios may push an at-risk intent below its floor."),
        ("no_eps_inflation", "- epsilon inflation", "R risk levers",
         {"v2.lambda_risk": 0.0}, "intactv3", base,
         "eps_eff = eps (lambda_risk = 0)."),
        ("no_rescue", "- rescue term", "O objective terms",
         {"v2.predictive_rescue_gain": 0.0}, "intactv3", base,
         "Pure risk-weighted signed-margin objective."),
        ("plus_c4", "+ claim deficit C4 (INTACTv2)", "O objective terms", {},
         "intactv2", base, "INTACTv2: subordinate C4 claim-service criterion."),
        ("plus_inner_veto", "+ inner per-write veto", "E execution",
         {"v2.threshold_protection_policy": "no_new_victims"}, "intactv3",
         base, "Restores the per-write no-new-victims safety veto."),
        ("plus_local_dose", "inner_local_optimize: false -> true",
         "D design switches", {"v2.inner_local_optimize": True}, "intactv3",
         base, "Inner loop may choose a partial dose instead of the "
         "grid-nearest request."),
        ("C_s_blend_off", "C: predictive_s_blend: true -> false",
         "D design switches", {**cl, "v2.predictive_s_blend": False},
         "intactv3", classic, "Blending disabled inside the classic objective."),
        ("C_local_dose", "C: inner_local_optimize: false -> true",
         "D design switches", {**cl, "v2.inner_local_optimize": True},
         "intactv3", classic, "Local dose optimisation inside the classic "
         "objective."),
    ]
    o = 300
    for name, lab, comp, ovs, mode, ref, desc in ablations:
        o += 1
        add(MethodSpec(f"abl_{tag}_{name}", lab, mode, ablation_risk,
                       _ovd(ovs), "ablation", ("ablation",), ref, comp,
                       description=desc, order=o))

    # ---- one-at-a-time sensitivity ------------------------------------
    sbase = PROPOSED if sensitivity_risk == "mild" else "INTACTv3-A"
    stag = "M" if sensitivity_risk == "mild" else "A"
    sweeps = [
        ("v2.predictive_risk_gain", "risk weight gain g", 1.0, [0.5, 2.0, 4.0]),
        ("v2.predictive_rescue_gain", "rescue gain", 0.25, [0.1, 0.5, 1.0]),
        ("v2.risk_margin_reserve", "pre-failure reserve", 0.08, [0.04, 0.16]),
        ("v2.wif_temperature", "rescue temperature", 0.03, [0.015, 0.06]),
        ("v2.lambda_risk", "epsilon inflation lambda_risk", 0.5, [1.0]),
        ("outer.theta_Lambda", "tenant-deficit weight theta_Lambda", 0.5, [2.0]),
        ("outer.lambda_u", "urgency lambda_u", 1.0, [4.0]),
        ("v2.wif_performance_slack", "secondary tie-break slack", 0.0,
         [0.001, 0.005]),
        ("inner.max_step_frac", "trust region max_step_frac", 0.25,
         [0.10, 0.50]),
    ]
    if sensitivity_risk != "mild":
        sweeps.append(("inner.tau_risk", "protection threshold tau_risk",
                       0.65, [0.50, 0.80]))
    o = 500
    for param, plabel, default, values in sweeps:
        for v in values:
            o += 1
            safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{param}_{v}").strip("_")
            add(MethodSpec(f"sens_{stag}_{safe}", f"{plabel} = {v:g}",
                           "intactv3", sensitivity_risk, _ov(**{param: v}),
                           "sensitivity", ("sensitivity",), sbase, plabel,
                           param=param, value=v, nominal_default=default,
                           description=f"{param} = {v} (default {default})",
                           order=o))
    for v in (0.25, 1.0):
        o += 1
        add(MethodSpec(f"sens_constant_p_{str(v).replace('.', '_')}",
                       (f"INTACTv3 p=1 [oracle forecast]" if v == 1.0
                        else f"constant p = {v:g}"), "intactv3", "constant",
                       _ov(**{"risk.constant_p": v}), "sensitivity",
                       ("sensitivity",), "INTACTv3-C05", "constant risk p",
                       param="risk.constant_p", value=v, nominal_default=0.5,
                       description=f"uninformative constant p = {v}",
                       order=o))
    # ---- confirmation suite (added after the v10 results) -------------
    # "Lean" = the frozen method with every component that showed NO
    # detectable effect in the ablation removed; only the predictive
    # (forecast-regime) sensitivity blend and the risk source are kept.
    # The P1 family decomposes the exploratory finding that a constant
    # p = 1 was the best configuration in the sensitivity suite.
    lean_off = {
        "v2.predictive_rescue_gain": 0.0,      # rescue term: no effect
        "v2.risk_margin_reserve": 0.0,         # pre-failure reserve: no effect
        "v2.joint_protection": False,          # joint protection: no effect
        "v2.lambda_risk": 0.0,                 # epsilon inflation: no effect
        "v2.predictive_risk_gain": 0.0,        # (1+g p) urgency weight: no effect
        "outer.weight_use_urgency": False,     # u(p) tie-break weight: no effect
        "v2.predictive_use_pi": False,         # pi in primary: no effect
        "outer.weight_use_pi": False,          # pi in tie-break: no effect
        "outer.weight_use_omega": False,       # omega tie-break: no effect
        "outer.theta_Lambda": 0.0,             # tenant deficit: no effect
    }
    o = 700
    confirm = [
        ("LEAN-MILD", "INTACTv3-MILD lean (blend only)", "mild", lean_off,
         PROPOSED, "Proposed method with every no-effect component removed."),
        ("LEAN-A", "INTACTv3-A lean (blend only)", "analytical", lean_off,
         "INTACTv3-A", "Analytical method with every no-effect component "
         "removed."),
        ("P1-noJP", "p=1 without joint protection", "constant",
         {"risk.constant_p": 1.0, "v2.joint_protection": False},
         "sens_constant_p_1_0", "Isolates universal joint protection."),
        ("P1-noBlend", "p=1 without forecast blend", "constant",
         {"risk.constant_p": 1.0, "v2.predictive_s_blend": False},
         "sens_constant_p_1_0", "Isolates the full forecast-slope blend."),
        ("A-protectAll", "analytical + protect every intent", "analytical",
         {"inner.tau_risk": 0.001}, "INTACTv3-A",
         "Analytical blend, joint protection applied to every intent."),
        ("LEAN-P1", "p=1 lean (blend + protection)", "constant",
         {k: v for k, v in {**lean_off, "risk.constant_p": 1.0}.items()
          if k not in ("v2.joint_protection", "v2.lambda_risk")},
         "sens_constant_p_1_0", "Constant p=1 keeping only blend and joint "
         "protection."),
    ]
    # ---- regime-decomposition (v10.3) ---------------------------------
    # 2x2: does the planning regime come from the TENANT or the CELL, and
    # from a FORECAST or from the MEASURED load?  Isolates "per-tenant
    # regime resolution" from "look-ahead".
    p1 = {"risk.constant_p": 1.0}
    confirm += [
        ("RG-tenant-now", "INTACT-RA (proposed)", "constant",
         {**p1, "v2.planning_regime_source": "tenant_now"},
         "sens_constant_p_1_0",
         "Per-tenant regime resolution with no forecast read at all."),
        ("RG-cell-fc", "p=1, cell regime from FORECAST", "constant",
         {**p1, "v2.planning_regime_source": "cell"}, "INTACTv3-0",
         "Look-ahead only: one cell-wide regime, taken from the forecast."),
        ("RG-cell-now", "p=1, cell regime from MEASURED load", "constant",
         {**p1, "v2.planning_regime_source": "cell_now"}, "INTACTv3-0",
         "Neither per-tenant resolution nor look-ahead; should equal p=0."),
        ("INTACT-RA", "INTACT-RA-lean (no contract weights)", "constant",
         {**p1, "v2.planning_regime_source": "tenant_now", **lean_off},
         "sens_constant_p_1_0",
         "The frozen method: per-tenant measured regime, p=1, every "
         "no-effect component removed, no forecaster."),
    ]
    # C3 slack: let the contractual tie-break actually bind by allowing the
    # secondary criterion to spend a small amount of primary utility.
    for sl in (0.002, 0.005, 0.010):
        confirm.append((
            f"RA-slack-{str(sl).replace('.', '_')}",
            f"INTACT-RA, C3 tie-break slack {sl:g}", "constant",
            {**p1, "v2.planning_regime_source": "tenant_now",
             "v2.wif_performance_slack": sl},
            "RG-tenant-now",
            "Proposed method with a slack window in which the contractual "
            "weight (pi, omega, tenant deficit Lambda) may override the "
            "primary objective; tests whether C3 can be actively enforced."))
    for key, lab, risk, ovs, ref, desc in confirm:
        o += 1
        grp = ("regime" if key.startswith(("RG-", "INTACT-RA", "RA-slack"))
               else "confirm")
        add(MethodSpec(key, lab, "intactv3", risk, _ovd(ovs), grp,
                       ("confirm",), ref,
                       "confirmation" if grp == "confirm" else "regime source",
                       description=desc, order=o))
    # already-cached reference methods are part of the confirmation suite
    for k in ("B3", "INTACTv3-0", "INTACTv3-A", PROPOSED):
        if k in reg:
            sp = reg[k]
            reg[k] = MethodSpec(**{**sp.__dict__, "suites": tuple(
                dict.fromkeys(sp.suites + ("confirm",)))})
    # reuse the cached constant-p=1 runs inside the confirmation suite
    if "sens_constant_p_1_0" in reg:
        old_spec = reg["sens_constant_p_1_0"]
        reg["sens_constant_p_1_0"] = MethodSpec(**{
            **old_spec.__dict__,
            "suites": tuple(dict.fromkeys(old_spec.suites + ("confirm",)))})
    # ---- INTACT v1 ablation suite (conference study) -------------------
    # These variants ablate the ORIGINAL v1 objective
    #   S = argmax_S [ sum_j (V_j + theta_D D_j) - sum_{j,k} H(j,k) ]
    # one component at a time.  They run in mode "intact" (v1), so they are
    # only meaningful on the cell-homogeneous scenario family built by
    # scripts/build_v1_suite.py.
    v1abl = [
        ("v1_no_harm", "v1 without pair-harm H(j,k)",
         {"outer.coupled_pairs": []},
         "No interaction penalty: claims are valued independently."),
        ("v1_no_deficit", "v1 without claim deficit D_j",
         {"outer.theta_D": 0.0},
         "No starvation ledger in the selection objective."),
        ("v1_no_floor", "v1 without tenant floor term",
         {"outer.theta_Lambda": 0.0},
         "C3 deficit no longer biases the intent weights."),
        ("v1_no_urgency", "v1 without urgency u(p)",
         {"outer.weight_use_urgency": False},
         "Weights ignore how close an intent is to its floor."),
        ("v1_no_pi", "v1 without intent class pi",
         {"outer.weight_use_pi": False},
         "Weights ignore the contracted intent class."),
        ("v1_no_omega", "v1 without tenant priority omega",
         {"outer.weight_use_omega": False},
         "Weights ignore the commercial priority of the tenant."),
    ]
    o = 900
    for key, lab, ovs, desc in v1abl:
        o += 1
        add(MethodSpec(key, lab, "intact", "analytical", _ovd(ovs),
                       "ablation", ("v1ablation",), "INTACTv1",
                       "v1 component", description=desc, order=o))
    if "INTACTv1" in reg:
        sp = reg["INTACTv1"]
        reg["INTACTv1"] = MethodSpec(**{**sp.__dict__, "suites": tuple(
            dict.fromkeys(sp.suites + ("v1ablation",)))})

    # ---- INTACT v1 targeted tuning suite -----------------------------
    # This is deliberately small and hypothesis-led.  The 20-scenario v1
    # ablation showed that urgency was harmful, pair harm was also costly,
    # and claim deficit / pi / omega were beneficial.  The grid therefore
    # concentrates on urgency, evaluation-time exploration, their interaction
    # with H, and two local values around the nominal theta_D=0.6.  It does
    # not tune the scenario generator.
    v1tune = [
        ("v1_tune_E0", "v1 tune: no evaluation exploration",
         {"estimation.eps_exp": 0.0}, "evaluation exploration"),
        ("v1_tune_E0_U0", "v1 tune: E=0, urgency off",
         {"estimation.eps_exp": 0.0,
          "outer.weight_use_urgency": False}, "exploration + urgency"),
        ("v1_tune_U0_H0", "v1 tune: urgency off, pair harm off",
         {"outer.weight_use_urgency": False,
          "outer.coupled_pairs": []}, "urgency + pair harm"),
        ("v1_tune_E0_U0_H0", "v1 tune: E=0, urgency off, pair harm off",
         {"estimation.eps_exp": 0.0,
          "outer.weight_use_urgency": False,
          "outer.coupled_pairs": []}, "exploration + urgency + pair harm"),
        ("v1_tune_E0_U0_L0", "v1 tune: E=0, urgency/floor boost off",
         {"estimation.eps_exp": 0.0,
          "outer.weight_use_urgency": False,
          "outer.theta_Lambda": 0.0}, "exploration + urgency + floor"),
        ("v1_tune_E0_U0_D03", "v1 tune: E=0, urgency off, theta_D=0.3",
         {"estimation.eps_exp": 0.0,
          "outer.weight_use_urgency": False,
          "outer.theta_D": 0.3}, "claim deficit weight"),
        ("v1_tune_E0_U0_D09", "v1 tune: E=0, urgency off, theta_D=0.9",
         {"estimation.eps_exp": 0.0,
          "outer.weight_use_urgency": False,
          "outer.theta_D": 0.9}, "claim deficit weight"),
        ("v1_tune_E0_Uabs010", "v1 tune: E=0, absolute urgency lambda=0.1",
         {"estimation.eps_exp": 0.0,
          "outer.lambda_u": 0.1}, "absolute urgency strength"),
        ("v1_tune_E0_Uabs025", "v1 tune: E=0, absolute urgency lambda=0.25",
         {"estimation.eps_exp": 0.0,
          "outer.lambda_u": 0.25}, "absolute urgency strength"),
        ("v1_tune_E0_Urel010", "v1 tune: E=0, relative urgency lambda=0.1",
         {"estimation.eps_exp": 0.0,
          "outer.urgency_mode": "relative", "outer.lambda_u": 0.1},
         "relative urgency strength"),
        ("v1_tune_E0_Urel025", "v1 tune: E=0, relative urgency lambda=0.25",
         {"estimation.eps_exp": 0.0,
          "outer.urgency_mode": "relative", "outer.lambda_u": 0.25},
         "relative urgency strength"),
    ]
    o = 950
    for key, lab, ovs, component in v1tune:
        o += 1
        add(MethodSpec(key, lab, "intact", "analytical", _ovd(ovs),
                       "sensitivity", ("v1tune",), "INTACTv1", component,
                       param="joint v1 configuration", value=str(ovs),
                       nominal_default="INTACTv1",
                       description="Targeted post-ablation development "
                                   "configuration; validate on fresh holdout "
                                   "scenarios.", order=o))
    for key in ("INTACTv1", "B6plus", "B6ppU", "B6pp", "v1_no_urgency"):
        if key in reg:
            sp = reg[key]
            reg[key] = MethodSpec(**{**sp.__dict__, "suites": tuple(
                dict.fromkeys(sp.suites + ("v1tune",)))})

    # ---- full-component INTACT v1 + MILD tuning suite ----------------
    # Unlike ``v1tune``, this grid is NOT an ablation search.  Every INTACT
    # component remains active: pi, omega, urgency, tenant-floor deficit,
    # claim deficit, pair harm, and evaluation exploration.  MILD supplies
    # p_hat to both urgency and the inner risk-aware mediator.  Relative
    # urgency is emphasised because calibrated per-intent MILD probabilities
    # can be saturated even when their within-epoch ordering is informative.
    full_mild_common = {
        "outer.urgency_mode": "relative",
        "outer.theta_D": 0.9,
        "outer.theta_Lambda": 0.5,
    }
    v1fullmild = [
        # urgency strength, centred below the analytical-risk sweep because
        # relative ranks span the full [0, 1] range at each informative epoch
        ("v1m_rel_u001_e002_d09_tl05",
         "v1 full MILD: relative urgency 0.01, exploration 0.02",
         {**full_mild_common, "outer.lambda_u": 0.01,
          "estimation.eps_exp": 0.02}, "urgency strength"),
        ("v1m_rel_u002_e002_d09_tl05",
         "v1 full MILD: relative urgency 0.02, exploration 0.02",
         {**full_mild_common, "outer.lambda_u": 0.02,
          "estimation.eps_exp": 0.02}, "urgency strength"),
        ("v1m_rel_u005_e002_d09_tl05",
         "v1 full MILD: relative urgency 0.05, exploration 0.02",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "estimation.eps_exp": 0.02}, "urgency strength"),
        ("v1m_rel_u010_e002_d09_tl05",
         "v1 full MILD: relative urgency 0.10, exploration 0.02",
         {**full_mild_common, "outer.lambda_u": 0.10,
          "estimation.eps_exp": 0.02}, "urgency strength"),
        # positive exploration sweep; no candidate sets eps_exp to zero
        ("v1m_rel_u005_e001_d09_tl05",
         "v1 full MILD: relative urgency 0.05, exploration 0.01",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "estimation.eps_exp": 0.01}, "exploration strength"),
        ("v1m_rel_u005_e004_d09_tl05",
         "v1 full MILD: relative urgency 0.05, exploration 0.04",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "estimation.eps_exp": 0.04}, "exploration strength"),
        ("v1m_rel_u005_e008_d09_tl05",
         "v1 full MILD: relative urgency 0.05, exploration 0.08",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "estimation.eps_exp": 0.08}, "exploration strength"),
        # local claim-deficit sweep around the earlier analytical optimum
        ("v1m_rel_u005_e002_d03_tl05",
         "v1 full MILD: claim deficit 0.3",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "outer.theta_D": 0.3, "estimation.eps_exp": 0.02},
         "claim deficit strength"),
        ("v1m_rel_u005_e002_d06_tl05",
         "v1 full MILD: claim deficit 0.6",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "outer.theta_D": 0.6, "estimation.eps_exp": 0.02},
         "claim deficit strength"),
        ("v1m_rel_u005_e002_d12_tl05",
         "v1 full MILD: claim deficit 1.2",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "outer.theta_D": 1.2, "estimation.eps_exp": 0.02},
         "claim deficit strength"),
        # tenant-floor deficit remains active at every value
        ("v1m_rel_u005_e002_d09_tl025",
         "v1 full MILD: tenant floor deficit 0.25",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "outer.theta_Lambda": 0.25, "estimation.eps_exp": 0.02},
         "tenant floor deficit strength"),
        ("v1m_rel_u005_e002_d09_tl10",
         "v1 full MILD: tenant floor deficit 1.0",
         {**full_mild_common, "outer.lambda_u": 0.05,
          "outer.theta_Lambda": 1.0, "estimation.eps_exp": 0.02},
         "tenant floor deficit strength"),
        # One conservative absolute-probability candidate tests whether
        # ranking is needed; eps0 damps the p -> 1 amplification.
        ("v1m_abs_u002_e002_d09_tl05_eps010",
         "v1 full MILD: damped absolute urgency",
         {"outer.urgency_mode": "absolute", "outer.lambda_u": 0.02,
          "outer.eps0": 0.10, "outer.theta_D": 0.9,
          "outer.theta_Lambda": 0.5, "estimation.eps_exp": 0.02},
         "urgency mapping"),
    ]
    o = 970
    add(MethodSpec(
        "INTACTv1-MILD", "INTACTv1 (full, MILD risk)", "intact", "mild",
        (), "proposed", ("v1fullmild",), None, "full MILD reference",
        description="Unmodified full INTACTv1 with scenario defaults and "
                    "scenario-specific MILD risk.", order=o))
    o += 1
    add(MethodSpec(
        "B6pp-MILD", "B6++ (MILD risk)", "b6plusplus", "mild", (),
        "baseline", ("v1fullmild",), None, "same-risk control",
        description="Constraint-compliant B6++ evaluated with the exact same "
                    "scenario-specific MILD predictor as INTACTv1.", order=o))
    for key, lab, ovs, component in v1fullmild:
        o += 1
        add(MethodSpec(
            key, lab, "intact", "mild", _ovd(ovs), "sensitivity",
            ("v1fullmild",), "INTACTv1-MILD", component,
            param="full-component MILD configuration", value=str(ovs),
            nominal_default="INTACTv1-MILD",
            description="Development-only full-component INTACTv1 tuning; "
                        "freeze the winner and test it once on fresh holdout "
                        "scenarios.", order=o))
    for key in ("B6plus", "B6ppU", "B6pp"):
        if key in reg:
            sp = reg[key]
            reg[key] = MethodSpec(**{**sp.__dict__, "suites": tuple(
                dict.fromkeys(sp.suites + ("v1fullmild",)))})

    # ---- frozen WCNC v1 paper suite ----------------------------------
    # Predeclared after the five-scenario development run and before the
    # fresh paper scenarios are generated.  No component is removed from
    # the proposed method: urgency, claim deficit, pair harm, tenant-floor
    # deficit, pi, omega, exploration, and both loops remain active.
    paper_cfg = {
        "outer.urgency_mode": "relative",
        "outer.lambda_u": 0.05,
        "outer.theta_D": 0.9,
        "outer.theta_Lambda": 0.5,
        "estimation.eps_exp": 0.02,
    }
    o = 1100
    add(MethodSpec(
        "INTACTv1-WCNC", "INTACTv1 (full, MILD)", "intact", "mild",
        _ovd(paper_cfg), "proposed", ("v1paper",), None,
        "frozen full method",
        description="Frozen full-component INTACTv1 configuration for the "
                    "WCNC comparison; scenario-specific MILD supplies risk.",
        order=o))

    # Risk-aware baselines use the exact same MILD model as the proposed
    # method.  The non-risk baselines (AR/B0/B1/B2/B4) need no duplicate.
    paper_baselines = [
        ("WCNC-B5", "B5 outer only", "outer_only"),
        ("WCNC-B6", "B6 inner only", "inner_only"),
        ("WCNC-B6plus", "B6+", "b6plus"),
        ("WCNC-B6ppU", "B6++-U", "b6pp_no_urgency"),
        ("WCNC-B6pp", "B6++", "b6plusplus"),
        ("WCNC-B7", "B7 single loop", "single_loop"),
        ("WCNC-B7plus", "B7+", "b7plus"),
    ]
    for key, label, mode in paper_baselines:
        o += 1
        add(MethodSpec(
            key, label, mode, "mild", (), "baseline", (), None,
            "same-risk paper baseline",
            description="Paper baseline evaluated with the proposed "
                        "method's scenario-specific MILD predictor.", order=o))

    paper_ablations = [
        ("v1paper_no_harm", "without pair-harm H(j,k)", "intact",
         {**paper_cfg, "outer.coupled_pairs": []}, "pair harm"),
        ("v1paper_no_deficit", "without claim deficit D_j", "intact",
         {**paper_cfg, "outer.theta_D": 0.0}, "claim deficit"),
        ("v1paper_no_floor", "without tenant-floor deficit", "intact",
         {**paper_cfg, "outer.theta_Lambda": 0.0}, "tenant floor"),
        ("v1paper_no_urgency", "without urgency u(p)", "intact",
         {**paper_cfg, "outer.weight_use_urgency": False}, "urgency"),
        ("v1paper_no_inner", "without inner loop", "outer_only",
         paper_cfg, "inner loop"),
        ("v1paper_reject", "reject instead of attenuate", "intact",
         {**paper_cfg, "inner.reject_instead_of_attenuate": True},
         "minimal attenuation"),
    ]
    for key, label, mode, ovs, component in paper_ablations:
        o += 1
        add(MethodSpec(
            key, label, mode, "mild", _ovd(ovs), "ablation",
            ("v1paper",), "INTACTv1-WCNC", component,
            description="WCNC leave-one-out ablation of the frozen full "
                        "method.", order=o))

    # ---- forecast-realism suite ---------------------------------------
    # Every earlier result used the EXACT schedule forecast ("oracle").
    # These variants replace it, for every consumer (xApps, the sensitivity
    # blend, MILD features), with an estimated or degraded forecast, so the
    # paper can report how much of the gain survives.
    sources = [
        ("persistence", "persistence forecast", {"v2.forecast_source": "persistence"}),
        ("holt", "Holt trend forecast", {"v2.forecast_source": "holt"}),
        ("auto", "self-selecting forecast", {"v2.forecast_source": "auto"}),
        ("noisy15", "oracle + 15% noise",
         {"v2.forecast_source": "noisy", "v2.forecast_noise_sd": 0.15}),
        ("noisy40", "oracle + 40% noise",
         {"v2.forecast_source": "noisy", "v2.forecast_noise_sd": 0.40}),
    ]
    bases = [("B3", "B3", "value_only", "analytical"),
             ("V30", "INTACTv3 (no risk)", "intactv3", "zero"),
             ("A", "INTACTv3-A", "intactv3", "analytical"),
             ("P1", "INTACTv3 p=1", "intactv3", "constant"),
             ("MILD", "INTACTv3-MILD", "intactv3", "mild")]
    ref_of = {"B3": "B3", "V30": "INTACTv3-0", "A": "INTACTv3-A",
              "P1": "sens_constant_p_1_0", "MILD": PROPOSED}
    # p = 1 plans entirely with the forecast regime, so it is the variant
    # most exposed to forecast error: it must be tested under every source.
    extra_ov = {"P1": {"risk.constant_p": 1.0}}
    o = 800
    for src, slab, sov in sources:
        for bkey, blab, mode, risk in bases:
            if risk == "mild" and src not in ("holt", "auto"):
                continue        # keep the expensive MILD runs to two sources
            o += 1
            add(MethodSpec(f"fc_{src}_{bkey}", f"{blab} [{slab}]", mode, risk,
                           _ovd({**sov, **extra_ov.get(bkey, {})}),
                           "forecast", ("forecast",), ref_of[bkey],
                           f"forecast: {slab}",
                           description=f"{blab} with v2.forecast_source="
                                       f"{sov['v2.forecast_source']}"
                                       + (f", sd={sov.get('v2.forecast_noise_sd')}"
                                          if "v2.forecast_noise_sd" in sov else ""),
                           order=o))
    for k in ("B3", "INTACTv3-0", "INTACTv3-A", PROPOSED,
              "sens_constant_p_1_0"):
        if k in reg:
            sp = reg[k]
            reg[k] = MethodSpec(**{**sp.__dict__, "suites": tuple(
                dict.fromkeys(sp.suites + ("forecast",)))})
    validate_registry(reg)
    return reg


# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_registry(reg: Dict[str, MethodSpec]) -> None:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    try:
        from run_v2 import ALL_MODES
    except Exception:           # pragma: no cover - import path problem
        ALL_MODES = []
    errs = []
    for key, s in reg.items():
        if not _KEY_RE.match(key):
            errs.append(f"{key}: key must match {_KEY_RE.pattern}")
        if ALL_MODES and s.mode not in list(ALL_MODES) + list(EXTRA_MODES):
            errs.append(f"{key}: unknown mode {s.mode}")
        if s.risk not in RISK_SOURCES:
            errs.append(f"{key}: unknown risk source {s.risk}")
        for k, _ in s.overrides:
            if k in ("mild.enabled", "mild.model_dir", "risk.predictor"):
                errs.append(f"{key}: {k} is set from 'risk', not overrides")
            if s.risk == "mild" and (k in MILD_FORBIDDEN_KEYS or any(
                    k.startswith(p) for p in MILD_FORBIDDEN_PREFIXES)):
                errs.append(f"{key}: override {k} is forbidden with a MILD "
                            "model (signature/tau check would reject it)")
        if s.reference is not None and s.reference not in reg:
            errs.append(f"{key}: reference {s.reference} not in registry")
    if errs:
        raise ValueError("method registry invalid:\n  " + "\n  ".join(errs))


def risk_overrides(risk: str, scenario_stem: str,
                   mild_model_root: Optional[str]) -> Dict[str, object]:
    if risk == "mild":
        if not mild_model_root:
            raise ValueError("a MILD method needs --mild-model-root")
        from pathlib import Path
        return {"mild.enabled": True,
                "mild.model_dir": str(Path(mild_model_root) / scenario_stem)}
    name = {"analytical": "analytical", "zero": "zero",
            "constant": "constant", "forecast": "forecast_diagnostic"}[risk]
    return {"mild.enabled": False, "risk.predictor": name}


def check_override_paths(cfg: Dict, overrides: Dict[str, object]) -> List[str]:
    """Return override keys that do not exist in the merged configuration.

    ``load_config`` silently CREATES a missing dotted key.  A typo such as
    ``v2.predictive_rsik_gain`` would therefore produce an ablation that
    changes nothing -- exactly the failure this check prevents.
    """
    missing = []
    for dotted in overrides:
        if dotted in CREATABLE_KEYS:
            continue
        cur = cfg
        ok = True
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if not ok:
            missing.append(dotted)
    return missing


def comparison_pairs(reg: Dict[str, MethodSpec],
                     keys: Iterable[str],
                     ablation_base: Optional[str] = None
                     ) -> List[Tuple[str, str, str]]:
    """(method, reference, why) pairs used for decision-divergence checks."""
    keys = set(keys)
    pairs = []
    for k in keys:
        ref = reg[k].reference
        if ref and ref in keys:
            pairs.append((k, ref, "registered reference"))
    ladder = [k for k, _ in LADDER if k in keys]
    for a, b in zip(ladder[1:], ladder[:-1]):
        pairs.append((a, b, "ladder step"))
    if ablation_base and ablation_base in keys:
        for k in keys:
            if k != ablation_base and reg[k].group in ("ablation",
                                                       "sensitivity"):
                pairs.append((k, ablation_base, "vs full method"))
    seen, out = set(), []
    for a, b, why in pairs:
        if (a, b) in seen or a == b:
            continue
        seen.add((a, b))
        out.append((a, b, why))
    return sorted(out)


def select_methods(reg: Dict[str, MethodSpec], suites: Iterable[str],
                   extra: Iterable[str] = (), drop_mild: bool = False
                   ) -> List[MethodSpec]:
    suites = [s.strip() for s in suites if s and s.strip()]
    unknown = [s for s in suites if s not in
               ("main", "core", "ladder", "ablation", "sensitivity",
                "confirm", "forecast", "v1ablation", "v1tune",
                "v1fullmild", "v1paper")]
    if unknown:
        raise ValueError(f"unknown suite(s): {unknown}")
    keys = [k for k, s in reg.items() if set(s.suites) & set(suites)]
    for k in extra:
        k = k.strip()
        if not k:
            continue
        if k not in reg:
            raise ValueError(f"unknown method key {k!r}; see --list-methods")
        keys.append(k)
    keys = list(dict.fromkeys(keys))
    # A variant is only interpretable next to its reference, so the direct
    # reference of every ablation/sensitivity variant is always included.
    for k in list(keys):
        ref = reg[k].reference
        if reg[k].group in ("ablation", "sensitivity", "ladder",
                            "confirm", "forecast") and ref:
            if ref not in keys:
                keys.append(ref)
    out = [reg[k] for k in keys]
    if drop_mild:
        out = [s for s in out if s.risk != "mild"]
    return sorted(out, key=lambda s: s.order)
