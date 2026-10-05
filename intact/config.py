"""
intact/config.py
================
Configuration loading, validation, and hashing.

WHY A CONFIG FILE AT ALL:
Every number in the framework is either (a) a CONTRACT TERM agreed at
onboarding, or (b) an OPEN DESIGN PARAMETER that must be swept and reported
(v11 §12).  Neither should ever be hard-coded in the algorithm.  Putting
them all in one YAML file means a reviewer can see every knob at once, and
means every run folder carries an exact record of what produced it.
"""
from __future__ import annotations
import hashlib, json, copy
from pathlib import Path
from typing import Any, Dict
import yaml

from .types import Tenant, Intent, Claim, Scope, Kind, Direction


# These maps describe a complete generated v2 scenario, not a partial patch.
# Recursively merging them resurrects controls from base.yaml which the
# scenario generator deliberately omitted.  That happened in s001: four
# controls (quota_T2/T3/T4 and schedw_T3) silently reappeared.
_V2_ATOMIC_SECTIONS = (
    "ran.initial_controls",
    "ran.slices",
    "ran.envelopes",
    "sweep_domains",
)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge `override` into a COPY of `base`. Scenario files
    only need to state what they change from configs/base.yaml."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _path_get(mapping: Dict, dotted: str):
    cur = mapping
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def _path_set(mapping: Dict, dotted: str, value) -> None:
    cur = mapping
    keys = dotted.split(".")
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = copy.deepcopy(value)


def _find_in_list(lst, key):
    """
    Address a list element by INDEX ("0") or by IDENTITY ("j1", "i3", "x_H_eng").
    Identity matching is what you actually want:  --set claims.j1.step=0.5
    is readable and survives reordering, whereas claims.0.step does not.
    """
    if key.lstrip("-").isdigit():
        return lst[int(key)]
    for el in lst:
        if isinstance(el, dict) and any(el.get(f) == key
                                        for f in ("jid", "iid", "tid", "name")):
            return el
    have = [e.get("jid") or e.get("iid") or e.get("tid") or e.get("name")
            for e in lst if isinstance(e, dict)]
    raise KeyError("no list element with id %r; available: %r" % (key, have))


def _descend(cur, k):
    if isinstance(cur, list):
        return _find_in_list(cur, k)
    return cur.setdefault(k, {})


def _assign(cur, k, val):
    if isinstance(cur, list):
        _find_in_list(cur, k)                # will raise if absent
        if k.lstrip("-").isdigit():
            cur[int(k)] = val
        else:
            raise KeyError("cannot replace a whole list element by identity; "
                           "address a field inside it, e.g. claims.j1.step")
    else:
        cur[k] = val


def load_config(base_path: str, scenario_path: str | None = None,
                overrides: Dict[str, Any] | None = None) -> Dict:
    """
    Load base.yaml, optionally merge a scenario file on top, then apply
    command-line overrides (dotted keys, e.g. {"outer.theta_D": 1.5}).
    """
    cfg = yaml.safe_load(Path(base_path).read_text())
    if scenario_path:
        scenario = yaml.safe_load(Path(scenario_path).read_text()) or {}
        explicit = scenario.pop("_replace_sections", []) or []
        replace = set(str(p) for p in explicit)
        cfg = _deep_merge(cfg, scenario)
        replaced = []
        for dotted in sorted(replace):
            value, present = _path_get(scenario, dotted)
            if present:
                _path_set(cfg, dotted, value)
                replaced.append(dotted)
        if replaced:
            cfg["_scenario_atomic_sections"] = replaced
            cfg.setdefault("_notes", []).append(
                "Scenario atomically replaced: " + ", ".join(replaced))
        elif "v2" in scenario:
            inherited = []
            for dotted in _V2_ATOMIC_SECTIONS:
                base_value, base_present = _path_get(
                    yaml.safe_load(Path(base_path).read_text()) or {}, dotted)
                scen_value, scen_present = _path_get(scenario, dotted)
                if (base_present and scen_present and isinstance(base_value, dict)
                        and isinstance(scen_value, dict)):
                    extra = sorted(set(base_value) - set(scen_value))
                    if extra:
                        inherited.append(f"{dotted}={extra}")
            if inherited:
                cfg.setdefault("_notes", []).append(
                    "Legacy deep-merge inherited base keys: "
                    + "; ".join(inherited)
                    + ". Regenerate the scenario to make this explicit.")
    if overrides:
        for dotted, val in overrides.items():
            cur, keys = cfg, dotted.split(".")
            for k in keys[:-1]:
                cur = _descend(cur, k)
            _assign(cur, keys[-1], val)
    validate_config(cfg)
    return cfg


def config_hash(cfg: Dict) -> str:
    """Short deterministic hash of the config -- goes in the run folder name
    so two runs with identical settings are obviously identical."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def validate_config(cfg: Dict) -> None:
    """
    Fail LOUDLY and EARLY on anything that would silently produce nonsense.
    Each check maps to a specific claim in the theory document.
    """
    errs = []
    warns = []

    # A generated v2 scenario must not contain an actuator which is absent
    # from both the claim catalogue and the sweep domain.  This catches the
    # exact class of base/scenario merge contamination fixed above.
    if cfg.get("v2") and cfg.get("_scenario_atomic_sections"):
        claimed = {str(c["param"]) for c in cfg.get("claims", [])}
        controls = set((cfg.get("ran", {}).get("initial_controls", {}) or {}).keys())
        domains = set((cfg.get("sweep_domains", {}) or {}).keys())
        missing_controls = sorted(claimed - controls)
        missing_domains = sorted(claimed - domains)
        extra_controls = sorted(controls - claimed)
        if missing_controls:
            errs.append("v2 claims reference controls missing from "
                        f"ran.initial_controls: {missing_controls}")
        if missing_domains:
            errs.append("v2 claims reference parameters missing from "
                        f"sweep_domains: {missing_domains}")
        if extra_controls:
            errs.append("v2 ran.initial_controls contains unclaimed parameters "
                        f"(usually a base/scenario merge leak): {extra_controls}")

    # --- v11 §5:  lambda_u must be > 0, and u(p) must be >= 1 always. -------
    # If someone substitutes raw risk for the urgency multiplier, a healthy
    # intent gets w_i ~ 0 and the scheduler goes BLIND whenever the network
    # is healthy.  That is a live bug, not a simplification.
    if cfg["outer"]["lambda_u"] <= 0:
        errs.append("outer.lambda_u must be > 0 (v11 §5: urgency is a MULTIPLIER, "
                    "bounded below by 1; raw risk would zero out healthy intents)")

    # --- v11 §8.4:  admission test.  Contracted rates must be achievable. ---
    # Claims that share a knob are mutually exclusive (C1), so their rates
    # must sum to <= 1.  Two claims each demanding 80% need 160% -> infeasible.
    by_param: Dict[str, float] = {}
    for c in cfg["claims"]:
        by_param[c["param"]] = by_param.get(c["param"], 0.0) + c["r_j"]
    for p, tot in by_param.items():
        if tot > 1.0 + 1e-9:
            # *** INTACTv2 CHANGE. ***  v1 made this a HARD ERROR, which
            # meant every scenario the generator could emit had
            # sum(r_j) <= 1.0 on every C1 group.  Measured across the v1
            # 50-scenario ensemble: 60 of 60 groups fitted.  Both competitors
            # could therefore always be served, the deficit counter simply
            # rotated them, and priority only changed WHICH EPOCH each fired
            # -- which an 800-epoch average integrates away.  The finding
            # "weights do nothing" was thus GUARANTEED BY AN ADMISSION RULE,
            # not discovered from data.
            #
            # Over-subscription is normal in real contracting (airlines
            # overbook; cloud providers oversubscribe).  The contracts are
            # jointly infeasible ON PURPOSE, and rationing them by priority
            # is exactly the scheduler's job.  So v2 permits it behind an
            # explicit opt-in and reports the implied shortfall, rather than
            # refusing to represent the only regime in which priority can
            # possibly bind.
            if cfg.get("v2", {}).get("allow_oversubscription", False):
                warns.append(
                    f"C1 group '{p}' is OVER-SUBSCRIBED: contracted rates sum "
                    f"to {tot:.2f} > 1.0. At most 1.0 can be served, so at "
                    f"least {tot - 1.0:.2f} of contracted actuation rate MUST "
                    f"go unmet each epoch. Priority decides who goes short -- "
                    f"this is the intended INTACTv2 regime.")
            else:
                errs.append(f"C1 admission test FAILED for parameter '{p}': contracted "
                            f"rates sum to {tot:.2f} > 1.0.  These claims are mutually "
                            f"exclusive so this is infeasible (v11 §8.4). Renegotiate r_j. "
                            f"Set v2.allow_oversubscription: true to permit it deliberately.")

    # --- v11 §8.3.1:  C2 admission test, per (tenant, resource). -----------
    # sum of d_bar over a tenant's ALLOCATIVE claims on resource r must fit
    # E_{n,r}.  NOTE this is only a WARNING at config time, because a failing
    # sum does not make the config invalid -- it just means those claims can
    # never be co-eligible.  The mask (outer/constraints.py) handles it.
    env = {t["tid"]: t.get("envelope", {}) for t in cfg["tenants"]}
    alloc: Dict[tuple, float] = {}
    for c in cfg["claims"]:
        if c["kind"] == "allocative":
            key = (c["tenant"], c["resource"])
            alloc[key] = alloc.get(key, 0.0) + c["d_bar"]
    cfg.setdefault("_notes", [])
    for (tid, res), tot in alloc.items():
        cap = env.get(tid, {}).get(res)
        if cap is not None and tot > cap:
            cfg["_notes"].append(
                f"C2 will BIND within {tid}/{res}: sum d_bar={tot:.1f} > E={cap:.1f}. "
                f"Those allocative claims cannot be co-eligible (v11 §8.3.1). "
                f"This is expected and is what the admissibility mask encodes.")

    # --- cross-tenant sanity:  envelopes must fit the cell ------------------
    tot_env = sum(t.get("envelope", {}).get("PRB", 0.0) for t in cfg["tenants"])
    if tot_env > cfg["ran"]["n_prb"]:
        errs.append(f"Sum of tenant PRB envelopes ({tot_env}) exceeds the cell "
                    f"({cfg['ran']['n_prb']}).  Cross-tenant over-commitment must be "
                    f"impossible by construction (v11 §8.3.1) -- fix the envelopes.")

    # --- v11 §10.2:  the risk horizon must outrun margin erosion -----------
    if cfg["risk"]["horizon_slots"] < cfg["ran"]["slots_per_epoch"]:
        errs.append("risk.horizon_slots < ran.slots_per_epoch: the risk predictor "
                    "cannot warn before an epoch completes (v11 §10.2 ratchet).")

    # --- cell-scoped parameters must be host-owned (v11 §2.1) --------------
    hosts = {t["tid"] for t in cfg["tenants"] if t.get("is_host")}
    for c in cfg["claims"]:
        if str(c["scope"]).upper() == "CELL" and c["tenant"] not in hosts:
            errs.append(f"Claim {c['jid']} is CELL-scoped but owned by non-host "
                        f"tenant {c['tenant']}.  v11 §2.1: cell scope is host-only.")

    if errs:
        raise ValueError("CONFIG VALIDATION FAILED:\n  - " + "\n  - ".join(errs))
    for w in warns:
        print(f"[config warning] {w}")


# ---------------------------------------------------------------------------
# Builders: turn the plain dicts from YAML into the typed objects.
# ---------------------------------------------------------------------------
def build_tenants(cfg: Dict) -> Dict[str, Tenant]:
    return {t["tid"]: Tenant(tid=t["tid"], omega=t["omega"], rho_min=t["rho_min"],
                             envelope=t.get("envelope", {}),
                             B_n=t.get("B_n", 20), Bbar_n=t.get("Bbar_n", 200.0),
                             is_host=bool(t.get("is_host", False)))
            for t in cfg["tenants"]}


def build_intents(cfg: Dict) -> Dict[str, Intent]:
    return {i["iid"]: Intent(iid=i["iid"], tenant=i["tenant"], kpi=i["kpi"],
                             target=i["target"],
                             direction=Direction(i["direction"]),
                             pi_class=i["pi_class"], eta=i.get("eta", 0.95),
                             epsilon=i.get("epsilon", cfg["inner"]["epsilon_default"]),
                             clip=float(cfg.get("margins", {}).get("clip", float("inf"))))
            for i in cfg["intents"]}


def build_claims(cfg: Dict) -> Dict[str, Claim]:
    return {c["jid"]: Claim(jid=c["jid"], xapp=c["xapp"], tenant=c["tenant"],
                            param=c["param"], scope=Scope(c["scope"]),
                            kind=Kind(c["kind"]),
                            domain=tuple(c["domain"]), step=c["step"],
                            r_j=c["r_j"], resource=c.get("resource"),
                            d_bar=c.get("d_bar", 0.0),
                            max_step_frac=float(c.get("max_step_frac",
                                cfg.get("inner", {}).get("max_step_frac", 1.0))))
            for c in cfg["claims"]}
