"""Reproducibility fingerprints for v2 run and ensemble resumption."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence


MANIFEST_SCHEMA = 2


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_sha256(project_root: str | Path) -> str:
    """Hash all Python source which can affect a run."""
    root = Path(project_root)
    h = hashlib.sha256()
    files = sorted((root / "intact").rglob("*.py"))
    files += sorted((root / "scripts").glob("*.py"))
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(file_sha256(path)))
    return h.hexdigest()


def build_run_manifest(project_root: str | Path, base: str | Path,
                       scenario: str | Path, modes: Sequence[str],
                       seeds: Sequence[int], train_epochs: int,
                       eval_epochs: int, overrides: Dict) -> Dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "base_name": Path(base).name,
        "base_sha256": file_sha256(base),
        "scenario_name": Path(scenario).name,
        "scenario_sha256": file_sha256(scenario),
        "code_sha256": code_sha256(project_root),
        "modes": list(modes),
        "seeds": [int(s) for s in seeds],
        "train_epochs": int(train_epochs),
        "eval_epochs": int(eval_epochs),
        "overrides": dict(sorted(overrides.items())),
    }


def manifest_mismatches(actual: Dict, expected: Dict) -> list[str]:
    keys = sorted(set(actual) | set(expected))
    return [k for k in keys if actual.get(k) != expected.get(k)]


def atomic_write_json(path: str | Path, obj: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)
