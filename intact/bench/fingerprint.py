"""Fingerprints that decide whether a cached benchmark result is reusable.

Only code that can change a *simulation result* is hashed: the simulator
package ``intact/`` (excluding the report-only statistics module and this
file), plus the two scripts whose functions the benchmark executes
(``scripts/run_v2.py`` and ``scripts/all_reject_baseline.py``).  Editing a
figure or a table in ``scripts/benchmark_report.py`` therefore never forces
an expensive recomputation, while editing any controller, predictor, RAN or
metric code always does.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Files that cannot influence a simulation result.
_EXCLUDED = {
    "intact/bench/stats.py",
    "intact/bench/fingerprint.py",
    "intact/bench/registry.py",   # method specs are hashed per task instead
}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_code_sha256(project_root: str | Path) -> str:
    root = Path(project_root)
    files = sorted((root / "intact").rglob("*.py"))
    files += [root / "scripts" / "run_v2.py",
              root / "scripts" / "all_reject_baseline.py"]
    h = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in _EXCLUDED or "__pycache__" in rel:
            continue
        h.update(rel.encode())
        h.update(bytes.fromhex(file_sha256(path)))
    return h.hexdigest()


def json_sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                     default=str).encode()
    return hashlib.sha256(raw).hexdigest()
