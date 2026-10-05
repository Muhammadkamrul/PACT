#!/usr/bin/env python3
"""Check that this repository's code is byte-identical to the code that produced a run.

run_w3.py stores in <run>/manifest.json a SHA-256 over: every .py file under intact/
(sorted, path + bytes), scripts/run_w3.py, scripts/run_intact_w_revised.py,
configs/base.yaml and the registration file.  This script recomputes the same hash.

Usage:
  python scripts/verify_code_hash.py --run runs_w3v_confirmation --registration w3_docs/REGISTRATION_W3V.json
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def code_hash(registration: str) -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / 'intact').rglob('*.py')):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    for p in (ROOT / 'scripts/run_w3.py', ROOT / 'scripts/run_intact_w_revised.py',
              ROOT / 'configs/base.yaml', ROOT / registration):
        h.update(p.read_bytes())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run folder containing manifest.json')
    ap.add_argument('--registration', default='w3_docs/REGISTRATION_W3V.json')
    a = ap.parse_args()
    man = json.loads((Path(a.run) / 'manifest.json').read_text())
    mine = code_hash(a.registration)
    ok = mine == man['code_sha256']
    print(('MATCH' if ok else 'MISMATCH') + f": repository {mine[:16]}...  run {man['code_sha256'][:16]}...")
    if not ok:
        print('The code differs from the code that produced this run (check line endings: the repository must not convert CRLF/LF).')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
