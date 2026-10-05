#!/usr/bin/env python3
"""INTACT-W3R confirmation / development runner (parallel over seeds, resumable).

Per seed: base calibration (sweep + MILD + paired effects, w_revised code, cached
per episode) -> request-aware paired calibration (w3, cached per episode) ->
every registered method for --epochs inner epochs (checkpointed every 50).
Rerun the SAME command to resume.  A changed code hash or argument set refuses
to reuse the directory (use a new --out).
"""
import argparse, hashlib, json, sys, os, pickle, multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))

_rp=next((sys.argv[i+1] for i,a in enumerate(sys.argv) if a=='--registration'),'w3_docs/REGISTRATION_W3.json')
REG=json.loads((ROOT/_rp).read_text()); CAND=REG['candidate']

def job(args):
    seed,out,methods,epochs=args
    import logging,warnings; logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
    from run_intact_w_revised import scenario
    from intact.w_revised.core import calibrate, atomic
    from intact.w3.core3 import calibrate_req, evaluate3, w3_reliability
    cfg=scenario(seed,REG['scope']); cfg['w']['lease']=REG['lease_inner_epochs']
    d=Path(out)/f's{seed}'; atomic(d/'config.json',cfg)
    cal=calibrate(cfg,seed,d/'calibration',REG['calibration_episodes'],REG['calibration_length'],REG['risk_training_epochs'])
    cal['_w3_rel']=w3_reliability(d/'calibration',cal)
    cal['req']=calibrate_req(cfg,seed,d/'calibration_req',d/'calibration',REG['calibration_episodes'],REG['calibration_length'])
    lines=[]
    for m in methods:
        r=evaluate3(cfg,seed,cal,m,d/m,epochs); lines.append(f"{seed} {m} WIF={r['WIF']:.4f}")
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',default='runs_w3_confirmation')
    ap.add_argument('--split',default='confirmation',choices=['confirmation','development','smoke'])
    ap.add_argument('--seeds',default='',help='override (development/smoke only)')
    ap.add_argument('--methods',default='',help='override method list (development/smoke only)')
    ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--registration',default='w3_docs/REGISTRATION_W3.json')
    a=ap.parse_args()
    if a.split=='confirmation' and (a.seeds or a.methods or a.epochs):
        sys.exit('confirmation uses the registered seeds/methods/epochs only ('+a.registration+')')
    seeds=[int(s) for s in a.seeds.split(',')] if a.seeds else REG['confirmation_seeds']
    if a.split!='confirmation' and set(seeds)&set(REG['confirmation_seeds']):
        sys.exit('refusing: development/smoke runs may not touch registered confirmation seeds')
    if a.split=='confirmation' and set(seeds)&set(REG['inspected_do_not_reuse']):
        sys.exit('refusing: registered confirmation overlaps inspected seeds')
    methods=a.methods.split(',') if a.methods else REG['primary']+REG['secondary']+REG['ablations']
    epochs=a.epochs or REG['evaluation_inner_epochs']
    h=hashlib.sha256()
    for p in sorted((ROOT/'intact').rglob('*.py')): h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    for p in (Path(__file__),ROOT/'scripts/run_intact_w_revised.py',ROOT/'configs/base.yaml',ROOT/a.registration): h.update(p.read_bytes())
    ident={'split':a.split,'seeds':seeds,'methods':methods,'epochs':epochs,'code_sha256':h.hexdigest()}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); man=out/'manifest.json'
    if man.exists() and json.loads(man.read_text())!=ident: sys.exit('Identity changed (code, seeds, methods or epochs): use a new --out')
    man.write_text(json.dumps(ident,indent=2))
    jobs=[(s,str(out),methods,epochs) for s in seeds]
    if a.workers<=1:
        for j in jobs: print(job(j),flush=True)
    else:
        with mp.get_context('fork').Pool(a.workers) as pool:
            for msg in pool.imap_unordered(job,jobs): print(msg,flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
