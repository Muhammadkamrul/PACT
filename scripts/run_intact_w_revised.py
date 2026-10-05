#!/usr/bin/env python3
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from intact.config import load_config
from intact.w_revised.core import World,atomic,calibrate,evaluate,contract_audit,METHODS

def scenario(seed,scope='feasible'):
    cfg=load_config(str(ROOT/'configs/base.yaml'));rng=np.random.default_rng(seed)
    cfg['v2']={'regime_load_edges':[.9,1.25],'forecast_source':'persistence'}
    cfg['mild']['enabled']=False;cfg['outer'].pop('theta_D',None);cfg['outer'].pop('deficit_cap',None)
    cfg['outer'].update(theta_Lambda=.3,lambda_u=.1,urgency_mode='absolute')
    cfg['w']={'slots':4,'horizon':6,'scope':scope,'calibration':'paired','min_gain':.003,'lease':2}
    cfg['ran'].update(slot_ms=125,slots_per_epoch=2,pre_slots=2,post_slots=2,
                      actuation_time_constant_epochs=0.,max_queue_kb=6.)
    cfg['ran']['load_profile']={'enabled':True,'levels':[.8,1.,1.25,1.05],
      'hold_slots':int(rng.integers(100,150)),'ramp_slots':40,'per_tenant_phase':True,'tenant_phase_stride':1}
    # Fixed SLA targets supported by offered traffic at the lowest load.
    targets={'i1':1.25,'i2':1.5,'i3':80.,'i4':.25,'i5':10**((42.5-30)/10)}
    cfg['ran']['slices']['T3']['load_mbps_per_ue']=1.7
    cfg['ran']['initial_controls']['quota_T3']=23
    cfg['ran']['envelopes']['T3']['PRB']=25
    for t in cfg['tenants']:
        if t['tid']=='T3':t['envelope']['PRB']=25
        t['rho_min']=.85 if not t.get('is_host') else .65
    for i in cfg['intents']:
        i['target']=targets[i['iid']]
        if i['iid']=='i5':i['kpi']='radiated_power_w'
    for t in ['T1','T2','T3']:
        cfg['intents'].append({'iid':'delay_'+t,'tenant':t,'kpi':'delay_ms','target':100.,
                               'direction':'lower_better','pi_class':.6,'epsilon':.02}) if t!='T3' else None
    for sl in cfg['ran']['slices'].values():sl['load_mbps_per_ue']*=float(rng.uniform(.9,1.1))
    for c in cfg['claims']:
        c.pop('r_j',None);c['max_step_frac']=.25
        if c['param']=='prbcap_T2':c['domain']=[8,35];c['step']=1
        if c['param']=='txpower':c['domain']=[38,46];c['step']=.5
    cfg['sweep_domains']['prbcap_T2']=[8,35];cfg['sweep_domains']['txpower']=[38,46]
    cfg['ran']['initial_controls']['prbcap_T2']=22
    for x in cfg['xapps']:
        x['forecast_aware']=False
        if x['param']=='prbcap_T2':x['domain']=[8,35];x['step']=1
        if x['param']=='txpower':x['domain']=[38,46];x['step']=.5
        if x['kind']=='throughput':x['target']=1.6 if x['tenant']=='T1' else 1.8;x['gain']=8.
        if x['kind']=='energy':
            x['target']=41.5 if x['param']=='txpower' else 18
            x.pop('target_alt',None);x['period']=0;x['gain']=.5
        if x['kind']=='robustness':x['kind']='adaptive_mcs';x['threshold']=4.
    cfg['xapps'].append({'name':'coverage_H','tenant':'H','kind':'coverage','param':'txpower','domain':[38,46],
                         'step':.5,'gain':.75,'watch_tenant':'T3','delay_target':80.})
    cfg['claims'].append({'jid':'j8','xapp':'coverage_H','tenant':'H','param':'txpower','scope':'cell',
                          'kind':'regulative','domain':[38,46],'step':.5,'max_step_frac':.25})
    cfg['outer']['coupled_pairs']=[['j4','j6'],['j4','j8'],['j3','j6'],['j5','j6']]
    if scope=='contested':
        cfg['ran']['slices']['T3']['load_mbps_per_ue']*=1.2
        cfg['intents'][0]['target']=1.45;cfg['intents'][1]['target']=1.7
    if scope=='slow':cfg['ran']['actuation_time_constant_epochs']=2.
    return cfg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs_w_revised');ap.add_argument('--scope',default='feasible',choices=['feasible','contested','slow'])
    ap.add_argument('--seeds',default='1101,1102,1103');ap.add_argument('--epochs',type=int,default=300)
    ap.add_argument('--episodes',type=int,default=12);ap.add_argument('--length',type=int,default=64);ap.add_argument('--risk-epochs',type=int,default=30)
    ap.add_argument('--lease',type=int,default=2,choices=[1,2]);ap.add_argument('--methods',default=','.join(METHODS));ap.add_argument('--split',default='development',choices=['development','confirmation'])
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    for p in sorted((ROOT/'intact').rglob('*.py')):h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes());h.update((ROOT/'configs/base.yaml').read_bytes())
    identity={**vars(args),'out':str(out),'code_sha256':h.hexdigest()}
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise SystemExit('Identity changed: use a new --out')
    atomic(manifest,identity)
    methods=args.methods.split(',')
    if set(methods)-set(METHODS):raise ValueError('Unknown method')
    for seed in map(int,args.seeds.split(',')):
        job=out/args.split/args.scope/f's{seed}';cfg=scenario(seed,args.scope);cfg['w']['lease']=args.lease;atomic(job/'config.json',cfg)
        print(f'{seed}: calibration',flush=True)
        cal=calibrate(cfg,seed,job/'calibration',args.episodes,args.length,args.risk_epochs)
        results=[]
        for m in methods:
            r=evaluate(cfg,seed,cal,m,job/m,args.epochs);results.append(r)
            print(f'{seed}: {m}: WIF={r["WIF"]:.4f}',flush=True)
        atomic(job/'contracts.json',contract_audit(cfg,results))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
