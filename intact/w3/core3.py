"""INTACT-W3: decision-objective repair on top of the frozen w_revised harness.

w_revised is NOT modified (frozen provenance of the 2101-2120 confirmation).
This module adds methods and diagnostics only:

  W-prob   : sum_i w_i * Phi( (h(g_i) + alpha_i + sum_j beta_ji Z_j + sum gamma Z Z) / sigma_i )
             sigma_i = held-out residual SD of the joint predictor (computed by
             calibrate() and stored as cal['residual_sd'] but unused by W).
             This is a calibrated probability-of-fulfilment objective: when the
             predicted level is uncertain it tends to a linear beta weighting,
             when it is certain it tends to the threshold.  alpha, beta, gamma,
             MILD (urgency + inner risk), pi, omega and C3 keep their roles.
  W-lin    : sum_i w_i * (sum_j beta_ji Z_j + sum gamma Z Z)   (no level threshold)
  W-prob-* : ablations of W-prob (no-alpha, no-beta, no-gamma, scalar, no-MILD,
             no-urgency, no-pi, no-omega, no-C3)
"""
from __future__ import annotations
import copy, json, os, pickle, time
from pathlib import Path
import numpy as np
from scipy.stats import norm, spearmanr
from ..w_revised import core as C

_ORIG_SELECT = C.select
_ORIG_SCORES = C.scores

NEW = ['W3R-V','W3R-V-nogamma','W3R-V-noalpha','W3R-V-nopi','W3R-V-noomega','W3R-V-noC3','W3R-V-noMILD','W3R-V-nourgency','W3R-V-nobeta','W3R-V-scalar','W3R-V-lin','W3R','W3R-noalpha','W3R-nobeta','W3R-nogamma','W3R-scalar','W3R-noMILD','W3R-nourgency',
       'W3R-nopi','W3R-noomega','W3R-noC3','W3R-lin','W3','W3-noalpha','W3-nobeta','W3-nogamma','W3-scalar','W3-noMILD','W3-nourgency',
       'W3-nopi','W3-noomega','W3-noC3','W3-nocontext','W-prob','W-lin','W-prob-noalpha','W-prob-nobeta','W-prob-nogamma','W-prob-scalar',
       'W-prob-noMILD','W-prob-nourgency','W-prob-nopi','W-prob-noomega','W-prob-noC3']
METHODS3 = C.METHODS + NEW

def _model_key(method):
    if method in ('W3-scalar','W3-nocontext'): return 'scalar'
    for suf,key in (('noalpha','no-alpha'),('nobeta','no-beta'),('nogamma','no-gamma'),('scalar','scalar')):
        if method.endswith(suf): return key
    return 'full'

def _weight_method(method):
    method=method.replace('W3R-V','W3R')
    return {'W-prob-nopi':'no-pi','W-prob-noomega':'no-omega','W-prob-nourgency':'no-urgency',
            'W-prob-noC3':'no-C3','W3-nopi':'no-pi','W3-noomega':'no-omega',
            'W3-nourgency':'no-urgency','W3-noC3':'no-C3','W3R-nopi':'no-pi','W3R-noomega':'no-omega',
            'W3R-nourgency':'no-urgency','W3R-noC3':'no-C3'}.get(method,'W')

def predicted(w,cal,method):
    x=w.state(); n=len(w.sets)
    model=cal['fits'][_model_key(method)]
    return model.predict(np.tile(x,(n,1)),None,w.Z)          # (n_sets, I): alpha+beta+gamma

def w3_reliability(cal_dir, cal):
    """kappa_i (held-out skill of alpha) and sigma_i (held-out residual SD of the
    W3 level predictor), from the CACHED paired calibration episodes, using only
    the held-out (last 25%) whole episodes.  Stored next to calibration.pkl."""
    cal_dir=Path(cal_dir); f=cal_dir/'w3_reliability.json'
    if f.exists():
        d=json.loads(f.read_text()); return np.array(d['kappa']),np.array(d['sigma']),np.array(d['sigma_noalpha'])
    eps=sorted(cal_dir.glob('paired_episode_*.pkl')); n=len(eps); test=eps[int(.75*n):]
    X=[];A=[];J=[];Zs=[]
    for e in test:
        d=pickle.loads(e.read_bytes())
        X+=d['x']; A+=d['a']; J+=d['joint']; Zs+=d['z']
    X=np.array(X);A=np.array(A);Jt=np.array(J);Z=np.array(Zs)
    m=cal['fits']['full']; a_hat,b_hat,g_hat=m.components(X)
    var=np.maximum(A.var(axis=0),1e-9); mse=((a_hat-A)**2).mean(axis=0)
    kappa=np.clip(1-mse/var,0,1)
    inc=m.predict(X,None,Z)-a_hat                      # beta/gamma part only
    resid=Jt-(kappa[None,:]*a_hat+inc); resid0=Jt-inc
    sigma=np.maximum(resid.std(axis=0),.03); sigma0=np.maximum(resid0.std(axis=0),.03)
    C.atomic(f,{'kappa':kappa.tolist(),'sigma':sigma.tolist(),'sigma_noalpha':sigma0.tolist(),
                'alpha_mse':mse.tolist(),'alpha_var':var.tolist(),'heldout_snapshots':int(len(X))})
    return kappa,sigma,sigma0

def scores3(w,cal,p,method):
    method=method.replace('W3R-V','W3R')
    wt=C.weights(w,p,_weight_method(method))
    if method.startswith('W3'):
        req=method.startswith('W3R')
        cc=cal['req'] if req else cal
        kappa,sig,sig0=cc['_w3_rel']
        x=xreq(w) if req else w.state(); n=len(w.sets)
        mk=method.replace('W3R','W3')
        model=cc['fits'][_model_key(mk)]
        method=mk
        a,b,g=model.components(x[None,:])
        a=a[0]; b=b[0]; g=g[0]
        if method=='W3-nobeta': b=np.zeros_like(b)
        if method=='W3-nogamma': g=np.zeros_like(g)
        inc=w.Z@b+(w.P@g if len(w.pairs) else 0.)
        if method=='W3-lin': return inc@wt
        if method=='W3-noalpha': level=np.clip(w.g,-1,1)[None,:]+inc; s_=sig0
        else: level=np.clip(w.g,-1,1)[None,:]+kappa[None,:]*a[None,:]+inc; s_=sig
        return norm.cdf(level/s_[None,:])@wt
    dg=predicted(w,cal,method)
    if method=='W-lin':
        a=cal['fits']['full'].components(w.state()[None,:])[0]   # alpha is common to all sets
        return (dg-a)@wt
    sig=np.asarray(cal['residual_sd'])
    level=np.clip(w.g[None,:],-1,1)+dg
    return norm.cdf(level/sig[None,:])@wt

def select3(w,cal,p,method):
    if method not in NEW: return _ORIG_SELECT(w,cal,p,method)
    u=scores3(w,cal,p,method); best=int(np.argmax(u))
    empty=next(i for i,a in enumerate(w.sets) if not a)
    if u[best]<=u[empty]+w.cfg['w'].get('min_gain_prob',0.001): return set()
    return w.sets[best]

def risk_for(method,w,cal):
    method=method.replace('W3R-V','W3R')
    if method in ('no-MILD','reactive-refit','B3-contract-reactive','W-prob-noMILD','W3-noMILD','W3R-noMILD'): return C.reactive(w)
    return cal['risk'].predict(w.state())[0]

def evaluate3(cfg,seed,cal,method,path,epochs):
    """Same loop and metrics as w_revised.core.evaluate; dispatches select3."""
    return _evaluate(cfg,seed,cal,method,path,epochs)

def _evaluate(cfg,seed,cal,method,path,epochs):
    # copy of C.evaluate with risk_for() so new no-MILD variant is handled
    path=Path(path); final=path/'result.json'; checkpoint=path/'checkpoint.pkl'
    if final.exists(): return json.loads(final.read_text())
    path.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists(): w,records=pickle.loads(checkpoint.read_bytes())
    else:
        w=C.World(cfg,seed,cal['sens']); records=[]
        if method=='binary-veto': w.inner.reject_instead_of_attenuate=True
    while w.epoch<epochs:
        p=risk_for(method,w,cal)
        start=time.perf_counter(); outer=w.epoch%cfg['w']['lease']==0
        if outer: w.lease_set=select3(w,cal,p,method)
        S=w.lease_set; lat=time.perf_counter()-start
        adv=method if method in C.METHODS else ('QACM-contract' if method.startswith('W3R-V') else 'W')
        writes,c1,c2=w.advance(S,p,adv)
        records.append({'epoch':w.epoch,'selected':sorted(S),'g':w.g.tolist(),'risk':p.tolist(),
                        'Lambda':w.L.tolist(),'controls':w.ran.current_controls(),'writes':writes,
                        'c1':c1,'c2':c2,'outer_decision':outer,'selection_ms':lat*1000,
                        'cell':{k:float(v) for k,v in w.kpm['_cell'].items() if isinstance(v,(int,float))},
                        'tenants':{t:{k:float(v) for k,v in w.kpm[t].items() if isinstance(v,(int,float))}
                                   for t in w.ran.slices}})
        if w.epoch%50==0: C.atomic(checkpoint,(w,records),True)
    f=np.mean(w.history,axis=0); rho=w.tenant_rates(f)
    pi=np.array([w.intents[i].pi_class for i in w.I]); omega=np.array([w.tenants[w.intents[i].tenant].omega for i in w.I])
    floors=np.array([w.tenants[t].rho_min for t in w.T])
    tw=np.array([w.tenants[t].omega for t in w.T])
    tpairs=[(a,b) for a in range(len(w.T)) for b in range(len(w.T)) if tw[a]>tw[b]]
    tinv=[float(ra[a]<ra[b]) for fa in w.history for ra in [w.tenant_rates(fa)] for a,b in tpairs]
    res={'method':method,'seed':seed,'epochs':epochs,'WIF':float(np.average(f,weights=pi)),
         'contract_weighted_fulfillment':float(np.average(f,weights=pi*omega)),
         'tenant_priority_inversion':float(np.mean(tinv)) if tinv else 0.,
         'C3_shortfall':float(np.maximum(0,floors-rho).mean()),'C3_breach_fraction':float(np.mean(rho<floors-1e-9)),
         'C1':sum(r['c1'] for r in records),'C2':sum(r['c2'] for r in records),
         'writes_per_epoch':sum(len(r['writes']) for r in records)/epochs,
         'applied_per_epoch':sum(d['outcome']!='reject' for r in records for d in r['writes'])/epochs,
         'selection_p95_ms':float(np.percentile([r['selection_ms'] for r in records if r['outer_decision']],95)),
         'tenant_fulfillment':dict(zip(w.T,rho.tolist())),'intent_fulfillment':dict(zip(w.I,f.tolist()))}
    tmp=path/'telemetry.jsonl.tmp'
    with tmp.open('w') as out:
        for r in records: out.write(json.dumps(r)+'\n')
    os.replace(tmp,path/'telemetry.jsonl'); C.atomic(checkpoint,(w,records),True); C.atomic(final,res)
    return res

# ---------------------------------------------------------------- diagnostic
def ranking_diagnostic(cfg,seed,cal,methods,n_states=60,tail=4,driver='B3-contract',out=None):
    """At states visited by ``driver``, roll out EVERY admissible portfolio from the
    same snapshot and RNG (offline twin) and compare each method's choice with the
    realised best.  Realised value = mean over (lease + `tail` held epochs) of
    sum_i w_i 1[g_i >= 0] with the method-independent contract weights."""
    w=C.World(cfg,seed,cal['sens']); rows=[]
    lease=cfg['w']['lease']; t=0
    while len(rows)<n_states:
        p=cal['risk'].predict(w.state())[0]
        if w.epoch%lease==0:
            wt=C.weights(w,p,'W')
            real=np.zeros(len(w.sets))
            for k,S in enumerate(w.sets):
                arm=copy.deepcopy(w); vals=[]
                for e in range(lease+tail):
                    pp=cal['risk'].predict(arm.state())[0]
                    arm.advance(S if e<lease else set(),pp); vals.append((arm.g>=0).astype(float)@wt)
                real[k]=np.mean(vals)
            row={'epoch':w.epoch,'best':float(real.max()),'empty':float(real[[i for i,a in enumerate(w.sets) if not a][0]])}
            for m in methods:
                S=select3(w,cal,p,m)
                k=w.sets.index(S) if S in w.sets else 0
                row[m]=float(real[k])
                if m in NEW or m=='W':
                    u=scores3(w,cal,p,m) if m in NEW else _ORIG_SCORES(w,cal,p,m)
                    rho=spearmanr(u,real).correlation if np.std(u)>0 and np.std(real)>0 else np.nan
                    row[m+'_spearman']=float(rho) if np.isfinite(rho) else None
            rows.append(row)
            S=select3(w,cal,p,driver); w.lease_set=S
        w.advance(w.lease_set if hasattr(w,'lease_set') else set(),p,driver)
    if out: C.atomic(Path(out),rows)
    return rows

# ---------------------------------------------------------------- request-aware calibration
def pending_requests(w):
    """Normalised pending request of every claim at the start of a lease:
    (feasible requested value - current value) / domain span, 0 if the xApp is idle.
    Computed exactly as B3 computes it (copy of the app, next-epoch clock)."""
    ctl=w.ran.current_controls(); out=[]
    for j,c in w.claims.items():
        app=copy.deepcopy(w.apps[c.xapp])
        if hasattr(app,'_t'): app._t=w.epoch+1
        req=app.propose(w.kpm,ctl,np.random.default_rng(1))
        d=0. if req is None else w.feasible_value(c,req,ctl[c.param])-ctl[c.param]
        out.append(d/max(c.domain[1]-c.domain[0],1e-9))
    return np.array(out)

def xreq(w): return np.r_[w.state(),pending_requests(w)]

def calibrate_req(cfg,seed,path,base_dir,episodes=12,length=64):
    """Paired-twin alpha/beta/gamma with the pending-request vector in the context.
    Reuses the frozen sweep + MILD model of the base calibration (base_dir)."""
    path=Path(path); done=path/'calibration_req.pkl'
    if done.exists(): return pickle.loads(done.read_bytes())
    base=pickle.loads((Path(base_dir)/'calibration.pkl').read_bytes())
    sens,risk=base['sens'],base['risk']; path.mkdir(parents=True,exist_ok=True)
    data={k:[] for k in ['x','a','b','g','z','joint','episode']}
    for e in range(episodes):
        f=path/f'reqpaired_episode_{e:03d}.pkl'
        if f.exists(): d=pickle.loads(f.read_bytes())
        else:
            w=C.World(cfg,seed+40000+e,sens); d={k:[] for k in data}
            for t in range(length):
                if t%2==0:
                    x=xreq(w)
                    held=C.paired_rollout(w,set(),risk); a=held-np.clip(w.g,-1,1)
                    B=np.array([C.paired_rollout(w,{j},risk)-held for j in w.J])
                    G=np.array([C.paired_rollout(w,{w.J[u],w.J[v]},risk)-held-B[u]-B[v] for u,v in w.pairs])
                    idx=int(w.rng.integers(len(w.sets)))
                    jt=C.paired_rollout(w,w.sets[idx],risk)-np.clip(w.g,-1,1)
                    for k,v in [('x',x),('a',a),('b',B),('g',G),('z',w.Z[idx]),('joint',jt),('episode',e)]: d[k].append(v)
                S=w.sets[int(w.rng.integers(len(w.sets)))]
                for _ in range(cfg['w']['lease']): w.advance(S,risk.predict(w.state())[0])
            C.atomic(f,d,True)
        for k,v in d.items(): data[k].extend(v)
    data={k:np.array(v) for k,v in data.items()}; tr=data['episode']<int(.75*episodes); te=~tr
    w=C.World(cfg,seed,sens)
    model=C.PairedEffects(w.I,w.J,w.pairs).fit(data['x'][tr],data['a'][tr],data['b'][tr],data['g'][tr])
    fits={}
    for name in ['full','scalar','no-alpha','no-beta','no-gamma']:
        m=copy.copy(model); m.variant=name; fits[name]=m
    a_hat,b_hat,g_hat=model.components(data['x'][te])
    var=np.maximum(data['a'][te].var(axis=0),1e-9)
    kappa=np.clip(1-((a_hat-data['a'][te])**2).mean(axis=0)/var,0,1)
    inc=model.predict(data['x'][te],None,data['z'][te])-a_hat
    sig=np.maximum((data['joint'][te]-(kappa*a_hat+inc)).std(axis=0),.03)
    sig0=np.maximum((data['joint'][te]-inc).std(axis=0),.03)
    ev={'beta_RMSE':float(np.sqrt(((b_hat-data['b'][te])**2).mean())),
        'beta_mean_abs_truth':float(np.abs(data['b'][te]).mean()),
        'gamma_RMSE':float(np.sqrt(((g_hat-data['g'][te])**2).mean())),
        'gamma_mean_abs_truth':float(np.abs(data['g'][te]).mean()),
        'joint_RMSE':float(np.sqrt(((model.predict(data['x'][te],None,data['z'][te])-data['joint'][te])**2).mean())),
        'kappa':kappa.tolist(),'sigma':sig.tolist(),'heldout_snapshots':int(te.sum())}
    res={'sens':sens,'risk':base['risk'],'fits':fits,'_w3_rel':(kappa,sig,sig0),'evidence_req':ev,
         'residual_sd':base['residual_sd']}
    C.atomic(path/'evidence_req.json',ev); C.atomic(done,res,True); return res
