"""Auditable INTACT-W research harness. No oracle future features or planted KPI effects."""
from __future__ import annotations
import copy, itertools, json, logging, os, pickle, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression
from ..config import build_tenants, build_intents, build_claims
from .plant import RevisedRAN as AnalyticRAN
from .plant import build_apps
from ..estimation.margins import margin
from ..estimation.sensitivity import sweep, regime_key
from ..inner.arbiter import InnerLoop
from ..outer.constraints import AdmissibilityMask
from ..outer.weights import compute_weights
from ..xapps import build_xapps
from ..types import Write, Outcome
from ..mild.model import MildMoE
from ..mild.losses import lead_time_focal_bce

LOG = logging.getLogger('intact.w')
METHODS = ['W','QACM-style','QACM-contract','B3','B3-contract','B3-raw','B3-contract-raw',
           'B3-contract-reactive','reactive-refit','inner-priority','inner-random','inner-contract',
           'all-reject','v1-scalar','no-alpha','no-beta','no-gamma','no-context',
           'no-MILD','no-urgency','no-pi','no-omega','no-C3','binary-veto','outer-only']

def atomic(path, obj, binary=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    if binary:
        with tmp.open('wb') as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    else: tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

class World:
    def __init__(self,cfg,seed,sens=None):
        self.cfg=copy.deepcopy(cfg); self.cfg['ran']['seed']=int(seed)
        self.tenants=build_tenants(cfg); self.intents=build_intents(cfg); self.claims=build_claims({**cfg,'claims':[{**c,'r_j':0.} for c in cfg['claims']]})
        self.I=list(self.intents); self.J=list(self.claims); self.T=list(self.tenants)
        self.ran=AnalyticRAN(self.cfg,list(self.cfg['ran']['slices']))
        self.apps=build_apps(cfg); self.rng=np.random.default_rng(seed+131)
        self.mask=AdmissibilityMask(self.claims,self.tenants,LOG)
        groups=list(self.mask.knob_groups.values())
        self.sets=[set(j for j in x if j is not None) for x in itertools.product(*[[None]+g for g in groups])]
        self.sets=[s for s in self.sets if self.mask.is_admissible(s)]
        self.Z=np.array([[float(j in s) for j in self.J] for s in self.sets])
        self.pairs=[(self.J.index(a),self.J.index(b)) for a,b in cfg['outer']['coupled_pairs']
                    if any(a in s and b in s for s in self.sets)]
        self.P=np.array([[z[a]*z[b] for a,b in self.pairs] for z in self.Z])
        self.sens=sens
        if sens is not None: self.inner=InnerLoop(self.claims,self.intents,self.tenants,sens,cfg,LOG)
        self.kpm=self.ran.step(cfg['w']['slots']); self.g=self.margins(self.kpm)
        self.prev=self.g.copy(); self.epoch=0
        self.L=np.zeros(len(self.T)); self.history=[]
    def margins(self,k): return np.array([margin(self.intents[i],k) for i in self.I])
    def state(self):
        c=self.kpm['_cell']; ctl=self.ran.current_controls()
        # Readbacks are PRE-treatment state. Conditioning on them is valid for
        # randomized current authority, even though previous authority affected them.
        params={c.param:c for c in self.claims.values()}
        levels=[(ctl[p]-c.domain[0])/max(c.domain[1]-c.domain[0],1e-9) for p,c in params.items()]
        loads=[self.kpm[t]['offered_input_ratio'] for t in sorted(self.ran.slices)]
        queues=[self.kpm[t]['buffer_kb']/self.cfg['ran']['max_queue_kb'] for t in sorted(self.ran.slices)]
        return np.r_[np.clip(self.g,-3,3),np.clip(self.g-self.prev,-1,1),
                     loads,queues,c['prb_util_pct']/100,c['retx_prb']/self.ran.n_prb,levels]
    def basis(self,contextual=True):
        if not contextual: return np.ones(1)
        return np.r_[1.,self.kpm['_cell']['offered_input_ratio']-1,
                     np.clip(self.g,-1,1)]
    def advance(self,S,p,method='W'):
        self.epoch+=1; ctl=self.ran.current_controls(); reg=regime_key(self.kpm,self.cfg)
        g_running=dict(zip(self.I,self.g)); risk=dict(zip(self.I,p))
        writes=[]; parameters=set(); c1=c2=0
        if hasattr(self.sens,'set_state'): self.sens.set_state(self.state())
        for j in sorted(S):
            c=self.claims[j]; app=self.apps[c.xapp]
            if hasattr(app,'_t'): app._t=self.epoch
            # No forecast-aware apps; each queried at most once per epoch.
            req=app.propose(self.kpm,ctl,self.rng)
            if req is None: continue
            if method.startswith('QACM'):
                slope=np.array([self.sens.get(reg,c.param,i) for i in self.I])
                wt=weights(self,p,method) if method.endswith('contract') else np.ones(len(self.I))/len(self.I)
                grid=sorted(set(self.feasible_value(c,v,ctl[c.param]) for v in c.grid()))
                req=min(grid,key=lambda v:(float(np.sum(wt*np.minimum(self.g+slope*(v-ctl[c.param]),0)**2)),abs(v-ctl[c.param])))
            old=ctl[c.param]
            wr=Write(j,c.param,req,old,self.epoch,0)
            if method in ('outer-only','B3-raw','B3-contract-raw'):
                q=self.feasible_value(c,req,old); out=Outcome.ADMIT
            else:
                d=self.inner.decide(wr,reg,g_running,risk,self.ran); q=d.nu_star; out=d.outcome
            if out!=Outcome.REJECT:
                c1+=int(c.param in parameters); parameters.add(c.param)
                self.ran.apply(c.param,q); ctl[c.param]=q
                for i in self.I:
                    g_running[i]+=self.sens.get(reg,c.param,i)*(q-old)
                if c.resource:
                    env,used=self.ran.headroom(c.tenant,c.resource); c2+=int(used>env+1e-8)
            writes.append({'claim':j,'parameter':c.param,'requested':float(req),'old':float(old),
                           'executed':float(q),'outcome':out.value})
        self.prev=self.g.copy(); self.kpm=self.ran.step(self.cfg['w']['slots']); self.g=self.margins(self.kpm)
        self.history.append((self.g>=0).astype(float))
        recent=np.mean(self.history[-40:],axis=0); rho=self.tenant_rates(recent)
        self.L=np.minimum(10,np.maximum(0,self.L+np.array([t.rho_min for t in self.tenants.values()])-rho))
        return writes,c1,c2
    def tenant_rates(self,f):
        out=[]
        for t in self.T:
            idx=[k for k,i in enumerate(self.I) if self.intents[i].tenant==t]
            wt=[self.intents[self.I[k]].pi_class for k in idx]
            out.append(float(np.average(f[idx],weights=wt)) if idx else 1.)
        return np.array(out)
    def feasible_value(self,c,req,old):
        step=c.max_step_frac*(c.domain[1]-c.domain[0]); v=float(np.clip(req,old-step,old+step))
        grid=c.grid()
        if c.resource:
            env,used=self.ran.headroom(c.tenant,c.resource); grid=[x for x in grid if x<=env-used+old+1e-9]
        return min(grid,key=lambda x:abs(x-v)) if grid else old

class PairedEffects:
    """Frozen nonparametric contextual alpha/beta/gamma learned from paired twins.
    beta is a finite binary-authority effect, NOT a per-unit movement slope.
    """
    def __init__(self,I,J,pairs,variant='full'):
        self.I=I;self.J=J;self.pairs=pairs;self.variant=variant
    def fit(self,X,A,B,G):
        self.K=len(self.I);self.meanB=B.mean(0);self.meanG=G.mean(0)
        Y=np.c_[A,B.reshape(len(X),-1),G.reshape(len(X),-1)]
        self.output_scale=np.maximum(Y.std(axis=0),.03)
        self.model=ExtraTreesRegressor(n_estimators=48,min_samples_leaf=4,max_features=.9,random_state=411,n_jobs=1).fit(X,Y/self.output_scale)
        return self
    def components(self,x):
        y=self.model.predict(np.atleast_2d(x))*self.output_scale;K=self.K;J=len(self.J)
        a=y[:,:K];b=y[:,K:K+J*K].reshape(-1,J,K);g=y[:,K+J*K:].reshape(-1,len(self.pairs),K)
        if self.variant=='scalar':b=np.broadcast_to(self.meanB,b.shape);g=np.broadcast_to(self.meanG,g.shape)
        if self.variant=='no-alpha':a=np.zeros_like(a)
        if self.variant=='no-beta':b=np.zeros_like(b)
        if self.variant=='no-gamma':g=np.zeros_like(g)
        return a,b,g
    def predict(self,x,basis,z):
        x=np.atleast_2d(x);z=np.atleast_2d(z)
        # All candidate sets share one current state; avoid redundant tree traversals.
        shared=len(x)>1 and np.all(x==x[0])
        a,b,g=self.components(x[:1] if shared else x)
        p=np.stack([z[:,u]*z[:,v] for u,v in self.pairs],axis=1)
        return a+np.einsum('bj,bjk->bk',z,b)+np.einsum('bp,bpk->bk',p,g)

class Risk:
    """Existing numpy MILD MoE; new state features and held-control labels.
    Fit/early-stop/calibration/test are separate whole-episode partitions.
    """
    def fit(self,rows,I,epochs=30):
        x=np.array(rows['risk_x']); t=np.array(rows['ttf']); ep=np.array(rows['risk_episode'])
        groups=np.unique(ep)
        groups=np.array_split(groups,4); train=np.isin(ep,np.r_[groups[0],groups[1]])
        valid=np.isin(ep,groups[2]); test=np.isin(ep,groups[3])
        self.mean=x[train].mean(0); self.scale=np.maximum(x[train].std(0),.05)
        X=np.clip((x-self.mean)/self.scale,-8,8).astype('float32'); H=rows['horizon']
        self.model=MildMoE(x.shape[1],I,enc_units=(32,32),gate_units=16,seed=812)
        rng=np.random.default_rng(920); best=np.inf; best_weights=None
        for _ in range(epochs):
            idx=rng.permutation(np.flatnonzero(train))
            for batch in np.array_split(idx,max(1,len(idx)//64)):
                p,g=self.model.forward(X[batch]); loss,dp=lead_time_focal_bce(t[batch],p,H)
                self.model.backward(dp,np.zeros_like(g),.002)
            pv=self.model.predict(X[valid]); loss=float(np.mean((pv-(t[valid]>0))**2))
            if loss<best: best=loss; best_weights=self.model.get_weights()
        self.model.set_weights(best_weights)
        pv=self.model.predict(X[valid]); self.cal=[]
        for k in range(len(I)):
            self.cal.append(IsotonicRegression(out_of_bounds='clip').fit(pv[:,k],(t[valid,k]>0).astype(float)))
        pt=self.predict(x[test]); y=t[test]>0; cur=np.asarray(rows['risk_g'])[test]
        self.evidence=[]
        for k,i in enumerate(I):
            safe=cur[:,k]>=0; ys=y[safe,k]; ps=pt[safe,k]
            rate=float(ys.mean()) if len(ys) else 0.
            ap=float(average_precision_score(ys,ps)) if 0<rate<1 else None
            self.evidence.append({'intent':i,'safe_samples':int(safe.sum()),'future_events':int(ys.sum()),
                 'safe_PR_AUC':ap,'base_rate':rate,'lift':ap/rate if ap is not None else None,
                 'brier':float(np.mean((pt[:,k]-y[:,k])**2)),
                 'gate':bool(len(ys)>=30 and ys.sum()>=5 and ap is not None and ap>1.1*rate)})
        return self
    def predict(self,x):
        X=np.clip((np.atleast_2d(x)-self.mean)/self.scale,-8,8).astype('float32')
        p=self.model.predict(X)
        return np.stack([c.predict(p[:,k]) for k,c in enumerate(self.cal)],axis=1)

def reactive(w): return np.clip(.5-3*w.g-4*(w.g-w.prev),0,1)

def paired_rollout(w,S,risk):
    arm=copy.deepcopy(w);values=[]
    for _ in range(w.cfg['w']['lease']):
        p=reactive(arm) if risk is None else risk.predict(arm.state())[0]
        arm.advance(S,p);values.append(np.clip(arm.g,-1,1))
    return np.mean(values,axis=0)

def calibrate(cfg,seed,path,episodes=12,length=64,risk_epochs=30):
    path=Path(path);done=path/'calibration.pkl'
    if done.exists():return pickle.loads(done.read_bytes())
    path.mkdir(parents=True,exist_ok=True)
    sp=path/'sensitivity.pkl'
    if sp.exists():sens=pickle.loads(sp.read_bytes())
    else:
        world=World(cfg,seed)
        sens=sweep(world.ran,cfg,world.intents,sorted(cfg['sweep_domains']),LOG)
        atomic(sp,sens,True)
    data={k:[] for k in ['risk_x','risk_g','risk_episode','ttf']};data['horizon']=cfg['w']['horizon']
    for e in range(episodes):
        file=path/f'risk_episode_{e:03d}.pkl'
        if file.exists():d=pickle.loads(file.read_bytes())
        else:
            w=World(cfg,seed+30000+e,sens);d={k:[] for k in data if k!='horizon'}
            for t in range(length):
                if t%2==0:
                    clone=w.ran.clone();ttf=np.zeros(len(w.I))
                    for h in range(1,cfg['w']['horizon']+1):
                        future=w.margins(clone.step(cfg['w']['slots']));ttf[(ttf==0)&(future<0)]=h
                    d['risk_x'].append(w.state());d['risk_g'].append(w.g.copy());d['ttf'].append(ttf);d['risk_episode'].append(e)
                w.advance(w.sets[int(w.rng.integers(len(w.sets)))],reactive(w))
            atomic(file,d,True)
        for k,v in d.items():data[k].extend(v)
    risk=Risk().fit(data,list(build_intents(cfg)),risk_epochs)
    data={k:[] for k in ['x','a','b','g','z','joint','episode','ar','br','gr','jointr']}
    for e in range(episodes):
        file=path/f'paired_episode_{e:03d}.pkl'
        if file.exists():d=pickle.loads(file.read_bytes())
        else:
            w=World(cfg,seed+40000+e,sens);d={k:[] for k in data}
            for t in range(length):
                x=w.state();p=risk.predict(x)[0]
                if t%2==0:
                    # OFFLINE ONLY. Same snapshot and RNG for all intervention arms.
                    held=paired_rollout(w,set(),risk);a=held-np.clip(w.g,-1,1)
                    B=[]
                    for j in w.J:
                        arm=paired_rollout(w,{j},risk);B.append(arm-held)
                    B=np.array(B);G=[]
                    for u,v in w.pairs:
                        arm=paired_rollout(w,{w.J[u],w.J[v]},risk)
                        G.append(arm-held-B[u]-B[v])
                    idx=int(w.rng.integers(len(w.sets)))
                    arm=paired_rollout(w,w.sets[idx],risk)
                    for k,v in [('x',x),('a',a),('b',B),('g',np.array(G)),('z',w.Z[idx]),('joint',arm-np.clip(w.g,-1,1)),('episode',e)]:d[k].append(v)
                    # Matched reactive-risk refit on exactly the same pre-treatment snapshots.
                    hr=paired_rollout(w,set(),None)
                    br=np.array([paired_rollout(w,{j},None)-hr for j in w.J])
                    gr=np.array([paired_rollout(w,{w.J[u],w.J[v]},None)-hr-br[u]-br[v] for u,v in w.pairs])
                    jr=paired_rollout(w,w.sets[idx],None)-np.clip(w.g,-1,1)
                    for k,v in [('ar',hr-np.clip(w.g,-1,1)),('br',br),('gr',gr),('jointr',jr)]:d[k].append(v)
                S=w.sets[int(w.rng.integers(len(w.sets)))]
                for _ in range(cfg['w']['lease']):w.advance(S,risk.predict(w.state())[0])
            atomic(file,d,True)
        for k,v in d.items():data[k].extend(v)
    data={k:np.array(v) for k,v in data.items()};train=data['episode']<int(.75*episodes);test=~train
    w=World(cfg,seed,sens);base=PairedEffects(w.I,w.J,w.pairs).fit(data['x'][train],data['a'][train],data['b'][train],data['g'][train])
    fits={};evidence={}
    for name in ['full','scalar','no-alpha','no-beta','no-gamma']:
        f=copy.copy(base);f.variant=name;fits[name]=f
        pred=f.predict(data['x'][test],None,data['z'][test]);err=pred-data['joint'][test]
        evidence[name]={'heldout_RMSE':float(np.sqrt(np.mean(err**2))),
                       'zero_RMSE':float(np.sqrt(np.mean(data['joint'][test]**2))),
                       'per_intent_RMSE':dict(zip(w.I,np.sqrt(np.mean(err**2,axis=0)).tolist()))}
    ap,bp,gp=base.components(data['x'][test])
    evidence['paired_components']={
        'alpha_RMSE':float(np.sqrt(np.mean((ap-data['a'][test])**2))),
        'beta_RMSE':float(np.sqrt(np.mean((bp-data['b'][test])**2))),
        'gamma_RMSE':float(np.sqrt(np.mean((gp-data['g'][test])**2))),
        'beta_mean_abs_truth':float(np.mean(abs(data['b'][test]))),
        'gamma_mean_abs_truth':float(np.mean(abs(data['g'][test]))),
        'heldout_snapshots':int(sum(test)),
        'whole_episode_split':True,'paired_twin_requirement':True}
    residual=data['joint'][test]-fits['full'].predict(data['x'][test],None,data['z'][test])
    refit=PairedEffects(w.I,w.J,w.pairs).fit(data['x'][train],data['ar'][train],data['br'][train],data['gr'][train])
    fits['reactive-refit']=refit
    err=refit.predict(data['x'][test],None,data['z'][test])-data['jointr'][test]
    evidence['reactive-refit']={'heldout_RMSE':float(np.sqrt(np.mean(err**2))),
                              'matched_reactive_mediator':True}
    result={'sens':sens,'fits':fits,'risk':risk,'evidence':evidence,'mild_evidence':risk.evidence,
            'residual_sd':np.maximum(np.std(residual,axis=0),.03)}
    atomic(path/'evidence.json',{'effects':evidence,'MILD':risk.evidence})
    atomic(done,result,True);return result

def weights(w,p,method):
    cfg=copy.deepcopy(w.cfg)
    if method=='no-pi': cfg['outer']['weight_use_pi']=False
    if method=='no-omega': cfg['outer']['weight_use_omega']=False
    if method=='no-urgency': cfg['outer']['weight_use_urgency']=False
    if method=='no-C3': cfg['outer']['theta_Lambda']=0
    wi=compute_weights(w.intents,w.tenants,dict(zip(w.I,p)),dict(zip(w.T,w.L)),cfg)
    a=np.array([wi[i] for i in w.I]); return a/a.sum()

def scores(w,cal,p,method):
    x=w.state(); b=w.basis(); n=len(w.sets)
    wt=weights(w,p,method)
    utility=np.zeros(n)
    if not method.startswith(('B3','inner-','QACM')):
        key=method if method in ('no-alpha','no-beta','no-gamma','reactive-refit') else ('scalar' if method in ('no-context','v1-scalar') else 'full')
        model=cal['fits'][key]
        dg=model.predict(np.tile(x,(n,1)),np.tile(b,(n,1)),w.Z)
        # alpha matters here: outcome utility is nonlinear in predicted margin.
        utility=np.clip((np.clip(w.g[None,:],-1,1)+dg)/.2,-1,1)@wt
        if method=='v1-scalar': utility=dg@wt
    if method.startswith('B3'):
        # Greedy signed-sensitivity B3 adaptation with common inner mediation.
        V=[]; controls=w.ran.current_controls(); reg=regime_key(w.kpm,w.cfg)
        for j,c in w.claims.items():
            app=copy.deepcopy(w.apps[c.xapp])
            if hasattr(app,'_t'): app._t=w.epoch+1
            req=app.propose(w.kpm,controls,np.random.default_rng(1))
            delta=0 if req is None else w.feasible_value(c,req,controls[c.param])-controls[c.param]
            s=np.array([w.sens.get(reg,c.param,i) for i in w.I])
            V.append(float(s@ (wt if 'contract' in method else np.ones(len(w.I))/len(w.I)) *delta))
        utility=w.Z@np.array(V)
    if method in ('inner-priority','inner-contract','QACM-style','QACM-contract'):
        V=np.array([sum(wt[k] for k,i in enumerate(w.I) if w.intents[i].tenant==c.tenant)
                    if method in ('inner-contract','QACM-contract') else w.tenants[c.tenant].omega for c in w.claims.values()])
        utility=w.Z@V/len(w.J)
    if method=='inner-random': utility=w.rng.random(n)
    return utility

def select(w,cal,p,method):
    if method=='all-reject': return set()
    u=scores(w,cal,p,method)
    if method.startswith('B3'):
        # Incremental greedy maximisation; skip nonpositive marginal changes.
        s=set(); index={frozenset(s):i for i,s in enumerate(w.sets)}
        while True:
            options=[s|{j} for j in w.J if j not in s and frozenset(s|{j}) in index]
            if not options: break
            best=max(options,key=lambda a:u[index[frozenset(a)]])
            if u[index[frozenset(best)]]<=u[index[frozenset(s)]]+1e-12: break
            s=best
        return s
    best=int(np.argmax(u));empty=next(i for i,a in enumerate(w.sets) if not a)
    if method in ('W','reactive-refit','no-alpha','no-beta','no-gamma','no-context','no-MILD','no-urgency','no-pi','no-omega','no-C3','binary-veto','outer-only') and u[best]<=u[empty]+w.cfg['w']['min_gain']:return set()
    return w.sets[best]

def evaluate(cfg,seed,cal,method,path,epochs):
    path=Path(path); final=path/'result.json'; checkpoint=path/'checkpoint.pkl'
    if final.exists(): return json.loads(final.read_text())
    path.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():
        w,records=pickle.loads(checkpoint.read_bytes())
    else:
        w=World(cfg,seed,cal['sens']); records=[]
        if method=='binary-veto': w.inner.reject_instead_of_attenuate=True
    while w.epoch<epochs:
        p=reactive(w) if method in ('no-MILD','reactive-refit','B3-contract-reactive') else cal['risk'].predict(w.state())[0]
        # All primary methods share the fitted MILD inner risk. no-MILD is
        # explicitly a policy-shift ablation and uses reactive risk throughout.
        start=time.perf_counter();outer_decision=w.epoch%cfg['w']['lease']==0
        if outer_decision:w.lease_set=select(w,cal,p,method)
        S=w.lease_set;latency=time.perf_counter()-start
        writes,c1,c2=w.advance(S,p,method)
        records.append({'epoch':w.epoch,'time_s':w.epoch*cfg['w']['slots']*cfg['ran']['slot_ms']/1000,
             'selected':sorted(S),'g':w.g.tolist(),'risk':p.tolist(),'Lambda':w.L.tolist(),
             'kpm':w.kpm,'controls':w.ran.current_controls(),'writes':writes,
             'c1':c1,'c2':c2,'outer_decision':outer_decision,'selection_ms':latency*1000})
        if w.epoch%50==0: atomic(checkpoint,(w,records),True)
    f=np.mean(w.history,axis=0); rho=w.tenant_rates(f)
    pi=np.array([w.intents[i].pi_class for i in w.I]); omega=np.array([w.tenants[w.intents[i].tenant].omega for i in w.I])
    granted=np.mean([[j in r['selected'] for j in w.J] for r in records],axis=0)
    floors=np.array([w.tenants[t].rho_min for t in w.T])
    intent_pairs=[(a,b) for a in range(len(w.I)) for b in range(len(w.I)) if pi[a]>pi[b]]
    tenant_weights=np.array([w.tenants[t].omega for t in w.T])
    tenant_pairs=[(a,b) for a in range(len(w.T)) for b in range(len(w.T)) if tenant_weights[a]>tenant_weights[b]]
    intent_inversions=[float(fa[a]<fa[b]) for fa in w.history for a,b in intent_pairs]
    tenant_inversions=[float(ra[a]<ra[b]) for fa in w.history for ra in [w.tenant_rates(fa)] for a,b in tenant_pairs]
    result={'method':method,'seed':seed,'epochs':epochs,'WIF':float(np.average(f,weights=pi)),
        'contract_weighted_fulfillment':float(np.average(f,weights=pi*omega)),
        'intent_mean':float(f.mean()),
        'intent_priority_inversion':float(np.mean(intent_inversions)) if intent_inversions else 0.,
        'tenant_priority_inversion':float(np.mean(tenant_inversions)) if tenant_inversions else 0.,'C3_shortfall':float(np.maximum(0,floors-rho).mean()),
        'C3_breach_fraction':float(np.mean(rho<floors-1e-9)),
        'C1':sum(r['c1'] for r in records),'C2':sum(r['c2'] for r in records),
        'writes_per_epoch':sum(len(r['writes']) for r in records)/epochs,
        'applied_per_epoch':sum(d['outcome']!='reject' for r in records for d in r['writes'])/epochs,
        'selection_p95_ms':float(np.percentile([r['selection_ms'] for r in records if r['outer_decision']],95)),
        'tenant_fulfillment':dict(zip(w.T,rho.tolist())), 'intent_fulfillment':dict(zip(w.I,f.tolist())),
        'claim_grant_rate':dict(zip(w.J,granted.tolist()))}
    tmp=path/'telemetry.jsonl.tmp'
    with tmp.open('w') as out:
        for r in records: out.write(json.dumps(r)+'\n')
    os.replace(tmp,path/'telemetry.jsonl'); atomic(checkpoint,(w,records),True); atomic(final,result)
    return result

def contract_audit(cfg,results):
    w=World(cfg,0); floors=np.array([t.rho_min for t in w.tenants.values()])
    witnesses=[x['method'] for x in results if np.all(np.array([x['tenant_fulfillment'][t] for t in w.T])>=floors)]
    return {'C3_observed_finite_run_witnesses':witnesses,
            'C3_status':'empirical witness only' if witnesses else 'unresolved; absence of witness does not prove infeasible',
            'claim_service_contract':False}
