#!/usr/bin/env python3
"""Self-interpreting report for W3R runs (confirmation or development).

Writes <root>/report_w3/: REPORT.md (verdicts in words), method_means.csv,
paired.csv, ablations.csv and figures/*.{pdf,png}: method means, per-seed paired
differences, ablation deltas, and RAN telemetry of one seed (offered load, PRB
utilisation, radiated power, per-tenant throughput and delay, writes, selected
claims) for W3R vs B3-contract vs all-reject.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':7.5,'axes.grid':True,'grid.alpha':.3,'savefig.bbox':'tight','savefig.dpi':300})
_rp=next((sys.argv[i+1] for i,a in enumerate(sys.argv) if a=='--registration'),'w3_docs/REGISTRATION_W3.json')
REG=json.loads((ROOT/_rp).read_text()); CAND=REG['candidate']

def signflip(d,n=99999,seed=7):
    d=np.asarray(d,float); obs=abs(d.mean()); rng=np.random.default_rng(seed)
    s=rng.choice([-1.,1.],(n,len(d))); return float((1+np.sum(np.abs((s*d).mean(1))>=obs-1e-15))/(n+1))
def holm(p):
    k=sorted(p,key=p.get); m=len(k); out={}; run=0.
    for r,x in enumerate(k): run=max(run,min(1.,(m-r)*p[x])); out[x]=run
    return out
def boot(d,n=20000,seed=3):
    d=np.asarray(d,float); rng=np.random.default_rng(seed); b=rng.choice(d,(n,len(d))).mean(1)
    return float(np.quantile(b,.025)),float(np.quantile(b,.975))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='runs_w3_confirmation'); ap.add_argument('--label',default=''); ap.add_argument('--registration',default='w3_docs/REGISTRATION_W3.json')
    a=ap.parse_args(); root=Path(a.root); out=root/'report_w3'; (out/'figures').mkdir(parents=True,exist_ok=True)
    R={}
    for f in root.glob('s*/*/result.json'):
        r=json.loads(f.read_text()); R.setdefault(r['method'],{})[r['seed']]=r
    seeds=sorted(R.get(CAND,{}))
    if not seeds: sys.exit('no '+CAND+' results under '+str(root))
    split=a.label or ('CONFIRMATION' if set(seeds)<=set(REG['confirmation_seeds']) else 'DEVELOPMENT (not evidence for the paper)')
    complete={m:all(s in R[m] for s in seeds) for m in R}
    L=[f'# INTACT {CAND} report — {split}','',f'Seeds: {seeds} (n={len(seeds)}). C4 is removed everywhere.','',
       '## 1. Integrity','']
    exp=REG['primary']+REG['secondary']+REG['ablations']
    miss=[m for m in exp if m not in R or not complete.get(m)]
    L.append(f"- [{'PASS' if not miss else 'INCOMPLETE'}] registered methods complete for every seed"+(f" (missing: {', '.join(miss)})" if miss else ''))
    c12=max(R[CAND][s]['C1']+R[CAND][s]['C2'] for s in seeds)
    L.append(f"- [{'PASS' if c12==0 else 'FAIL'}] {CAND} records zero C1 and C2 violations")
    if split.startswith('CONF') and len(seeds)<len(REG['confirmation_seeds']): L.append('- WARNING: partial confirmation set; do not quote as final.')
    wif=lambda m:np.array([R[m][s]['WIF'] for s in seeds])
    rows=[]
    L+=['','## 2. Method means','','| Method | WIF | tenant-weighted | C3 shortfall | tenant inversion | writes/epoch |','|---|---:|---:|---:|---:|---:|']
    for m in sorted([m for m in R if complete[m]],key=lambda m:-wif(m).mean()):
        g=lambda k:np.mean([R[m][s][k] for s in seeds])
        L.append(f"| {m} | {g('WIF'):.4f} | {g('contract_weighted_fulfillment'):.4f} | {g('C3_shortfall'):.4f} | {g('tenant_priority_inversion'):.3f} | {g('writes_per_epoch'):.2f} |")
        rows.append((m,g('WIF'),g('contract_weighted_fulfillment'),g('C3_shortfall'),g('tenant_priority_inversion'),g('writes_per_epoch')))
    (out/'method_means.csv').write_text('method,WIF,tenant_weighted,C3_shortfall,tenant_inversion,writes\n'+'\n'.join(','.join([r[0]]+[f'{v:.5f}' for v in r[1:]]) for r in rows))
    def family(names,title,fname):
        names=[n for n in names if n in R and complete[n]]; P={}; S={}
        for n in names:
            d=100*(wif(CAND)-wif(n)); P[n]=signflip(d); S[n]=(d.mean(),*boot(d),int((d>0).sum()),int((d<0).sum()))
        H=holm(P) if P else {}
        L.extend(['',f'## {title}','',f'| Comparator | {CAND} minus (pp) | 95% CI | wins/losses | Holm p | verdict |','|---|---:|---:|---:|---:|---|'])
        lines=['comparator,delta_pp,lo,hi,wins,losses,p,p_holm']
        for n in names:
            dm,lo,hi,wn,ls=S[n]
            v=(f'{CAND} significantly better' if H[n]<.05 and dm>0 else f'{CAND} significantly WORSE' if H[n]<.05 else 'not resolved (no superiority, no equivalence claimed)')
            L.append(f'| {n} | {dm:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {wn}/{ls} | {H[n]:.4f} | {v} |')
            lines.append(f'{n},{dm:.4f},{lo:.4f},{hi:.4f},{wn},{ls},{P[n]:.5f},{H[n]:.5f}')
        (out/fname).write_text('\n'.join(lines)); return S,H
    Sp,Hp=family(REG['primary_comparisons'],'3. Registered primary comparisons (WIF)','paired.csv')
    sup=[n for n in Sp if Hp[n]<.05 and Sp[n][0]>0]; worse=[n for n in Sp if Hp[n]<.05 and Sp[n][0]<0]
    L+=['','**Automatic verdict:** '+(f"{CAND} is significantly better than ALL registered primary comparators." if len(sup)==len(Sp) and Sp else
        f"{CAND} is significantly better than {', '.join(sup) if sup else 'none'}; significantly worse than {', '.join(worse) if worse else 'none'}; others unresolved. Superiority over all primary comparators is NOT established.")]
    Sa,Ha=family(REG['ablations'],f'4. Component ablations ({CAND} minus variant; positive = component helps)','ablations.csv')
    helps=[n for n in Sa if Ha[n]<.05 and Sa[n][0]>0]; nulls=[n for n in Sa if not (Ha[n]<.05)]
    L+=['',f"Components with a significant WIF contribution: {', '.join(helps) if helps else 'none'}. Unresolved (do not claim): {', '.join(nulls) if nulls else 'none'}."]
    # ----- figures
    ms=sorted([m for m in R if complete[m]],key=lambda m:wif(m).mean())
    fig,ax=plt.subplots(figsize=(3.5,.18*len(ms)+.6)); y=np.arange(len(ms))
    ax.errorbar([wif(m).mean() for m in ms],y,xerr=[wif(m).std()/np.sqrt(len(seeds)) for m in ms],fmt='o',ms=3,c='k')
    for k,m in enumerate(ms):
        if m==CAND: ax.plot(wif(m).mean(),k,'o',c='#1F4E9E',ms=5)
    ax.set_yticks(y); ax.set_yticklabels(ms); ax.set_xlabel('WIF (mean ± s.e.)'); fig.savefig(out/'figures/fig_method_means.pdf'); fig.savefig(out/'figures/fig_method_means.png'); plt.close(fig)
    comps=[n for n in REG['primary_comparisons'] if n in Sp]
    fig,ax=plt.subplots(figsize=(3.5,1.8))
    for k,n in enumerate(comps):
        d=100*(wif(CAND)-wif(n)); ax.scatter(np.full(len(d),k)+np.random.default_rng(k).uniform(-.12,.12,len(d)),d,s=8,c='gray')
        ax.errorbar(k,Sp[n][0],yerr=[[Sp[n][0]-Sp[n][1]],[Sp[n][2]-Sp[n][0]]],fmt='D',c='k',ms=4,capsize=3)
    ax.axhline(0,c='k',lw=.8); ax.set_xticks(range(len(comps))); ax.set_xticklabels(comps,rotation=20,fontsize=6); ax.set_ylabel(f'{CAND} − comparator (pp)')
    fig.savefig(out/'figures/fig_paired.pdf'); fig.savefig(out/'figures/fig_paired.png'); plt.close(fig)
    ab=[n for n in REG['ablations'] if n in Sa]
    if ab:
        fig,ax=plt.subplots(figsize=(3.5,.2*len(ab)+.6))
        for k,n in enumerate(ab):
            c='#B2182B' if Ha[n]<.05 and Sa[n][0]>0 else 'gray'
            ax.errorbar(Sa[n][0],k,xerr=[[Sa[n][0]-Sa[n][1]],[Sa[n][2]-Sa[n][0]]],fmt='o',c=c,ms=3,capsize=2)
        ax.axvline(0,c='k',lw=.8); ax.set_yticks(range(len(ab))); ax.set_yticklabels(ab); ax.set_xlabel(f'{CAND} − variant (pp WIF); red = significant')
        fig.savefig(out/'figures/fig_ablation.pdf'); fig.savefig(out/'figures/fig_ablation.png'); plt.close(fig)
    s0=seeds[0]; tel={}
    for m in (CAND,'QACM-contract','B3-contract','all-reject'):
        f=root/f's{s0}'/m/'telemetry.jsonl'
        if f.exists(): tel[m]=[json.loads(x) for x in f.read_text().splitlines()]
    if tel and 'cell' in next(iter(tel.values()))[0]:
        rec=next(iter(tel.values()))[0]; tens=sorted(rec['tenants'])
        panels=[('offered load ratio',lambda r:r['cell'].get('offered_input_ratio',np.nan)),
                ('PRB utilisation (%)',lambda r:r['cell'].get('prb_util_pct',np.nan)),
                ('radiated power (W)',lambda r:r['cell'].get('radiated_power_w',np.nan)),
                ('Tx power (dBm)',lambda r:r['cell'].get('txpower_dbm',np.nan))]
        for t in tens:
            panels.append((f'{t} thr. (Mb/s)',lambda r,t=t:r['tenants'][t].get('throughput_mbps',np.nan)))
            panels.append((f'{t} delay (ms)',lambda r,t=t:r['tenants'][t].get('delay_ms',np.nan)))
        panels.append(('cumulative writes',None)); panels.append(('claims authorised',lambda r:len(r['selected'])))
        n=len(panels); fig,axs=plt.subplots((n+1)//2,2,figsize=(7.16,1.15*((n+1)//2)),sharex=True); axs=axs.ravel(); fig.subplots_adjust(wspace=.3,hspace=.35)
        sty={CAND:('-','#1F4E9E'),'QACM-contract':('-.','#7570B3'),'B3-contract':('--','k'),'all-reject':(':','#D95F02')}
        for k,(lab,fn) in enumerate(panels):
            for m,T in tel.items():
                x=np.arange(len(T))*0.5
                yv=np.cumsum([len(r['writes']) for r in T]) if fn is None else [fn(r) for r in T]
                axs[k].plot(x,yv,sty[m][0],c=sty[m][1],lw=.7,label=m)
            axs[k].set_title(lab,fontsize=6.5,pad=2)
        for k in range(n,len(axs)): axs[k].axis('off')
        axs[n-1].set_xlabel('time (s)'); axs[n-2].set_xlabel('time (s)')
        axs[0].legend(fontsize=5.5); fig.suptitle(f'RAN telemetry, seed {s0}',fontsize=7.5)
        fig.savefig(out/'figures/fig_telemetry.pdf'); fig.savefig(out/'figures/fig_telemetry.png'); plt.close(fig)
    L+=['',f'Figures: {out/"figures"}.']
    (out/'REPORT.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))
if __name__=='__main__': main()
