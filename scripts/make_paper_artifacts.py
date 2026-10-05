#!/usr/bin/env python3
"""Generate EVERY paper artefact for the frozen INTACT-W (= W3R-V) confirmation.

Reads only finished outputs (no new simulation):
  <root>/s<seed>/<method>/result.json, telemetry.jsonl, config.json,
  <root>/s<seed>/calibration/evidence.json, calibration_req/evidence_req.json
Writes <out>/:
  tables/   *.tex (IEEE-ready), *.md, *.csv
  figures/  *.pdf, *.png
  text/     environment.md, hyperparameters.md, results.md, claims_allowed.md
  ARTIFACT_INDEX.md   which artefact goes where in the paper

Usage:
  python scripts/make_paper_artifacts.py --root runs_w3v_confirmation \
      --registration w3_docs/REGISTRATION_W3V.json --out paper_artifacts
"""
from __future__ import annotations
import argparse, json, sys, statistics
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.grid': True, 'grid.alpha': .3,
                     'savefig.bbox': 'tight', 'savefig.dpi': 300})
W1, W2 = 3.5, 7.16

# ------------------------------------------------------------------ naming
PROPOSED = 'W3R-V'
DISPLAY = {
    'W3R-V': 'INTACT-W (proposed)',
    'QACM-style': 'QACM-style value synthesis',
    'B3': 'Greedy value arbitration (B3)',
    'inner-contract': 'Inner arbitration only',
    'all-reject': 'No arbitration (all-reject)',
    'QACM-contract': 'Value synthesis + INTACT weights (no outer selection)',
    'B3-contract': 'Greedy + INTACT weights',
    'B3-contract-raw': 'Greedy + INTACT weights, no inner mediator',
    'B3-contract-reactive': 'Greedy + INTACT weights, reactive risk',
    'W3R': 'INTACT-W with projection inner loop',
    'W': 'INTACT-W, previous objective',
    'W3R-V-noalpha': r'$-\alpha$ (drift)', 'W3R-V-nobeta': r'$-\beta$ (authority effects)',
    'W3R-V-nogamma': r'$-\gamma$ (pair interactions)', 'W3R-V-scalar': r'scalar $\beta,\gamma$ (no context; v1-style)',
    'W3R-V-noMILD': 'MILD $\\rightarrow$ reactive risk', 'W3R-V-nourgency': r'$-$ urgency $u(\hat p)$',
    'W3R-V-nopi': r'$-\pi$ (intent priority)', 'W3R-V-noomega': r'$-\omega$ (tenant priority)',
    'W3R-V-noC3': r'$-$ C3 deficit $\Lambda$', 'W3R-V-lin': r'linear objective (no $\Phi$)'}
PLAIN = {k: v.replace('$', '').replace('\\rightarrow', '->').replace('\\', '') for k, v in DISPLAY.items()}
PUBLISHED = ['QACM-style', 'B3', 'inner-contract', 'all-reject']             # principle-level prior art
MATCHED = ['QACM-contract', 'B3-contract', 'B3-contract-raw', 'B3-contract-reactive']  # given INTACT's weights/MILD/C3
DESIGN = ['W3R', 'W']
ABL = ['W3R-V-nobeta', 'W3R-V-scalar', 'W3R-V-noMILD', 'W3R-V-noalpha', 'W3R-V-nogamma', 'W3R-V-nourgency',
       'W3R-V-nopi', 'W3R-V-noomega', 'W3R-V-noC3', 'W3R-V-lin']


# ------------------------------------------------------------------ statistics
def signflip(d, n=99999, seed=7):
    d = np.asarray(d, float)
    if np.allclose(d, 0): return 1.0
    s = np.random.default_rng(seed).choice([-1., 1.], (n, len(d)))
    return float((1 + np.sum(np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-15)) / (n + 1))

def holm(p):
    out, run, k = {}, 0., sorted(p, key=p.get)
    for r, x in enumerate(k):
        run = max(run, min(1., (len(k) - r) * p[x])); out[x] = run
    return out

def boot(d, n=20000, seed=3):
    d = np.asarray(d, float); b = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


# ------------------------------------------------------------------ io helpers
def service_metrics(run_dir: Path, cfg: dict) -> dict:
    """Service-protection metrics of one run, from telemetry.jsonl (per-epoch post-decision
    margins).  Definitions follow the WCNC v1 manuscript exactly:
      intent miss     100 (1 - mean_i Phi_i)                       (unweighted)
      floor breach    100 * share of intent-holding tenants with F_n < rho_min (run level)
      safety crossing 100 * #(g_i >= eps_i at t-1 and g_i < eps_i at t) / ((T-1) * I)
      intent inversion: epoch-level pairs with unequal pi (excluding duplicate intents with the
                      same tenant, KPI, target and direction); inversion if the higher-pi
                      intent fails while the lower-pi one is met; rate over eligible pair-epochs
      tenant inversion: non-host tenant pairs with unequal omega; inversion if the higher-omega
                      tenant has lower epoch-level pi-weighted fulfilment
    Cached as service_metrics.json next to result.json."""
    cache = run_dir / 'service_metrics.json'
    tel = run_dir / 'telemetry.jsonl'
    if cache.exists() and (not tel.exists() or cache.stat().st_mtime >= tel.stat().st_mtime):
        return json.loads(cache.read_text())
    G = np.array([json.loads(z)['g'] for z in tel.read_text().splitlines()], float)
    ints = cfg['intents']; I = len(ints)
    eps = np.array([it.get('epsilon', 0.02) for it in ints]); pi = np.array([it['pi_class'] for it in ints])
    met = (G >= 0).astype(float); Phi = met.mean(0)
    tens = [t['tid'] for t in cfg['tenants']]; host = {t['tid'] for t in cfg['tenants'] if t.get('is_host')}
    omega = {t['tid']: t['omega'] for t in cfg['tenants']}; floor = {t['tid']: t['rho_min'] for t in cfg['tenants']}
    owner = [it['tenant'] for it in ints]
    def tf(mrow):
        out = {}
        for t in tens:
            idx = [k for k in range(I) if owner[k] == t]
            if idx: out[t] = float(np.dot(pi[idx], mrow[idx]) / pi[idx].sum())
        return out
    Frun = tf(Phi)
    breach = 100.0 * np.mean([Frun[t] < floor[t] - 1e-12 for t in Frun])
    cross = 100.0 * float(((G[:-1] >= eps) & (G[1:] < eps)).sum()) / max(1, (len(G) - 1) * I)
    pairs = [(a, b) for a in range(I) for b in range(I) if pi[a] > pi[b] and not (
        owner[a] == owner[b] and ints[a]['kpi'] == ints[b]['kpi'] and ints[a]['target'] == ints[b]['target']
        and ints[a]['direction'] == ints[b]['direction'])]
    iinv = 100.0 * float(np.mean([(met[:, a] == 0) & (met[:, b] == 1) for a, b in pairs])) if pairs else 0.0
    nh = [t for t in Frun if t not in host]
    tp = [(a, b) for a in nh for b in nh if omega[a] > omega[b]]
    vals = []
    for row in met:
        f = tf(row); vals += [float(f[a] < f[b] - 1e-12) for a, b in tp]
    tinv = 100.0 * float(np.mean(vals)) if vals else 0.0
    res = {'intent_miss_pct': float(100 * (1 - Phi.mean())), 'floor_breach_pct': float(breach),
           'safety_crossing_pct': cross, 'intent_inversion_pct': iinv, 'tenant_inversion_pct': tinv}
    cache.write_text(json.dumps(res, indent=2))
    return res


def save_table(out, name, header, rows, caption, label, align=None):
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    md = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    (out / 'tables' / f'{name}.md').write_text(f'**{caption}**\n\n' + '\n'.join(md) + '\n')
    (out / 'tables' / f'{name}.csv').write_text('\n'.join([','.join(header)] + [','.join('"' + str(c).replace('"', "'") + '"' for c in r) for r in rows]) + '\n')
    al = align or ('l' + 'r' * (len(header) - 1))
    esc = lambda s: str(s).replace('%', r'\%').replace('_', r'\_').replace('&', r'\&') if '$' not in str(s) else str(s)
    tex = [r'\begin{table}[t]', r'\centering', r'\scriptsize', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
           rf'\begin{{tabular}}{{{al}}}', r'\hline', ' & '.join(esc(h) for h in header) + r' \\', r'\hline']
    tex += [' & '.join(esc(c) for c in r) + r' \\' for r in rows] + [r'\hline', r'\end{tabular}', r'\end{table}']
    (out / 'tables' / f'{name}.tex').write_text('\n'.join(tex) + '\n')

def savefig(fig, out, name):
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'): fig.savefig(out / 'figures' / f'{name}.{ext}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='runs_w3v_confirmation')
    ap.add_argument('--registration', default='w3_docs/REGISTRATION_W3V.json')
    ap.add_argument('--out', default='paper_artifacts')
    ap.add_argument('--allow-incomplete', action='store_true', help='development/smoke use only')
    ap.add_argument('--name', default='PACT', help='paper name of the proposed method')
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    for k in list(DISPLAY):
        DISPLAY[k] = DISPLAY[k].replace('INTACT-W', a.name).replace('INTACT weights', a.name + ' weights')
    for k in list(PLAIN):
        PLAIN[k] = PLAIN[k].replace('INTACT-W', a.name).replace('INTACT weights', a.name + ' weights')
    REG = json.loads((ROOT / a.registration).read_text())
    out.mkdir(parents=True, exist_ok=True)
    log = []

    # ---------------------------------------------------------- load
    R = {}
    for f in root.glob('s*/*/result.json'):
        r = json.loads(f.read_text()); R.setdefault(r['method'], {})[r['seed']] = r
    seeds = sorted(R.get(PROPOSED, {}))
    if not seeds: sys.exit(f'no {PROPOSED} results in {root}')
    is_conf = set(seeds) == set(REG['confirmation_seeds'])
    expected = REG['primary'] + REG['secondary'] + REG['ablations']
    missing = [m for m in expected if m not in R or any(s not in R[m] for s in seeds)]
    if missing and not a.allow_incomplete:
        sys.exit(f'incomplete results for {missing}; finish the run (or pass --allow-incomplete for development)')
    methods = [m for m in expected if m in R and all(s in R[m] for s in seeds)]
    V = lambda m, k='WIF': np.array([R[m][s][k] for s in seeds], float)
    cfgs = {s: json.loads((root / f's{s}' / 'config.json').read_text()) for s in seeds}
    cfg = cfgs[seeds[0]]
    label = 'CONFIRMATION' if is_conf else 'DEVELOPMENT / PARTIAL (NOT paper evidence)'
    log.append(f'# Data: {root} — {label}; seeds {seeds[0]}–{seeds[-1]} (n={len(seeds)}); methods {len(methods)}')
    c12 = int(sum(R[PROPOSED][s]['C1'] + R[PROPOSED][s]['C2'] for s in seeds))

    # ---------------------------------------------------------- T1 environment
    rows = [['Cell', f"{cfg['ran']['n_prb']} PRB, {cfg['ran']['prb_bandwidth_hz']/1e3:.0f} kHz/PRB, carrier {cfg['ran']['carrier_ghz']} GHz"],
            ['Inner epoch', f"{cfg['w']['slots']} x {cfg['ran']['slot_ms']} ms = {cfg['w']['slots']*cfg['ran']['slot_ms']/1000:.2f} s"],
            ['Authority lease (outer)', f"{cfg['w']['lease']} inner epochs = {cfg['w']['lease']*cfg['w']['slots']*cfg['ran']['slot_ms']/1000:.1f} s"],
            ['Evaluation length', f"{REG['evaluation_inner_epochs']} inner epochs = {REG['evaluation_inner_epochs']*cfg['w']['slots']*cfg['ran']['slot_ms']/1000:.0f} s per scenario"],
            ['Scenarios', f"{len(seeds)} independent seeds ({seeds[0]}–{seeds[-1]}), one trajectory each"],
            ['Actuation lag', f"{cfg['ran']['actuation_time_constant_epochs']} epochs (none)"],
            ['Load profile', 'piecewise levels ' + ', '.join(str(x) for x in cfg['ran']['load_profile']['levels']) +
             f"; hold {min(c['ran']['load_profile']['hold_slots'] for c in cfgs.values())}–{max(c['ran']['load_profile']['hold_slots'] for c in cfgs.values())} slots, ramp {cfg['ran']['load_profile']['ramp_slots']} slots, per-tenant phase"],
            ['Queue limit', f"{cfg['ran']['max_queue_kb']} kB per slice"]]
    for t, sl in cfg['ran']['slices'].items():
        lo = min(c['ran']['slices'][t]['load_mbps_per_ue'] for c in cfgs.values()); hi = max(c['ran']['slices'][t]['load_mbps_per_ue'] for c in cfgs.values())
        rows.append([f'Slice {t}', f"{sl['n_ue']} UEs, {lo:.2f}–{hi:.2f} Mb/s per UE (seed-dependent)"])
    save_table(out, 'tab_environment', ['Item', 'Setting'], rows, 'Simulation environment.', 'tab:env', 'll')
    trow = [[t['tid'] + (' (host)' if t.get('is_host') else ''), t['omega'], t['rho_min'], t['envelope'].get('PRB', 0)] for t in cfg['tenants']]
    save_table(out, 'tab_tenants', ['Tenant', r'$\omega_n$', r'$\rho_n^{\min}$', 'PRB envelope'], trow, 'Tenants and contracts.', 'tab:tenants')
    irow = [[i['iid'], i['tenant'], i['kpi'].replace('_', ' '), f"{i['target']:.3g}", 'higher' if i['direction'].startswith('higher') else 'lower', i['pi_class']] for i in cfg['intents']]
    save_table(out, 'tab_intents', ['Intent', 'Tenant', 'KPI', 'Target', 'Better', r'$\pi_i$'], irow, 'Intents.', 'tab:intents')
    xk = {x['name']: x['kind'] for x in cfg['xapps']}
    crow = [[c['jid'], c['xapp'], xk.get(c['xapp'], ''), c['tenant'], c['param'].replace('_', ' '), f"[{c['domain'][0]}, {c['domain'][1]}]", c['step'], c['kind']] for c in cfg['claims']]
    save_table(out, 'tab_claims', ['Claim', 'xApp', 'Kind', 'Tenant', 'Parameter', 'Domain', 'Step', 'Type'], crow,
               'Claims (xApp, parameter) and control domains. Same-parameter claims form C1 groups.', 'tab:claims')

    # ---------------------------------------------------------- T2 hyperparameters
    o, inn, w = cfg['outer'], cfg['inner'], cfg['w']
    hp = [[r'Urgency gain $\lambda_u$ / $\epsilon_0$', f"{o['lambda_u']} / {o['eps0']}"],
          [r'C3 gain $\theta_\Lambda$; deficit window / cap', f"{o['theta_Lambda']}; 40 inner epochs / 10"],
          ['Hold threshold (min objective gain)', REG['frozen_settings'].get('min_gain_prob', 0.001)],
          [r'Inner risk threshold $\tau_{risk}$ / harm deadband $\delta$', f"{inn['tau_risk']} / {inn['delta']}"],
          [r'Protection floor $\epsilon_i$', inn['epsilon_default']],
          [r'Slew limit $\kappa$ (fraction of domain)', cfg['claims'][0]['max_step_frac']],
          ['Calibration', f"{REG['calibration_episodes']} episodes x {REG['calibration_length']} decisions; last 25% of whole episodes held out"],
          ['Paired arms per snapshot', 'empty, every single claim, every declared pair, one random portfolio (common random numbers)'],
          [r'Effect model ($\alpha,\beta,\gamma$)', 'ExtraTrees, 48 trees, min leaf 4, max features 0.9; context = state + pending requests'],
          [r'Reliability $\kappa_i$, $\sigma_i$', 'held-out skill of alpha; held-out residual SD of the level predictor'],
          ['MILD', f"MoE, encoder 32-32, gate 16; horizon {w['horizon']} inner epochs ({w['horizon']*w['slots']*cfg['ran']['slot_ms']/1000:.0f} s); lead-time focal loss; isotonic calibration; {REG['risk_training_epochs']} epochs"],
          ['Statistics', '20 seeds; paired sign-flip test (99,999 draws); Holm within each family; bootstrap 95% CI']]
    save_table(out, 'tab_hyperparameters', ['Hyperparameter', 'Value'], hp, 'INTACT-W hyperparameters (frozen before confirmation).', 'tab:hp', 'll')

    # ---------------------------------------------------------- T3 main comparison
    order = [PROPOSED] + [m for m in PUBLISHED + MATCHED + DESIGN if m in methods]
    def row(m):
        g = lambda k: np.mean([R[m][s][k] for s in seeds])
        return [PLAIN.get(m, m), f"{g('WIF'):.4f}", f"{g('contract_weighted_fulfillment'):.4f}", f"{g('C3_shortfall'):.4f}",
                f"{g('tenant_priority_inversion'):.3f}", f"{g('writes_per_epoch'):.2f}", int(sum(R[m][s]['C1'] for s in seeds)),
                int(sum(R[m][s]['C2'] for s in seeds)), f"{np.percentile([R[m][s]['selection_p95_ms'] for s in seeds], 50):.2f}"]
    save_table(out, 'tab_main', ['Method', 'WIF', 'Tenant-wt.', 'C3 shortf.', 'Ten. inv.', 'Writes/ep', 'C1', 'C2', 'Sel. ms'],
               [row(m) for m in order], f'Scenario means over {len(seeds)} fresh seeds.', 'tab:main')

    # ---------------------------------------------------------- T4 paired tests
    def family(names, metric='WIF', scale=100.):
        P, S = {}, {}
        for n in names:
            d = scale * (V(PROPOSED, metric) - V(n, metric)); P[n] = signflip(d)
            S[n] = (d.mean(), *boot(d), int((d > 1e-12).sum()), int((d < -1e-12).sum()))
        return S, holm(P) if P else {}
    prim = [m for m in REG['primary_comparisons'] if m in methods]
    Sp, Hp = family(prim)
    expl = [m for m in ['QACM-style', 'B3', 'W3R', 'W'] if m in methods]
    Se, He = family(expl)
    prow = [[PLAIN[n] + ' [registered]', f'{Sp[n][0]:+.2f}', f'[{Sp[n][1]:+.2f}, {Sp[n][2]:+.2f}]', f'{Sp[n][3]}/{Sp[n][4]}', f'{Hp[n]:.4f}'] for n in prim]
    prow += [[PLAIN[n] + ' [exploratory]', f'{Se[n][0]:+.2f}', f'[{Se[n][1]:+.2f}, {Se[n][2]:+.2f}]', f'{Se[n][3]}/{Se[n][4]}', f'{He[n]:.4f}'] for n in expl]
    save_table(out, 'tab_paired', ['Comparator', r'$\Delta$WIF (pp)', '95% CI', 'W/L', 'Holm $p$'], prow,
               'INTACT-W minus comparator. Registered family and exploratory family are Holm-corrected separately.', 'tab:paired')
    # secondary endpoints (exploratory)
    sec = []
    for metric, nm, sc, better in [('writes_per_epoch', 'writes/epoch', 1., 'lower'), ('C3_shortfall', 'C3 shortfall (pp)', 100., 'lower'),
                                   ('contract_weighted_fulfillment', 'tenant-weighted fulf. (pp)', 100., 'higher'),
                                   ('tenant_priority_inversion', 'tenant inversion (pp)', 100., 'lower')]:
        cs = [m for m in ['QACM-style', 'QACM-contract', 'B3-contract'] if m in methods]
        S2, H2 = family(cs, metric, sc)
        for n in cs: sec.append([nm, PLAIN[n], f'{S2[n][0]:+.3f}', f'[{S2[n][1]:+.3f}, {S2[n][2]:+.3f}]', f'{H2[n]:.4f}', better])
    save_table(out, 'tab_secondary', ['Endpoint', 'Comparator', r'$\Delta$', '95% CI', 'Holm $p$', 'Better'], sec,
               'Secondary endpoints, INTACT-W minus comparator (exploratory; Holm within each endpoint).', 'tab:secondary')

    # ---------------------------------------------------------- T5 ablation
    abl = [m for m in ABL if m in methods] + [m for m in ['W3R', 'QACM-contract'] if m in methods]
    Sa, Ha = family([m for m in ABL if m in methods])
    Sd, Hd = family([m for m in ['W3R', 'QACM-contract'] if m in methods])
    arow = []
    for n in abl:
        S_, H_ = (Sa, Ha) if n in Sa else (Sd, Hd)
        lab = PLAIN[n] if n not in ('W3R', 'QACM-contract') else ('inner: projection instead of value synthesis' if n == 'W3R' else 'outer: authorise all (no selection)')
        g = lambda k: np.mean([R[n][s][k] for s in seeds])
        verdict = 'helps' if (H_[n] < .05 and S_[n][0] > 0) else ('hurts' if H_[n] < .05 else 'n.s.')
        arow.append([lab, f'{S_[n][0]:+.2f}', f'[{S_[n][1]:+.2f}, {S_[n][2]:+.2f}]', f'{H_[n]:.4f}', verdict, f"{g('C3_shortfall'):.4f}", f"{g('writes_per_epoch'):.2f}"])
    save_table(out, 'tab_ablation', ['Variant', r'$\Delta$WIF (pp)', '95% CI', 'Holm $p$', 'Verdict', 'C3 shortf.', 'Writes/ep'], arow,
               r'Ablation: INTACT-W minus variant (positive = removed element helps). Component and design families Holm-corrected separately.', 'tab:ablation')

    # ---------------------------------------------------------- T6 model evidence
    ev = []
    for s in seeds:
        f = root / f's{s}' / 'calibration_req' / 'evidence_req.json'
        if f.exists(): ev.append(json.loads(f.read_text()))
    mild = []
    for s in seeds:
        f = root / f's{s}' / 'calibration' / 'evidence.json'
        if f.exists(): mild += json.loads(f.read_text())['MILD']
    mrow = []
    if ev:
        m_ = lambda k: np.mean([e[k] for e in ev])
        mrow += [[r'$\beta$ held-out RMSE / mean $|\beta|$', f"{m_('beta_RMSE'):.4f} / {m_('beta_mean_abs_truth'):.4f}"],
                 [r'$\gamma$ held-out RMSE / mean $|\gamma|$', f"{m_('gamma_RMSE'):.4f} / {m_('gamma_mean_abs_truth'):.4f}"],
                 ['Joint level RMSE (bounded margin)', f"{m_('joint_RMSE'):.3f}"]]
        K = np.array([e['kappa'] for e in ev]); Sg = np.array([e['sigma'] for e in ev])
        for k, it in enumerate(cfg['intents']):
            mrow.append([rf"$\kappa$, $\sigma$ for {it['iid'].replace('_', ' ')}", f'{K[:, k].mean():.2f}, {Sg[:, k].mean():.2f}'])
    if mild:
        gp = np.mean([m['gate'] for m in mild]); lifts = [m['lift'] for m in mild if m['lift'] is not None]
        mrow.append(['MILD safe-state gates passed', f'{100*gp:.0f}% of intent-scenario cases'])
        if lifts: mrow.append(['MILD median PR-AUC lift (where defined)', f'{statistics.median(lifts):.2f}'])
    if mrow: save_table(out, 'tab_model_evidence', ['Quantity', 'Value'], mrow, 'Held-out evidence for the frozen models (mean over seeds).', 'tab:evidence', 'll')

    # ---------------------------------------------------------- T7 per-tenant fulfilment
    tens = [t['tid'] for t in cfg['tenants']]; floors = {t['tid']: t['rho_min'] for t in cfg['tenants']}
    km = [m for m in [PROPOSED, 'QACM-style', 'QACM-contract', 'B3-contract', 'inner-contract', 'all-reject'] if m in methods]
    trow = [[t, floors[t]] + [f"{np.mean([R[m][s]['tenant_fulfillment'][t] for s in seeds]):.3f}" for m in km] for t in tens]
    save_table(out, 'tab_tenants_fulfilment', ['Tenant', r'$\rho^{\min}$'] + [PLAIN[m] for m in km], trow, 'Per-tenant fulfilment (mean over seeds).', 'tab:tenantful')

    # ---------------------------------------------------------- C3 attainable-floor analysis (no new simulation)
    # A tenant floor is ATTAINABLE in a scenario if at least one evaluated (non-ablation) method reaches it
    # there; a scenario's floors are JOINTLY attainable if one single method reaches all of them.
    att_methods = [m for m in methods if not m.startswith(PROPOSED + '-')]
    fl = {sd: {t['tid']: t['rho_min'] for t in cfgs[sd]['tenants']} for sd in seeds}
    own = {sd: {i['tenant'] for i in cfgs[sd]['intents']} for sd in seeds}
    pairs_tn = [(sd, t) for sd in seeds for t in sorted(own[sd])]
    attain = {(sd, t): max(R[m][sd]['tenant_fulfillment'][t] for m in att_methods) >= fl[sd][t] - 1e-12 for sd, t in pairs_tn}
    joint = {sd: any(all(R[m][sd]['tenant_fulfillment'][t] >= fl[sd][t] - 1e-12 for t in own[sd]) for m in att_methods) for sd in seeds}
    n_att = sum(attain.values()); n_all = len(pairs_tn)
    frow = []
    for m in [mm for mm in order if mm in att_methods]:
        br = [R[m][sd]['tenant_fulfillment'][t] < fl[sd][t] - 1e-12 for sd, t in pairs_tn]
        bra = [b for (k, b) in zip(pairs_tn, br) if attain[k]]
        allmet = np.mean([all(R[m][sd]['tenant_fulfillment'][t] >= fl[sd][t] - 1e-12 for t in own[sd]) for sd in seeds])
        frow.append([PLAIN.get(m, m), f'{100*np.mean(br):.1f}', f'{100*np.mean(bra):.1f}' if bra else 'n/a', f'{100*allmet:.0f}'])
    save_table(out, 'tab_floor_attainability', ['Method', 'Floor breach, all (%)', 'Breach, attainable floors (%)', 'Scenarios with all floors met (%)'],
               frow, f'C3 floor attainability: {n_att}/{n_all} tenant-runs have a floor attained by at least one evaluated method; '
               f'all floors are jointly attainable by a single method in {sum(joint.values())}/{len(seeds)} scenarios.', 'tab:floorattain')
    per_t = {t: (sum(attain[(sd, t)] for sd in seeds if t in own[sd]), sum(1 for sd in seeds if t in own[sd])) for t in sorted({t for _, t in pairs_tn})}
    best_cmp = 'QACM-style' if 'QACM-style' in att_methods else None
    def br_att(m):
        b = [R[m][sd]['tenant_fulfillment'][t] < fl[sd][t] - 1e-12 for sd, t in pairs_tn if attain[(sd, t)]]
        return 100 * np.mean(b) if b else float('nan')
    unatt = [f'{t} ({c - a}/{c})' for t, (a, c) in per_t.items() if a < c]
    para = (f"A floor breach is meaningful only where the floor can be met. Over the {n_all} tenant-runs, {n_att} floors "
            f"({100*n_att/n_all:.0f}\\%) are attained by at least one evaluated method, and all floors of a scenario are "
            f"jointly attainable by a single method in {sum(joint.values())} of {len(seeds)} scenarios"
            + (f"; the unattainable floors belong to {', '.join(unatt)}." if unatt else ".")
            + f" Restricted to attainable floors, {a.name} breaches {br_att(PROPOSED):.1f}\\%"
            + (f" versus {br_att(best_cmp):.1f}\\% for {PLAIN[best_cmp]}." if best_cmp else "."))
    (out / 'text').mkdir(exist_ok=True)
    (out / 'text' / 'floor_attainability.tex').write_text(para + '\n')
    log.append('C3 attainability: ' + para.replace('\\%', '%'))

    # ---------------------------------------------------------- figures
    fm = [m for m in order]
    fig, ax = plt.subplots(figsize=(W1, .2 * len(fm) + .6)); fm_s = sorted(fm, key=lambda m: V(m).mean())
    for k, m in enumerate(fm_s):
        c = '#1F4E9E' if m == PROPOSED else ('#D95F02' if m in MATCHED else ('#7F7F7F' if m in DESIGN else 'k'))
        ax.errorbar(V(m).mean(), k, xerr=V(m).std(ddof=1) / np.sqrt(len(seeds)), fmt='o', c=c, ms=3.5, capsize=2)
    ax.set_yticks(range(len(fm_s))); ax.set_yticklabels([PLAIN[m] for m in fm_s]); ax.set_xlabel('WIF (mean $\\pm$ s.e.)')
    savefig(fig, out, 'fig_main_wif')
    comps = prim + [m for m in ['QACM-style', 'B3'] if m in methods]
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for k, n in enumerate(comps):
        d = 100 * (V(PROPOSED) - V(n)); ax.scatter(k + np.random.default_rng(k).uniform(-.15, .15, len(d)), d, s=6, c='0.6', zorder=2)
        st = Sp.get(n) or Se.get(n); ax.errorbar(k, st[0], yerr=[[st[0] - st[1]], [st[2] - st[0]]], fmt='D', c='k', ms=4, capsize=3, zorder=3)
    ax.axhline(0, c='k', lw=.8); ax.set_xticks(range(len(comps))); ax.set_xticklabels([PLAIN[n].split(' (')[0] for n in comps], rotation=25, ha='right', fontsize=5.5)
    ax.set_ylabel('INTACT-W $-$ comparator (pp WIF)'); savefig(fig, out, 'fig_paired')
    fig, ax = plt.subplots(figsize=(W1, .22 * len(abl) + .6))
    for k, n in enumerate(abl):
        S_, H_ = (Sa, Ha) if n in Sa else (Sd, Hd); c = '#B2182B' if H_[n] < .05 and S_[n][0] > 0 else '0.5'
        ax.errorbar(S_[n][0], k, xerr=[[S_[n][0] - S_[n][1]], [S_[n][2] - S_[n][0]]], fmt='o', c=c, ms=3.5, capsize=2)
    ax.axvline(0, c='k', lw=.8); ax.set_yticks(range(len(abl))); ax.set_yticklabels([r[0] for r in arow]); ax.set_xlabel('INTACT-W $-$ variant (pp WIF); red = significant')
    savefig(fig, out, 'fig_ablation')
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for m in order:
        c = '#1F4E9E' if m == PROPOSED else ('#D95F02' if m in MATCHED else ('#7F7F7F' if m in DESIGN else 'k'))
        x, y = V(m, 'writes_per_epoch').mean(), V(m).mean(); ax.scatter(x, y, c=c, s=18, marker='*' if m == PROPOSED else 'o', zorder=3)
        ax.annotate(PLAIN[m].split(' (')[0][:26], (x, y), fontsize=5, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel('E2 writes per inner epoch'); ax.set_ylabel('WIF'); savefig(fig, out, 'fig_wif_vs_writes')
    fig, ax = plt.subplots(figsize=(W1, 1.8)); x = np.arange(len(tens)); bw = .8 / len(km)
    for k, m in enumerate(km):
        ax.bar(x + k * bw, [np.mean([R[m][s]['tenant_fulfillment'][t] for s in seeds]) for t in tens], bw, label=PLAIN[m].split(' (')[0][:22],
               color='#1F4E9E' if m == PROPOSED else None, edgecolor='k', lw=.3, hatch=['', '//', '..', 'xx', '\\\\', '--'][k % 6])
    ax.scatter(x + .4 - bw / 2, [floors[t] for t in tens], marker='_', s=200, c='r', label=r'floor $\rho^{\min}$', zorder=4)
    ax.set_xticks(x + .4 - bw / 2); ax.set_xticklabels(tens); ax.set_ylim(.5, 1.02); ax.set_ylabel('tenant fulfilment'); ax.legend(fontsize=4.8, ncol=2)
    savefig(fig, out, 'fig_tenants')
    # telemetry: seed with the MEDIAN difference vs the strongest comparator (typical case, not cherry-picked)
    ref = 'QACM-contract' if 'QACM-contract' in methods else prim[0]
    dd = V(PROPOSED) - V(ref); rep = seeds[int(np.argsort(dd)[len(dd) // 2])]
    tel = {}
    for m in [PROPOSED, ref, 'QACM-style', 'all-reject']:
        f = root / f's{rep}' / m / 'telemetry.jsonl'
        if f.exists():
            T = [json.loads(z) for z in f.read_text().splitlines()]
            if T and 'cell' in T[0]: tel[m] = T
    if tel:
        sl = sorted(next(iter(tel.values()))[0]['tenants'])
        pan = [('offered load ratio', lambda r: r['cell'].get('offered_input_ratio')), ('PRB utilisation (%)', lambda r: r['cell'].get('prb_util_pct')),
               ('Tx power (dBm)', lambda r: r['cell'].get('txpower_dbm')), ('radiated power (W)', lambda r: r['cell'].get('radiated_power_w'))]
        for t in sl:
            pan += [(f'{t} throughput (Mb/s)', lambda r, t=t: r['tenants'][t].get('throughput_mbps')), (f'{t} delay (ms)', lambda r, t=t: r['tenants'][t].get('delay_ms'))]
        pan += [('cumulative E2 writes', None), ('claims authorised', lambda r: len(r['selected']))]
        n = len(pan); fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(W2, 1.15 * ((n + 1) // 2)), sharex=True); axs = axs.ravel()
        fig.subplots_adjust(wspace=.3, hspace=.4)
        sty = {PROPOSED: ('-', '#1F4E9E'), ref: ('--', 'k'), 'QACM-style': ('-.', '#7570B3'), 'all-reject': (':', '#D95F02')}
        for k, (lab, fn) in enumerate(pan):
            for m, T in tel.items():
                xs = np.arange(len(T)) * cfg['w']['slots'] * cfg['ran']['slot_ms'] / 1000
                ys = np.cumsum([len(r['writes']) for r in T]) if fn is None else [fn(r) for r in T]
                axs[k].plot(xs, ys, sty.get(m, ('-', 'g'))[0], c=sty.get(m, ('-', 'g'))[1], lw=.7, label=PLAIN[m].split(' (')[0])
            axs[k].set_title(lab, fontsize=6.5, pad=2)
        for k in range(n, len(axs)): axs[k].axis('off')
        axs[n - 1].set_xlabel('time (s)'); axs[n - 2].set_xlabel('time (s)'); axs[0].legend(fontsize=5)
        fig.suptitle(f'RAN telemetry, seed {rep} (median-difference seed vs {PLAIN[ref].split(" (")[0]})', fontsize=7.5)
        savefig(fig, out, 'fig_telemetry')

    # ---------------------------------------------------------- text
    (out / 'text').mkdir(exist_ok=True)
    nt = len(cfg['tenants']); ni = len(cfg['intents']); nc = len(cfg['claims'])
    env = (f"We evaluate on a flow-level analytical RAN model with {cfg['ran']['n_prb']} PRBs and per-slice queues, "
           f"retransmissions and a shared PRB budget. One scenario has {nt-1} service tenants and a host, {ni} intents and {nc} claims "
           f"over {len(set(c['param'] for c in cfg['claims']))} parameters; C1 groups arise where two xApps claim the same parameter "
           f"(transmit power: energy vs coverage restoration; T2 PRB cap: streaming vs host capacity). Offered load follows a "
           f"piecewise profile ({', '.join(str(x) for x in cfg['ran']['load_profile']['levels'])} of nominal) with per-tenant phases; "
           f"per-UE loads vary by seed. An inner epoch lasts {cfg['w']['slots']*cfg['ran']['slot_ms']/1000:.1f} s and an authority lease "
           f"{cfg['w']['lease']} inner epochs. Each of the {len(seeds)} scenarios is simulated for "
           f"{REG['evaluation_inner_epochs']*cfg['w']['slots']*cfg['ran']['slot_ms']/1000:.0f} s with identical random streams for all methods. "
           f"The model is not a validated testbed; radiated power is not site electrical energy.")
    (out / 'text' / 'environment.md').write_text(env + '\n')
    (out / 'text' / 'hyperparameters.md').write_text('\n'.join(f'- {r[0]}: {r[1]}' for r in hp) + '\n')
    best_other = max([m for m in methods if m != PROPOSED and not m.startswith(PROPOSED)], key=lambda m: V(m).mean())
    sig_beat = [PLAIN[n] for n in prim if Hp[n] < .05 and Sp[n][0] > 0]
    unres = [PLAIN[n] for n in prim if not Hp[n] < .05]
    helps = [r[0] for r in arow if r[4] == 'helps']; nulls = [r[0] for r in arow if r[4] == 'n.s.']; hurts = [r[0] for r in arow if r[4] == 'hurts']
    others = [m for m in methods if not m.startswith(PROPOSED + '-') and m != PROPOSED]
    top = V(PROPOSED).mean() >= max(V(m).mean() for m in others)
    scen = (f"{len(seeds)} fresh scenarios registered before evaluation" if is_conf else f"{len(seeds)} DEVELOPMENT scenarios (not evidence)")
    res = [f"Across {scen}, INTACT-W attains a mean WIF of {V(PROPOSED).mean():.4f}"
           + (f", the highest of the {len(others)+1} evaluated methods; the next best is " if top else
              f"; it is NOT the highest of the {len(others)+1} evaluated methods: the best is ")
           + f"{PLAIN[best_other]} ({V(best_other).mean():.4f}).",
           "It is significantly better than " + (', '.join(sig_beat) if sig_beat else 'none of the registered comparators') +
           (". The difference to " + ', '.join(unres) + " is not statistically resolved and we do not claim superiority over it." if unres else '.')]
    for n in prim:
        res.append(f"- vs {PLAIN[n]}: {Sp[n][0]:+.2f} pp (95% CI [{Sp[n][1]:+.2f}, {Sp[n][2]:+.2f}], {Sp[n][3]}/{Sp[n][4]} scenarios, Holm p = {Hp[n]:.4f}).")
    for n in expl:
        res.append(f"- (exploratory) vs {PLAIN[n]}: {Se[n][0]:+.2f} pp (95% CI [{Se[n][1]:+.2f}, {Se[n][2]:+.2f}], Holm p = {He[n]:.4f}).")
    if 'QACM-style' in methods:
        dw = V(PROPOSED, 'writes_per_epoch').mean(); dq = V('QACM-style', 'writes_per_epoch').mean()
        res.append(f"INTACT-W issues {dw:.2f} E2 writes per inner epoch versus {dq:.2f} for QACM-style value synthesis ({100*(1-dw/dq):.0f}% fewer).")
    res.append(f"INTACT-W records {c12} C1/C2 violations in total.")
    res.append("Ablation: removing or replacing " + (', '.join(helps) if helps else 'no element') + " significantly lowers WIF; "
               + (', '.join(nulls) if nulls else 'no element') + " has no resolved effect on WIF" + ("; removing " + ', '.join(hurts) + " slightly raises it." if hurts else '.'))
    (out / 'text' / 'results.md').write_text('\n'.join(res) + '\n')
    allowed = ['## Claims supported by this run', ''] + [f'- INTACT-W significantly outperforms {x}.' for x in sig_beat] + \
              [f'- Element contributes to WIF: {x}.' for x in helps] + \
              ['', '## Claims NOT supported (do not write)', ''] + [f'- Superiority over {x} (not resolved).' for x in unres] + \
              [f'- Any WIF contribution of {x}.' for x in nulls] + ['- Equivalence with any method (non-significance is not equivalence).',
              '- End-to-end near-RT latency (selection time excludes E2 transport).', '- Live-cell calibration (paired twins need an offline simulator).']
    (out / 'text' / 'claims_allowed.md').write_text('\n'.join(allowed) + '\n')

    # ---------------------------------------------------------- per-tenant fulfilment tests
    # PACT minus comparator on each tenant's run-level fulfilment F_n (result.json tenant_fulfillment);
    # paired sign-flip test, Holm across the comparators within each tenant (same rule as other endpoints).
    tcomp = [m for m in ['QACM-style', 'QACM-contract', 'B3-contract'] if m in methods]
    TF = lambda m, t: np.array([R[m][sd]['tenant_fulfillment'][t] for sd in seeds], float)
    ttrows = []; tt = {}
    for t in tens:
        P = {}; S_ = {}
        for n in tcomp:
            d = 100 * (TF(PROPOSED, t) - TF(n, t)); P[n] = signflip(d)
            S_[n] = (d.mean(), *boot(d), int((d > 1e-12).sum()), int((d < -1e-12).sum()))
        H_ = holm(P)
        for n in tcomp:
            tt[(t, n)] = (S_[n], H_[n])
            ttrows.append([t + (' (host)' if t in {x['tid'] for x in cfg['tenants'] if x.get('is_host')} else ''), PLAIN[n],
                           f"{TF(PROPOSED, t).mean():.3f}", f"{TF(n, t).mean():.3f}", f'{S_[n][0]:+.2f}',
                           f'[{S_[n][1]:+.2f}, {S_[n][2]:+.2f}]', f'{S_[n][3]}/{S_[n][4]}', f'{H_[n]:.4f}'])
    save_table(out, 'tab_tenant_tests', ['Tenant', 'Comparator', a.name, 'Comparator mean', r'$\Delta$ (pp)', '95% CI', 'W/L', 'Holm $p$'],
               ttrows, f'Per-tenant fulfilment, {a.name} minus comparator (positive = {a.name} better; Holm within each tenant).', 'tab:tenanttests')
    host = [x['tid'] for x in cfg['tenants'] if x.get('is_host')]
    if host and 'QACM-style' in tcomp:
        (dm, lo, hi, wn, ls), hp_ = tt[(host[0], 'QACM-style')]
        pstr = '$<$0.001' if hp_ < 0.001 else f'{hp_:.3f}'
        (out / 'text').mkdir(exist_ok=True)
        (out / 'text' / 'host_row.tex').write_text(
            f"Host (operator) fulfillment $\\uparrow$ & \\textbf{{{TF(PROPOSED, host[0]).mean():.3f}}} & {TF('QACM-style', host[0]).mean():.3f} & "
            f"${dm:+.1f}$\\pp & {pstr}\\\\\n"
            f"% host: 95% CI [{lo:+.2f}, {hi:+.2f}] pp, {wn}/{ls} scenarios better/worse, Holm p = {hp_:.4f}\n")

    # ---------------------------------------------------------- service protection (Fig. 1 style)
    SVC = [('intent_miss_pct', 'Intent miss'), ('floor_breach_pct', 'Floor breach'), ('safety_crossing_pct', 'Safety crossing'),
           ('intent_inversion_pct', 'Intent inversion'), ('tenant_inversion_pct', 'Tenant inversion')]
    fig_m = [m for m in [PROPOSED, 'QACM-style', 'QACM-contract', 'B3-contract', 'B3', 'inner-contract', 'all-reject'] if m in methods]
    svc = {}
    for m in fig_m:
        for sd in seeds:
            d = root / f's{sd}' / m
            if (d / 'telemetry.jsonl').exists() or (d / 'service_metrics.json').exists():
                svc.setdefault(m, {})[sd] = service_metrics(d, cfgs[sd])
    have_svc = all(m in svc and len(svc[m]) == len(seeds) for m in fig_m)
    if have_svc:
        SV = lambda m, k: np.array([svc[m][sd][k] for sd in seeds])
        srow = [[PLAIN[m]] + [f"{SV(m, k).mean():.2f}" for k, _ in SVC] for m in fig_m]
        save_table(out, 'tab_service', ['Method'] + [lab + ' (%)' for _, lab in SVC], srow,
                   'Service-protection rates (mean over seeds; lower is better). C1 and C2 violations are zero for every method.', 'tab:service')
        comps_s = [m for m in ['QACM-style', 'QACM-contract', 'B3-contract'] if m in fig_m]
        trows = []; stats = {}
        for k, lab in SVC:
            P = {}; S_ = {}
            for n in comps_s:
                d = SV(PROPOSED, k) - SV(n, k); P[n] = signflip(d); S_[n] = (d.mean(), *boot(d), int((d < -1e-12).sum()), int((d > 1e-12).sum()))
            H_ = holm(P)
            for n in comps_s:
                stats[(k, n)] = (S_[n], H_[n])
                trows.append([lab, PLAIN[n], f'{S_[n][0]:+.2f}', f'[{S_[n][1]:+.2f}, {S_[n][2]:+.2f}]', f'{S_[n][3]}/{S_[n][4]}', f'{H_[n]:.4f}'])
        save_table(out, 'tab_service_tests', ['Metric', 'Comparator', r'$\Delta$ (pp)', '95% CI', 'better/worse', 'Holm $p$'], trows,
                   f'{a.name} minus comparator on service-protection rates (negative = {a.name} better; exploratory, Holm within each metric).', 'tab:servicetests')
        # figure: (a) WIF, (b) E2 writes, (c) miss + breach, (d) crossing + inversions
        cols = {PROPOSED: '#1F4E9E'}
        rows_m = fig_m[::-1]; y = np.arange(len(rows_m))
        SHORT = {PROPOSED: f'{a.name} (ours)', 'QACM-style': 'QACM-style value synthesis',
                 'QACM-contract': f'Value synthesis + {a.name} weights', 'B3-contract': f'Greedy + {a.name} weights',
                 'B3': 'Greedy value arbitration', 'inner-contract': 'Inner arbitration only', 'all-reject': 'No arbitration'}
        fig, axs = plt.subplots(1, 4, figsize=(W2, .21 * len(rows_m) + 0.85), sharey=True,
                                gridspec_kw={'width_ratios': [1.0, 0.85, 1.2, 1.2], 'wspace': 0.10})
        def col(m): return cols.get(m, '#D95F02' if m in MATCHED else '0.25')
        for k, m in enumerate(rows_m):
            ec = 'k' if m == PROPOSED else col(m)
            axs[0].plot(V(m).mean(), k, 'h', ms=5, c=col(m), mec=ec, mew=.6)
            axs[1].plot(V(m, 'writes_per_epoch').mean(), k, '*', ms=6.5, c=col(m), mec=ec, mew=.5)
        axs[0].set_title('(a) Fulfilment $\\uparrow$'); axs[0].set_xlabel('WIF, mean')
        axs[1].set_title('(b) Control cost $\\downarrow$'); axs[1].set_xlabel('E2 writes / epoch, mean'); axs[1].set_xlim(left=-0.2)
        mk = {'intent_miss_pct': ('o', 'Intent miss'), 'floor_breach_pct': ('D', 'Floor breach'), 'safety_crossing_pct': ('P', 'Safety crossing'),
              'intent_inversion_pct': ('^', 'Intent inversion'), 'tenant_inversion_pct': ('s', 'Tenant inversion')}
        for ax, keys, title in ((axs[2], ['intent_miss_pct', 'floor_breach_pct'], '(c) Service outcome $\\downarrow$'),
                                (axs[3], ['safety_crossing_pct', 'intent_inversion_pct', 'tenant_inversion_pct'], '(d) Protection, priority $\\downarrow$')):
            off = np.linspace(-0.16, 0.16, len(keys)) if len(keys) > 1 else [0.0]
            for q, kk in enumerate(keys):
                for k, m in enumerate(rows_m):
                    ax.plot(SV(m, kk).mean(), k + off[q], mk[kk][0], ms=4, c=col(m), mec='k' if m == PROPOSED else col(m), mew=.5)
            ax.set_title(title); ax.set_xlabel('rate (%), mean')
        fig.legend(handles=[plt.Line2D([], [], ls='', marker=mk[kk][0], c='0.35', ms=4.5, label=mk[kk][1]) for kk in mk],
                   loc='upper center', bbox_to_anchor=(0.62, 1.07), ncol=5, fontsize=6.3, frameon=False, handletextpad=0.25, columnspacing=1.0)
        axs[0].set_yticks(y); axs[0].set_yticklabels([SHORT.get(m, PLAIN[m]) for m in rows_m])
        for t, m in zip(axs[0].get_yticklabels(), rows_m):
            t.set_color(col(m)); t.set_fontweight('bold' if m == PROPOSED else 'normal')
        for ax in axs: ax.grid(axis='y', alpha=0)
        savefig(fig, out, 'fig_service_protection')
        # paste-ready LaTeX paragraph
        best = 'QACM-style' if 'QACM-style' in fig_m else comps_s[0]
        def phr(k):
            (dm, lo, hi, nb, nw), hp_ = stats[(k, best)]
            sig = hp_ < .05
            rel = ('significantly lower' if sig and dm < 0 else 'significantly higher' if sig and dm > 0 else 'not significantly different')
            return (f"{dict(SVC)[k].lower()} {SV(PROPOSED, k).mean():.2f}\\% vs.\\ {SV(best, k).mean():.2f}\\% "
                    f"({rel}, Holm $p={hp_:.3f}$)")
        para = (f"Fig.~\\ref{{fig:service}} reports the service-protection rates. Against {PLAIN[best]}, the strongest baseline, "
                f"{a.name} obtains " + '; '.join(phr(k) for k, _ in SVC) + ". "
                f"C1 and C2 violations are zero for every evaluated method, because every method selects from admissible authority sets.")
        (out / 'text' / 'service_protection.tex').write_text(para + '\n')
    else:
        print('NOTE: telemetry missing for some runs; service-protection artefacts skipped')

    # ---------------------------------------------------------- index
    idx = [f'# Paper artefacts — {label}', '', f'Generated from `{root}` with `{a.registration}`; INTACT-W = `{PROPOSED}`.', '',
           '| Artefact | Paper location |', '|---|---|',
           '| tables/tab_environment, tab_tenants, tab_intents, tab_claims | Sec. V-A Setup (environment, contracts, intents, claims) |',
           '| text/environment.md | Sec. V-A first paragraph |',
           '| tables/tab_hyperparameters, text/hyperparameters.md | Sec. V-A / Table of hyperparameters |',
           '| tables/tab_main + figures/fig_main_wif | Sec. V-D(a) main comparison (Table III, Fig. 2a) |',
           '| tables/tab_paired + figures/fig_paired | Sec. V-D(a) paired statistics |',
           '| tables/tab_secondary + figures/fig_wif_vs_writes | Sec. V-D(b) cost / writes and contract endpoints |',
           '| tables/tab_ablation + figures/fig_ablation | Sec. V-D(d) ablation (Table IV) |',
           '| tables/tab_tenants_fulfilment + figures/fig_tenants | Sec. V-D(a) C3 discussion |',
           '| tables/tab_model_evidence | Sec. IV-A / V-D effect-model and MILD evidence (advisor comment 8) |',
           '| figures/fig_telemetry | Sec. V-D(b) arbitration behaviour (RAN telemetry, median seed) |',
           '| figures/fig_service_protection + tables/tab_service, tab_service_tests | Fig. 2 (double column, figure*) + Sec. V-D(a) |',
           '| text/service_protection.tex | paste-ready paragraph for Sec. V-D(a) |',
           '| tables/tab_floor_attainability + text/floor_attainability.tex | C3 attainable-floor analysis (advisor comment 1.3); Sec. V-D item (iii) |',
           '| tables/tab_tenant_tests, text/host_row.tex | per-tenant tests; tested host row for Table IV |',
           '| text/results.md | Sec. V-D paste-ready result sentences |',
           '| text/claims_allowed.md | Checklist before writing the abstract/conclusion |', '',
           'Integrity: ' + ('PASS' if not missing else f'INCOMPLETE ({missing})') + f'; INTACT-W C1+C2 = {c12}.']
    (out / 'ARTIFACT_INDEX.md').write_text('\n'.join(idx + [''] + log) + '\n')
    print('\n'.join(idx)); print('\n'.join(res))


if __name__ == '__main__':
    main()
