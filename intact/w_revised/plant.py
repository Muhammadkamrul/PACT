"""Small explicit changes to the supplied plant/controller workload, not KPI bonuses."""
import numpy as np
from ..ran.analytic import AnalyticRAN
from ..xapps import build_xapps,XApp

class RevisedRAN(AnalyticRAN):
    def step(self,n_slots):
        k=super().step(n_slots)
        k['_cell']['radiated_power_w']=10**((k['_cell']['txpower_dbm']-30)/10)
        return k

class CoverageXApp(XApp):
    """Host-owned cell power controller responding to a monitored service KPI."""
    def propose(self,kpm,controls,rng):
        cur=controls[self.param];row=kpm[self.cfg['watch_tenant']]
        error=row['delay_ms']/self.cfg['delay_target']-1
        # Coverage restoration only; energy xApp independently asks to reduce power.
        if error<=0:return None
        return self._quantise(cur+min(self.gain*error,2.))

class AdaptiveMCS(XApp):
    """Simple bidirectional buffer-driven MCS controller; no access to plant internals."""
    def propose(self,kpm,controls,rng):
        b=kpm[self.tenant]['buffer_kb'];cur=controls[self.param]
        d=-1 if b>self.cfg['threshold'] else (1 if b<self.cfg['threshold']*.2 else 0)
        return self._quantise(cur+d*self.step) if d else None

def build_apps(cfg):
    basic={**cfg,'xapps':[x for x in cfg['xapps'] if x['kind'] not in ('coverage','adaptive_mcs')]}
    out=build_xapps(basic)
    for x in cfg['xapps']:
        if x['kind'] not in ('coverage','adaptive_mcs'):continue
        cls=CoverageXApp if x['kind']=='coverage' else AdaptiveMCS
        out[x['name']]=cls(name=x['name'],tenant=x['tenant'],param=x['param'],domain=x['domain'],
                           step=x['step'],gain=x['gain'],cfg=x)
    return out
