"""Conservative causal personal reference; no outcome or oracle inputs."""
import numpy as np
from src.synthetic.config import SIGNALS

class PersonalBaseline:
    def __init__(self,config):self.c=config['baseline'];self.floors=np.array(self.c['scale_floors']);self.initialized=False;self.last_cut=0;self.reference=None;self.samples=[[] for _ in SIGNALS];self.hours=[[] for _ in SIGNALS];self.frozen=False;self.freeze_count=0
    def _summarize(self):
        self.mean=[];self.median=[];self.scale=[];self.quantiles=[];self.clock=[];self.count=[]
        for j,x in enumerate(self.samples):
            a=np.asarray(x);h=np.asarray(self.hours[j]);n=len(a);median=float(np.median(a)) if n else np.nan
            sd=max(float(1.4826*np.median(abs(a-median))),self.floors[j]) if n else self.floors[j]
            self.mean.append(float(a.mean()) if n else np.nan);self.median.append(median);self.scale.append(sd);self.count.append(n)
            self.quantiles.append(np.quantile(a,[.1,.9]).tolist() if n else [np.nan,np.nan]);clock=[]
            for bucket in range(4):
                v=a[(h%24//6)==bucket];clock.append(float(np.median(v)) if len(v)>=self.c['minimum_count'] else median)
            self.clock.append(clock)
        self.mean=np.array(self.mean);self.median=np.array(self.median);self.scale=np.array(self.scale);self.quantiles=np.array(self.quantiles);self.clock=np.array(self.clock);self.count=np.array(self.count)
    def update(self,stream,landmark):
        value=stream['value'];t=np.arange(len(value));arrived=(stream['available_hour']<=landmark)&~stream['missingness_mask']&np.isfinite(value);eligible_end=landmark-self.c['recent_exclusion_hours']
        if not self.initialized:
            eligible_end=min(eligible_end,self.c['initial_trusted_hours'])
            for j in range(len(SIGNALS)):
                good=arrived[:,j]&(t<eligible_end);self.samples[j]=value[good,j].astype(float).tolist();self.hours[j]=t[good].tolist()
            self._summarize();self.reference=self.median.copy();self.initialized=True;self.last_cut=eligible_end
        # Freeze using received observable residuals, never hidden event/state labels.
        recent=arrived&(t[:,None]>landmark-48)&(t[:,None]<=landmark);zs=[]
        for j in range(len(SIGNALS)):
            good=recent[:,j];expected=self.clock[j,(t[good]%24)//6];delta=value[good,j]-expected
            zs.append(float(np.nanmedian(delta)/self.scale[j]) if len(delta) and np.isfinite(expected).any() else np.nan)
        groups=[[0,1],[2,3],[5,6],[7],[8],[9],[10]]
        deviating=sum(any(np.isfinite(zs[j]) and abs(zs[j])>=self.c['freeze_z'] for j in group) for group in groups)
        self.frozen=deviating>=self.c['freeze_systems'];self.freeze_count+=int(self.frozen)
        old_reference=self.reference.copy();old_scale=self.scale.copy()
        if not self.frozen and eligible_end>self.last_cut:
            for j in range(len(SIGNALS)):
                good=arrived[:,j]&(t>=self.last_cut)&(t<eligible_end)
                indices=np.flatnonzero(good)
                if np.isfinite(self.reference[j]):
                    expected=self.clock[j,(indices%24)//6];indices=indices[abs(value[indices,j]-expected)<=self.c['sample_acceptance_z']*self.scale[j]]
                self.samples[j].extend(value[indices,j].astype(float).tolist());self.hours[j].extend(indices.tolist())
            self._summarize()
            # Conservative anchor movement even when long-history median shifts.
            proposed=np.where(np.isfinite(old_reference),old_reference+self.c['update_rate']*(self.median-old_reference),self.median)
            bound=self.c['maximum_update_scale_per_day']*old_scale
            proposed=np.where(np.isfinite(old_reference),np.clip(proposed,old_reference-bound,old_reference+bound),proposed)
            offset=proposed-self.median;self.clock+=offset[:,None];self.reference=proposed
        self.last_cut=eligible_end
        return self.snapshot()
    def snapshot(self):
        return {'mean':self.mean.copy(),'median':self.reference.copy(),'sd':self.scale.copy(),'q10':self.quantiles[:,0].copy(),'q90':self.quantiles[:,1].copy(),'clock':self.clock.copy(),
          'count':self.count.copy(),'confidence':np.minimum(self.count/np.array([96,5,8,8,8,8,8,4,96,4,4]),1),'frozen':bool(self.frozen),'freeze_count':self.freeze_count,'trusted_cutoff_hour':self.last_cut}
