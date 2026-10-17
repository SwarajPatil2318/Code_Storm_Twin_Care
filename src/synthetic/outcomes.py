"""Causal public feature allowlist and disjoint future-event label intervals."""
import numpy as np
from src.synthetic.config import SIGNALS

EHR_FEATURES=['age','female','bmi','ef','hf_severity','hypertension','diabetes','ckd','beta_blocker','ace_arb','diuretic','mra','sglt2','creatinine','sodium','hemoglobin','bnp']
STATISTICS=['mean','sd','last','slope_per_day','change_recent_vs_prior','coverage']
MODEL_FEATURES=['ehr__'+k for k in EHR_FEATURES]+['phys__'+s+'__'+k for s in SIGNALS for k in STATISTICS]
FORBIDDEN={'state','latent','vulnerability','profile','path','event_time','event_hazard','seed','future_state','baseline','stable_daily','index'}


def labels(event_time,landmark,followup_end):
    if landmark+72>followup_end:raise ValueError('Insufficient72h followup')
    delta=None if event_time is None or event_time<=landmark else float(event_time-landmark)
    return {'event_0_24h':int(delta is not None and 0<delta<=24),'event_24_48h':int(delta is not None and 24<delta<=48),
      'event_48_72h':int(delta is not None and 48<delta<=72),'event_within_72h':int(delta is not None and 0<delta<=72),'time_to_event_hours':delta}


def features(context,stream,landmark,history=168):
    hours=np.arange(len(stream['value']));out={'ehr__'+k:context[k] for k in EHR_FEATURES}
    for j,signal in enumerate(SIGNALS):
        observed=(hours>landmark-history)&(hours<=landmark)&~stream['missingness_mask'][:,j]&(stream['available_hour'][:,j]<=landmark)
        t=hours[observed];v=stream['value'][observed,j];scheduled=stream['scheduled'][:,j]& (hours>landmark-history)&(hours<=landmark)
        if len(v):
            recent=v[t>landmark-48];past=v[t<=landmark-48]
            values={'mean':float(v.mean()),'sd':float(v.std()),'last':float(v[-1]),
              'slope_per_day':float(np.polyfit((t-t[0])/24,v,1)[0]) if len(v)>1 else None,
              'change_recent_vs_prior':float(recent.mean()-past.mean()) if len(recent) and len(past) else None,
              'coverage':float(len(v)/max(sum(scheduled),1))}
        else:values={k:None for k in STATISTICS};values['coverage']=0.
        out.update({'phys__'+signal+'__'+k:v for k,v in values.items()})
    if set(out)!=set(MODEL_FEATURES):raise ValueError('Model feature allowlist mismatch')
    return out


def landmarks(context,stream,event_time,config):
    l=config['landmarks'];end=config['days']*24
    for hour in range(l['history_hours'],end-l['horizon_hours']+1,l['stride_hours']):
        if l['end_before_first_event'] and event_time is not None and hour>=event_time:break
        yield hour,features(context,stream,hour,l['history_hours']),labels(event_time,hour,end)
