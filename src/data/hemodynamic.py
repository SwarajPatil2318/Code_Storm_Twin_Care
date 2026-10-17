"""Retrospective target construction and causal risk sets; no predictor training."""
from collections import Counter
import bisect
import math
import numpy as np
from src.data.eicu_spec import CANDIDATES,PROTOCOL,PRIMARY_SIGNALS


def covered(times,values,start,end,fraction=.8,max_gap=10,step=5):
    """No interpolation: require observed support and bounded gaps, including edges."""
    if end<=start:return False
    i=np.searchsorted(times,start,'left');j=np.searchsorted(times,end,'right')
    t=times[i:j][np.isfinite(values[i:j])];t=np.unique(t)
    if len(t)<math.ceil((end-start)/step*fraction):return False
    edges=np.r_[start,t,end]
    return bool(np.max(np.diff(edges))<=max_gap)


def episodes(times,values,definition):
    """Missing/invalid samples and excessive gaps break sustained runs.

    Duration is actual first-to-last low observation span, not count*nominal rate.
    Low runs separated by <=120 min are recurrent components of one episode.
    """
    if len(times)!=len(values) or np.any(np.diff(times)<0):raise ValueError('Timeline ordering/alignment violation')
    runs=[];start=None;last=None;count=0;confirmation=None
    def finish(reason):
        if start is not None and last-start>=definition['duration'] and count>=definition['min_points']:
            runs.append({'onset':float(start),'last_low':float(last),'confirmation':float(confirmation),
                         'observed_low_span_minutes':float(last-start),'low_observations':count,'termination':reason})
    for t,value in zip(times,values):
        low=np.isfinite(value) and (value<=definition['threshold'] if definition['inclusive'] else value<definition['threshold'])
        gap=last is not None and t-last>definition['max_gap']
        if not low or gap:
            finish('gap_or_invalid' if gap or not np.isfinite(value) else 'observed_recovery')
            start=None;last=None;count=0;confirmation=None
        if low:
            if start is None:start=t
            last=t;count+=1
            if confirmation is None and t-start>=definition['duration'] and count>=definition['min_points']:confirmation=t
    finish('right_censored_series_end')
    if definition.get('drop'):
        accepted=[]
        for r in runs:
            t=r['onset'];idx=(times>=t-60)&(times<t)&np.isfinite(values)
            if sum(idx)>=10 and covered(times,values,t-60,t-5,.8,10):
                baseline=float(np.median(values[idx]));first=float(values[np.flatnonzero(times==t)[0]])
                if baseline>=80 and baseline-first>=definition['drop']:
                    accepted.append({**r,'prior_baseline_map':baseline,'drop_from_prior_baseline':baseline-first})
        runs=accepted
    result=[]
    for run in runs:
        if result and run['onset']-result[-1]['last_low']<=PROTOCOL['episode_merge_minutes']:
            result[-1]['runs'].append(run);result[-1]['last_low']=run['last_low']
        else:result.append({**run,'runs':[run]})
    return result


def asof_context(patient,t0,labs=()):
    """Presence is a documented-positive flag; absent evidence stays unknown."""
    flags={k:None for k in patient['ever']};ids=[]
    for r in patient['conditions']:
        if r['available_offset']<=t0 and r['observed_offset']<=t0:
            for key,v in r['flags'].items():
                if v:flags[key]=1
            ids.append({'table':r['source_table'],'record':r['source_record'],'available_offset':r['available_offset']})
    recent={}
    for r in labs:
        if r['available_offset']<=t0 and r['observed_offset']<=t0:
            key=r['concept']
            if key not in recent or (r['observed_offset'],r['available_offset'])>(recent[key]['observed_offset'],recent[key]['available_offset']):recent[key]=r
    return flags,ids,recent


def features(patient,series,t0,lookback,labs=()):
    """Strict feature allowlist; future labels/episode summaries never enter inputs."""
    end=t0-PROTOCOL['feature_lag_minutes'];start=end-lookback
    flags,context_ids,latest=asof_context(patient,t0,labs)
    r=patient['raw'];age=r['age'];age_over89=age.strip()=='> 89'
    x={'age':None if age_over89 or not age else float(age),'age_over89':age_over89,
       'gender':r['gender'] or None,'unit_type':r['unittype'] or None,'unit_admission_source':r['unitadmitsource'] or None,
       **{f'documented_{k}':v for k,v in flags.items()}}
    provenance={'static_availability':'index-semantic age/gender/admission context; capture timestamps unverified',
                'context_records':context_ids,'labs':[],'dynamic':{},'cutoff':t0,'feature_end':end}
    for field in PRIMARY_SIGNALS:
        if field not in series:continue
        times=series['offset'];value=series[field];idx=(times>=start)&(times<=end)&np.isfinite(value)
        t=times[idx];v=value[idx];name=field
        x[name+'_n']=int(len(v));x[name+'_missing']=not len(v)
        for stat in ['last','mean','median','std','min','max','slope_per_hour']:x[name+'_'+stat]=None
        x[name+'_time_since_last_minutes']=None
        if len(v):
            x.update({name+'_last':float(v[-1]),name+'_mean':float(np.mean(v)),name+'_median':float(np.median(v)),
                      name+'_std':float(np.std(v,ddof=1)) if len(v)>1 else None,name+'_min':float(min(v)),name+'_max':float(max(v)),
                      name+'_time_since_last_minutes':float(t0-t[-1])})
            if len(v)>1 and np.ptp(t)>0:x[name+'_slope_per_hour']=float(np.polyfit((t-t[0])/60,v,1)[0])
        provenance['dynamic'][field]={'min_offset':float(t[0]) if len(t) else None,'max_offset':float(t[-1]) if len(t) else None,'n':len(v)}
    for concept in ['creatinine','potassium','sodium','lactate','bun','hgb','glucose']:
        lab=latest.get(concept)
        x['lab_'+concept]=None if lab is None else lab['value']
        x['lab_'+concept+'_unit']=None if lab is None else lab['unit']
        if lab:provenance['labs'].append(lab)
    return x,provenance


def prediction_windows(patient,series,events,horizon,lookback,step=5):
    if not patient['adult']:return [],{'not_adult_stay':1}
    times=series['offset'];values=series['systemicmean'];end=patient['end'];exclusions=Counter();result=[]
    lag=PROTOCOL['feature_lag_minutes'];minimum=lookback+lag
    for t0 in range(math.ceil(minimum/horizon)*horizon,int(end-horizon-30)+1,horizon):
        if not covered(times,values,t0-lookback-lag,t0-lag,PROTOCOL['history_coverage'],max(PROTOCOL['max_coverage_gap'],step),step):
            exclusions['insufficient_past_map_support']+=1;continue
        # Washout is causal: no known low MAP <65 in the preceding 120 minutes.
        prior=(times>=t0-PROTOCOL['washout_minutes'])&(times<=t0-lag)&np.isfinite(values)
        if np.any(values[prior]<65):exclusions['recent_low_map_washout']+=1;continue
        if not covered(times,values,t0,t0+horizon+30,PROTOCOL['future_coverage'],max(PROTOCOL['max_coverage_gap'],step),step):
            exclusions['future_label_not_observable']+=1;continue
        underway=[e for e in events if e['onset']<=t0 and e['last_low']>=t0]
        if underway:exclusions['event_underway_at_cutoff']+=1;continue
        future=[e for e in events if t0<e['onset']<=t0+horizon]
        if any(e['onset']<=t0+PROTOCOL['min_lead_minutes'] for e in future):
            exclusions['onset_inside_15min_lead_gap']+=1;continue
        outcome=int(bool(future));match=future[0] if future else None
        flags,_,_=asof_context(patient,t0)
        result.append({'patient_id':patient['patient_id'],'icu_stay_id':patient['icu_stay_id'],
            't0':t0,'horizon_minutes':horizon,'lookback_minutes':lookback,'label':outcome,
            'event_onset':match['onset'] if match else None,'lead_minutes':match['onset']-t0 if match else None,
            'hf_known_at_t0':flags['hf']==1,'cv_known_at_t0':flags['cardiovascular']==1,
            'hf_ever_during_stay':patient['ever']['hf'],'cv_ever_during_stay':patient['ever']['cardiovascular']})
    return result,dict(exclusions)


def summarize_windows(rows):
    patients={r['patient_id'] for r in rows};positive_patients={r['patient_id'] for r in rows if r['label']}
    return {'windows':len(rows),'positive_windows':sum(r['label'] for r in rows),'controls':sum(not r['label'] for r in rows),
            'patients':len(patients),'positive_patients':len(positive_patients),
            'patients_with_controls':len({r['patient_id'] for r in rows if not r['label']}),
            'negative_only_patients':len(patients-positive_patients),
            'window_prevalence':sum(r['label'] for r in rows)/len(rows) if rows else None,
            'patient_event_prevalence':len(positive_patients)/len(patients) if patients else None,
            'icu_stays':len({r['icu_stay_id'] for r in rows})}
