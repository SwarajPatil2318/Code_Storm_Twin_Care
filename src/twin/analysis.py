"""Frozen secondary analyses; oracle annotations are evaluation-only."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from src.twin.protocol import PRIMARY_SIGNALS,EVIDENCE
from src.twin.metrics import basic,paired_ap
from src.twin.inference import finite
from src.synthetic.config import SIGNALS


def summary(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return {'n':len(a),'mean':finite(a.mean()) if len(a) else None,'median':finite(np.median(a)) if len(a) else None,'q10_q90':np.quantile(a,[.1,.9]).tolist() if len(a) else None}

def paired_mean(values,ids,n=1000,seed=2026):
    """Mean per patient, then paired patient-bootstrap of that estimand."""
    values=np.asarray(values,float);ids=np.asarray(ids);valid=np.isfinite(values);unique=np.unique(ids[valid])
    if not len(unique):return {'difference':None,'interval95':None,'patients':0}
    sums=np.zeros(len(unique));counts=np.zeros(len(unique));group=np.searchsorted(unique,ids[valid]);np.add.at(sums,group,values[valid]);np.add.at(counts,group,1);means=sums/counts;rng=np.random.default_rng(seed)
    samples=[float(means[rng.integers(len(means),size=len(means))].mean()) for _ in range(n)]
    return {'difference':float(means.mean()),'interval95':np.quantile(samples,[.025,.975]).tolist(),'patients':len(unique),'estimand':'Equal-weight patient mean difference','requested':n}

def shadow_quality(shadow,data,mask,bootstraps=1000):
    result=shadow.quality(data,mask);forecast=shadow.predict(data)
    for j,s in enumerate(PRIMARY_SIGNALS):
        good=mask&np.isfinite(data['recent'][:,j]);actual=data['recent'][good,j];err=abs(actual-forecast['expected'][good,j]);pop=abs(actual-forecast['population'][good,j])
        result[s]['paired_patient_mae_gain']=paired_mean(pop-err,data['patient_ids'][good],bootstraps)
        # Private patient-level errors remain useful for inspecting heterogeneous fit.
        result[s]['patient_error_distribution']=summary([np.mean(err[data['patient_ids'][good]==p]) for p in np.unique(data['patient_ids'][good])])
    return {'signals':result,'expected_response':'48h received-observation mean; descriptive retrospective bin templates are a separate output',**EVIDENCE}

def oracle_annotations(cohort,role,data):
    """Only call inside the already frozen joint evaluation. No feature writes."""
    rows={r['patient_id']:r for r in cohort.rows('ground_truth',role,oracle=True)};n=len(data['patient_ids']);state=np.empty(n,np.uint8);stable48=np.zeros(n,bool);healthy=np.empty_like(data['reference']);decoy=np.empty(n,object);path=np.empty(n,object);event=np.full(n,np.nan);recovery=np.zeros(n,bool)
    for pid,stream in cohort.streams(role,oracle=True):
        indices=np.flatnonzero(data['patient_ids']==pid);truth=rows[pid]
        for i in indices:
            hour=int(data['landmark_hours'][i]);cut=min(hour,len(stream['state'])-1);state[i]=stream['state'][cut];stable48[i]=bool(np.all(stream['state'][max(0,cut-47):cut+1]==0));healthy[i]=stream['stable_daily'][min(cut//24,len(stream['stable_daily'])-1),[SIGNALS.index(s) for s in PRIMARY_SIGNALS]]
            event[i]=truth['event_time_hours'] if truth['event_time_hours'] is not None else np.nan
            decoy[i]=[d['kind'] for d in truth['decoys'] if d['start_hour']<=hour and d['end_hour']>hour-48]
            episodes=[e for e in truth['episodes'] if e['onset_hour']<=hour<e['recovery_end_hour']];path[i]=episodes[-1]['path'] if episodes else 'no active episode';recovery[i]=any(e['recovery_hour']<=hour<e['recovery_end_hour'] for e in episodes)
    return {'truth':rows,'state':state,'stable48':stable48,'healthy_reference':healthy,'decoys':decoy,'path':path,'event_hour':event,'recovery':recovery}

def baseline_quality(data,oracle):
    result={}
    for j,s in enumerate(PRIMARY_SIGNALS):
        error=abs(data['reference'][:,j]-oracle['healthy_reference'][:,j]);stable=oracle['stable48'];stress=(oracle['state']>=1)&(oracle['state']<=4)
        result[s]={'stable_reference_error':summary(error[stable]),'deterioration_reference_error':summary(error[stress]),'trusted_history_count':summary(data['baseline_count'][:,j]),'baseline_confidence':summary(data['confidence'][:,j])}
    fraction=lambda mask:float(np.mean(data['baseline_frozen'][mask])) if mask.any() else None
    return {'signals':result,'freeze_fraction_stable':fraction(oracle['stable48']), 'freeze_fraction_deterioration':fraction((oracle['state']>=1)&(oracle['state']<=4)), 'oracle_scope':'Exogenous simulated healthy daily level; not an observed clinical reference',**EVIDENCE}

def tdi_analysis(data,dr,oracle):
    y=data['labels'].sum(axis=1);tdi=dr['tdi'];good=np.isfinite(tdi);result={'stable':summary(tdi[oracle['stable48']]),'deteriorating':summary(tdi[(oracle['state']>=1)&(oracle['state']<=4)]),'event_next72':summary(tdi[y>0]),'nonevent_next72':summary(tdi[y==0]),'missing_entirely':int((~good).sum())}
    if len(set(y[good]))==2:result.update(auprc72=float(average_precision_score(y[good],tdi[good])),auroc72=float(roc_auc_score(y[good],tdi[good])))
    result['by_path']={p:summary(tdi[oracle['path']==p]) for p in sorted(set(oracle['path']))};result['component_scope']='Normalized absolute standardized residual shares; not an additive covariance-distance decomposition';result.update(EVIDENCE);return result

def event_aligned(data,prob,dr,oracle,config):
    result={};curves=[]
    for lag in config['event_lags_hours']:
        indices=[];offsets=[]
        for pid,r in oracle['truth'].items():
            event=r['event_time_hours']
            if event is None:continue
            target=event-lag;eligible=np.flatnonzero((data['patient_ids']==pid)&(data['landmark_hours']<=target))
            # Daily decision schedule: use most recent issued prediction, never
            # interpolate with a future landmark. Actual lag differs by <24h.
            if len(eligible):
                i=eligible[np.argmax(data['landmark_hours'][eligible])]
                if target-data['landmark_hours'][i]<24:indices.append(i);offsets.append(event-data['landmark_hours'][i])
        ix=np.array(indices,int);result[str(lag)]={'patients':len(ix),'actual_lead_hours':summary(offsets),'risk24':summary(prob[ix,0]),'risk48':summary(prob[ix,1]),'risk72':summary(prob[ix,2]),'tdi':summary(dr['tdi'][ix]),'physiological_z':{s:summary(dr['z'][ix,j]) for j,s in enumerate(PRIMARY_SIGNALS)}}
    for pid,r in oracle['truth'].items():
        if r['event_time_hours'] is None:continue
        ix=np.flatnonzero((data['patient_ids']==pid)&(data['landmark_hours']<=r['event_time_hours'])&(data['landmark_hours']>=r['event_time_hours']-192))
        if len(ix):curves.append({'patient_id':pid,'lead_hours':(r['event_time_hours']-data['landmark_hours'][ix]).tolist(),'risk':prob[ix].tolist(),'tdi':[finite(x) for x in dr['tdi'][ix]],'z':[[finite(x) for x in row] for row in dr['z'][ix]]})
        if len(curves)>=12:break
    return {'aggregate':result,'representative_private_curves':curves,'schedule':'Daily predictions; lag0 uses the last pre-event issued prediction, not an event-time observation',**EVIDENCE}

def warning_times(data,prob,thresholds,oracle):
    result={}
    for j in range(3):
        horizon=(j+1)*24;result[str(horizon)]={}
        for name,t in thresholds[str(horizon)].items():
            leads=[];eligible_events=0
            for pid,r in oracle['truth'].items():
                event=r['event_time_hours']
                if event is None:continue
                ix=np.flatnonzero((data['patient_ids']==pid)&(data['landmark_hours']<event)&(data['landmark_hours']>=event-horizon))
                if not len(ix):continue
                eligible_events+=1;alerts=ix[prob[ix,j]>=t]
                if len(alerts):leads.append(float(event-min(data['landmark_hours'][alerts])))
            result[str(horizon)][name]={'eligible_events':eligible_events,'warned_events':len(leads),'event_sensitivity':len(leads)/eligible_events if eligible_events else None,'warning_time_hours':summary(leads),'warning_definition':'Earliest daily alert inside this horizon before first generated event; undetected events excluded from conditional warning median'}
    return result

def decoy_results(data,prob,thresholds,oracle):
    negative=data['labels'].sum(axis=1)==0;result={};kinds=sorted({k for row in oracle['decoys'] for k in row})
    for kind in kinds+['recovery','no_decoy_or_recovery']:
        mask=oracle['recovery'] if kind=='recovery' else np.array([not row for row in oracle['decoys']])&~oracle['recovery'] if kind=='no_decoy_or_recovery' else np.array([kind in row for row in oracle['decoys']]);mask &= negative
        result[kind]={'rows':int(mask.sum()),'patients':len(set(data['patient_ids'][mask])),'false_alerts72':int(np.sum(prob[mask,2]>=thresholds['72']['balanced'])),'false_alert_fraction72':float(np.mean(prob[mask,2]>=thresholds['72']['balanced'])) if mask.any() else None,'risk72':summary(prob[mask,2])}
    return {'groups':result,'eligibility':'No generated event within72h; groups can overlap; synthetic episode annotations are oracle evaluation only',**EVIDENCE}

def uncertainty_results(data,prob,ensemble,quality):
    sd=np.std(ensemble,axis=0,ddof=1);truth=np.cumsum(data['labels'],axis=1);error=np.mean((prob-truth)**2,axis=1);result={}
    for name,values in [('model_sd72',sd[:,2]),('data_quality',quality)]:
        result[name]={'distribution':summary(values),'quartile_brier':[]}
        bins=np.array_split(np.argsort(values,kind='stable'),4)
        for ix in bins:result[name]['quartile_brier'].append({'rows':len(ix),'uncertainty':summary(values[ix]),'brier':float(error[ix].mean())})
        result[name]['correlation_with_squared_error']=finite(np.corrcoef(values,error)[0,1])
    result.update(ensemble_members=len(ensemble),interval_scope='5/95% range across three conditional bootstrap predictions is descriptive, not validated coverage of unobserved true patient risk',**EVIDENCE);return result,sd
