"""Role-gated frozen cohort access and deterministic causal feature construction."""
from collections import Counter
import json
import hashlib
from pathlib import Path
import numpy as np
from src.data.io import read_jsonl,file_hash
from src.data.eicu_validate import restore_series
from src.data.hemodynamic import features,asof_context,covered
from src.data.splits import stratified_split,validate_splits
from src.predictive.protocol import SIGNALS,BP,SCALES,RANGES,PROTOCOL


def guard(rows,assignments,allowed):
    if not rows or any(assignments.get(r['patient_id']) not in allowed for r in rows):
        raise ValueError('Patient role violation')


def weights(rows):
    n=Counter(r['patient_id'] for r in rows)
    w=np.array([1/n[r['patient_id']] for r in rows]);return w/w.mean()


def take(series,field,start,end):
    t=series['offset'];v=series[field];a=np.searchsorted(t,start,'left');b=np.searchsorted(t,end,'right')
    valid=np.isfinite(v[a:b]);return t[a:b][valid],v[a:b][valid]


def slope(t,v):
    return float(np.polyfit((t-t[0])/60,v,1)[0]) if len(v)>1 and np.ptp(t)>0 else None


def temporal(series,field,end):
    result={}
    for width in SCALES:
        t,v=take(series,field,end-width,end);n=len(v);prefix=f'phys__{field}__{width}__'
        stats=dict(n=n,missing=not n,coverage=min(1.,n/(width/5+1)),last=None,mean=None,median=None,
                   minimum=None,maximum=None,q10=None,q90=None,sd=None,cv=None,slope=None,
                   staleness=None,short_variability=None,low_supported_minutes=0.,high_supported_minutes=0.)
        if n:
            sd=float(np.std(v,ddof=1)) if n>1 else None
            stats.update(last=float(v[-1]),mean=float(np.mean(v)),median=float(np.median(v)),minimum=float(min(v)),maximum=float(max(v)),
                         q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)),sd=sd,
                         cv=sd/abs(float(np.mean(v))) if sd is not None and abs(np.mean(v))>1 else None,
                         slope=slope(t,v),staleness=float(end+5-t[-1]))
            if n>1:
                dt=np.diff(t);supported=(dt>0)&(dt<=10);lo,hi=RANGES[field]
                stats['short_variability']=float(np.mean(np.abs(np.diff(v)[supported]))) if any(supported) else None
                stats['low_supported_minutes']=float(sum(dt[supported&(v[:-1]<lo)&(v[1:]<lo)]))
                stats['high_supported_minutes']=float(sum(dt[supported&(v[:-1]>hi)&(v[1:]>hi)]))
        result.update({prefix+k:val for k,val in stats.items()})
    t1,v1=take(series,field,end-30,end);t0,v0=take(series,field,end-60,end-30-1)
    s1=slope(t1,v1);s0=slope(t0,v0)
    result[f'phys__{field}__recent_delta']=float(np.median(v1)-np.median(v0)) if len(v1) and len(v0) else None
    result[f'phys__{field}__acceleration']=2*(s1-s0) if s1 is not None and s0 is not None else None
    return result


def personal(series,field,end):
    t,v=take(series,field,end-360,end-60);recent_t,recent_v=take(series,field,end-30,end)
    dt=np.diff(t);connected=float(sum(dt[(dt>0)&(dt<=10)]))/60
    usable=len(v)>=30 and connected>=3
    last_t,last_v=take(series,field,end-360,end)
    out={'n':len(v),'coverage':min(1.,len(v)/61),'connected_hours':connected,'available':usable,'zero_variability':None,
         'mean':None,'median':None,'sd':None,'current_deviation':None,'standardized_deviation':None,
         'recent_reference_delta':None,'deviation_persistence_minutes':None}
    if usable:
        med=float(np.median(v));sd=float(np.std(v,ddof=1));out.update(mean=float(np.mean(v)),median=med,sd=sd,zero_variability=sd<1e-6)
        if len(last_v):out['current_deviation']=float(last_v[-1]-med)
        if len(recent_v):out['recent_reference_delta']=float(np.median(recent_v)-med)
        if sd>=1e-6:
            out['standardized_deviation']=out['current_deviation']/sd if len(last_v) else None
            if len(recent_v):
                sign=np.sign(recent_v[-1]-med);start=recent_t[-1];last=recent_t[-1]
                for ti,vi in reversed(list(zip(recent_t,recent_v))):
                    if abs(vi-med)<sd or np.sign(vi-med)!=sign or last-ti>10:break
                    start=ti;last=ti
                out['deviation_persistence_minutes']=float(recent_t[-1]-start)
    return {f'personal__{field}__{k}':val for k,val in out.items()}


def sequence(series,end):
    """Each bin closes at its own time; missing values remain NaN, never ffilled."""
    grid=np.arange(end-360,end+1,5);value=np.full((73,7),np.nan);mask=np.zeros((73,7));age=np.ones((73,7))
    for j,field in enumerate(SIGNALS):
        raw_t=series['offset'];raw_v=series[field];valid=np.isfinite(raw_v)&(raw_t>=0)&(raw_t<=end)
        t=raw_t[valid];v=raw_v[valid];right=np.searchsorted(t,grid,'right')
        left=np.searchsorted(t,np.r_[grid[0],grid[1:]-5+1e-8],'left')
        for i,(a,b) in enumerate(zip(left,right)):
            if b>a:value[i,j]=float(np.median(v[a:b]));mask[i,j]=1
            if b:age[i,j]=min(1.,float(grid[i]-t[b-1])/360)
    return np.concatenate([value,mask,age],axis=1)


def construct(patient,series,t0):
    end=t0-5;old,provenance=features(patient,series,t0,360,patient['labs'])
    approved=['age','age_over89','gender','unit_type','unit_admission_source']+[k for k in old if k.startswith('documented_') or k.startswith('lab_')]
    x={'ehr__'+k:old[k] for k in approved}
    _,_,labs=asof_context(patient,t0,patient['labs'])
    for concept in ['creatinine','potassium','sodium','lactate','bun','hgb','glucose']:
        x['ehr__lab_'+concept+'_age_hours']=(t0-labs[concept]['observed_offset'])/60 if concept in labs else None
    for field in SIGNALS:x.update(temporal(series,field,end));x.update(personal(series,field,end))
    current_t,current_v=take(series,'systemicmean',0,t0);t,v=take(series,'systemicmean',end-120,end)
    _,washout_values=take(series,'systemicmean',t0-120,end)
    m30=x['phys__systemicmean__30__minimum'];m60=x['phys__systemicmean__60__minimum'];last=x['phys__systemicmean__30__last'];sl=x['phys__systemicmean__60__slope']
    audit={'map_at_t0':float(current_v[-1]) if len(current_v) else None,
           'map_at_t0_offset':float(current_t[-1]) if len(current_t) else None,
           'buffered_latest_map':last,'minimum_30m':m30,'minimum_60m':m60,'map_slope':sl,
           'distance_from65':None if last is None else last-65,'recent_low_excursions':int(sum(v<65)),
           'hf':old['documented_hf']==1,'max_physiology_offset':end,
           'history_supported':covered(series['offset'],series['systemicmean'],end-360,end,.8,10),
           'washout_pass':not any(washout_values<65),
           'maximum_available_clinical_offset':max([r['available_offset'] for r in provenance['context_records']+provenance['labs']],default=None)}
    cm=audit['map_at_t0'];fresh=t0-audit['map_at_t0_offset'] if cm is not None else None
    audit['map_stratum']='missing' if cm is None else 'below_threshold' if cm<65 else 'near_threshold' if cm<70 else 'moderate' if cm<80 else 'clearly_normal'
    audit['hard']=bool(cm is not None and cm>=80 and fresh<=10 and last is not None and last>=80 and m60 is not None and m60>=75 and sl is not None and sl>=-1 and not any(v<65))
    return x,sequence(series,end),audit


def columns(modality,keys):
    if modality not in PROTOCOL['modalities']:raise ValueError('Unknown modality')
    chosen=[]
    for key in sorted(keys):
        ehr=key.startswith('ehr__');phys=key.startswith('phys__');personalized=key.startswith('personal__')
        signal=key.split('__')[1] if phys or personalized else None
        keep={'ehr':ehr,'map_only':phys and signal=='systemicmean','bp_only':phys and signal in BP,
              'phys_no_map':phys and signal!='systemicmean','phys_no_bp':phys and signal not in BP,
              'phys':phys,'fused':ehr or phys,'personalized':ehr or phys or personalized}[modality]
        if keep:chosen.append(key)
    return chosen


class Cohort:
    def __init__(self,root='data/processed/eicu_demo'):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        for name,meta in self.manifest['artifacts'].items():
            if file_hash(self.root/name)!=meta['sha256']:raise ValueError('Frozen source artifact changed')
        self.assignments=json.loads((self.root/'splits.json').read_text())['assignments']
        self.patients={p['icu_stay_id']:p for p in read_jsonl(self.root/'patient_context.jsonl')}
        self.series=restore_series(self.root/'vitalPeriodic.npz')
        self.events=[r for r in read_jsonl(self.root/'events.jsonl') if r['candidate']=='map65_30min']
        # Frozen labels are touched for integrity/count metadata, never for model choice.
        self.labels={r['window_id']:r for r in read_jsonl(self.root/'labels.jsonl')}
        self.indices=[]
        for row in read_jsonl(self.root/'predictors.jsonl'):
            self.indices.append({k:row[k] for k in ['window_id','patient_id','icu_stay_id','t0']})
        labels={pid:int(any(self.labels[r['window_id']]['label'] for r in self.indices if r['patient_id']==pid)) for pid in self.assignments}
        expected=stratified_split(labels,2026)['assignments']
        if expected!=self.assignments:raise ValueError('Frozen deterministic split differs')
        validate_splits({'assignments':self.assignments},set(self.assignments))

    def batch(self,role,selection_seal=None):
        if role not in {'train','validation','test'}:raise ValueError('Unknown role')
        if role=='test':
            if selection_seal is None:raise ValueError('Final test feature access requires frozen selection seal')
            sealpath=Path(selection_seal);seal=json.loads(sealpath.read_text())
            if seal.get('status')!='FROZEN' or seal.get('dataset_manifest_sha256')!=file_hash(self.root/'manifest.json'):
                raise ValueError('Invalid final-test selection seal')
            if not (sealpath.parent/'evaluation_started.json').exists():raise ValueError('Test transaction has not started')
            for name,sha in seal['hashes'].items():
                if file_hash(sealpath.parent/name)!=sha:raise ValueError('Frozen modeling pipeline changed')
        result=[]
        for index in self.indices:
            if self.assignments[index['patient_id']]!=role:continue
            p=self.patients[index['icu_stay_id']];x,seq,audit=construct(p,self.series[p['source_stay_key']],index['t0'])
            label=self.labels[index['window_id']]
            result.append({**index,'role':role,'features':x,'sequence':seq,'audit':audit,
                           'label':label['label'],'event_onset':label['event_onset']})
        guard(result,self.assignments,{role});return result
