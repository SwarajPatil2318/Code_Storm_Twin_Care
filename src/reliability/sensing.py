"""One-measurement clarification with causal hide/reveal and relative burden.

The utility learner sees only the masked state. Revealed observations and labels
define TRAIN utility targets, never recommendation inputs. Confidence is not
claimed to be clinically calibrated.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.synthetic.config import SIGNALS
from src.twin.baseline import PersonalBaseline
from src.twin.features import representation
from src.twin.models import cumulative
from src.twin.protocol import PRIMARY_SIGNALS
from src.twin.inference import quality_uncertainty
from src.reliability.access import (root,save_arrays,save_json,save_model,timed,
    deterministic_patients,routed_rows,routed_streams,phase6_identity,reference_artifact)
from src.reliability.development import reference_predict
from src.reliability.features import recovery_features, scores
from src.reliability.models import temporal_recovery_rule

CANDIDATES=['weight','bp','spo2','hrv_rmssd']
CHANNELS={'weight':['weight'],'bp':['systolic_bp','diastolic_bp'],'spo2':['spo2'],'hrv_rmssd':['hrv_rmssd']}


def latest_revealable(stream, hour, measurement):
    """A received reading newly available since the previous daily cutoff."""
    columns=[SIGNALS.index(s) for s in CHANNELS[measurement]]
    t=np.arange(len(stream['value']))
    valid=np.ones(len(t),bool)
    for j in columns:
        valid &= (t<=hour)&(t>hour-48)&(stream['available_hour'][:,j]<=hour)&(stream['available_hour'][:,j]>hour-24)
        valid &= ~stream['missingness_mask'][:,j]&np.isfinite(stream['value'][:,j])
    available=np.flatnonzero(valid)
    return (int(available[-1]),columns) if len(available) else None


def hide_readings(stream,held):
    masked={k:v.copy() for k,v in stream.items()}
    for item in held.values():
        if item is None:continue
        hour,columns=item
        for j in columns:
            masked['value'][hour,j]=np.nan;masked['raw_value'][hour,j]=np.nan
            masked['missingness_mask'][hour,j]=True;masked['quality_status'][hour,j]=1
            masked['available_hour'][hour,j]=np.nan
    return masked


def reveal_reading(masked,original,item,cutoff):
    revealed={k:v.copy() for k,v in masked.items()}
    if item is None:return revealed
    hour,columns=item
    for j in columns:
        if hour>cutoff or original['available_hour'][hour,j]>cutoff or not np.isfinite(original['value'][hour,j]):
            raise PermissionError('Cannot reveal an unavailable or future observation')
        for key in revealed:
            revealed[key][hour,j]=original[key][hour,j]
    return revealed


def pipeline_predict(data,bundle):
    h=reference_predict(data,bundle['reference_model'],bundle['reference_calibrator'])
    f=recovery_features(data)
    if bundle['winner']=='offset_logistic':adjusted=bundle['reliability_model'].predict(h,f)
    elif bundle['winner']=='temporal_recovery_rule':adjusted=temporal_recovery_rule(h,f,bundle['temporal_attenuation'])
    else:adjusted=h
    risk=cumulative(bundle['calibrator'].transform(adjusted))
    return risk,f,cumulative(h)


def utility_inputs(data,features,probability,threshold):
    p=probability[:,2]
    margin=abs(p-threshold)/max(threshold,.001)
    entropy=-(p*np.log(np.clip(p,1e-9,1))+(1-p)*np.log(np.clip(1-p,1e-9,1)))
    absolute=np.asarray(data['absolute']).reshape(len(p),len(PRIMARY_SIGNALS),10)
    choices=[]
    for k,m in enumerate(CANDIDATES):
        ix=[PRIMARY_SIGNALS.index(s) for s in CHANNELS[m]]
        age=np.max(absolute[:,ix,6],axis=1)/168
        coverage=np.mean(absolute[:,ix,5],axis=1)
        confidence=np.mean(np.asarray(data['confidence'])[:,ix],axis=1)
        indicator=np.zeros((len(p),4));indicator[:,k]=1
        choices.append(np.c_[p,margin,entropy,features,age,coverage,confidence,indicator])
    return np.stack(choices,1).astype(np.float32)


class SensingUtility:
    def fit(self,cases,config):
        if set(cases['roles'].tolist())!={'train'}:
            raise PermissionError('Sensing utility learning is TRAIN-only')
        x=cases['utility_inputs'].reshape(-1,cases['utility_inputs'].shape[-1])
        before=cases['before_risk'][:,2,None]
        after=cases['after_risk'][:,:,2]
        y=cases['labels'].sum(1)[:,None]
        t=cases['threshold72'][0]
        correction=((before>=t)!=y)-((after>=t)!=y).astype(float)
        gain=(before-y)**2-(after-y)**2+config['decision_utility_weight']*correction
        cost=np.array([config['burden_weights'][m] for m in CANDIDATES])*config['burden_brier_scale']
        target=gain-cost
        eligible=cases['available'].ravel()
        if eligible.sum()<20:raise ValueError('Insufficient observed hide/reveal examples')
        self.median=np.nanmedian(x[eligible],axis=0);self.median=np.nan_to_num(self.median)
        self.scaler=StandardScaler().fit(np.where(np.isfinite(x[eligible]),x[eligible],self.median))
        values=self.scaler.transform(np.where(np.isfinite(x[eligible]),x[eligible],self.median))
        self.model=Ridge(alpha=config['sensing_ridge_alpha']).fit(values,target.ravel()[eligible])
        self.cost=cost;self.config=config
        self.fit_patients=sorted(set(cases['patient_ids'].tolist()))
        return self

    def expected(self,inputs):
        shape=inputs.shape[:2];x=inputs.reshape(-1,inputs.shape[-1])
        return self.model.predict(self.scaler.transform(np.where(np.isfinite(x),x,self.median))).reshape(shape)


def recommend(data,features,risk,utility,threshold,config,cooldown=True):
    inputs=utility_inputs(data,features,risk,threshold)
    net=utility.expected(inputs)
    choice=np.argmax(net,axis=1)
    quality=quality_uncertainty(data)
    ambiguous=abs(risk[:,2]-threshold)<=max(config['ambiguity_absolute_margin'],config['ambiguity_relative_margin']*threshold)
    absolute=np.asarray(data['absolute']).reshape(len(risk),len(PRIMARY_SIGNALS),10)
    stale=np.any(absolute[:,[PRIMARY_SIGNALS.index(s) for s in ['weight','systolic_bp','spo2','hrv_rmssd']],6]>config['sensor_stale_hours'],axis=1)
    sufficient=(features[:,-1]>=.65)&(features[:,-5]>=3)&(quality<.25)&(risk[:,2]>1.35*threshold)
    needed=(ambiguous | ((quality>.3)&stale&(risk[:,2]>.5*threshold)))&~sufficient&(net[np.arange(len(net)),choice]>0)
    if cooldown:
        ids=np.asarray(data['patient_ids']);hours=np.asarray(data['landmark_hours']);last={}
        order=np.lexsort((hours,ids))
        for i in order:
            if needed[i]:
                previous=last.get(str(ids[i]),-np.inf)
                if hours[i]-previous<config['request_cooldown_hours']:needed[i]=False
                else:last[str(ids[i])]=int(hours[i])
    return {'measurement_needed':needed,'choice':choice,'expected_net_utility':net[np.arange(len(net)),choice],
        'expected_information_gain':net[np.arange(len(net)),choice]+utility.cost[choice],
        'current_data_quality':quality,'ambiguous':ambiguous,'sufficient_evidence':sufficient,
        'current_uncertainty':-(risk[:,2]*np.log(np.clip(risk[:,2],1e-9,1))+(1-risk[:,2])*np.log(np.clip(1-risk[:,2],1e-9,1))),
        'uncertainty_definition':'Predictive entropy, descriptive; not validated individual confidence'}


def case_landmarks(pid,seed,count=2):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:sensing:{pid}'.encode()).digest()[:8],'big'))
    return sorted((168+24*rng.choice(51,size=count,replace=False)).tolist())


def development_cases(config,bundle,data,count,stage):
    path=root(config)/(stage+'_sensing_cases.npz');proof=path.with_suffix('.json')
    identity={'stage':stage,'patients':count,'config_sha256':config['_sha256'],
        'bundle_sha256':bundle['identity_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if path.exists():
        p=json.loads(proof.read_text())
        from src.data.io import file_hash
        if any(p.get(k)!=v for k,v in identity.items()) or p['artifact_sha256']!=file_hash(path):raise ValueError('Stale hide/reveal cache')
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    if (root(config)/'PHASE6_5_FROZEN_MANIFEST.json').exists():raise PermissionError('Development sensing closed')
    start=time.monotonic();allowed_train=data['roles']=='train'
    if stage=='fast':allowed_train &= data['fast_subset']
    ids=deterministic_patients(data['patient_ids'][allowed_train],count,config['subset_seed'],stage+'_sensing')
    wanted=set(ids);context={r['patient_id']:r['context'] for r in routed_rows('ehr',wanted)}
    labels={(str(pid),int(h)):y for pid,h,y in zip(data['patient_ids'][allowed_train],data['landmark_hours'][allowed_train],data['labels'][allowed_train]) if pid in wanted}
    cases=[];phase6=phase6_identity()[0]
    for ordinal,(pid,stream) in enumerate(routed_streams(wanted)):
        targets=case_landmarks(pid,config['subset_seed'],config['sensing_landmarks_per_patient'])
        baseline=PersonalBaseline(phase6)
        for hour in range(168,max(targets)+1,24):
            if hour in targets:
                cases.append(make_case(pid,context[pid],stream,baseline,hour,labels[(pid,hour)],phase6,bundle,'train'))
            baseline.update(stream,hour)
        if (ordinal+1)%20==0:print({'phase65_sensing_history_replays':ordinal+1,'target':count,'stage':stage},flush=True)
    result=stack_cases(cases,bundle)
    save_arrays(path,**result)
    from src.data.io import file_hash
    save_json(proof,{**identity,'artifact_sha256':file_hash(path),'case_landmarks':len(cases),
        'mask_scope':'Latest newly received measurement per candidate, all candidates masked before current baseline update',
        'future_values_revealed':False,'spent_test_used':False})
    timed(config,stage+'_sensing_exact_hide_reveal',start,{'patients':count,'landmarks':len(cases)})
    return result


def make_case(pid,context,stream,prior_baseline,hour,label,phase6,bundle,role):
    held={m:latest_revealable(stream,hour,m) for m in CANDIDATES}
    masked=hide_readings(stream,held)
    def encode(values):
        b=copy.deepcopy(prior_baseline)
        ref=b.update(values,hour)
        f=representation(context,values,hour,ref,phase6)
        return {k:np.asarray(v)[None,...] for k,v in f.items()}
    before_data=encode(masked)
    before,features,_=pipeline_predict(before_data,bundle)
    after=[];readings=[]
    for m in CANDIDATES:
        revealed=reveal_reading(masked,stream,held[m],hour)
        p,_,_=pipeline_predict(encode(revealed),bundle);after.append(p[0])
        item=held[m]
        readings.append([float(stream['value'][item[0],j]) for j in item[1]] if item else [])
    threshold=bundle['thresholds']['72']['balanced']
    return {'patient_id':pid,'hour':hour,'role':role,'label':label,
        'before_risk':before[0],'after_risk':np.array(after),'available':np.array([held[m] is not None for m in CANDIDATES]),
        'utility_inputs':utility_inputs(before_data,features,before,threshold)[0],
        'features':features[0],'absolute':before_data['absolute'][0],'confidence':before_data['confidence'][0],
        'baseline_count':before_data['baseline_count'][0],'reading_values_json':json.dumps(readings)}


def stack_cases(cases,bundle):
    return {'patient_ids':np.array([x['patient_id'] for x in cases]),'landmark_hours':np.array([x['hour'] for x in cases]),
        'roles':np.array([x['role'] for x in cases]),'labels':np.stack([x['label'] for x in cases]),
        **{k:np.stack([x[k] for x in cases]) for k in ['before_risk','after_risk','available','utility_inputs','features','absolute','confidence','baseline_count']},
        'reading_values_json':np.array([x['reading_values_json'] for x in cases]),
        'threshold72':np.array([bundle['thresholds']['72']['balanced']])}
