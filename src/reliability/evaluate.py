"""Exactly one frozen seed-2027 replication transaction, with resumable outputs."""
import datetime as dt
import json
import os
from pathlib import Path
import time
import uuid

import joblib
import numpy as np
from numpy.lib.format import open_memmap
from threadpoolctl import threadpool_limits

from src.data.io import file_hash
from src.synthetic.generate import iter_rows
from src.synthetic.config import SIGNALS
from src.twin.access import OBS_KEYS
from src.twin.baseline import PersonalBaseline
from src.twin.features import representation
from src.twin.prepare import ARRAYS
from src.twin.metrics import patient_bootstrap,paired_ap
from src.twin.protocol import load as phase6_config,EVIDENCE
from src.twin.inference import quality_uncertainty
from src.reliability.access import load_config,root,save_json,save_arrays,timed
from src.reliability.sealing import verify
from src.reliability.sensing import (make_case,stack_cases,case_landmarks,pipeline_predict,recommend,CANDIDATES)
from src.reliability.features import scores
from src.reliability.development import summary


def begin(config):
    verify(config);out=root(config);path=out/'replication_evaluation_started.json'
    identity={'frozen_manifest_sha256':file_hash(out/'PHASE6_5_FROZEN_MANIFEST.json'),
        'replication_manifest_sha256':file_hash(Path(config['replication'])/'manifest.json')}
    if path.exists():
        d=json.loads(path.read_text())
        if any(d.get(k)!=v for k,v in identity.items()):raise ValueError('Cannot evaluate a changed frozen study')
        return d
    d={**EVIDENCE,**identity,'event_id':str(uuid.uuid4()),'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'logical_replication_evaluations':1,'status':'STARTED'}
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:json.dump(d,f,sort_keys=True,indent=2)
    return d


def annotation_flags(truth,hours):
    active=np.zeros(len(hours),bool);recovery=active.copy();decoy=active.copy();deterioration=active.copy()
    for e in truth['episodes']:
        active |= (hours>=e['onset_hour'])&(hours<e['recovery_end_hour'])
        recovery |= (hours>=e['recovery_hour'])&(hours<e['recovery_end_hour'])
        deterioration |= (hours>=e['onset_hour'])&(hours<e['recovery_hour'])
    for e in truth['decoys']:decoy |= (e['start_hour']<=hours)&(e['end_hour']>hours-48)
    return {'recovery':recovery,'deterioration':deterioration,'decoy':decoy,'ordinary_stable':~active&~decoy}


def prepare(config,bundle):
    out=root(config);directory=out/'replication_features';manifest=directory/'manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        if any(file_hash(directory/name)!=h for name,h in m['artifacts'].items()):raise ValueError('Replication cache changed')
        return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in directory.glob('*.npy')}
    start=time.monotonic();cohort=Path(config['replication']);meta=json.loads((cohort/'manifest.json').read_text())
    if meta['seed']!=2027 or meta['patients']!=2500:raise ValueError('Wrong replication cohort')
    audit=json.loads((cohort/'replication_audit.json').read_text())
    if not audit['all_passed']:raise PermissionError('Essential simulator gates did not pass')
    context={r['patient_id']:r['context'] for r in iter_rows(cohort/'ehr/records.jsonl.gz')}
    # Outcomes are a separate label-only interface, never included in features.
    labels={(r['patient_id'],r['landmark_hour']):[r['event_0_24h'],r['event_24_48h'],r['event_48_72h']] for r in iter_rows(cohort/'outcomes/records.jsonl.gz')}
    truth={r['patient_id']:r for r in iter_rows(cohort/'ground_truth/records.jsonl.gz')}
    from src.reliability.access import deterministic_patients
    sensing=set(deterministic_patients(list(context),config['replication_sensing_patients'],2027,'replication_sensing'))
    directory.mkdir(exist_ok=True,mode=0o700);n=127500
    arrays={k:open_memmap(directory/(k+'.npy'),mode='w+',dtype=np.float32,shape=(n,*shape)) for k,shape in ARRAYS.items()}
    y=np.empty((n,3),np.uint8);ids=[];landmarks=[];cases=[];row=0;pc=phase6_config()
    for name in meta['batches']:
        with np.load(cohort/'observations'/name,allow_pickle=False) as z:
            block={k:z[k] for k in OBS_KEYS};pids=z['patient_ids'].tolist()
        for ordinal,pid in enumerate(pids):
            stream={k:v[ordinal] for k,v in block.items()};baseline=PersonalBaseline(pc)
            selected=case_landmarks(pid,2027,config['sensing_landmarks_per_patient']) if pid in sensing else []
            for hour in range(168,1369,24):
                if hour in selected:
                    cases.append(make_case(pid,context[pid],stream,baseline,hour,labels[(pid,hour)],pc,bundle,'replication'))
                ref=baseline.update(stream,hour);features=representation(context[pid],stream,hour,ref,pc)
                for k,mem in arrays.items():mem[row]=features[k]
                y[row]=labels[(pid,hour)];ids.append(pid);landmarks.append(hour);row+=1
            if row//51%100==0:print({'phase65_replication_feature_patients':row//51,'target':2500},flush=True)
    if row!=n:raise ValueError('Incomplete replication feature schedule')
    for value in arrays.values():value.flush()
    hours=np.array(landmarks,np.int16);patient_ids=np.array(ids)
    extra={'labels':y,'patient_ids':patient_ids,'landmark_hours':hours}
    flags={k:np.zeros(n,bool) for k in ['recovery','deterioration','decoy','ordinary_stable']}
    for a in range(0,n,51):
        annotation=annotation_flags(truth[ids[a]],hours[a:a+51])
        for k in flags:flags[k][a:a+51]=annotation[k]
    extra.update(flags)
    for key,value in extra.items():np.save(directory/(key+'.npy'),value,allow_pickle=False)
    save_arrays(out/'replication_sensing_cases.npz',**stack_cases(cases,bundle))
    artifacts={p.name:file_hash(p) for p in directory.glob('*.npy')}
    save_json(manifest,{'patients':2500,'rows':n,'artifacts':artifacts,'cohort_manifest_sha256':file_hash(cohort/'manifest.json'),
        'frozen_manifest_sha256':file_hash(out/'PHASE6_5_FROZEN_MANIFEST.json'),'sensing_benchmark_patients':len(sensing),
        'sensing_benchmark_landmarks':len(cases),'sensing_sample_fixed_before_outcomes':True,'oracle_predictor_inputs':False})
    for p in directory.iterdir():p.chmod(0o600)
    timed(config,'replication_feature_and_hide_reveal_evaluation',start,{'patients':2500,'rows':n,'benchmark_landmarks':len(cases)})
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in directory.glob('*.npy')}


def sensing_metrics(cases,utility,bundle,config):
    data={k:cases[k] for k in ['absolute','confidence','patient_ids','landmark_hours']}
    before=cases['before_risk'];threshold=bundle['thresholds']['72']['balanced']
    rec=recommend(data,cases['features'],before,utility,threshold,config,cooldown=False)
    needed=rec['measurement_needed'];choice=rec['choice'];row=np.arange(len(before));available=cases['available'][row,choice]
    after=cases['after_risk'][row,choice];after=np.where((needed&available)[:,None],after,before)
    y=cases['labels'].sum(1);old=before[:,2]>=threshold;new=after[:,2]>=threshold
    corrected=needed&(old!=y)&(new==y);harmed=needed&(old==y)&(new!=y)
    gain=(before[:,2]-y)**2-(after[:,2]-y)**2
    net=gain+config['decision_utility_weight']*(corrected.astype(float)-harmed.astype(float))-needed*utility.cost[choice]
    entropy=lambda p:-(p*np.log(np.clip(p,1e-9,1))+(1-p)*np.log(np.clip(1-p,1e-9,1)))
    n=int(needed.sum());div=lambda value,mask:float(np.mean(value[mask])) if mask.any() else None
    useful_exists=np.any(cases['available']&(((cases['before_risk'][:,2,None]-y[:,None])**2-(cases['after_risk'][:,:,2]-y[:,None])**2)>0),axis=1)
    result={'benchmark_patients':len(np.unique(cases['patient_ids'])),'benchmark_landmarks':len(y),
        'benchmark_scope':'Deterministic two-landmark sample per500replication patients; all newly available candidate readings jointly hidden before revealing one',
        'ambiguous_cases':int(rec['ambiguous'].sum()),'useful_measurement_exists_fraction_ambiguous':div(useful_exists,rec['ambiguous']),
        'requests':n,'recommendation_fraction_benchmark':n/len(y),
        'requested_measurement_distribution':{m:int(np.sum(needed&(choice==j))) for j,m in enumerate(CANDIDATES)},
        'mean_expected_information_gain_requested':div(rec['expected_information_gain'],needed),
        'mean_observed_brier_gain_requested':div(gain,needed),'mean_observed_net_utility_requested':div(net,needed),
        'mean_absolute_probability_change_requested':div(abs(after[:,2]-before[:,2]),needed),
        'mean_predictive_entropy_reduction_requested':div(entropy(before[:,2])-entropy(after[:,2]),needed),
        'entropy_is_not_validated_uncertainty_reduction':True,
        'decisions_corrected':int(corrected.sum()),'decisions_harmed':int(harmed.sum()),
        'corrected_fraction_of_incorrect_requested':float(corrected.sum()/max(np.sum(needed&(old!=y)),1)),
        'harmed_fraction_of_correct_requested':float(harmed.sum()/max(np.sum(needed&(old==y)),1)),
        'alert_suppression_rate_requested':div(old&~new,needed),
        'deterioration_confirmation_rate_requested':div((y>0)&new&(after[:,2]>before[:,2]),needed),
        'unavailable_requests':int(np.sum(needed&~available)),
        'unnecessary_requests_nonpositive_observed_net_utility':int(np.sum(needed&(net<=0))),
        'policy_inputs_do_not_include_reveal_values_or_outcomes':True}
    return result,{'before':before,'after':after,'requested':needed,'available':available,'choice':choice,
        'corrected':corrected,'harmed':harmed,'observed_brier_gain':gain,'net_utility':net,'recommendation':rec}


def paired_recovery_ci(data,reference,selected,rt,st,n=1000):
    mask=data['recovery']&(data['labels'].sum(1)==0);ids=data['patient_ids']
    unique,groups=np.unique(ids,return_inverse=True);counts=np.bincount(groups[mask],minlength=len(unique))
    old=np.bincount(groups[mask],weights=(reference[mask,2]>=rt),minlength=len(unique))
    new=np.bincount(groups[mask],weights=(selected[mask,2]>=st),minlength=len(unique))
    rng=np.random.default_rng(2027);samples=[]
    for _ in range(n):
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));den=w@counts
        if den:samples.append(float(w@(new-old)/den))
    return {'difference':float((new.sum()-old.sum())/counts.sum()),'interval95':np.quantile(samples,[.025,.975]).tolist(),
        'requested':n,'valid_resamples':len(samples),'estimand':'Recovery-negative landmark alert-fraction difference, resampling whole patients'}


def run(config=None):
    c=config or load_config();out=root(c);verify(c)
    if (out/'replication_evaluation_complete.json').exists():return json.loads((out/'replication_results.json').read_text())
    transaction=begin(c);start=time.monotonic();bundle=joblib.load(out/'selected_pipeline.joblib');utility=joblib.load(out/'sensing_utility.joblib')
    data=prepare(c,bundle);pred_path=out/'replication_predictions.npz'
    if pred_path.exists():
        with np.load(pred_path,allow_pickle=False) as z:risk=z['adjusted_risk'];reference=z['raw_risk'];features=z['features']
    else:
        risk=np.empty((len(data['labels']),3));reference=risk.copy();features=np.empty((len(risk),21),np.float32)
        for a in range(0,len(risk),8192):
            sl=slice(a,a+8192);chunk={k:v[sl] for k,v in data.items()}
            risk[sl],features[sl],reference[sl]=pipeline_predict(chunk,bundle)
        save_arrays(pred_path,adjusted_risk=risk,raw_risk=reference,features=features)
    ref=summary(data,reference,bundle['reference_thresholds'],np.ones(len(risk),bool));selected=summary(data,risk,bundle['thresholds'],np.ones(len(risk),bool))
    boot_start=time.monotonic();intervals={}
    for name,p,t in [('reference',reference,bundle['reference_thresholds']),('selected',risk,bundle['thresholds'])]:
        checkpoint=out/('replication_bootstrap_'+name+'.json')
        if checkpoint.exists():intervals[name]=json.loads(checkpoint.read_text())['intervals']
        else:
            intervals[name]=patient_bootstrap(data['labels'],p,data['patient_ids'],t,n=1000,seed=2027)
            save_json(checkpoint,{'intervals':intervals[name],'prediction_artifact_sha256':file_hash(pred_path),'frozen_event_id':transaction['event_id']})
        print({'phase65_replication_bootstrap_complete':name},flush=True)
    paired=paired_ap(data['labels'].sum(1),risk[:,2],reference[:,2],data['patient_ids'],n=1000,seed=2027)
    recovery=paired_recovery_ci(data,reference,risk,bundle['reference_thresholds']['72']['balanced'],bundle['thresholds']['72']['balanced'])
    timed(c,'replication_headline_bootstraps',boot_start,{'resamples':1000,'models':2})
    with np.load(out/'replication_sensing_cases.npz',allow_pickle=False) as z:cases={k:z[k] for k in z.files}
    sense,sense_arrays=sensing_metrics(cases,utility,bundle,c)
    rec=recommend(data,features,risk,utility,bundle['thresholds']['72']['balanced'],c)
    sense.update(policy_requests_all_landmarks=int(rec['measurement_needed'].sum()),
        requests_per_eligible_patient_day=float(np.mean(rec['measurement_needed'])),
        policy_requested_measurement_distribution={m:int(np.sum(rec['measurement_needed']&(rec['choice']==j))) for j,m in enumerate(CANDIDATES)})
    save_arrays(out/'replication_sensing_updates.npz',**{k:v for k,v in sense_arrays.items() if k!='recommendation'})
    result={'event_id':transaction['event_id'],'timestamp_utc':transaction['timestamp_utc'],'logical_replication_evaluations':1,
        'patients':2500,'landmarks':127500,'reference':ref,'selected':selected,'intervals':intervals,
        'paired_auprc72':paired,'paired_recovery_false_alert_fraction':recovery,'adaptive_sensing':sense,
        'frozen_manifest_sha256':file_hash(out/'PHASE6_5_FROZEN_MANIFEST.json'),'original_phase6_test_used':False}
    save_json(out/'replication_results.json',result)
    from src.reliability.report import generate as report
    report(c,data,risk,reference,features,cases,sense_arrays,rec)
    save_json(out/'replication_evaluation_complete.json',{'status':'COMPLETE','event_id':transaction['event_id'],
        'logical_replication_evaluations':1,'results_sha256':file_hash(out/'replication_results.json'),
        'frozen_manifest_sha256':file_hash(out/'PHASE6_5_FROZEN_MANIFEST.json')})
    timed(c,'replication_evaluation_total',start,{'patients':2500,'landmarks':127500,'logical_evaluations':1})
    print({'phase65_replication_evaluation_complete':True,'patients':2500,'landmarks':127500},flush=True)
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
