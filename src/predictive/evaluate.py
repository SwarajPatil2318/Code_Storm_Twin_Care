"""One exclusive final-test transaction; comparator choices are frozen beforehand."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.data.io import file_hash
from src.predictive.data import Cohort,construct
from src.predictive.metrics import basic,operating,bootstrap,calibration_diagnostics,paired_bootstrap
from src.predictive.inference import prediction_states
from src.predictive.artifacts import check_seal,write_json,write_rows,artifact_hashes,verify_hashes,verify_previous


def counts(rows):
    return {'patients':len({r['patient_id'] for r in rows}),'positive_patients':len({r['patient_id'] for r in rows if r['label']}),
            'stays':len({r['icu_stay_id'] for r in rows}),'windows':len(rows),'positive_windows':sum(r['label'] for r in rows)}


def summarize(rows,p,rules,n=2000):
    return {'counts':counts(rows),'metrics':basic([r['label'] for r in rows],p),
            'calibration':calibration_diagnostics([r['label'] for r in rows],p),
            'operating_points':{k:operating(rows,p,t) for k,t in rules.items()},
            'patient_bootstrap95':bootstrap(rows,p,rules['balanced'],n=n),
            'sparse_support':len({r['patient_id'] for r in rows if r['label']})<10}


def assess(bundle,rows,n=2000):
    result={};probabilities={};y=[r['label'] for r in rows]
    for name,model in bundle['models'].items():
        raw=model.predict(rows);p=bundle['calibrators'][name].predict(raw);probabilities[name]={'raw':raw,'calibrated':p}
        result[name]=summarize(rows,p,bundle['thresholds'][name],n)
        result[name]['raw_metrics']=basic(y,raw)
        print({'evaluated_frozen_comparator':name},flush=True)
    selected=bundle['selected'];subsets={}
    for name in ['below_threshold','near_threshold','moderate','clearly_normal','hard','hf']:
        mask=np.array([r['audit'][name] if name in ['hard','hf'] else r['audit']['map_stratum']==name for r in rows],dtype=bool)
        subset=[r for r,yes in zip(rows,mask) if yes]
        subsets[name]={model:summarize(subset,probabilities[model]['calibrated'][mask],bundle['thresholds'][model],n) for model in result}
        print({'evaluated_predefined_stratum':name,'patients':len({r['patient_id'] for r in subset})},flush=True)
    winners=bundle['winners']
    paired={name:paired_bootstrap(rows,probabilities[winners[left]]['calibrated'],probabilities[winners[right]]['calibrated'],n=n)
            for name,left,right in [('fusion_minus_phys','fused','phys'),('fusion_minus_ehr','fused','ehr'),('personal_minus_fused','personalized','fused')]}
    return {'selected':selected,'models':result,'subsets':subsets,'paired_comparisons':paired,
            'bootstrap_replicates':n,'false_alert_exposure':'3 eligible observed hours per nonoverlapping retained landmark; conditional on complete follow-up, not total ICU time or deployed notifications',
            'test_selection_feedback':False},probabilities


def trajectories(cohort,bundle,rows):
    # Only event-bearing test patients; no model selection or threshold changes.
    anchors={}
    for row in rows:
        if row['label']:anchors.setdefault((row['icu_stay_id'],row['event_onset']),row)
    probes=[]
    for (stay,onset),anchor in anchors.items():
        patient=cohort.patients[stay];series=cohort.series[patient['source_stay_key']]
        for lead in bundle['protocol']['risk_trajectories_minutes_before_onset']:
            t0=onset-lead
            if t0<365:continue
            x,seq,audit=construct(patient,series,t0)
            if audit['buffered_latest_map'] is None:continue
            audit['diagnostic_outside_target_landmark']=lead<=15
            probes.append({**anchor,'t0':t0,'features':x,'sequence':seq,'audit':audit,'trajectory_lead_minutes':lead})
    if not probes:return []
    states=prediction_states(bundle,probes)
    return [{'patient_id':r['patient_id'],'icu_stay_id':r['icu_stay_id'],'event_onset':r['event_onset'],
             'minutes_before_onset':r['trajectory_lead_minutes'],'prediction_state':state,
             'physiology':{field:r['features'][f'phys__{field}__30__last'] for field in ['heartrate','systemicmean','systemicsystolic','systemicdiastolic','sao2','respiration','temperature']},
             'map_reference':{k:r['features']['personal__systemicmean__'+k] for k in ['mean','median','sd','available']},
             'diagnostic_only':True,'forecast_horizon_always_minutes':180} for r,state in zip(probes,states)]


TEST_REGISTRY=Path('data/results/phase4_test_access/evaluation_started.json')

def begin_transaction(root,registry=None):
    root=Path(root);seal=check_seal(root);completed=root/'evaluation_seal.json'
    if completed.exists():
        done=json.loads(completed.read_text());verify_hashes(root,done['hashes'])
        return False
    marker=root/'evaluation_started.json'
    if marker.exists():raise RuntimeError('Test transaction already started; incomplete runs cannot silently re-evaluate')
    registry=TEST_REGISTRY if registry is None else Path(registry)
    registry.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    try:
        global_fd=os.open(registry,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise RuntimeError('This Phase4 study already opened its test set; another experiment cannot re-evaluate it') from exc
    with os.fdopen(global_fd,'w') as handle:
        json.dump({'experiment_root':str(root.resolve()),'selection_seal_sha256':file_hash(root/'selection_seal.json'),'evaluation_id':'phase4_frozen_comparators_once'},handle)
        handle.flush();os.fsync(handle.fileno())
    try:
        fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise RuntimeError('Test transaction already started; incomplete runs cannot silently re-evaluate') from exc
    with os.fdopen(fd,'w') as handle:
        json.dump({'evaluation_id':'phase4_frozen_comparators_once','started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                   'selection_seal_sha256':file_hash(root/'selection_seal.json'),'comparators':seal['comparators'],'selected':seal['selected']},handle)
        handle.flush();os.fsync(handle.fileno())
    return True


def evaluate(root='data/results/phase4'):
    root=Path(root)
    if not begin_transaction(root):return {'status':'ALREADY_EVALUATED','results':str(root/'test_metrics.json')}
    bundle=joblib.load(root/'pipeline.joblib');cohort=Cohort()
    with threadpool_limits(limits=1):
        rows=cohort.batch('test',root/'selection_seal.json')
        results,p=assess(bundle,rows);states=prediction_states(bundle,rows);traces=trajectories(cohort,bundle,rows)
        write_json(root/'test_metrics.json',results);write_rows(root/'prediction_states.jsonl',states)
        write_rows(root/'test_predictions.jsonl',[{'window_id':r['window_id'],'patient_id':r['patient_id'],'icu_stay_id':r['icu_stay_id'],'t0':r['t0'],
                   'label':r['label'],'event_onset':r['event_onset'],'audit':r['audit'],
                   'predictions':{name:{kind:float(values[i]) for kind,values in ps.items()} for name,ps in p.items()}} for i,r in enumerate(rows)])
        write_rows(root/'event_trajectories.jsonl',traces)
        trajectory_summary={}
        for lead in bundle['protocol']['risk_trajectories_minutes_before_onset']:
            eligible=[r for r in traces if r['minutes_before_onset']==lead and r['prediction_state']['operating_state']!='outside_training_eligibility']
            all_rows=[r for r in traces if r['minutes_before_onset']==lead]
            trajectory_summary[str(lead)]={'available_event_probes':len(all_rows),'eligible_event_probes':len(eligible),
                    'median_risk':float(np.median([r['prediction_state']['calibrated_risk'] for r in eligible])) if eligible else None,
                    'median_map':float(np.median([r['physiology']['systemicmean'] for r in eligible])) if eligible else None}
        write_json(root/'trajectory_summary.json',trajectory_summary)
    verify_previous(json.loads((root/'preservation.json').read_text()))
    write_json(root/'evaluation_seal.json',{'status':'EVALUATED_ONCE','selected':bundle['selected'],
               'selection_seal_sha256':file_hash(root/'selection_seal.json'),
               'hashes':artifact_hashes(root,exclude=['evaluation_seal.json']),
               'test_evaluation_count':1,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    return {'status':'EVALUATED_ONCE','selected':bundle['selected'],'results':str(root/'test_metrics.json')}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/results/phase4');args=parser.parse_args();print(evaluate(args.root))
