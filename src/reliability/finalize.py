"""One full development run of the subset-selected pipeline, never TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.io import file_hash
from src.twin.models import cumulative
from src.reliability.access import (load_config,root,reference_artifact,save_model,save_arrays,
    save_json,timed,development_cache)
from src.reliability.models import OffsetReliability,HorizonCalibration,temporal_recovery_rule
from src.reliability.development import prepare_development,summary,thresholds_at_reference_sensitivity
from src.reliability.sensing import development_cases,SensingUtility,recommend
from src.reliability.features import recovery_features


def bundle_for(config,selection,data,scope):
    fast=data['fast_subset'] if scope=='fast' else np.ones(len(data['labels']),bool)
    train=fast&(data['roles']=='train');cal=fast&(data['roles']=='calibration');op=fast&(data['roles']=='operating')
    h=data['hazards'];features=data['features'];winner=selection['winner'];model=None
    if winner=='offset_logistic':
        model=OffsetReliability(config['reliability_l2']).fit(h[train],features[train],data['labels'][train],['train'])
        adjusted=model.predict(h,features)
    elif winner=='temporal_recovery_rule':adjusted=temporal_recovery_rule(h,features,.65)
    else:adjusted=h
    calibrator=HorizonCalibration().fit(adjusted[cal],data['labels'][cal],['calibration'],selection['calibration_methods'])
    risk=cumulative(calibrator.transform(adjusted));reference=cumulative(h)
    original_thresholds=json.loads(reference_artifact('thresholds.json').read_text())['thresholds']
    thresholds=thresholds_at_reference_sensitivity(risk,reference,data['labels'],op,original_thresholds)
    identity=hashlib.sha256((file_hash(root(config)/'fast_selection_final.json')+config['_sha256']+scope).encode()).hexdigest()
    bundle={'reference_model':joblib.load(reference_artifact('selected.joblib')),
        'reference_calibrator':joblib.load(reference_artifact('calibration.joblib')),
        'winner':winner,'reliability_model':model,'calibrator':calibrator,'thresholds':thresholds,
        'reference_thresholds':original_thresholds,'temporal_attenuation':.65,
        'identity_sha256':identity,'scope':scope,'calibration_fit_patients':sorted(set(data['patient_ids'][cal].tolist())),
        'threshold_fit_patients':sorted(set(data['patient_ids'][op].tolist()))}
    return bundle,risk,reference


def full_development(config=None):
    c=config or load_config();out=root(c)
    if (out/'PHASE6_5_FROZEN_MANIFEST.json').exists():raise PermissionError('Full development closed')
    if (out/'full_development_complete.json').exists():
        return json.loads((out/'full_development_complete.json').read_text())
    selection=json.loads((out/'fast_selection_final.json').read_text())
    if selection['config_sha256']!=c['_sha256']:raise ValueError('Selection config inconsistent')
    data,_=prepare_development(c)
    # Exact sensing utility can be explored on the fast subset without altering
    # the frozen reference/reliability/calibration choices selected above.
    fast_bundle,_,_=bundle_for(c,selection,data,'fast')
    cases=development_cases(c,fast_bundle,data,c['fast_sensing_train_patients'],'fast')
    fast_utility=SensingUtility().fit(cases,c)
    save_model(out/'fast_sensing_utility.joblib',fast_utility)
    start=time.monotonic()
    marker=out/'full_development_started.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['config_sha256']!=c['_sha256']:raise ValueError('Cannot resume a changed development pipeline')
    else:
        save_json(marker,{'config_sha256':c['_sha256'],'selection_sha256':file_hash(out/'fast_selection_final.json'),
            'logical_full_development_runs':1,'selection_fixed':True})
    bundle_path=out/'selected_pipeline.joblib'
    if bundle_path.exists():
        bundle=joblib.load(bundle_path)
        with np.load(out/'full_development_predictions.npz',allow_pickle=False) as z:
            risk=z['adjusted_risk'];reference=z['raw_risk']
    else:
        bundle,risk,reference=bundle_for(c,selection,data,'full')
        save_model(bundle_path,bundle)
        save_arrays(out/'full_development_predictions.npz',adjusted_risk=risk,raw_risk=reference)
    cases=development_cases(c,bundle,data,c['full_sensing_train_patients'],'full')
    utility=SensingUtility().fit(cases,c)
    save_model(out/'sensing_utility.joblib',utility)
    recommendations=[]
    for role in ['train','validation']:
        cached,_=development_cache(role)
        m=(data['roles']=='train') if role=='train' else (data['roles']!='train')
        rec=recommend(cached,data['features'][m],risk[m],utility,bundle['thresholds']['72']['balanced'],c)
        recommendations.append(rec)
    val=data['roles']!='train';op=data['roles']=='operating';sel=data['roles']=='selection';cal=data['roles']=='calibration'
    result={'config_sha256':c['_sha256'],'winner':selection['winner'],'calibration_methods':selection['calibration_methods'],
        'patients':8500,'train_patients':7000,'validation_patients':1500,'rows':len(data['labels']),
        'logical_full_development_runs':1,'reference':summary(data,reference,bundle['reference_thresholds'],val),
        'selected':summary(data,risk,bundle['thresholds'],val),
        'independent_selection_role_reference':summary(data,reference,bundle['reference_thresholds'],sel),
        'independent_selection_role_selected':summary(data,risk,bundle['thresholds'],sel),
        'validation_all_is_development_description':True,'validation_all_not_independent_holdout':True,
        'calibration_patients':len(np.unique(data['patient_ids'][cal])),
        'operating_patients':len(np.unique(data['patient_ids'][op])),
        'requests':sum(int(r['measurement_needed'].sum()) for r in recommendations),
        'requests_per_eligible_patient_day':sum(int(r['measurement_needed'].sum()) for r in recommendations)/len(data['labels']),
        'sensing_fit_patients':c['full_sensing_train_patients'],'sensing_hide_reveal_landmarks':len(cases['labels']),
        'sensing_algorithm':'TRAIN-only Ridge predicts observed hide/reveal Brier/decision gain minus configurable relative burden; request one positive-utility candidate on ambiguous/stale states with 72h cooldown',
        'source_scope':'All original TRAIN/validation caches; no original TEST clinical access',
        'artifact_sha256':{'selected_pipeline.joblib':file_hash(bundle_path),'sensing_utility.joblib':file_hash(out/'sensing_utility.joblib')}}
    save_json(out/'full_development_complete.json',result)
    timed(c,'full_development_selected_pipeline_once',start,{'patients':8500,'rows':len(data['labels']),'logical_runs':1})
    print({'phase65_full_development_complete':True,'winner':selection['winner'],'patients':8500,'requests':result['requests']},flush=True)
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):full_development()
