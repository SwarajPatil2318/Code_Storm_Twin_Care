"""Seal development choices then spend the untouched cohort in one exclusive evaluation event."""
import importlib.metadata,json,time,uuid
from pathlib import Path
import numpy as np
from .protocol import ROOT,CONFIG,SOURCE,config,digest,write_json,code_hash,assert_stage,verify_frozen,utc
from .data import build_cache,subset
from .models import ForecastModel,baseline,train_mask
from .risk import RiskBundle
from .final_training import predict_primary
from .metrics import forecast_metrics,confidence,paired_gain,subgroups,event_metrics,event_confidence
from .product import make_state


def seal():
    manifest=assert_stage('development')
    training=json.loads((ROOT/'final_training.json').read_text())
    selection=json.loads((ROOT/'development_selection.json').read_text())
    risk=json.loads((ROOT/'risk_selection.json').read_text())
    tests=json.loads((ROOT/'test_results.json').read_text())
    if not tests['passed'] or tests['D2_test_source_sha256']!=digest('tests/test_metabolic_d2.py'):raise RuntimeError('Current D2 tests must pass before freezing')
    assert training['selection_sha256']==digest(ROOT/'development_selection.json')
    assert training['context_sha256']==digest(ROOT/'context_experiments.json')
    paths=[CONFIG,SOURCE,ROOT/'participant_manifest.json',ROOT/'development_selection.json',ROOT/'context_experiments.json',
        ROOT/'final_training.json',ROOT/'risk_selection.json',ROOT/'spike_target_development.json',ROOT/'cgmacros_provenance.json',
        ROOT/'test_results.json',Path('tests/test_metabolic_d2.py'),Path('scripts/test_metabolic_d2.py'),
        Path('docs/METABOLIC_D2_PROTOCOL.md'),
        ROOT/'cv/oof_predictions.npz',ROOT/'context_cv/oof_predictions.npz',ROOT/'calibration_predictions.npz']
    for stage in ['development','calibration']:paths.extend([ROOT/'cache'/f'{stage}.npz',ROOT/'cache'/f'{stage}.json'])
    for name,record in training['models'].items():
        path=ROOT/'models'/(name+'.pkl')
        if record['artifact_sha256']!=digest(path):raise RuntimeError('Final model hash mismatch')
        model=ForecastModel.load(path)
        if not set(model.training_participants).issubset(manifest['model_development']):raise RuntimeError('Forecast training isolation violation')
        paths.extend([path,path.with_suffix('.json')])
    if risk['artifact_sha256']!=digest(ROOT/'models/risk.pkl'):raise RuntimeError('Risk artifact mismatch')
    bundle=RiskBundle.load(ROOT/'models/risk.pkl')
    assert set(bundle.training_participants).issubset(manifest['model_development'])
    assert set(bundle.calibration_participants)==set(manifest['calibration'])
    paths.append(ROOT/'models/risk.pkl')
    frozen={'created_at_utc':utc(),'code_sha256':code_hash(),'artifact_hashes':{str(p):digest(p) for p in paths},
        'protocol':config(),'forecast_selection':{'population':selection['population'],'warm_model':selection['warm_model'],
            'personalization_enabled':selection['personalization_enabled']},'spike_definition':'incident_two_readings_gt180',
        'risk_method':risk['selected_method'],'spike_threshold':risk['threshold'],
        'forecast_training_participants':64,'calibration_participants':16,'untouched_holdout_participants':20,
        'holdout_evaluation_budget':1,'comparators':list(training['models']),
        'candidate_roles':'Primary selection fixed; all predeclared comparators evaluated together without post-holdout selection.',
        'meal_episode_rule':'Source meal within 7.5 minutes after a complete forecast landmark; alignment is evaluation-only.',
        'example_rule':'Median absolute peak error among meal-aligned eligible examples; fallback all landmarks. Under/over require matching peak-error direction. Stable requires current ≤180, all observed/predicted horizons within 30 mg/dL of current, and native peak rise <30.',
        'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','xgboost-cpu','torch','scipy']}}
    write_json(ROOT/'FROZEN_MODEL_MANIFEST.json',frozen,exclusive=True);verify_frozen()
    print(json.dumps({'sealed':True,'holdout_opened':False,'code_sha256':frozen['code_sha256']}),flush=True)
    return frozen


def peak_metrics(data,p):
    predicted=p.max(1);error=predicted-data['true_peak'];subjects=np.unique(data['pid'])
    category=np.array([30,60,90,120])[np.abs(data['true_peak_minutes'][:,None]-np.array([30,60,90,120])).argmin(1)]
    predicted_category=np.array([30,60,90,120])[p.argmax(1)]
    per=np.array([[np.abs(error[data['pid']==s]).mean(),np.sqrt((error[data['pid']==s]**2).mean()),error[data['pid']==s].mean(),
        (predicted_category[data['pid']==s]==category[data['pid']==s]).mean(),
        np.abs(predicted_category[data['pid']==s]-data['true_peak_minutes'][data['pid']==s]).mean()] for s in subjects])
    draw=np.random.default_rng(2026).integers(0,len(subjects),(1000,len(subjects)))
    return {'participants':len(subjects),'windows':len(p),'subject_macro':dict(zip(['peak_MAE','peak_RMSE','peak_bias','nearest_category_agreement','timing_MAE_minutes_descriptive'],per.mean(0).tolist())),
        'ci95':dict(zip(['peak_MAE','peak_RMSE','peak_bias','nearest_category_agreement','timing_MAE_minutes_descriptive'],np.quantile(per[draw].mean(1),[.025,.975],axis=0).T.tolist())),
        'timing_scope':'Four nominal categories only; true peaks use native observations in (T,T+120]. Agreement bins true timing to nearest category, earlier on ties.'}


def collect_examples(data,p,probability,threshold,model_hashes):
    actual=data['spike'];predicted=probability>=threshold;error=np.abs(p.max(1)-data['true_peak'])
    current=data['sequence'][:,-1]
    stable=(current<=180)&(np.abs(data['y']-current[:,None]).max(1)<30)&(np.abs(p-current[:,None]).max(1)<30)&(data['true_peak']-current<30)
    masks={'correct_spike':actual&predicted,'correct_stable':~actual&~predicted&stable,
        'underpredicted_spike':actual&~predicted&(p.max(1)<data['true_peak']),
        'overpredicted_spike':~actual&predicted&(p.max(1)>data['true_peak'])}
    records={};arrays={};states=[]
    for name,mask in masks.items():
        use=mask&data['meal_episode'];meal_aligned=bool(use.any())
        if not use.any():use=mask
        candidates=np.flatnonzero(use)
        if not len(candidates):records[name]={'available':False};continue
        ordered=candidates[np.argsort(error[candidates],kind='stable')];i=int(ordered[len(ordered)//2])
        record={'available':True,'participant_id':str(data['pid'][i]),'landmark_index':i,'meal_aligned':meal_aligned,
            'true_spike':bool(actual[i]),'alert_probability':float(probability[i]),'absolute_peak_error':float(error[i]),
            'actual_peak_glucose':float(data['true_peak'][i]),'actual_peak_minutes':float(data['true_peak_minutes'][i]),
            'future_observed_horizons':data['y'][i].tolist(),'prediction':p[i].tolist()}
        records[name]=record
        arrays[name+'_history']=data['sequence'][i];arrays[name+'_observed']=data['y'][i];arrays[name+'_prediction']=p[i]
        state=make_state(data,i,p[i],probability[i],threshold,model_hashes['warm' if data['warm'][i] else 'cold']);states.append({'example':name,'state':state})
    write_json(ROOT/'representative_examples_private.json',records);write_json(ROOT/'product_states_private.json',states)
    path=ROOT/'representative_arrays_private.npz';np.savez_compressed(path,**arrays);path.chmod(0o600)
    return {k:{'available':v['available'],'meal_aligned':v.get('meal_aligned'),'absolute_peak_error':v.get('absolute_peak_error')} for k,v in records.items()}


def evaluate_once():
    frozen=verify_frozen()
    # This must fail before any target/data read if the evaluation has already been attempted.
    event={'event_id':str(uuid.uuid4()),'started_at_utc':utc(),'model_manifest_sha256':digest(ROOT/'FROZEN_MODEL_MANIFEST.json'),
        'code_sha256':frozen['code_sha256'],'status':'started','budget':1}
    write_json(ROOT/'HOLDOUT_EVALUATION_STARTED.json',event,exclusive=True)
    start=time.time();data=build_cache('holdout')
    selection=json.loads((ROOT/'development_selection.json').read_text())
    models={name:ForecastModel.load(ROOT/'models'/(name+'.pkl')) for name in frozen['comparators']}
    predictions={name:baseline(data,name) for name in ['persistence','trend']}
    for name,model in models.items():
        mask=train_mask(data,'meal' if name.startswith('meal_') else 'static' if name.startswith('static_') else model.variant)
        p=np.full_like(data['y'],np.nan);p[mask]=model.predict(subset(data,mask));predictions[name]=p
    primary=predict_primary(models,selection,data);predictions['primary']=primary
    risk=RiskBundle.load(ROOT/'models/risk.pkl');probabilities=risk.probabilities(data,primary)
    path=ROOT/'holdout_predictions.npz';np.savez_compressed(path,**predictions,**{'risk_'+k:v for k,v in probabilities.items()});path.chmod(0o600)
    # All predictions are persisted before computing metrics. No subsequent model fitting is permitted.
    write_json(ROOT/'HOLDOUT_PREDICTIONS_RECORDED.json',{'event_id':event['event_id'],'prediction_sha256':digest(path),'at_utc':utc()})
    scores={}
    for name,p in predictions.items():
        use=np.isfinite(p).all(1)
        scores[name]={'all':forecast_metrics(data['y'][use],p[use],data['pid'][use]),
            'warm':forecast_metrics(data['y'][use&data['warm']],p[use&data['warm']],data['pid'][use&data['warm']]),
            'insufficient_prefix':forecast_metrics(data['y'][use&~data['warm']],p[use&~data['warm']],data['pid'][use&~data['warm']])}
    results={'evaluation_event':event,'split_counts':{'development':80,'model_training':64,'calibration':16,'holdout':len(np.unique(data['pid']))},
        'windows':{'holdout':len(data['y']),'warm':int(data['warm'].sum()),'insufficient_prefix':int((~data['warm']).sum()),
            'meal_context':int(data['meal_eligible'].sum()),'meal_episodes':int(data['meal_episode'].sum())},
        'forecasts':scores,'primary_ci95':confidence(data['y'],primary,data['pid']),
        'primary_vs_persistence':paired_gain(data['y'],predictions['persistence'],primary,data['pid']),
        'primary_vs_trend':paired_gain(data['y'],predictions['trend'],primary,data['pid']),
        'cold_window_only':{'metrics':scores[selection['population']]['all'],
            'ci95':confidence(data['y'],predictions[selection['population']],data['pid']),
            'definition':'Population model uses only the current 120-minute CGM window; evaluated on all held-out landmarks irrespective of prefix availability.'},
        'warm_personalized':{'metrics':scores['primary']['warm'],
            'ci95':confidence(data['y'][data['warm']],primary[data['warm']],data['pid'][data['warm']])},
        'subgroups':subgroups(data,primary),'peak':peak_metrics(data,primary)}
    use=data['warm']
    results['personalization_gain']=paired_gain(data['y'][use],predictions[selection['population']][use],primary[use],data['pid'][use])
    results['matched_warm_training_personalization_gain']={}
    for family in ['boosting','gru']:
        results['matched_warm_training_personalization_gain'][family]=paired_gain(data['y'][use],
            predictions[family+'_warm_population'][use],predictions[family+'_personal'][use],data['pid'][use])
    results['context_gains']={}
    for variant in ['meal','static']:
        mask=train_mask(data,variant)
        results['context_gains'][variant]=paired_gain(data['y'][mask],predictions[variant+'_matched_cgm'][mask],predictions[variant+'_context'][mask],data['pid'][mask])
    results['spike_candidates']={name:event_metrics(data['spike'],p,risk.threshold,data['pid']) for name,p in probabilities.items()}
    selected_probability=probabilities[risk.method]
    results['spike_ci95']=event_confidence(data['spike'],selected_probability,data['pid'],risk.threshold)
    episode=data['meal_episode']
    results['meal_response']={'forecast':forecast_metrics(data['y'][episode],primary[episode],data['pid'][episode]),
        'peak':peak_metrics(subset(data,episode),primary[episode]),
        'spike':event_metrics(data['spike'][episode],selected_probability[episode],risk.threshold,data['pid'][episode])}
    results['representative_examples']=collect_examples(data,primary,selected_probability,risk.threshold,
        {'cold':digest(ROOT/'models'/(selection['population']+'.pkl')),'warm':digest(ROOT/'models'/(selection['warm_model']+'.pkl'))})
    results['seconds']=time.time()-start;results['completed_at_utc']=utc();results['prediction_sha256']=digest(path)
    results['holdout_cache_sha256']=digest(ROOT/'cache/holdout.npz');results['model_manifest_sha256']=digest(ROOT/'FROZEN_MODEL_MANIFEST.json')
    write_json(ROOT/'holdout_results.json',results,exclusive=True)
    write_json(ROOT/'HOLDOUT_EVALUATION_COMPLETED.json',{'event_id':event['event_id'],'completed_at_utc':results['completed_at_utc'],
        'result_sha256':digest(ROOT/'holdout_results.json'),'prediction_sha256':digest(path),'evaluation_count':1},exclusive=True)
    print(json.dumps({'holdout_evaluations':1,'participants':results['split_counts']['holdout'],
        'primary_MAE':scores['primary']['all']['subject_macro']['MAE'],'primary_RMSE':scores['primary']['all']['subject_macro']['RMSE'],
        'risk_method':risk.method,'AUPRC':results['spike_candidates'][risk.method]['AUPRC']}),flush=True)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['seal','evaluate']);args=parser.parse_args()
    seal() if args.action=='seal' else evaluate_once()
