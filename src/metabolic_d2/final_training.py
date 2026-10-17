"""Fit predeclared comparators on 64 people; calibrate on 16 different development people."""
import json,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from .protocol import ROOT,assert_stage,config,digest,write_json
from .data import build_cache,subset
from .models import ForecastModel,train_mask,subject_weights
from .development import CANDIDATES,training_hash
from .risk import RiskBundle
from .metrics import event_metrics


def predict_primary(models,selection,data):
    cold=models[selection['population']].predict(data)
    if selection['personalization_enabled']:
        use=data['warm'];cold[use]=models[selection['warm_model']].predict(subset(data,use))
    return cold


def run():
    manifest=assert_stage('development');data=build_cache('development');allowed=set(manifest['model_development'])
    selection=json.loads((ROOT/'development_selection.json').read_text())
    if selection['training_sha256']!=training_hash():raise RuntimeError('Stale selection')
    context=json.loads((ROOT/'context_experiments.json').read_text())
    family=selection['family'];directory=ROOT/'models';directory.mkdir(mode=0o700,exist_ok=True)
    models={};records={}
    plan=[(f,v,f+'_'+v,None) for f,v in CANDIDATES]
    for variant in ['meal','static']:
        plan.extend([(family,'population',variant+'_matched_cgm',variant),(family,variant,variant+'_context',variant)])
    for family,variant,name,restricted in plan:
        path=directory/(name+'.pkl');meta=path.with_suffix('.json')
        if path.exists():
            record=json.loads(meta.read_text())
            if record['training_sha256']!=training_hash() or record['artifact_sha256']!=digest(path):raise RuntimeError('Stale final model checkpoint')
            models[name]=ForecastModel.load(path);records[name]=record;continue
        start=time.time();training=subset(data,train_mask(data,restricted)) if restricted else data
        model=ForecastModel(family,variant).fit(training,allowed,config()['seed']);model.save(path)
        records[name]={'artifact_sha256':digest(path),'training_sha256':training_hash(),'seconds':time.time()-start,
            'training_participants':len(model.training_participants),'training_windows':model.train_windows,
            'variant':variant,'family':family,'path':str(path)}
        write_json(meta,records[name]);models[name]=model
        print(json.dumps({'final_fit':name,'participants':len(model.training_participants),'seconds':round(records[name]['seconds'],2)}),flush=True)
    with np.load(ROOT/'cv/oof_predictions.npz',allow_pickle=False) as z:
        oof=z[selection['population']].copy()
        if selection['personalization_enabled']:oof[data['warm']]=z[selection['warm_model']][data['warm']]
    # Labels selected from the model-development cohort; rise definitions remain diagnostics.
    targets={}
    current=data['sequence'][:,-1]
    for key in ['spike','rise30','rise50']:
        y=data[key];trivial=np.zeros(len(y));fold=np.array([manifest['folds'][pid] for pid in data['pid']])
        for f in range(5):
            train=fold!=f;valid=~train;weights=subject_weights(data['pid'][train])
            mean=np.average(current[train],weights=weights)
            sd=max(float(np.sqrt(np.average((current[train]-mean)**2,weights=weights))),1)
            z=(current-mean)/sd;x=np.column_stack([z,z*z])
            classifier=LogisticRegression(C=1,max_iter=1000,random_state=2026).fit(x[train],y[train],sample_weight=weights)
            trivial[valid]=classifier.predict_proba(x[valid])[:,1]
        if key=='spike':trivial[current>180]=0
        targets[key]={'prevalence':float(y.mean()),'positive_landmarks':int(y.sum()),
            'positive_participants':sum(bool(y[data['pid']==s].any()) for s in np.unique(data['pid'])),
            'current_glucose_only':event_metrics(y,trivial,.5,data['pid']),
            'current_glucose_model':'Five participant-fold out-of-fold regularized logistic regression with current glucose and its square; train-only scaling.'}
    write_json(ROOT/'spike_target_development.json',{'candidates':targets,'frozen_target':'incident_two_readings_gt180',
        'rationale':'Incident definition excludes already-high glucose and requires adjacent native readings; prevalence supports calibration.'})
    risk=RiskBundle().fit_auxiliary(data,oof,allowed)
    cal=build_cache('calibration');p=predict_primary(models,selection,cal)
    foldmap={pid:i%4 for i,pid in enumerate(manifest['calibration'])};folds=np.array([foldmap[pid] for pid in cal['pid']])
    result,crossfit=risk.fit_calibration(cal,p,folds);risk_path=directory/'risk.pkl';risk.save(risk_path)
    result['artifact_sha256']=digest(risk_path);result['path']=str(risk_path)
    write_json(ROOT/'risk_selection.json',result)
    path=ROOT/'calibration_predictions.npz';np.savez_compressed(path,forecast=p,**crossfit);path.chmod(0o600)
    write_json(ROOT/'final_training.json',{'models':records,'risk':result,'selection_sha256':digest(ROOT/'development_selection.json'),
        'context_sha256':digest(ROOT/'context_experiments.json'),'calibration_prediction_sha256':digest(path),
        'holdout_inspected':False,'forecast_fit_participants':64,'calibration_participants':16})
    print(json.dumps({'calibration_selected':risk.method,'frozen_threshold':risk.threshold,'holdout_opened':False}),flush=True)

if __name__=='__main__':run()
