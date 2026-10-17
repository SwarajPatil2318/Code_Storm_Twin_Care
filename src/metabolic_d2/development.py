"""Five grouped development folds; checkpointed once per predeclared candidate."""
import json,time
from pathlib import Path
import numpy as np
from .protocol import ROOT,assert_stage,config,digest,write_json,code_hash
from .data import build_cache,subset
from .models import ForecastModel,baseline,train_mask
from .metrics import forecast_metrics,paired_gain,subgroups

CANDIDATES=[('mean','population'),('ridge','population'),('boosting','population'),('boosting','personal'),('gru','population'),('gru','personal'),('boosting','warm_population'),('gru','warm_population')]


def training_hash():
    from hashlib import sha256
    h=sha256()
    for path in ['src/metabolic_d2/models.py','src/metabolic_d2/data.py','configs/metabolic_d2.json']:
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def run_cv():
    m=assert_stage('development');data=build_cache('development');c=config();start=time.time()
    fold=np.array([m['folds'][pid] for pid in data['pid']]);out=ROOT/'cv';out.mkdir(exist_ok=True,mode=0o700)
    predictions={name:baseline(data,name) for name in ['persistence','trend']}
    timings={}
    for family,variant in CANDIDATES:
        name=family+'_'+variant;p=np.full_like(data['y'],np.nan);durations=[]
        for f in range(5):
            artifact=out/f'{name}_fold{f}.npz';meta=artifact.with_suffix('.json')
            use=fold==f;allowed=set(data['pid'][~use]);eligible=train_mask(data,variant)&use
            if artifact.exists():
                record=json.loads(meta.read_text())
                if record['training_sha256']!=training_hash() or record['artifact_sha256']!=digest(artifact):raise RuntimeError('Stale CV training checkpoint')
                with np.load(artifact,allow_pickle=False) as z:p[z['indices']]=z['predictions']
                durations.append(record['seconds']);continue
            begin=time.time();model=ForecastModel(family,variant).fit(subset(data,~use),allowed,c['seed']+f)
            pred=model.predict(subset(data,eligible));ids=np.flatnonzero(eligible);p[ids]=pred
            np.savez_compressed(artifact,indices=ids,predictions=pred);artifact.chmod(0o600)
            duration=time.time()-begin;durations.append(duration)
            write_json(meta,{'candidate':name,'fold':f,'training_sha256':training_hash(),'artifact_sha256':digest(artifact),
                'training_participants':len(allowed),'validation_participants':len(set(data['pid'][eligible])),
                'training_windows':model.train_windows,'validation_windows':len(ids),'seconds':duration})
            print(json.dumps({'CV_candidate':name,'fold':f,'seconds':round(duration,2),
                'validation_subject_macro_MAE':forecast_metrics(data['y'][eligible],pred,data['pid'][eligible])['subject_macro']['MAE']}),flush=True)
        predictions[name]=p;timings[name]=durations
    scores={}
    for name,p in predictions.items():
        good=np.isfinite(p).all(1);scores[name]={'all':forecast_metrics(data['y'][good],p[good],data['pid'][good]),
            'warm':forecast_metrics(data['y'][good&data['warm']],p[good&data['warm']],data['pid'][good&data['warm']]),
            'folds':[forecast_metrics(data['y'][good&(fold==f)],p[good&(fold==f)],data['pid'][good&(fold==f)]) for f in range(5)],
            'subgroups':subgroups(subset(data,good),p[good])}
    # Select population family by subject-macro equal-horizon MAE; personalization by matched warm MAE.
    populations=['persistence','trend','mean_population','ridge_population','boosting_population','gru_population']
    ranking=sorted(populations,key=lambda k:(np.mean(scores[k]['all']['subject_macro']['MAE']),scores[k]['all']['subject_macro']['MAE'][3],populations.index(k)))
    population=ranking[0];family=population.split('_')[0]
    personal=family+'_personal' if family in ['boosting','gru'] else None
    gains={}
    for fam in ['boosting','gru']:
        mask=data['warm']&np.isfinite(predictions[fam+'_personal']).all(1)
        gains[fam]=paired_gain(data['y'][mask],predictions[fam+'_population'][mask],predictions[fam+'_personal'][mask],data['pid'][mask])
        gains[fam+'_matched_warm_training']=paired_gain(data['y'][mask],predictions[fam+'_warm_population'][mask],predictions[fam+'_personal'][mask],data['pid'][mask])
    enable_personal=personal is not None and np.mean(scores[personal]['warm']['subject_macro']['MAE'])<np.mean(scores[population]['warm']['subject_macro']['MAE'])
    selection={'population':population,'family':family,'warm_model':personal if enable_personal else population,
        'personalization_enabled':bool(enable_personal),'ranking':ranking,'scores':scores,'personalization':gains,
        'timing_seconds':timings,'elapsed_seconds':time.time()-start,'training_sha256':training_hash(),
        'holdout_inspected':False,'calibration_used_for_forecast_selection':False}
    path=out/'oof_predictions.npz';np.savez_compressed(path,**predictions);path.chmod(0o600)
    selection['oof_sha256']=digest(path)
    write_json(ROOT/'development_selection.json',selection)
    print(json.dumps({'forecast_selection':population,'warm_model':selection['warm_model'],'holdout_opened':False}),flush=True)
    return selection,data,predictions

if __name__=='__main__':run_cv()
