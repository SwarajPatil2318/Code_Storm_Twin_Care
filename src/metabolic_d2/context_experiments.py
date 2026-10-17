"""Matched conditional meal and onboarding-demographic ablations on development only."""
import json,time
import numpy as np
from .protocol import ROOT,assert_stage,config,digest,write_json
from .data import build_cache,subset
from .models import ForecastModel,train_mask
from .development import training_hash
from .metrics import forecast_metrics,paired_gain


def run():
    manifest=assert_stage('development');data=build_cache('development')
    selection=json.loads((ROOT/'development_selection.json').read_text())
    family=selection['family']
    if family not in ['boosting','gru']:raise ValueError('Context experiments require selected trainable family')
    fold=np.array([manifest['folds'][s] for s in data['pid']])
    directory=ROOT/'context_cv';directory.mkdir(mode=0o700,exist_ok=True)
    summary={};predictions={}
    for variant in ['meal','static']:
        # BOTH comparators fit the same training rows, preventing sample-composition confounding.
        eligible=train_mask(data,variant);paired={}
        for context in [False,True]:
            name=variant+('_context' if context else '_matched_cgm')
            pred=np.full_like(data['y'],np.nan);times=[]
            for f in range(5):
                path=directory/f'{name}_fold{f}.npz';meta=path.with_suffix('.json')
                train=eligible&(fold!=f);valid=eligible&(fold==f);ids=np.flatnonzero(valid)
                allowed=set(data['pid'][train])
                if path.exists():
                    record=json.loads(meta.read_text())
                    if record['training_sha256']!=training_hash() or record['artifact_sha256']!=digest(path):raise RuntimeError('Stale context checkpoint')
                    with np.load(path,allow_pickle=False) as z:pred[z['indices']]=z['predictions']
                    times.append(record['seconds']);continue
                start=time.time()
                model=ForecastModel(family,variant if context else 'population').fit(subset(data,train),allowed,config()['seed']+f)
                p=model.predict(subset(data,valid));pred[ids]=p
                np.savez_compressed(path,indices=ids,predictions=p);path.chmod(0o600)
                seconds=time.time()-start;times.append(seconds)
                write_json(meta,{'training_sha256':training_hash(),'artifact_sha256':digest(path),'seconds':seconds,
                    'training_participants':len(allowed),'training_windows':int(train.sum()),'validation_windows':len(ids)})
            paired[name]=pred;predictions[name]=pred
            summary[name]={'metrics':forecast_metrics(data['y'][eligible],pred[eligible],data['pid'][eligible]),
                'folds':[forecast_metrics(data['y'][eligible&(fold==f)],pred[eligible&(fold==f)],data['pid'][eligible&(fold==f)]) for f in range(5)],
                'seconds':times}
        summary[variant+'_gain']=paired_gain(data['y'][eligible],paired[variant+'_matched_cgm'][eligible],paired[variant+'_context'][eligible],data['pid'][eligible])
        for warm in [False,True]:
            use=eligible&(data['warm']==warm)
            summary[variant+('_warm_gain' if warm else '_cold_prefix_gain')]=paired_gain(data['y'][use],paired[variant+'_matched_cgm'][use],paired[variant+'_context'][use],data['pid'][use])
        print(json.dumps({'experiment':variant,'matched_MAE_reduction':summary[variant+'_gain']['MAE_reduction']}),flush=True)
    path=directory/'oof_predictions.npz';np.savez_compressed(path,**predictions);path.chmod(0o600)
    summary['prediction_sha256']=digest(path);summary['holdout_inspected']=False
    summary['meal_receipt_policy']=config()['availability_policy'];summary['static_policy']=config()['static_policy']
    summary['primary_release_policy']='Conditional context ablations only; selected CGM/personal primary remains unchanged.'
    write_json(ROOT/'context_experiments.json',summary)
    return summary

if __name__=='__main__':run()
