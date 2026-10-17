"""Within-cohort causal next-observation forecasts with patient-grouped selection."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.features.temporal import time_number, TEMPORAL_FEATURES
from .common import *

GLOBAL_FEATURES=('current_value','slope_per_day_7d','prior_count_7d','prior_count_30d',
                 'time_since_last_usable_days','current_missing','current_invalid')
PERSONAL_FEATURES=GLOBAL_FEATURES+tuple(k for k in TEMPORAL_FEATURES if k not in GLOBAL_FEATURES)
CANDIDATES=('persistence','rolling_median','rolling_mean','global_ridge_1','global_ridge_10','global_ridge_100',
            'personal_ridge_1','personal_ridge_10','personal_ridge_100')


def windows(rows,dataset):
    """Labels shift strictly forward. No future time/gap is a model feature."""
    groups=defaultdict(list)
    for r in rows:
        if r['dataset'] != dataset: raise ValueError('Mixed dataset forecast inputs')
        if r['current_value'] is not None:
            groups[r['concept'],r['patient_id']].append(r)
    result=[]
    for (signal,patient),group in sorted(groups.items()):
        group.sort(key=lambda r:time_number(r['timestamp'],r['time_basis']))
        if len({r['timestamp'] for r in group})!=len(group):raise ValueError('Duplicate timestamp summaries')
        for a,b in zip(group,group[1:]):
            cutoff=time_number(a['timestamp'],a['time_basis']); target=time_number(b['timestamp'],b['time_basis']);gap=target-cutoff
            if gap<=0 or a['unit']!=b['unit']:raise ValueError('Invalid future target or incompatible unit')
            eligible=abs(gap-1)<1e-8 if dataset=='chiron' else gap<=3
            if not eligible:continue
            f={'current_value':a['current_value'],**a['features']}
            if a['provenance']['max_prior_observed_time'] is not None and a['provenance']['max_prior_observed_time']>=cutoff:
                raise ValueError('Future baseline information')
            result.append({'patient_id':patient,'dataset':dataset,'signal':signal,'cutoff':a['timestamp'],
                           'timestamp':b['timestamp'],'cutoff_days':cutoff,'target_days':target,'gap_days':gap,
                           'unit':a['unit'],'unit_support':a['unit_support'],'features':f,'observed_value':float(b['current_value']),
                           'warmup':'warm' if f['prior_count_30d']>=2 else 'cold',
                           'input_observation_ids':a['current_observation_ids']+a['provenance']['prior_observation_ids'],
                           'target_observation_ids':b['current_observation_ids'],
                           'quality_status':'SUSPECT' if a['unit_support'].startswith('provisional') else 'VALID'})
    return result


def array(rows,names):
    return np.array([[np.nan if r['features'].get(n) is None else float(r['features'][n]) for n in names] for r in rows])


class PhysiologyModel:
    def __init__(self,name):self.name=name
    def fit(self,rows,assignments):
        ids=sorted({r['patient_id'] for r in rows});require_roles(ids,assignments,{'train'})
        self.fit_patient_ids_=ids
        self.training_scale_=max(float(np.std([r['observed_value'] for r in rows],ddof=1)),1e-8)
        self.names_=PERSONAL_FEATURES if self.name.startswith('personal') else GLOBAL_FEATURES
        self.pipeline_=None
        if 'ridge' in self.name:
            self.pipeline_=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                                     ('scale',StandardScaler()),('model',Ridge(alpha=float(self.name.rsplit('_',1)[1])))])
            counts={p:sum(r['patient_id']==p for r in rows) for p in ids}
            weights=np.array([1/counts[r['patient_id']] for r in rows]);weights*=len(rows)/sum(weights)
            # Both imputation/scaling and ridge are train-only; patient weights apply to loss.
            self.pipeline_.fit(array(rows,self.names_),[r['observed_value'] for r in rows],model__sample_weight=weights)
        return self
    def predict(self,rows):
        if self.pipeline_ is not None:return self.pipeline_.predict(array(rows,self.names_))
        key={'persistence':'current_value','rolling_median':'rolling_median_7d','rolling_mean':'rolling_mean_7d'}[self.name]
        return np.array([r['features'][key] if r['features'].get(key) is not None else r['features']['current_value'] for r in rows],dtype=float)
    def forecast(self,rows):
        p=self.predict(rows)
        q=self.interval_half_width_
        return [{'predicted_value':float(v),'lower_bound':float(v-q),'upper_bound':float(v+q),
                 'uncertainty':{'half_width':q,'nominal_level':.95,'method':'validation patient-balanced absolute residual percentile; exploratory'}} for v in p]


def score(rows,p,scale):
    actual=np.array([r['observed_value'] for r in rows]);e=actual-p
    patients={}
    for pid in sorted({r['patient_id'] for r in rows}):
        ix=np.array([r['patient_id']==pid for r in rows]);err=e[ix]
        patients[pid]={'n':int(sum(ix)),'mae':float(np.mean(abs(err))),'rmse':float(np.sqrt(np.mean(err**2)))}
    return {'n':len(rows),'patients':len(patients),'mae':float(np.mean(abs(e))), 'rmse':float(np.sqrt(np.mean(e**2))),
            'macro_patient_mae':float(np.mean([r['mae'] for r in patients.values()])),
            'macro_patient_rmse':float(np.mean([r['rmse'] for r in patients.values()])),
            'normalized_mae_training_sd':float(np.mean(abs(e))/scale),'per_patient':patients}


def residual_interval(rows,p,level=.95):
    ids={r['patient_id'] for r in rows};counts={pid:sum(r['patient_id']==pid for r in rows) for pid in ids}
    errors=np.abs(np.array([r['observed_value'] for r in rows])-p)
    weights=np.array([1/counts[r['patient_id']] for r in rows]);order=np.argsort(errors)
    cdf=np.cumsum(weights[order])/sum(weights)
    return float(errors[order[min(np.searchsorted(cdf,level),len(order)-1)]])


def support(allrows,folds):
    report={}
    for signal in sorted({r['signal'] for r in allrows}):
        rows=[r for r in allrows if r['signal']==signal]
        counts=[]
        for fold in folds:
            counts.append({role:{'windows':sum(fold['assignments'][r['patient_id']]==role for r in rows),
                                 'patients':len({r['patient_id'] for r in rows if fold['assignments'][r['patient_id']]==role})}
                           for role in ['train','validation','test']})
        enough=len({r['patient_id'] for r in rows})>=10 and len(rows)>=100 and all(
            c['train']['windows']>=30 and c['validation']['windows']>=10 and c['test']['windows']>=10
            and c['train']['patients']>=5 and c['validation']['patients']>=2 and c['test']['patients']>=2 for c in counts)
        gaps=[r['gap_days'] for r in rows]
        report[signal]={'supported':enough,'windows':len(rows),'patients':len({r['patient_id'] for r in rows}),
                        'fold_counts':counts,'unit':rows[0]['unit'],'unit_support':rows[0]['unit_support'],
                        'gap_days_quantiles':np.quantile(gaps,[0,.25,.5,.75,1]).tolist(),
                        'reason':'meets prespecified coverage gate' if enough else 'fails at least one patient/window coverage gate'}
    return report


def train(root='data/processed',output='data/results/phase3',datasets=('chiron','perheart')):
    summary={}
    for dataset in datasets:
        info=metadata(root,dataset);directory=sealed_directory(Path(output)/dataset)
        folds=load_json(Path(root)/dataset/'splits.json')['folds']
        rows=windows(read_jsonl(Path(root)/dataset/'temporal_features.jsonl'),dataset)
        coverage=support(rows,folds);write_json(directory/'coverage.json',coverage)
        selections=[]
        for signal,c in coverage.items():
            if not c['supported']:continue
            signalrows=[r for r in rows if r['signal']==signal]
            for fold in folds:
                assignments=fold['assignments'];number=fold['fold']
                tr=[r for r in signalrows if assignments[r['patient_id']]=='train']
                va=[r for r in signalrows if assignments[r['patient_id']]=='validation']
                models={};metrics={}
                for name in CANDIDATES:
                    m=PhysiologyModel(name).fit(tr,assignments);models[name]=m
                    metrics[name]=score(va,m.predict(va),m.training_scale_)
                best=min(v['macro_patient_mae'] for v in metrics.values())
                # Simpler models win if within 1% of best validation patient MAE.
                selected=next(n for n in CANDIDATES if metrics[n]['macro_patient_mae']<=best*1.01)
                globalname=min((n for n in CANDIDATES if n.startswith('global')),key=lambda n:metrics[n]['macro_patient_mae'])
                personalname=min((n for n in CANDIDATES if n.startswith('personal')),key=lambda n:metrics[n]['macro_patient_mae'])
                for name,m in models.items():
                    m.interval_half_width_=residual_interval(va,m.predict(va))
                    save_model(directory/f'{signal}_fold{number}_{name}.joblib',m)
                selections.append({'signal':signal,'fold':number,'selected':selected,'global':globalname,'personal':personalname,
                    'validation_metrics':metrics,'train_patients':sorted({r['patient_id'] for r in tr}),
                    'validation_patients':sorted({r['patient_id'] for r in va}),
                    'validation_interval_patients':len({r['patient_id'] for r in va}),
                    'interval_half_width':models[selected].interval_half_width_})
        write_json(directory/'selection.json',{'metadata':info,'selections':selections,'candidates':CANDIDATES,
            'features':{'global':GLOBAL_FEATURES,'personal':PERSONAL_FEATURES},
            'horizon':'exact next recorded source day' if dataset=='chiron' else 'next usable observed measurement, gap <=3 days; future gap never an input',
            'interval_limitations':'Empirical 95% interval; only 2-6 validation patients; shared tuning/calibration validation may bias intervals. No distribution-free coverage guarantee, no clinical reference interpretation.'})
        write_json(directory/'selection_seal.json',{'hashes':artifact_hashes(directory)})
        summary[dataset]={'supported_signals':[s for s,v in coverage.items() if v['supported']],'unsupported_signals':[s for s,v in coverage.items() if not v['supported']]}
    return summary


def shadow(row,prediction,scale,fold,model):
    residual=row['observed_value']-prediction['predicted_value']
    return {'patient_id':row['patient_id'],'dataset':row['dataset'],'timestamp':row['timestamp'],'prediction_cutoff':row['cutoff'],
            'signal':row['signal'],'expected_value':prediction['predicted_value'],'observed_value':row['observed_value'],
            'residual':residual,'normalized_residual':residual/scale,'normalization_scale':scale,
            'normalization_method':'outer training target standard deviation (minimum 1e-8); no target patient or future fit',
            'lower_bound':prediction['lower_bound'],'upper_bound':prediction['upper_bound'],'uncertainty':prediction['uncertainty'],
            'quality_status':row['quality_status'],'unit':row['unit'],'unit_support':row['unit_support'],
            'warmup':row['warmup'],'gap_days':row['gap_days'],'fold':fold,'model':model,
            'input_observation_ids':row['input_observation_ids'],'target_observation_ids':row['target_observation_ids']}


def paired_bootstrap(per_patient,reps=2000):
    differences=np.array([v['personal_minus_global_mae'] for v in per_patient.values()])
    rng=np.random.default_rng(SEED)
    means=[float(np.mean(rng.choice(differences,len(differences),replace=True))) for _ in range(reps)]
    return {'mean_personal_minus_global_mae':float(np.mean(differences)),'patient_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),
            'replicates':reps,'limitation':'descriptive across outer folds; training folds overlap, fit uncertainty excluded'}


def evaluate(root='data/processed',output='data/results/phase3',datasets=('chiron','perheart')):
    summary={}
    for dataset in datasets:
        directory=Path(output)/dataset;verify_hashes(directory,load_json(directory/'selection_seal.json')['hashes'])
        selected=load_json(directory/'selection.json');verify_inputs(root,selected['metadata'])
        if code_hash()!=selected['metadata']['code_sha256']:raise ValueError('Source changed after selection')
        if (directory/'test_metrics.json').exists():
            verify_hashes(directory,load_json(directory/'evaluation_seal.json')['hashes'])
            summary[dataset]='stored evaluation';continue
        folds=load_json(Path(root)/dataset/'splits.json')['folds']
        rows=windows(read_jsonl(Path(root)/dataset/'temporal_features.jsonl'),dataset)
        residuals=[];foldmetrics=[];allpredictions=defaultdict(list)
        for selection in selected['selections']:
            signal,number=selection['signal'],selection['fold'];assignments=folds[number]['assignments']
            test=[r for r in rows if r['signal']==signal and assignments[r['patient_id']]=='test']
            metrics={}
            for name in CANDIDATES:
                m=joblib.load(directory/f'{signal}_fold{number}_{name}.joblib')
                if set(m.fit_patient_ids_)&{r['patient_id'] for r in test}:raise ValueError('Forecast patient contamination')
                predictions=m.predict(test);metrics[name]=score(test,predictions,m.training_scale_)
                for row,value in zip(test,predictions):allpredictions[signal,name].append((row,float(value),m.training_scale_))
            m=joblib.load(directory/f"{signal}_fold{number}_{selection['selected']}.joblib")
            predictions=m.forecast(test)
            residuals.extend(shadow(row,p,m.training_scale_,number,selection['selected']) for row,p in zip(test,predictions))
            covered=[p['lower_bound']<=r['observed_value']<=p['upper_bound'] for r,p in zip(test,predictions)]
            cp=[np.mean([v for r,v in zip(test,covered) if r['patient_id']==pid]) for pid in sorted({r['patient_id'] for r in test})]
            foldmetrics.append({'signal':signal,'fold':number,'selected':selection['selected'],'global':selection['global'],'personal':selection['personal'],
                                'models':metrics,'interval_sample_coverage':float(np.mean(covered)),
                                'interval_macro_patient_coverage':float(np.mean(cp)),'interval_half_width':m.interval_half_width_})
        signals={}
        for signal in sorted({r['signal'] for r in residuals}):
            sr=[r for r in residuals if r['signal']==signal]; fs=[f for f in foldmetrics if f['signal']==signal]
            patient={};compare={}
            for f in fs:
                for pid,v in f['models'][f['selected']]['per_patient'].items():
                    if pid in patient:raise ValueError('Patient tested twice')
                    patient[pid]=v
                    g=f['models'][f['global']]['per_patient'][pid];p=f['models'][f['personal']]['per_patient'][pid]
                    compare[pid]={'global_mae':g['mae'],'personal_mae':p['mae'],'personal_minus_global_mae':p['mae']-g['mae']}
            e=np.array([r['residual'] for r in sr])
            macro=np.mean([p['mae'] for p in patient.values()])
            fixed={}
            for name in CANDIDATES:
                vals=allpredictions[signal,name]; rr=[v[0] for v in vals];pp=np.array([v[1] for v in vals])
                fixed[name]=score(rr,pp,1.)
                fixed[name].pop('normalized_mae_training_sd')
            cov={pid:float(np.mean([r['lower_bound']<=r['observed_value']<=r['upper_bound'] for r in sr if r['patient_id']==pid])) for pid in patient}
            signals[signal]={'unit':sr[0]['unit'],'n':len(sr),'patients':len(patient),'mae':float(np.mean(abs(e))),
                'rmse':float(np.sqrt(np.mean(e**2))),'macro_patient_mae':float(macro),
                'macro_patient_rmse':float(np.mean([p['rmse'] for p in patient.values()])),
                'fold_macro_mae':[f['models'][f['selected']]['macro_patient_mae'] for f in fs],
                'fold_macro_mae_std':float(np.std([f['models'][f['selected']]['macro_patient_mae'] for f in fs],ddof=1)),
                'selected_models':[f['selected'] for f in fs],'fixed_models':fixed,'per_patient':patient,
                'personalization':paired_bootstrap(compare),'paired_per_patient':compare,
                'interval_sample_coverage':float(np.mean([r['lower_bound']<=r['observed_value']<=r['upper_bound'] for r in sr])),
                'interval_macro_patient_coverage':float(np.mean(list(cov.values()))),
                'interval_mean_width':float(np.mean([r['upper_bound']-r['lower_bound'] for r in sr])),
                'normalized_mae_training_sd':float(np.mean([abs(r['normalized_residual']) for r in sr])),
                'warmup':{warm:{'n':sum(r['warmup']==warm for r in sr),'mae':float(np.mean([abs(r['residual']) for r in sr if r['warmup']==warm]))}
                          for warm in ['cold','warm'] if any(r['warmup']==warm for r in sr)}}
        write_rows(directory/'shadow_residuals.jsonl',residuals)
        write_json(directory/'test_metrics.json',{'dataset':dataset,'signals':signals,'fold_metrics':foldmetrics,
                                                'selection_sha256':file_hash(directory/'selection.json')})
        write_json(directory/'evaluation_seal.json',{'hashes':artifact_hashes(directory,('evaluation_seal.json',))})
        summary[dataset]={'signals':len(signals),'shadow_residuals':len(residuals)}
    return summary
