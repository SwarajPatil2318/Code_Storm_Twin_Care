"""Completion gates over frozen experiments; never retune or reopen clinical test labels."""
import argparse
from pathlib import Path
import numpy as np
from .common import *
from .clinical import read_predictors
from .forecasting import windows
from src.features.temporal import time_number
from src.data.splits import validate_splits


def audit(root='data/results/phase3',processed='data/processed',raw='Datasets'):
    root=Path(root);processed=Path(processed);checks=[]
    def check(condition,name):
        if not condition:raise ValueError(f'Completion gate failed: {name}')
        checks.append(name)
    verification=load_json(root/'verification.json')
    check(verification['status']=='PASS' and verification['failures']==0 and verification['errors']==0 and verification['skipped']==0,'full regression and modeling tests pass without skips')
    check(verification['models_prediction_identical'] and verification['selections_identical'] and not verification['clinical_test_reexecuted'],'repeat-training reproducibility without repeating clinical testing')
    check(verification['code_sha256']==code_hash(),'verification code version current')
    for ds in ['zigong','chiron','perheart','synthea']:
        validate_dataset(processed/ds);checks.append(f'{ds}: canonical integrity and causal validation')
    for item in load_json('data/audit/report.json')['inventory']:
        check(file_hash(Path(raw)/item['file'])==item['sha256'],'raw input immutable: '+item['file'])
    d=root/'clinical';s=load_json(d/'selection.json');m=load_json(d/'test_metrics.json')
    verify_hashes(d,load_json(d/'selection_seal.json')['hashes']);verify_hashes(d,load_json(d/'evaluation_seal.json')['hashes'])
    check((d/'TEST_EVALUATION_STARTED').exists() and m['test_evaluation_count']==1,'single final clinical test evaluation')
    a=load_json(processed/'zigong/splits.json')['assignments']
    model=joblib.load(d/'model.joblib')
    check(set(model.fit_patient_ids_)=={p for p,role in a.items() if role=='train'},'clinical fit patient roles')
    check(set(s['calibration_patient_ids']).isdisjoint(s['selection_patient_ids']),'calibration and model selection disjoint')
    check(set(model.calibration_patient_ids_)==set(s['calibration_patient_ids']),'calibration fit provenance')
    check(all(a[p]=='validation' for p in model.calibration_patient_ids_),'calibration excludes test')
    check(m['selection_sha256']==file_hash(d/'selection.json'),'clinical selected before test')
    check(s['metadata']['code_sha256']==code_hash(),'clinical code version frozen')
    ids,x=read_predictors(processed,a,{'test'});pred=list(read_jsonl(d/'test_predictions.jsonl'))
    check(ids==[r['patient_id'] for r in pred],'clinical prediction identity alignment')
    check(np.array_equal(model.predict(x),[r['clinical_vulnerability'] for r in pred]),'clinical serialization prediction equality (no labels reopened)')
    for ds in ['chiron','perheart']:
        d=root/ds;s=load_json(d/'selection.json');m=load_json(d/'test_metrics.json');coverage=load_json(d/'coverage.json')
        verify_hashes(d,load_json(d/'selection_seal.json')['hashes']);verify_hashes(d,load_json(d/'evaluation_seal.json')['hashes'])
        check(s['metadata']['code_sha256']==code_hash(),f'{ds}: code version frozen')
        folds=load_json(processed/ds/'splits.json');patients={p for p in folds['membership']};validate_splits(folds,patients)
        expected=windows(read_jsonl(processed/ds/'temporal_features.jsonl'),ds)
        residuals=list(read_jsonl(d/'shadow_residuals.jsonl'))
        check(len(residuals)==sum(v['windows'] for v in coverage.values() if v['supported']),f'{ds}: all eligible test windows exactly represented')
        keys=[(r['patient_id'],r['signal'],r['timestamp']) for r in residuals]
        check(len(keys)==len(set(keys)),f'{ds}: test windows unique')
        obs={r['observation_id']:r for r in read_jsonl(processed/ds/'observations.jsonl')}
        for selection in s['selections']:
            signal=selection['signal'];number=selection['fold'];a=folds['folds'][number]['assignments']
            model=joblib.load(d/f"{signal}_fold{number}_{selection['selected']}.joblib")
            check(all(a[p]=='train' for p in model.fit_patient_ids_),f'{ds}/{signal}/{number}: grouped model fit')
            check(all(a[p]=='validation' for p in selection['validation_patients']),f'{ds}/{signal}/{number}: validation residual interval roles')
            test=[r for r in expected if r['signal']==signal and a[r['patient_id']]=='test']
            actual=[r for r in residuals if r['signal']==signal and r['fold']==number]
            check(np.array_equal(model.predict(test),[r['expected_value'] for r in actual]),f'{ds}/{signal}/{number}: serialized predictions reproduce')
            for r in actual:
                cutoff=time_number(r['prediction_cutoff'],obs[r['input_observation_ids'][0]]['time_basis'])
                for oid in r['input_observation_ids']:
                    o=obs[oid]; t=time_number(o['observed_time'],o['time_basis'])
                    available=time_number(o['available_time'],o['time_basis']) if o['available_time'] is not None else t
                    if o['patient_id']!=r['patient_id'] or o['concept']!=signal or t>cutoff or available>cutoff:
                        raise ValueError('Residual provenance includes future or unrelated observation')
                if not np.isclose(r['observed_value']-r['expected_value'],r['residual']) or not np.isclose(r['residual']/r['normalization_scale'],r['normalized_residual']):
                    raise ValueError('Residual arithmetic failed')
                if a[r['patient_id']]!='test':raise ValueError('Non-test ShadowResidual')
            check(True,f'{ds}/{signal}/{number}: residual arithmetic, causal provenance, test roles')
        check(m['selection_sha256']==file_hash(d/'selection.json'),f'{ds}: selection sealed before outer evaluation')
    for doc in ['MODEL_BASELINES.md','CLINICAL_MODEL_CARD.md','PHYSIOLOGY_FORECASTING.md','MODEL_BASELINES_AGGREGATE.json']:
        check((Path('docs')/doc).exists(),'actual result documentation: '+doc)
    for d in [root/'clinical',root/'chiron',root/'perheart']:
        check(all((p.stat().st_mode&0o777)==0o600 for p in d.rglob('*') if p.is_file()),f'{d.name}: private artifact permissions')
        check(any(d.glob('plots/*.png')),f'{d.name}: evaluation plots generated')
    summary={'status':'PASS','checks':checks,'check_count':len(checks),'code_sha256':code_hash(),
             'limits':'engineering/experiment evidence only; does not certify clinical deterioration, external calibration, or fused twin'}
    write_json(root/'completion_audit.json',summary)
    print({'status':'PASS','checks':len(checks)})
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/results/phase3');p.add_argument('--processed',default='data/processed');p.add_argument('--raw',default='Datasets');a=p.parse_args();audit(a.root,a.processed,a.raw)
