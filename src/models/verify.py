"""Regression tests and repeat training; never evaluate clinical test again."""
import argparse
import unittest
from pathlib import Path
from .common import *
from .clinical import train as train_clinical, read_predictors
import numpy as np
from .forecasting import train as train_forecasting


def comparable(selection):
    selection=dict(selection)
    metadata=dict(selection['metadata']);metadata.pop('timestamp_utc')
    selection['metadata']=metadata
    return selection


def verify(root='data/results/phase3',replica='data/results/phase3_reproduction',processed='data/processed'):
    suite=unittest.defaultTestLoader.discover('tests')
    results=unittest.TextTestRunner(verbosity=1).run(suite)
    if not results.wasSuccessful() or results.skipped:raise ValueError('Full dependency-enabled tests must pass without skips')
    root,replica=Path(root),Path(replica)
    if root.resolve()==replica.resolve():raise ValueError('Repeat-training directory must differ')
    if not (replica/'clinical/selection.json').exists():train_clinical(processed,replica/'clinical')
    for ds in ['chiron','perheart']:
        if not (replica/ds/'selection.json').exists():train_forecasting(processed,replica,(ds,))
    model_count=0
    identical_byte_count=0
    clinical_assignments=load_json(Path(processed)/'zigong/splits.json')['assignments']
    _,clinical_x=read_predictors(processed,clinical_assignments,{'train','validation'})
    for ds in ['clinical','chiron','perheart']:
        for run in [root,replica]:
            selection=load_json(run/ds/'selection.json')
            if selection['metadata']['code_sha256']!=code_hash():raise ValueError('Training source version changed')
            verify_hashes(run/ds,load_json(run/ds/'selection_seal.json')['hashes'])
        if comparable(load_json(root/ds/'selection.json'))!=comparable(load_json(replica/ds/'selection.json')):
            raise ValueError('Repeated selection or validation metrics differ')
        paths=list((root/ds).glob('*.joblib'))
        for p in paths:
            if file_hash(p)==file_hash(replica/ds/p.name):
                identical_byte_count+=1
            elif ds=='clinical':
                one=joblib.load(p);two=joblib.load(replica/ds/p.name)
                if not np.array_equal(one.predict(clinical_x),two.predict(clinical_x)) or not np.array_equal(one.raw(clinical_x),two.raw(clinical_x)):
                    raise ValueError('Repeated clinical predictions differ')
                if one.pipeline_ is not None and hasattr(one.pipeline_.named_steps['model'],'estimators_'):
                    left=one.pipeline_.named_steps['model'].estimators_.ravel();right=two.pipeline_.named_steps['model'].estimators_.ravel()
                    if not all(np.array_equal(a.tree_.__getstate__()['nodes'],b.tree_.__getstate__()['nodes']) and np.array_equal(a.tree_.value,b.tree_.value) for a,b in zip(left,right)):
                        raise ValueError('Repeated learned clinical tree states differ')
            else:raise ValueError('Repeated forecasting model differs')
        model_count+=len(paths)
    if any(replica.rglob('test_metrics.json')):raise ValueError('Reproduction must not repeat test evaluation')
    value={'status':'PASS','tests_run':results.testsRun,'failures':len(results.failures),'errors':len(results.errors),'skipped':len(results.skipped),
           'models_byte_identical':identical_byte_count==model_count,'byte_identical_model_count':identical_byte_count,'models_prediction_identical':True,'serialized_model_count':model_count,'selections_identical':True,'clinical_test_reexecuted':False,
           'code_sha256':code_hash(),'test_sha256':{str(p):file_hash(p) for p in sorted(Path('tests').glob('test_*.py'))},
           'replica_directory':str(replica),'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(root/'verification.json',value)
    print({k:value[k] for k in ['status','tests_run','serialized_model_count','models_prediction_identical','byte_identical_model_count']})
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/results/phase3');p.add_argument('--replica',default='data/results/phase3_reproduction');p.add_argument('--processed',default='data/processed');a=p.parse_args();verify(a.root,a.replica,a.processed)
