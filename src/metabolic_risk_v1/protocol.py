"""New risk experiment: immutable D2, development cache/profile reuse, no spent loaders."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic_d2.protocol import digest,verify_frozen,write_json
from src.metabolic_d35.protocol import DevelopmentScope,profile_permutation
from src.metabolic_d35.data import load_dynamic,SAFE

ROOT=Path('data/metabolic_risk_v1')
SEED=5102026


def load_inputs():
    verify_frozen()
    scope=DevelopmentScope()
    data=load_dynamic(scope)
    path=Path('data/metabolic_d35/participant_static.parquet')
    profiles=pd.read_parquet(path)
    scope.assert_allowed(profiles['participant_id'])
    if set(profiles['participant_id'])!=scope.allowed or profiles['participant_id'].duplicated().any():
        raise PermissionError('Safe profiles must cover exactly the permitted development cohort')
    audit=json.loads(Path('data/metabolic_d35/input_audit.json').read_text())
    if audit['holdout_feature_rows_decoded']!=0 or audit['source_sha256']!=digest('Datasets/diabetes_datasets.zip'):
        raise PermissionError('Unverified safe-profile provenance')
    if not set(SAFE)<=set(profiles.columns):raise ValueError('Safe profile schema missing')
    mapping=profiles.set_index('participant_id')
    for pid,cutoff in zip(data['pid'],data['time']):
        if mapping.loc[pid,'available_at_assumed_ns']>cutoff:raise PermissionError('Future clinical profile')
    return data,profiles,scope


def folds(ids,n=5,seed=SEED):
    ordered=np.random.default_rng(seed).permutation(sorted(set(ids)))
    return {str(pid):i%n for i,pid in enumerate(ordered)}


def preserve_snapshot():
    roots=['data/metabolic_d2','data/metabolic_d3','data/metabolic_d35','src/metabolic_d2','src/metabolic_d3','configs','data/synthetic','data/product']
    return {str(p):digest(p) for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def assert_preserved(before):
    if any(not Path(p).exists() or digest(p)!=h for p,h in before.items()):raise RuntimeError('Frozen scientific evidence modified')
    verify_frozen()
    return {'files_unchanged':len(before),'spent_holdout_payloads_opened':False}
