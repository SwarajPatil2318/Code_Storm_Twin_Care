"""Role-specific cached arrays; no hidden truth or test access during development."""
import json,hashlib
from pathlib import Path
import numpy as np
from src.twin.protocol import EHR_COLUMNS,ABS_FEATURES,PERSONAL_FEATURES,SHADOW_FEATURES,TDI_FEATURES,validate_features,require_role,feature_code_hash
from src.twin.access import Cohort
from src.data.io import file_hash


def read_cache(role,config,consumer=False,check_hashes=True):
    cohort=Cohort(config['cohort'],config['output'],config);cohort.authorize(role)
    root=Path(config['output'])/'features'/('consumer_'+role if consumer else role);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['role']!=role or manifest['config_sha256']!=config['_sha256'] or manifest['cohort_manifest_sha256']!=cohort.metadata_sha or manifest['feature_code_sha256']!=feature_code_hash():raise ValueError('Stale feature cache')
    if bool(manifest['consumer_subset'])!=bool(consumer):raise ValueError('Consumer cache boundary mismatch')
    if check_hashes:
        for name,h in manifest['artifacts'].items():
            if file_hash(root/name)!=h:raise ValueError('Feature cache changed')
    data={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in root.glob('*.npy')}
    require_role(data['patient_ids'],cohort.roles,{role});return data

def validation_groups(patient_ids,config):
    ids=sorted(set(patient_ids.astype(str)),key=lambda p:hashlib.sha256((p+':phase6_validation').encode()).hexdigest());v=config['validation_roles'];a=v['selection'];b=a+v['calibration']
    if len(ids)!=sum(v.values()):raise ValueError('Expected complete authoritative validation cohort')
    roles={p:'selection' if i<a else 'calibration' if i<b else 'operating' for i,p in enumerate(ids)}
    return {name:np.array([roles[str(p)]==name for p in patient_ids]) for name in v},roles

def take(data,mask):return {k:v[mask] for k,v in data.items()}

def matrix(data,divergence):
    names=EHR_COLUMNS+ABS_FEATURES+PERSONAL_FEATURES+SHADOW_FEATURES+TDI_FEATURES;validate_features(names)
    x=np.concatenate([data['ehr'],data['absolute'],data['personal'],divergence['shadow_features'],divergence['tdi_features']],axis=1).astype(np.float32)
    if x.shape[1]!=len(names):raise ValueError('Feature tensor/schema disagreement')
    return x,names
