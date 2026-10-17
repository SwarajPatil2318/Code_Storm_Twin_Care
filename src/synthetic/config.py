"""Versioned JSON-subset YAML configuration, with validated numeric contracts."""
import hashlib,json
from pathlib import Path
import numpy as np

DEFAULT=Path('configs/synthetic_v1.yaml')
SIGNALS=['heart_rate','hrv_rmssd','systolic_bp','diastolic_bp','map','spo2','respiration','weight','steps','sleep_hours','temperature']
UNITS=['bpm','ms','mmHg','mmHg','mmHg','%','/min','kg','count/hour','h','degC']
EVIDENCE='SYNTHETIC POC'
STATUS='NOT VALIDATED'


def load(path=DEFAULT):
    path=Path(path);data=json.loads(path.read_text())
    if data['version']!='synthetic_v1':raise ValueError('Unknown simulator version')
    if len(data['phenotypes'])!=len(data['phenotype_proportions']) or abs(sum(data['phenotype_proportions'])-1)>1e-9:raise ValueError('Phenotype proportions must sum to one')
    if data['days']<30 or data['days']>90:raise ValueError('History must contain30–90days')
    if data['grid_hours']!=1:raise ValueError('v1 supports hourly summaries only')
    for signal in SIGNALS:
        lo,hi=data['signal_bounds'][signal]
        if lo>=hi:raise ValueError('Invalid physiological bounds')
    data['_config_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def rng(seed,index,component):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(index),int(component)]))
