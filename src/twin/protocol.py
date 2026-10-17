"""Frozen Phase 6 contracts; all choices precede test clinical decoding."""
import hashlib,json
from pathlib import Path
from src.synthetic.config import SIGNALS,UNITS
from src.synthetic.outcomes import EHR_FEATURES

CONFIG=Path('configs/phase6_v1.json')
EVIDENCE={'evidence_label':'SYNTHETIC POC VALIDATION','clinical_validation':'NOT CLINICALLY VALIDATED'}
ABS_STATS=['mean','sd','last','slope_per_day','recent_change','coverage','staleness_hours','recent_mean','recent_count','suspect_fraction']
PERSONAL_STATS=['mean','median','sd','q10','q90','count','confidence','frozen','recent_deviation','recent_z','persistence','circadian_recent_reference','day_reference','night_reference']
PRIMARY_SIGNALS=[s for s in SIGNALS if s!='map']
ABS_FEATURES=['observed__'+s+'__'+k for s in PRIMARY_SIGNALS for k in ABS_STATS]
EHR_COLUMNS=['ehr__'+s for s in EHR_FEATURES]
PERSONAL_FEATURES=['personal__'+s+'__'+k for s in PRIMARY_SIGNALS for k in PERSONAL_STATS]
SHADOW_FEATURES=['shadow__'+s+'__'+k for s in PRIMARY_SIGNALS for k in ['expected','scale','residual','z','persistence','acceleration','count','width']]
TDI_FEATURES=['tdi__value','tdi__systems','tdi__coverage','tdi__confidence','tdi__trend']+['tdi__component__'+s for s in PRIMARY_SIGNALS]
ALLOWLIST=EHR_COLUMNS+ABS_FEATURES+PERSONAL_FEATURES+SHADOW_FEATURES+TDI_FEATURES
FORBIDDEN={'state','latent','profile','path','vulnerability','event_time','event_time_hours','event_time_hours','event_hazard','seed','stable_daily','true_physiology','index','deterioration_onset'}

def load(path=CONFIG):
    p=Path(path);c=json.loads(p.read_text());c['_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    if c['history_hours']!=168 or c['bin_hours']!=6:raise ValueError('Unsupported temporal contract')
    return c

def code_hash():
    h=hashlib.sha256()
    for p in sorted(Path('src/twin').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def require_role(ids,roles,allowed):
    if not len(ids) or any(roles.get(str(p)) not in allowed for p in ids):raise ValueError('Patient split access violation')

def validate_features(keys):
    if not set(keys)<=set(ALLOWLIST) or any(k.split('__')[-1] in FORBIDDEN for k in keys):raise ValueError('Unapproved model feature')

def columns(variant,names,consumer_signals=None):
    validate_features(names);out=[]
    for i,n in enumerate(names):
        group=n.split('__')[0];signal=n.split('__')[1]
        keep={'ehr_only':group=='ehr','wearable_only':group=='observed','population_fusion':group in ['ehr','observed'],
          'personalized_fusion':group in ['ehr','observed','personal'],'full_twin':True,'full_no_ehr':group!='ehr',
          'full_no_personal':group not in ['personal','shadow','tdi'],'full_no_tdi':group!='tdi',
          'personal_normal_only':group in ['personal','shadow','tdi'],'consumer_twin':group=='ehr' or signal in (consumer_signals or []) or group=='tdi' and signal!='component'}[variant]
        if keep:out.append(i)
    return out


def feature_code_hash():
    h=hashlib.sha256()
    for name in ["access.py","baseline.py","features.py","prepare.py","protocol.py"]:
        p=Path("src/twin")/name;h.update(name.encode());h.update(p.read_bytes())
    return h.hexdigest()
