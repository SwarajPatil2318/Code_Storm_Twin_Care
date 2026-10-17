"""Lazy canonical observations over efficient synthetic hourly NPZ batches.

This view opens public observation/patient artifacts only. Ground truth and
outcomes are deliberately absent from the reader interface.
"""
import datetime as dt
import hashlib,json
from pathlib import Path
import numpy as np
from src.synthetic.generate import iter_rows
from src.synthetic.config import SIGNALS,UNITS
from src.synthetic.sensors import QUALITY,REASONS,SOURCES
from src.data.schema import Observation,SourceValue,Provenance


def canonical_observations(root,patient_ids=None,limit=None):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['evidence']!='SYNTHETIC POC':raise ValueError('Synthetic reader cannot read a real cohort')
    patient_rows={r['patient_id']:r for r in iter_rows(root/'patients/records.jsonl.gz')};wanted=set(patient_ids) if patient_ids is not None else set(patient_rows)
    statuses={v:k for k,v in QUALITY.items()};reasons={v:k for k,v in REASONS.items()};emitted=0
    for name in manifest['batches']:
        with np.load(root/'observations'/name,allow_pickle=False) as arrays:
            for i,pid in enumerate(arrays['patient_ids'].tolist()):
                if pid not in wanted:continue
                origin=dt.datetime.fromisoformat(patient_rows[pid]['observed_time'])
                for hour,j in np.argwhere(arrays['scheduled'][i]):
                    raw=arrays['raw_value'][i,hour,j];value=arrays['value'][i,hour,j];available=arrays['available_hour'][i,hour,j];quality=statuses[int(arrays['quality_status'][i,hour,j])];reason=reasons[int(arrays['reason_code'][i,hour,j])]
                    concept=SIGNALS[j];identity=hashlib.sha256(f'{pid}:{concept}:{hour}'.encode()).hexdigest()[:24];path='observations/'+name
                    provenance=Provenance('synthetic',path,f'{i}:{hour}:{j}',concept,concept,'synthetic canonical view; preserve units,QCs and receipt time',manifest['artifacts'][path]['sha256'],UNITS[j],'synthetic_scenario')
                    source=SourceValue(float(raw) if np.isfinite(raw) else None,float(value) if np.isfinite(value) else None,quality,[] if reason=='NONE' else [reason],UNITS[j],provenance)
                    scheduled=(origin+dt.timedelta(hours=int(hour))).isoformat()
                    yield Observation('synthetic:'+identity,pid,'synthetic',concept,scheduled if np.isfinite(raw) else None,'absolute_utc',
                      (origin+dt.timedelta(hours=float(available))).isoformat() if np.isfinite(available) else None,'simulated_receipt',source,
                      metadata={'scheduled_opportunity_time':scheduled,'synthetic':True,'evidence':'SYNTHETIC POC','evidence_label':'NOT VALIDATED','sensor_source':SOURCES[j]})
                    emitted+=1
                    if limit is not None and emitted>=limit:return


def canonical_contexts(root):
    """Map generated historical EHR to the shared clinical-context dataclass."""
    from src.data.schema import PatientClinicalContext
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['evidence']!='SYNTHETIC POC':raise ValueError('Synthetic reader cannot read a real cohort')
    patients={r['patient_id']:r for r in iter_rows(root/'patients/records.jsonl.gz')}
    units={'age':'a','bmi':'kg/m2','ef':'%','hf_severity':'1','creatinine':'mg/dL','sodium':'mmol/L','hemoglobin':'g/dL','bnp':'pg/mL'}
    source='ehr/records.jsonl.gz'
    for i,row in enumerate(iter_rows(root/source)):
        pid=row['patient_id'];origin=dt.datetime.fromisoformat(patients[pid]['observed_time']);attributes={}
        for name,value in row['context'].items():
            provenance=Provenance('synthetic',source,str(i),name,name,'synthetic historical EHR mapping',manifest['artifacts'][source]['sha256'],units.get(name),'synthetic_scenario')
            attributes[name]=SourceValue(value,value,'MISSING' if value is None else 'VALID',[] if value is not None else ['synthetic_unavailable_lab'],units.get(name),provenance)
        yield PatientClinicalContext(pid,'synthetic',attributes,(origin-dt.timedelta(days=21)).isoformat(),
          (origin+dt.timedelta(hours=row['available_hour'])).isoformat(),'synthetic_historical_context')
