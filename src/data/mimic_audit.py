"""MIMIC demo integration-only audit; no clinical model evaluation."""
from collections import Counter,defaultdict
import datetime as dt
from pathlib import Path
from src.data.adapters.eicu import table_rows,number
from src.data.io import file_hash


def audit_mimic(root):
    tables={};patients={r['subject_id']:r for _,r in table_rows(root,'mimic_iv_demo/hosp/patients.csv.gz')}
    admissions={r['hadm_id']:r for _,r in table_rows(root,'mimic_iv_demo/hosp/admissions.csv.gz')}
    stays={r['stay_id']:r for _,r in table_rows(root,'mimic_iv_demo/icu/icustays.csv.gz')}
    if len(patients)!=100:raise ValueError('Unexpected MIMIC demo population; re-audit release')
    mapping=[];items={}
    for _,r in table_rows(root,'mimic_iv_demo/icu/d_items.csv.gz'):
        items[r['itemid']]=r
        label=r['label'].lower()
        signal= ('heart_rate' if label=='heart rate' else
                 'invasive_map' if label=='arterial blood pressure mean' else
                 'noninvasive_map' if label=='non invasive blood pressure mean' else
                 'invasive_systolic_bp' if label=='arterial blood pressure systolic' else
                 'invasive_diastolic_bp' if label=='arterial blood pressure diastolic' else
                 'noninvasive_systolic_bp' if label=='non invasive blood pressure systolic' else
                 'noninvasive_diastolic_bp' if label=='non invasive blood pressure diastolic' else
                 'spo2' if label=='o2 saturation pulseoxymetry' else
                 'respiratory_rate' if label in {'respiratory rate','respiratory rate (total)'} else
                 'temperature' if label in {'temperature celsius','temperature fahrenheit'} else
                 'cvp' if label=='cvp' else None)
        if signal:mapping.append({'canonical_concept':signal,'itemid':r['itemid'],'label':r['label'],'unit':r['unitname'],'table':r['linksto']})
    chosen={r['itemid'] for r in mapping};signalrows=Counter();signalpatients=defaultdict(set);signalstays=defaultdict(set);units=defaultdict(Counter)
    delays=[];orphan_counts={};unique_stay_subjects={r['subject_id'] for r in stays.values()}
    for path in sorted((Path(root)/'mimic_iv_demo').rglob('*.csv.gz')):
        relative=str(path.relative_to(root));count=0;ps=set();hs=set();ss=set();orphan=0;time_fields=None
        for _,r in table_rows(root,relative):
            count+=1
            if 'subject_id' in r:
                ps.add(r['subject_id']);orphan+=r['subject_id'] not in patients
            if 'hadm_id' in r and r['hadm_id']:
                hs.add(r['hadm_id']);orphan+=r['hadm_id'] not in admissions
                if r['hadm_id'] in admissions and r.get('subject_id') and admissions[r['hadm_id']]['subject_id']!=r['subject_id']:raise ValueError('MIMIC subject/admission mismatch')
            if 'stay_id' in r and r['stay_id']:
                ss.add(r['stay_id']);orphan+=r['stay_id'] not in stays
                if r['stay_id'] in stays and r.get('subject_id') and stays[r['stay_id']]['subject_id']!=r['subject_id']:raise ValueError('MIMIC subject/stay mismatch')
            if time_fields is None:time_fields=[k for k in r if 'time' in k or k in {'chartdate','dod'}]
            if relative.endswith('icu/chartevents.csv.gz') and r['itemid'] in chosen:
                signalrows[r['itemid']]+=1;signalpatients[r['itemid']].add(r['subject_id']);signalstays[r['itemid']].add(r['stay_id']);units[r['itemid']][r['valueuom']]+=1
                if r['charttime'] and r['storetime']:
                    delays.append((dt.datetime.fromisoformat(r['storetime'])-dt.datetime.fromisoformat(r['charttime'])).total_seconds()/60)
        tables[relative.split('mimic_iv_demo/')[1]]={'rows':count,'patients':len(ps),'hospital_stays':len(hs),'icu_stays':len(ss),
            'orphan_references':orphan,'timestamp_fields':time_fields,'sha256':file_hash(path)}
        if orphan:raise ValueError('Orphan MIMIC references')
    from src.data.adapters.eicu import quantiles
    for r in mapping:
        k=r['itemid'];r.update(measurements=signalrows[k],patients=len(signalpatients[k]),icu_stays=len(signalstays[k]),recorded_units=dict(units[k]))
    return {'patients':len(patients),'hospital_stays':len(admissions),'icu_stays':len(stays),
            'patients_with_icu':len(unique_stay_subjects),'tables':tables,'physiology_mapping':mapping,
            'chart_to_store_delay_minutes_quantiles':quantiles(delays),'negative_store_delays':sum(v<0 for v in delays),
            'admission_mortality':dict(Counter(r['hospital_expire_flag'] for r in admissions.values())),
            'role':'schema compatibility and ingestion testing only; no clinical performance claim',
            'diagnosis_availability':'discharge billing ICD diagnoses have no availability timestamp; exclude from index-time features'}
