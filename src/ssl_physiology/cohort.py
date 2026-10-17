"""Role routing before clinical deserialization; causal canonical 5min tensors."""
import csv,gzip,hashlib,json,re
from pathlib import Path
import numpy as np
from src.vitaldb_feasibility.source import ROOT as VITAL,private_json,sha,cases
from src.vitaldb_feasibility.signals import usable_minutes
from src.ssl_physiology.protocol import ROOT,SIGNALS,EICU_FIELDS,PROTOCOL

def role_rows(path,allowed,audit=None):
    """Only patient identity is decoded for rejected mixed-source rows."""
    with Path(path).open('rb') as stream:
        for line in stream:
            matches=re.findall(rb'"patient_id"\s*:\s*"([^"\\]+)"',line)
            if len(matches)!=1:raise ValueError('Ambiguous row routing identity')
            patient=matches[0].decode('ascii')
            if patient not in allowed:
                if audit is not None:audit['rejected_identity_rows']=audit.get('rejected_identity_rows',0)+1
                continue
            row=json.loads(line)
            if row['patient_id'] not in allowed:raise ValueError('Role routing failure')
            if audit is not None:audit['decoded_development_rows']=audit.get('decoded_development_rows',0)+1
            yield row

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    np.savez_compressed(path,**arrays);path.chmod(0o600)

def grouped_roles(subjects):
    subjects=sorted(set(subjects),key=lambda x:sha(('2026:'+x).encode()))
    cutoff=int(.8*len(subjects));return {p:'train' if i<cutoff else 'validation' for i,p in enumerate(subjects)}

def tensor(values,observed):
    values=np.asarray(values,float).copy();observed=np.asarray(observed,bool)&np.isfinite(values)
    values[~observed]=np.nan;age=np.ones_like(values)
    for j in range(values.shape[-1]):
        last=None
        for i in range(len(values)):
            if observed[i,j]:last=i
            age[i,j]=1 if last is None else min(1,(i-last)*5/360)
    return np.concatenate([values,observed.astype(float),age],axis=-1)

def gap_length(mask):
    locations=np.flatnonzero(mask);return int(max(np.diff(np.r_[-1,locations,len(mask)])-1))

def acceptable(seq):
    return all(np.mean(seq[:,6+j])>=.8 and gap_length(seq[:,6+j].astype(bool))<=2 for j in range(5))

def vital_build():
    cohort=cases();selection=json.loads((VITAL/'multisignal_selection.json').read_text())['case_ids'];tracks_by_case={};counts={};out={}
    for caseid in selection:
        if float(cohort[caseid]['age'])<18:continue
        values={}
        for concept in SIGNALS:
            path=VITAL/'numeric'/concept/(caseid+'.npz')
            if not path.exists():continue
            meta=json.loads(path.with_suffix('.json').read_text())
            if sha(path.read_bytes())!=meta['cache_sha256']:raise ValueError('VitalDB cache changed')
            with np.load(path,allow_pickle=False) as z:summary={k:z[k] for k in z.files}
            interval=meta['qc']['observed_sampling_interval_seconds'];mask=usable_minutes(summary,interval['median'] if interval else 2)
            # Each 5min bin uses only closed minute bins (t-5,t], never subsequent source samples.
            grid=np.arange(5,len(mask),5);v=[];m=[]
            for t in grid:
                good=mask[t-4:t+1];median=summary['median'][t-4:t+1]
                ok=sum(good)>=4;v.append(float(np.nanmedian(median[good])) if ok else np.nan);m.append(ok)
            values[concept]=(np.array(v),np.array(m))
        if not all(c in values for c in SIGNALS[:5]):continue
        length=max(len(v[0]) for v in values.values());v=np.full((length,6),np.nan);m=np.zeros((length,6),bool)
        for j,concept in enumerate(SIGNALS):
            if concept in values:n=len(values[concept][0]);v[:n,j]=values[concept][0];m[:n,j]=values[concept][1]
        tracks_by_case[caseid]=(tensor(v,m),'vitaldb:subject:'+cohort[caseid]['subjectid'],float(cohort[caseid]['age']))
    subjects={p for seq,p,age in tracks_by_case.values() if any(acceptable(seq[i:i+12]) for i in range(0,max(0,len(seq)-12+1),6))}
    roles=grouped_roles(subjects);private_json(ROOT/'vitaldb_splits.json',{'assignments':roles,'seed':2026,'labels_used':False})
    for minutes in PROTOCOL['vitaldb_sequences_minutes']:
        steps=minutes//5;rows=[];sequences=[];future=[]
        for caseid,(seq,pid,age) in sorted(tracks_by_case.items()):
            if pid not in roles:continue
            for start in range(0,max(0,len(seq)-steps-3+1),steps):
                x=seq[start:start+steps];y=seq[start+steps:start+steps+3]
                if not acceptable(x):continue
                sequences.append(x);future.append(y[:,:6]);rows.append({'case_id':caseid,'patient_id':pid,'role':roles[pid],'age':age,'start_minutes':(start+1)*5,'end_minutes':(start+steps)*5})
        if not sequences:raise ValueError('No VitalDB sequences')
        save_npz(ROOT/f'vitaldb_{minutes}.npz',sequence=np.stack(sequences),future=np.stack(future))
        private_json(ROOT/f'vitaldb_{minutes}_rows.json',rows)
        counts[str(minutes)]={role:{'sequences':sum(r['role']==role for r in rows),'cases':len({r['case_id'] for r in rows if r['role']==role}),'subjects':len({r['patient_id'] for r in rows if r['role']==role})} for role in ['train','validation']}
    private_json(ROOT/'vitaldb_cohort.json',{'sequence_durations':counts,'primary_minutes':60,'reason':'60minutes provides context while retaining short-recording subjects;30/120counts disclosed; no outcome optimization','source_cases_considered':len(selection),'models_trained':0})
    print({'vitaldb_sequences':counts});return counts

def eicu_build():
    source=Path('data/processed/eicu_demo');assign=json.loads((source/'splits.json').read_text())['assignments']
    allowed={p for p,r in assign.items() if r in ['train','validation']};routing={}
    indices=list(role_rows(source/'predictors.jsonl',allowed,routing));labels={r['window_id']:r for r in role_rows(source/'labels.jsonl',allowed,routing)}
    required_stays={r['icu_stay_id'] for r in indices};contexts=[r for r in role_rows(source/'patient_context.jsonl',allowed,routing) if r['icu_stay_id'] in required_stays]
    patients={int(r['source_stay_key']):r for r in contexts};by_stay={r['icu_stay_id']:r for r in contexts}
    target=ROOT/'development_source/eicu_demo/vitalPeriodic.csv.gz';target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    raw=Path('data/raw/phase3_5/eicu_demo/vitalPeriodic.csv.gz');kept=0;skipped=0
    with gzip.open(raw,'rb') as inp,gzip.open(target,'wb') as dest:
        header=inp.readline();fields=next(csv.reader([header.decode()]));stay_column=fields.index('patientunitstayid');dest.write(header)
        for line in inp:
            identity=int(line.split(b',',stay_column+1)[stay_column])
            if identity not in patients:skipped+=1;continue
            dest.write(line);kept+=1
    target.chmod(0o600)
    from src.data.adapters.eicu import vital_arrays
    from src.predictive.data import construct
    _,_,_,series,qc=vital_arrays(ROOT/'development_source','vitalPeriodic',patients)
    rows=[];sequences=[]
    for index in indices:
        patient=by_stay[index['icu_stay_id']];features,seq,audit=construct(patient,series[int(patient['source_stay_key'])],index['t0'])
        oldfields=['heartrate','systemicmean','systemicsystolic','systemicdiastolic','sao2','respiration','temperature'];cols=[oldfields.index(f) for f in EICU_FIELDS]
        canonical=np.concatenate([seq[:,cols],seq[:,np.array(cols)+7],seq[:,np.array(cols)+14]],axis=1)
        y=labels[index['window_id']]
        sequences.append(canonical);rows.append({k:index[k] for k in ['window_id','patient_id','icu_stay_id','t0']})
        rows[-1].update(features=features,audit=audit,label=y['label'],event_onset=y['event_onset'],original_role=assign[index['patient_id']])
    if any(r['patient_id'] not in allowed for r in rows):raise ValueError('Forbidden eICU patient materialized')
    save_npz(ROOT/'eicu_development.npz',sequence=np.stack(sequences));private_json(ROOT/'eicu_development_rows.json',rows)
    private_json(ROOT/'development_access.json',{'role_routing':routing,'allowed_patients':sorted(allowed),'materialized_patients':sorted({r['patient_id'] for r in rows}),
        'vital_source_identity_rejected_rows':skipped,'vital_development_rows':kept,'test_clinical_rows_decoded':0,'test_labels_decoded':0,'test_predictions_files_opened':0,
        'shared_source_note':'compressed mixed-source byte streams routed using identity only; rejected clinical fields never JSON/CSV deserialized, no whole-cohort NumPy array opened',
        'source_paths':[str(source/'splits.json'),str(source/'predictors.jsonl'),str(source/'labels.jsonl'),str(source/'patient_context.jsonl'),str(raw)],'qc':qc})
    print({'eicu_development_patients':len({r['patient_id'] for r in rows}),'windows':len(rows),'positives':sum(r['label'] for r in rows),'test_clinical_rows_decoded':0})
    return rows

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True,mode=0o700);private_json(ROOT/'protocol.json',PROTOCOL);vital_build();eicu_build()
