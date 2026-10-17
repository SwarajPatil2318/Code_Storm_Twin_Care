"""Validate every candidate/window and primary causal snapshot; no model scores."""
import argparse
from collections import defaultdict,Counter
from pathlib import Path
import json
import numpy as np
from src.data.io import read_jsonl,file_hash,dumps,ArtifactWriter
from src.data.splits import validate_splits
from src.data.eicu_spec import CANDIDATES,PROTOCOL
from src.data.hemodynamic import episodes,prediction_windows,features,summarize_windows
from src.data.eicu_feasibility import source_hash


def restore_series(path):
    with np.load(path,allow_pickle=False) as data:
        raw=data['raw'];fields=list(raw.dtype.names[3:]);values={f:data['usable_'+f] for f in fields}
        if np.any(np.diff(raw['stay'])<0):raise ValueError('Stay ordering invalid')
        starts=np.r_[0,1+np.flatnonzero((raw['stay'][1:]!=raw['stay'][:-1])|(raw['offset'][1:]!=raw['offset'][:-1]))]
        compact_stay=raw['stay'][starts];times=raw['offset'][starts];compact={}
        for f in fields:
            status=data['status_'+f];v=values[f]
            if len(v)!=len(raw) or len(status)!=len(raw):raise ValueError('Array alignment failure')
            if np.any(np.isfinite(v)&(status!=0)):raise ValueError('Unusable quality state has a usable value')
            if np.any(np.isfinite(v)&(v!=raw[f])):raise ValueError('Value silently changed/clipped')
            count=np.add.reduceat(np.isfinite(v).astype(int),starts)
            if np.any(count>1):raise ValueError('Duplicate usable values at one timestamp')
            compact[f]=np.where(count>0,np.add.reduceat(np.nan_to_num(v,nan=0),starts),np.nan)
        if 'sao2' in fields and np.any(np.isfinite(values['sao2'])&((values['sao2']<0)|(values['sao2']>100))):raise ValueError('Impossible SpO2 usable')
        stays=np.unique(compact_stay);indices=np.searchsorted(compact_stay,stays);result={}
        for i,stay in enumerate(stays):
            a=indices[i];b=indices[i+1] if i+1<len(indices) else len(times)
            if np.any(np.diff(times[a:b])<=0):raise ValueError('Nonunique or out-of-order timeline')
            result[int(stay)]={'offset':times[a:b],**{f:v[a:b] for f,v in compact.items()}}
    return result


def validate(root='data/processed/eicu_demo',raw='data/raw/phase3_5',archive='Datasets/aibhs_datasets-main.zip',write=True):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());checks=[]
    def check(condition,message):
        if not condition:raise ValueError(message)
        checks.append(message)
    for name,meta in manifest['artifacts'].items():check(file_hash(root/name)==meta['sha256'],'artifact integrity: '+name)
    for name,sha in manifest['inputs']['files'].items():check(file_hash(Path(raw)/name)==sha,'raw extraction immutable: '+name)
    check(file_hash(archive)==manifest['inputs']['archive_sha256'],'original ZIP immutable')
    check(manifest['source_code_sha256']==source_hash(),'source version recorded accurately')
    contexts=list(read_jsonl(root/'patient_context.jsonl'));patients={p['source_stay_key']:p for p in contexts}
    check(len(patients)==len(contexts),'source ICU stays unique')
    pidmap={p['icu_stay_id']:p['patient_id'] for p in contexts}
    periodic=restore_series(root/'vitalPeriodic.npz');aperiodic=restore_series(root/'vitalAperiodic.npz')
    windows=list(read_jsonl(root/'prediction_windows.jsonl'));stored=defaultdict(list)
    for r in windows:
        check(pidmap[r['icu_stay_id']]==r['patient_id'],'window identity valid')
        stored[r['candidate'],r['horizon_minutes'],r['lookback_minutes']].append(r)
        if r['label']:check(r['t0']+15<r['event_onset']<=r['t0']+r['horizon_minutes'],'strict forecast label boundary')
    events=list(read_jsonl(root/'events.jsonl'));computed_events=[];report=json.loads((root/'report.json').read_text())
    for candidate,spec in CANDIDATES.items():
        series=periodic if spec['source']=='vitalPeriodic' else aperiodic;per_stay={}
        for stay,r in sorted(series.items()):
            p=patients[stay]
            if not p['adult']:continue
            per_stay[stay]=episodes(r['offset'],r[spec['field']],spec)
            actual=[e for e in events if e['candidate']==candidate and e['icu_stay_id']==p['icu_stay_id']]
            check(len(actual)==len(per_stay[stay]),'episode count and identity recomputed')
            for a,b in zip(actual,per_stay[stay]):
                check(all(a[k]==v for k,v in b.items()),'episode definition and recurrence recomputed')
        for h in PROTOCOL['horizons_minutes']:
            for l in PROTOCOL['lookbacks_minutes']:
                expected=[]
                for stay,r in sorted(series.items()):
                    window_series=r if spec['source']=='vitalPeriodic' else {**r,'systemicmean':r[spec['field']]}
                    rows,_=prediction_windows(patients[stay],window_series,per_stay.get(stay,[]),h,l,step=5 if spec['source']=='vitalPeriodic' else 15)
                    for row in rows:row['candidate']=candidate
                    expected.extend(rows)
                check(dumps(expected)==dumps(stored[candidate,h,l]),'all window labels, support, censoring and washout recomputed')
                check(report['candidates'][candidate]['window_matrix'][f'h{h//60}_l{l//60}']['full']==summarize_windows(expected),'window report aggregates agree')
    primary=[r for r in windows if r['candidate']==PROTOCOL['primary_candidate'] and r['horizon_minutes']==PROTOCOL['primary_horizon'] and r['lookback_minutes']==PROTOCOL['primary_lookback']]
    predictors=list(read_jsonl(root/'predictors.jsonl'));labels=list(read_jsonl(root/'labels.jsonl'))
    check(len(primary)==len(predictors)==len(labels),'primary predictors/labels separated and aligned')
    context_by_id={p['icu_stay_id']:p for p in contexts}
    for row,x,y in zip(primary,predictors,labels):
        p=context_by_id[row['icu_stay_id']];s=periodic[p['source_stay_key']]
        expected,provenance=features(p,s,row['t0'],row['lookback_minutes'],p['labs'])
        check(dumps(x['features'])==dumps(expected) and dumps(x['provenance'])==dumps(provenance),'every predictor is a causal allowlisted snapshot')
        check(x['window_id']==y['window_id'] and y['label']==row['label'] and y['event_onset']==row['event_onset'],'primary outcome separated correctly')
        forbidden=['discharge','mortality','apache','pressor','drug','outcome','label','event_onset','hf_ever']
        check(not any(any(token in k.lower() for token in forbidden) for k in x['features']),'future/outcome-defined features excluded')
        for r in x['provenance']['context_records']+x['provenance']['labs']:check(r['available_offset']<=x['t0'],'clinical input available by cutoff')
    splits=json.loads((root/'splits.json').read_text());validate_splits(splits,{r['patient_id'] for r in primary})
    check(all(splits['assignments'][x['patient_id']] in {'train','validation','test'} for x in predictors),'all windows inherit patient role')
    check(len({(r['patient_id'],r['icu_stay_id'],r['t0'],r['candidate'],r['horizon_minutes'],r['lookback_minutes']) for r in windows})==len(windows),'window keys unique')
    check(not any(root.glob('*.joblib')),'no final predictor trained')
    check(all(p.stat().st_mode&0o777==0o600 for p in root.iterdir() if p.is_file()),'record artifacts private')
    result={'status':'PASS','checks':len(checks),'gate_types':sorted(set(checks)),'windows_verified':len(windows),'primary_snapshots_verified':len(predictors),
            'no_model_trained':True,'source_code_sha256':source_hash()}
    if write:
        w=ArtifactWriter(root);w.json('validation',result)
    print({k:result[k] for k in ['status','checks','windows_verified','primary_snapshots_verified']})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/processed/eicu_demo');p.add_argument('--raw',default='data/raw/phase3_5');p.add_argument('--archive',default='Datasets/aibhs_datasets-main.zip');a=p.parse_args();validate(a.root,a.raw,a.archive)
