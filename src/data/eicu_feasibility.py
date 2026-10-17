"""Build reproducible Phase 3.5 source audits, target candidates and isolated risk sets."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from src.data.io import ArtifactWriter,file_hash,dumps,read_jsonl
from src.data.splits import stratified_split,validate_splits
from src.data.adapters.eicu import read_patients,context_audit,vital_arrays,quantiles,table_rows,SCHEMAS
from src.data.eicu_spec import CANDIDATES,PROTOCOL,PERIODIC,APERIODIC
from src.data.hemodynamic import episodes,prediction_windows,summarize_windows,features,asof_context
from src.data.mimic_audit import audit_mimic


def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path('src/data').rglob('*')):
        if p.suffix in {'.py','.json'}:h.update(str(p).encode());h.update(p.read_bytes())
    return h.hexdigest()


def extract(archive,raw):
    archive=Path(archive);raw=Path(raw);raw.mkdir(parents=True,exist_ok=True,mode=0o700)
    sha=file_hash(archive)
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        roots={i.filename.split('/')[0] for i in members if '/eicu_demo/patient.csv.gz' in i.filename}
        if len(roots)!=1:raise ValueError('Archive must have one verified eICU root')
        prefix=next(iter(roots))+'/'
        selected=list(SCHEMAS)+['eicu_demo/LICENSE.txt','eicu_demo/SHA256SUMS.txt','mimic_iv_demo/LICENSE.txt','mimic_iv_demo/SHA256SUMS.txt','mimic_iv_demo/README.txt']
        inputs={}
        for name in selected:
            payload=z.read(prefix+name);out=raw/name
            out.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            if out.exists() and out.read_bytes()!=payload:raise ValueError('Extracted raw source differs from archive')
            if not out.exists():out.write_bytes(payload)
            out.chmod(0o600);inputs[name]=file_hash(out)
        writer=ArtifactWriter(raw)
        writer.json('archive_inventory',{'archive':str(archive),'sha256':sha,'members':[{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size} for i in members]})
        writer.json('extraction_manifest',{'archive_sha256':sha,'files':inputs})
        writer.close()
    if file_hash(archive)!=sha:raise ValueError('Raw archive changed during extraction')
    checksum_verification={}
    for cohort in ['eicu_demo','mimic_iv_demo']:
        checked=0
        for line in (raw/cohort/'SHA256SUMS.txt').read_text().splitlines():
            expected,name=line.split(maxsplit=1);path=raw/cohort/name.lstrip('*')
            if path.is_file():
                if file_hash(path)!=expected:raise ValueError('Supplier checksum mismatch')
                checked+=1
        checksum_verification[cohort]=checked
    return {'archive_sha256':sha,'files':inputs,'supplier_checksums_verified':checksum_verification}


def overlap(patients,series):
    result={}
    for cohort,predicate in [('full_adult',lambda p:p['adult']),('hf_ever',lambda p:p['adult'] and p['ever']['hf']),
                              ('hf_history',lambda p:p['adult'] and p['ever']['hf_history']),
                              ('cv_ever',lambda p:p['adult'] and p['ever']['cardiovascular'])]:
        ids=[s for s,p in patients.items() if predicate(p)];any_dynamic=[];map_dynamic=[];hours=0;adequate=[]
        for stay in ids:
            if stay not in series:continue
            r=series[stay];t=r['offset'];m=r['systemicmean'];valid=t[np.isfinite(m)]
            if any(np.any(np.isfinite(r[f])) for f in ['heartrate','systemicmean','sao2']):any_dynamic.append(stay)
            if len(valid):
                map_dynamic.append(stay);diff=np.diff(valid);hours+=float(sum(diff[(diff>0)&(diff<=10)]))/60
                if len(valid)>=60 and valid[-1]-valid[0]>=360:adequate.append(stay)
        result[cohort]={'patients':len({patients[s]['patient_id'] for s in ids}),'icu_stays':len(ids),
            'hospital_stays':len({patients[s]['hospital_stay_id'] for s in ids}),
            'static_and_any_dynamic_stays':len(any_dynamic),'static_and_any_dynamic_patients':len({patients[s]['patient_id'] for s in any_dynamic}),
            'static_and_invasive_map_stays':len(map_dynamic),'static_and_invasive_map_patients':len({patients[s]['patient_id'] for s in map_dynamic}),
            'invasive_map_connected_hours':hours,'potential_6h_map_stays':len(adequate),
            'potential_6h_map_patients':len({patients[s]['patient_id'] for s in adequate})}
    return result


def build(archive='Datasets/aibhs_datasets-main.zip',raw='data/raw/phase3_5',output='data/processed/eicu_demo'):
    frozen_source_hash=source_hash()
    inputs=extract(archive,raw);output=Path(output)
    if output.resolve()==Path(raw).resolve() or Path(raw).resolve() in output.resolve().parents:raise ValueError('Raw and output roots must be disjoint')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new output directory for a repeat build; completed cohorts are not overwritten')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.eicu-stage-',dir=output.parent));stage.chmod(0o700)
    w=ArtifactWriter(stage)
    try:
        patients=read_patients(raw);tables,labs,pressor=context_audit(raw,patients)
        periodic_raw,periodic_status,periodic_usable,periodic,audit_p=vital_arrays(raw,'vitalPeriodic',patients)
        aperiodic_raw,aperiodic_status,aperiodic_usable,aperiodic,audit_a=vital_arrays(raw,'vitalAperiodic',patients)
        for name,array,status,usable in [('vitalPeriodic',periodic_raw,periodic_status,periodic_usable),('vitalAperiodic',aperiodic_raw,aperiodic_status,aperiodic_usable)]:
            path=stage/(name+'.npz');np.savez_compressed(path,raw=array,**{f'status_{f}':v for f,v in status.items()},**{f'usable_{f}':v for f,v in usable.items()});path.chmod(0o600)
        # Raw lexical rows remain lossless in extracted .csv.gz; NPZ links source primary IDs.
        for stay,p in sorted(patients.items()):
            w.emit('patient_context',{'source_stay_key':stay,**p,'labs':labs.get(stay,[]),'documented_pressor_offsets':sorted(pressor.get(stay,[]))})
        physiology_overlap=overlap(patients,periodic)
        all_windows=[];target_report={};all_events={}
        for name,definition in CANDIDATES.items():
            source=periodic if definition['source']=='vitalPeriodic' else aperiodic
            event_list=[];by_stay={}
            for stay,r in sorted(source.items()):
                p=patients[stay]
                if not p['adult']:continue
                event=episodes(r['offset'],r[definition['field']],definition);by_stay[stay]=event
                for i,e in enumerate(event):
                    flags,_,_=asof_context(p,e['onset'])
                    row={'candidate':name,'event_id':f'{name}:{p["icu_stay_id"]}:{i}','patient_id':p['patient_id'],
                         'icu_stay_id':p['icu_stay_id'],'hf_ever':p['ever']['hf'],'hf_known_at_onset':flags['hf']==1,
                         'cv_ever':p['ever']['cardiovascular'],**e}
                    event_list.append(row);w.emit('events',row)
            all_events[name]=by_stay
            affected=Counter(e['patient_id'] for e in event_list)
            candidate={'definition':definition,'episodes':len(event_list),'qualifying_low_runs':sum(len(e['runs']) for e in event_list),
                'affected_patients':len(affected),'affected_icu_stays':len({e['icu_stay_id'] for e in event_list}),
                'patients_with_recurrence':sum(n>1 for n in affected.values()),
                'run_observed_duration_minutes_quantiles':quantiles([r['observed_low_span_minutes'] for e in event_list for r in e['runs']]),
                'episode_span_minutes_quantiles':quantiles([e['last_low']-e['onset'] for e in event_list]),
                'hf_ever_episodes':sum(e['hf_ever'] for e in event_list),'hf_known_at_onset_episodes':sum(e['hf_known_at_onset'] for e in event_list),
                'hf_ever_affected_patients':len({e['patient_id'] for e in event_list if e['hf_ever']}),
                'hf_known_affected_patients':len({e['patient_id'] for e in event_list if e['hf_known_at_onset']}),
                'cv_ever_episodes':sum(e['cv_ever'] for e in event_list),'window_matrix':{},
                'first_positive_pressor_record_before_episode':sum(bool(pressor.get(by,[])) and min(pressor[by])<=e['onset'] for by,p in patients.items() for e in by_stay.get(by,[])),
                'first_positive_pressor_record_after_episode_within_3h':sum(bool(pressor.get(by,[])) and e['onset']<min(pressor[by])<=e['onset']+180 for by,p in patients.items() for e in by_stay.get(by,[]))}
            if True: # All candidates receive the same horizon/lookback feasibility audit.
                for horizon in PROTOCOL['horizons_minutes']:
                    for lookback in PROTOCOL['lookbacks_minutes']:
                        rows=[];excluded=Counter()
                        for stay,r in sorted(source.items()):
                            window_series=r if definition['source']=='vitalPeriodic' else {**r,'systemicmean':r[definition['field']]}
                            cohort,counts=prediction_windows(patients[stay],window_series,by_stay.get(stay,[]),horizon,lookback,step=5 if definition['source']=='vitalPeriodic' else 15)
                            for row in cohort:row['candidate']=name
                            rows.extend(cohort);excluded.update(counts)
                        key=f'h{horizon//60}_l{lookback//60}'
                        candidate['window_matrix'][key]={'full':summarize_windows(rows),
                            'hf_known':summarize_windows([r for r in rows if r['hf_known_at_t0']]),
                            'hf_ever_retrospective':summarize_windows([r for r in rows if r['hf_ever_during_stay']]),
                            'cv_known':summarize_windows([r for r in rows if r['cv_known_at_t0']]),'exclusions':dict(excluded)}
                        all_windows.extend(rows)
            target_report[name]=candidate
        for row in all_windows:w.emit('prediction_windows',row)
        primary=[r for r in all_windows if r['candidate']==PROTOCOL['primary_candidate'] and r['horizon_minutes']==PROTOCOL['primary_horizon'] and r['lookback_minutes']==PROTOCOL['primary_lookback']]
        labels={pid:int(any(r['label'] for r in primary if r['patient_id']==pid)) for pid in sorted({r['patient_id'] for r in primary})}
        split=stratified_split(labels,seed=2026);w.json('splits',split)
        by_canonical={p['icu_stay_id']:stay for stay,p in patients.items()}
        clinical_fusion_windows=0;clinical_fusion_patients=set();clinical_fusion_stays=set()
        for row in primary:
            stay=by_canonical[row['icu_stay_id']];x,provenance=features(patients[stay],periodic[stay],row['t0'],row['lookback_minutes'],labs.get(stay,[]))
            if any(v==1 for k,v in x.items() if k.startswith('documented_')) or provenance['labs']:
                clinical_fusion_windows+=1;clinical_fusion_patients.add(row['patient_id']);clinical_fusion_stays.add(row['icu_stay_id'])
            key=f"{row['icu_stay_id']}:{row['t0']}"
            w.emit('predictors',{'window_id':key,'patient_id':row['patient_id'],'icu_stay_id':row['icu_stay_id'],'t0':row['t0'],'features':x,'provenance':provenance})
            w.emit('labels',{'window_id':key,'patient_id':row['patient_id'],'target':PROTOCOL['primary_candidate'],'label':row['label'],'event_onset':row['event_onset']})
        partitions={part:summarize_windows([r for r in primary if split['assignments'][r['patient_id']]==part]) for part in ['train','validation','test']}
        hfpartitions={part:summarize_windows([r for r in primary if r['hf_known_at_t0'] and split['assignments'][r['patient_id']]==part]) for part in ['train','validation','test']}
        full=summarize_windows(primary);hf=summarize_windows([r for r in primary if r['hf_known_at_t0']])
        gates=PROTOCOL['go_gates'];go_checks={'eligible_patients':full['patients']>=gates['eligible_patients'],
            'positive_patients':full['positive_patients']>=gates['positive_patients'],'controls':full['negative_only_patients']>=gates['negative_patients'],
            'train_positive_patients':partitions['train']['positive_patients']>=gates['development_positive_patients'],
            'validation_positive_patients':partitions['validation']['positive_patients']>=gates['validation_positive_patients'],
            'test_positive_patients':partitions['test']['positive_patients']>=gates['test_positive_patients']}
        decision={'status':'GO' if all(go_checks.values()) else 'NO-GO','gates':go_checks,'gate_thresholds':gates,
                  'hf_evaluation_gate':hf['positive_patients']>=gates['hf_positive_patients'],'primary':full,'hf_known_primary':hf,
                  'partitions':partitions,'hf_partitions':hfpartitions,'no_model_trained':True,
                  'causal_clinical_plus_dynamic_overlap':{'patients':len(clinical_fusion_patients),'icu_stays':len(clinical_fusion_stays),'windows':clinical_fusion_windows}}
        demographic={field:dict(Counter(p['raw'][field] for p in patients.values())) for field in ['gender','ethnicity','unittype','unitdischargestatus','hospitaldischargestatus']}
        counts={'patients':len({p['patient_id'] for p in patients.values()}),'hospital_stays':len({p['hospital_stay_id'] for p in patients.values()}),'icu_stays':len(patients),
            'adult_stays':sum(p['adult'] for p in patients.values()),'adult_patients':len({p['patient_id'] for p in patients.values() if p['adult']}),
            'unknown_age_stays':sum(not p['raw']['age'] for p in patients.values()),'topcoded_age_stays':sum(p['raw']['age'].strip()=='> 89' for p in patients.values()),
            'age_quantiles_not_topcoded':quantiles([float(p['raw']['age']) for p in patients.values() if p['raw']['age'] and '>' not in p['raw']['age']]),
            'unit_length_hours_quantiles':quantiles([p['end']/60 for p in patients.values()]),
            'condition_counts':{flag:{'patients':len({p['patient_id'] for p in patients.values() if p['ever'][flag]}),'stays':sum(p['ever'][flag] for p in patients.values())} for flag in ['hf','hf_history','hypertension','diabetes','ckd','cardiovascular']},
            'coded_hospital_ids':len({p['raw']['hospitalid'] for p in patients.values()}),
            'coded_ward_ids':len({p['raw']['wardid'] for p in patients.values()}),
            'repeated_stay_patients':sum(n>1 for n in Counter(p['patient_id'] for p in patients.values()).values()),
            'demographics':demographic}
        report={'protocol':PROTOCOL,'counts':counts,'tables':tables,'physiology':{'vitalPeriodic':audit_p,'vitalAperiodic':audit_a},
                'cohort_overlap':physiology_overlap,'candidates':target_report,'decision':decision,
                'source_availability':'documented entry offsets; physiology additional 5-min conservative buffer; not certified live latency'}
        w.json('report',report);w.json('mimic_audit',audit_mimic(raw));w.ensure_tables(['events','prediction_windows','patient_context','predictors','labels']);w.close()
        for name,sha in inputs['files'].items():
            if file_hash(Path(raw)/name)!=sha:raise ValueError('Extracted raw file changed')
        if file_hash(archive)!=inputs['archive_sha256']:raise ValueError('Archive changed')
        if source_hash()!=frozen_source_hash:raise ValueError('Source changed during build')
        w.manifest(inputs,2026,PROTOCOL,frozen_source_hash)
        if output.exists():output.rmdir()
        stage.rename(output)
        return {'status':decision['status'],'patients':counts['patients'],'hf':counts['condition_counts']['hf'],
                'primary_windows':len(primary),'primary_positive_patients':full['positive_patients'],'models_trained':0}
    except BaseException:
        w.close();shutil.rmtree(stage);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',default='Datasets/aibhs_datasets-main.zip');p.add_argument('--raw',default='data/raw/phase3_5');p.add_argument('--output',default='data/processed/eicu_demo');a=p.parse_args();print(build(a.archive,a.raw,a.output))
