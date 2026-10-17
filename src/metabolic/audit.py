"""Read local archives without extracting PHI into tracked paths or emitting IDs.

Run: .venv/bin/python -m src.metabolic.audit
Outputs under ignored data/metabolic, chmod 0600. Only aggregate JSON is printed.
"""
import hashlib, io, json, re, zipfile, warnings
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .windows import Series, MINUTE, HISTORIES, HORIZONS, prepare, window_masks
from .schema import ParticipantStatic

ROOT = Path('Datasets')
OUT = Path('data/metabolic')
MISSING = {'', '/', '-', '--', 'nan', 'none', 'na', 'n/a', 'nat'}


def present(s):
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(MISSING)


def safe_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')
    path.chmod(0o600)


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def times(s):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return pd.to_datetime(s, errors='coerce', format='mixed').astype('int64').to_numpy()


def summarize(series):
    result={}
    for population in ['all','T2D']:
        cohort=[s for s in series if population=='all' or s.diabetes_status=='T2D']
        qcs=[s.quality for s in cohort]
        total=lambda key: sum(q.get(key,0) for q in qcs)
        spans=[q['span_days'] for q in qcs]
        result[population]={'participants':len(cohort),'glucose_rows':total('rows'),
            'usable_observations':total('usable_observations'),
            'unusable_rows':total('unusable_rows'),'duplicate_extra_rows':total('duplicate_extra_rows'),
            'conflicting_timestamps':total('conflicting_timestamps'),
            'out_of_40_400_rows':total('out_of_40_400_rows'),
            'backward_intervals_original':total('backward_intervals_original'),
            'observed_equivalent_days':total('observed_equivalent_days'),
            'calendar_days_any':total('calendar_days_any'),
            'calendar_days_70pct':total('calendar_days_70pct'),
            'span_days_quantiles':np.quantile(spans,[0,.25,.5,.75,1]).tolist() if spans else [],
            'participants_at_least_3_span_days':sum(v>=3 for v in spans),
            'participants_at_least_7_span_days':sum(v>=7 for v in spans),
            'interval_bins':dict(sum((Counter(q['interval_bins']) for q in qcs),Counter())),
            'histories':{}}
        for history in HISTORIES:
            rows=[]
            for s in cohort:
                valid,masks,_=window_masks(s,history)
                rows.append({'participant_id':s.participant_id,'valid_landmarks':int(valid.sum()),
                    'landmark_days':int(len(np.unique(s.time[valid]//(MINUTE*1440)))),
                    'targets':{str(h):int(masks[h].sum()) for h in HORIZONS},
                    'all_horizons':int(np.logical_and.reduce(list(masks.values())).sum()) if masks else 0})
            landmarks=sum(r['valid_landmarks'] for r in rows)
            result[population]['histories'][str(history)]={
                'participants':sum(r['valid_landmarks']>0 for r in rows),
                'T2D_participants':sum(s.diabetes_status=='T2D' and r['valid_landmarks']>0 for s,r in zip(cohort,rows)),
                'usable_landmark_days':sum(r['landmark_days'] for r in rows),
                'valid_landmarks':landmarks,
                'targets':{str(h):sum(r['targets'][str(h)] for r in rows) for h in HORIZONS},
                'target_participants':{str(h):sum(r['targets'][str(h)]>0 for r in rows) for h in HORIZONS},
                'all_horizons':sum(r['all_horizons'] for r in rows),
                'missing_target_rate':{str(h):1-sum(r['targets'][str(h)] for r in rows)/landmarks if landmarks else None for h in HORIZONS}}
            safe_json(OUT/'participant_counts'/f'{series[0].dataset if series else "empty"}_{population}_{history}.json',rows)
    return result


def frame_missingness(df, idcol):
    return {str(c):{'rows':len(df),'available':int(present(df[c]).sum()),
                    'missing':int((~present(df[c])).sum()),
                    'participants_available':int(df.loc[present(df[c]),idcol].nunique())}
            for c in df.columns if c != idcol and c != '_id'}


def join_events(df, columns):
    def combine(row):
        values=[f'{c}: {v}' for c,v in row.items() if pd.notna(v) and str(v).strip().lower() not in MISSING]
        return ' | '.join(values) if values else pd.NA
    return df[columns].apply(combine,axis=1) if columns else pd.Series(pd.NA,index=df.index,dtype='string')


def normalize_id(x):
    return str(x).strip().split('_')[0]


def shanghai():
    series=[]; cohort_report={}; clinical_all=[]; frames={}; mappings=[]
    with zipfile.ZipFile(ROOT/'diabetes_datasets.zip') as z:
        for status,folder in [('T2D','Shanghai_T2DM'),('T1D','Shanghai_T1DM')]:
            clinical_name=next(n for n in z.namelist() if '/' not in n and n.endswith('.xlsx') and folder in n)
            clinical=pd.read_excel(io.BytesIO(z.read(clinical_name)),keep_default_na=False)
            clinical['_id']=clinical['Patient Number'].map(normalize_id)
            cohort_report[status]={'recording_summary_rows':len(clinical),
                'participants':clinical['_id'].nunique(), 'status_counts':clinical['Type of Diabetes'].value_counts().to_dict(),
                'static_missingness':frame_missingness(clinical,'_id'),
                'repeated_participants':int((clinical.groupby('_id').size()>1).sum()),
                'static_joint_coverage':{},
                'static_conflicting_participants_by_field':{c:int((clinical.groupby('_id')[c].nunique()>1).sum()) for c in clinical if c not in ['_id','Patient Number']}}
            joint_sets={'age+BMI+HbA1c':['Age (years)','BMI (kg/m2)','HbA1c (mmol/mol)'],
                'age+BMI+HbA1c+medication':['Age (years)','BMI (kg/m2)','HbA1c (mmol/mol)','Hypoglycemic Agents']}
            for label,fields in joint_sets.items():
                mask=np.logical_and.reduce([present(clinical[c]).to_numpy() for c in fields])
                cohort_report[status]['static_joint_coverage'][label]={'recordings':int(mask.sum()),'participants':int(clinical.loc[mask,'_id'].nunique()),'receipt_verified':0}
            for _,row in clinical.iterrows():
                pid=f'Shanghai{status}:{row["_id"]}'
                fields={c: str(v) if isinstance(v,(pd.Timestamp,)) else v for c,v in row.items() if c not in ['_id','Patient Number']}
                num=lambda name: float(row[name]) if name in row and present(pd.Series([row[name]])).iloc[0] else None
                text=lambda name: str(row[name]) if name in row and present(pd.Series([row[name]])).iloc[0] else None
                duration=next(c for c in row.index if c.lower().startswith('duration of diabetes'))
                clinical_all.append(asdict(ParticipantStatic(pid,'Shanghai'+status,status,
                    recording_id=str(row['Patient Number']),age=num('Age (years)'),
                    sex={1:'female',2:'male'}.get(row['Gender (Female=1, Male=2)']),BMI=num('BMI (kg/m2)'),
                    HbA1c=num('HbA1c (mmol/mol)'),diabetes_duration=num(duration),
                    fasting_glucose=num('Fasting Plasma Glucose (mg/dl)'),fasting_insulin=num('Fasting Insulin (pmol/L)'),
                    medications=text('Hypoglycemic Agents'),comorbidities=text('Comorbidities'),original_fields=fields,
                    original_units={'HbA1c':'mmol/mol IFCC','fasting_insulin':'pmol/L','fasting_glucose':'mg/dL','BMI':'kg/m2'},
                    timing_basis='recording-linked summary; individual measurement/receipt times absent',
                    data_quality=['laboratory timing ambiguous','hypoglycemia summary may encode recording outcomes'])))
            by_id=defaultdict(list)
            for n in z.namelist():
                if not n.startswith(folder+'/') or not n.endswith(('.xls','.xlsx')):continue
                d=pd.read_excel(io.BytesIO(z.read(n)),keep_default_na=False)
                d['_source_file']=n;d['_recording_id']=Path(n).stem
                d['_time_ns']=times(d['Date'])
                glucose=next(c for c in d if c.strip() in ['CGM (mg / dl)','CGM'])
                d['_glucose']=pd.to_numeric(d[glucose],errors='coerce')
                pid='Shanghai'+status+':'+normalize_id(Path(n).stem)
                by_id[pid].append(d)
                mappings.append({'raw_source_file':n,'participant_id':pid,'recording_id':Path(n).stem,
                    'clinical_match_rows':int((clinical['Patient Number'].astype(str)==Path(n).stem).sum())})
            dynamic_counts=Counter(); dynamic_subjects=defaultdict(set)
            for pid,ds in by_id.items():
                d=pd.concat(ds,ignore_index=True);frames[pid]=d
                t,v,q=prepare(d['_time_ns'],d['_glucose'],15)
                s=Series(pid,'Shanghai'+status,status,t,v,15,q);series.append(s)
                for c in d:
                    if c.startswith('_') or c=='Date':continue
                    mask=present(d[c]);dynamic_counts[c]+=int(mask.sum())
                    if mask.any():dynamic_subjects[c].add(pid)
                # Preserve all raw fields, source units in headers, and missingness; parquet is private.
                p=OUT/'canonical'/('Shanghai'+status)/(pid.split(':')[1]+'.parquet')
                p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                normalized=d.copy().astype({c:'string' for c in d if c not in ['_time_ns','_glucose']})
                normalized['participant_id']=pid;normalized['source_dataset']='Shanghai'+status
                normalized['timestamp']=pd.to_datetime(d['_time_ns']);normalized['glucose_mg_dl']=d['_glucose']
                normalized['glucose_source']='FreeStyle Libre H; source publication';normalized['availability_time']=pd.NaT
                for field in ['HR','steps','activity','sleep','SpO2','carbs_g','protein_g','fat_g','fiber_g','calories']:normalized[field]=np.nan
                mealcols=[c for c in ['Dietary intake','饮食'] if c in d]
                normalized['meal_event']=join_events(d,mealcols)
                normalized['insulin_event']=join_events(d,[c for c in d if ('insulin' in c.lower() and 'non-insulin' not in c.lower()) or '胰岛素' in c])
                normalized['medication_event']=join_events(d,['Non-insulin hypoglycemic agents'] if 'Non-insulin hypoglycemic agents' in d else [])
                normalized['data_quality']='raw event receipt unknown; no nutritional macros inferred'
                normalized['timezone']=pd.NA
                normalized['availability_basis']='unknown';normalized['is_observed_glucose']=True
                normalized.to_parquet(p,index=False);p.chmod(0o600)
            cohort_report[status]['dynamic_available']={c:{'rows':v,'participants':len(dynamic_subjects[c])} for c,v in dynamic_counts.items()}
            cohort_report[status]['dynamic_rows']=sum(len(frames[pid]) for pid in by_id)
    safe_json(OUT/'static_shanghai.json',clinical_all)
    safe_json(OUT/'raw_recording_map.json',mappings)
    return series,frames,cohort_report


def cgmacros():
    series=[];frames={};static=[]
    with zipfile.ZipFile(ROOT/'CGMacros/CGMacros_dateshifted365.zip') as z:
        bio=pd.read_csv(z.open(next(n for n in z.namelist() if Path(n).name=='bio.csv')),keep_default_na=False)
        bio['_id']=bio.subject.astype(str)
        a=pd.to_numeric(bio['A1c PDL (Lab)'],errors='coerce')
        status=dict(zip(bio['_id'],np.where(a>6.4,'T2D',np.where(a>=5.7,'Prediabetes','No diabetes'))))
        report={'participants':len(bio),'diabetes_status_counts':dict(Counter(status.values())),
            'static_missingness':frame_missingness(bio,'_id'),
            'T2D_static_missingness':frame_missingness(bio[a>6.4],'_id'),
            'HbA1c_range':[float(a.min()),float(a.max())],
            'HbA1c_unit_resolution':'dictionary says mmol/mol; original paper Table 2 says percent; preserve both',
            'food_photos':sum(n.lower().endswith('.jpg') for n in z.namelist()),
            'published_CGM_interpolated':True,
            'activity_column_variants':{'METs':'dictionary x10 MET; paper 20-minute mean filter with unspecified alignment','Intensity':'11 participants; undocumented units, preserve without conversion'}}
        for _,row in bio.iterrows():
            static.append(asdict(ParticipantStatic('CGMacros:'+str(row.subject),'CGMacros',status[str(row.subject)],
                age=float(row['Age']),sex=str(row['Gender']),BMI=float(row['BMI']),HbA1c=float(row['A1c PDL (Lab)']),
                fasting_glucose=float(row['Fasting GLU - PDL (Lab)']),fasting_insulin=float(row['Insulin ']),
                timing_basis='screening before CGM placement per original protocol; clock-only timestamps locally',
                original_fields=row.drop('_id').to_dict(),original_units={'HbA1c':'% (paper); dictionary contradiction preserved','fasting_insulin':'mcU/mL'},
                data_quality=['exact lab receipt absent','medication exposure absent'])))
        dynamic=defaultdict(Counter); linear=defaultdict(list); pairdiff=[];fusion=defaultdict(Counter)
        for n in z.namelist():
            if not n.endswith('.csv') or 'CGMacros-' not in Path(n).stem:continue
            sid=str(int(Path(n).stem.split('-')[-1]));pid='CGMacros:'+sid
            d=pd.read_csv(z.open(n),keep_default_na=False)
            if 'Mets' in d and 'METs' not in d: d=d.rename(columns={'Mets':'METs'})
            d['_time_ns']=times(d.Timestamp);frames[pid]=d
            pop=status[sid]
            for c in d:
                if c.startswith('_'):continue
                mask=present(d[c]);dynamic[c]['rows']+=int(mask.sum());dynamic[c]['participants']+=int(mask.any())
                dynamic[c]['T2D_rows']+=int(mask.sum()) if pop=='T2D' else 0
                dynamic[c]['T2D_participants']+=int(mask.any()) if pop=='T2D' else 0
            for field,device in [('Dexcom GL','Dexcom'),('Libre GL','Libre')]:
                v0=pd.to_numeric(d[field],errors='coerce');t,v,q=prepare(d['_time_ns'],v0,1)
                s=Series(pid,'CGMacros_'+device,pop,t,v,1,q);series.append(s)
                vfinite=v0.dropna().to_numpy();dd=np.diff(vfinite,2)
                linear[device].append({'noninteger':int((vfinite%1!=0).sum()),'values':len(vfinite),
                    'linear_triples':int(np.isclose(dd,0,atol=1e-8).sum()),'triples':len(dd)})
            dex=pd.to_numeric(d['Dexcom GL'],errors='coerce').notna()
            libre=pd.to_numeric(d['Libre GL'],errors='coerce').notna()
            hr=pd.to_numeric(d.HR,errors='coerce').notna()
            activity=pd.to_numeric(d['METs'] if 'METs' in d else d['Intensity'],errors='coerce').notna()
            mt=d.loc[present(d['Meal Type']),'_time_ns'].to_numpy()
            idx=np.searchsorted(mt,d['_time_ns'],side='right')-1
            recent=(idx>=0)&((d['_time_ns'].to_numpy()-mt[np.maximum(idx,0)])<=120*MINUTE) if len(mt) else np.zeros(len(d),bool)
            for device,glu in [('Dexcom',dex),('Libre',libre)]:
                for label,mask in [('CGM+HR',glu&hr),('CGM+activity',glu&activity),('CGM+recent_meal',glu&recent),
                    ('CGM+HR+activity+recent_meal',glu&hr&activity&recent),('CGM+clinical_age_BMI_HbA1c',glu),
                    ('CGM+clinical+medication',np.zeros(len(d),bool))]:
                    key=device+':'+label
                    fusion[key]['rows']+=int(mask.sum());fusion[key]['participants']+=int(mask.any())
                    if pop=='T2D':fusion[key]['T2D_rows']+=int(mask.sum());fusion[key]['T2D_participants']+=int(mask.any())
            vv=pd.to_numeric(d['Dexcom GL'],errors='coerce')-pd.to_numeric(d['Libre GL'],errors='coerce')
            pairdiff.extend(vv.dropna().tolist())
            p=OUT/'canonical'/'CGMacros'/(sid+'.parquet');p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            d['participant_id']=pid;d['source_dataset']='CGMacros';d['availability_time']=pd.NaT
            d['availability_basis']='unknown';d['is_observed_glucose']=pd.NA
            d['data_quality']='published glucose interpolation; MET 20-minute filter alignment unspecified'
            # Wide original device values retained; do not merge two sensors or fabricate missing signals.
            d=d.astype({c:'string' for c in d if c not in ['_time_ns','availability_time','is_observed_glucose']})
            d.to_parquet(p,index=False);p.chmod(0o600)
            for field,device in [('Dexcom GL','Dexcom'),('Libre GL','Libre')]:
                normalized=d.copy()
                normalized['timestamp']=pd.to_datetime(d['_time_ns'])
                normalized['glucose_mg_dl']=pd.to_numeric(d[field],errors='coerce')
                normalized['glucose_source']=device
                for target,col in [('HR','HR'),('steps','Steps'),('activity','METs' if 'METs' in d else 'Intensity'),('carbs_g','Carbs'),('protein_g','Protein'),('fat_g','Fat'),('fiber_g','Fiber'),('calories','Calories')]:
                    normalized[target]=pd.to_numeric(d[col],errors='coerce') if col in d else np.nan
                for target in ['sleep','SpO2']:normalized[target]=np.nan
                for target in ['insulin_event','medication_event','timezone']:normalized[target]=pd.NA
                normalized['meal_event']=d['Meal Type'].where(present(d['Meal Type']),pd.NA)
                normalized['source_file']=n;normalized['recording_id']=Path(n).stem
                pp=p.parent/(sid+'_'+device+'.parquet');normalized.to_parquet(pp,index=False);pp.chmod(0o600)
        for fn in ['microbes.csv','gut_health_test.csv']:
            d=pd.read_csv(z.open(next(n for n in z.namelist() if Path(n).name==fn)),keep_default_na=False)
            report[fn]={'rows':len(d),'columns':len(d.columns),'missing_cells':int(sum((~present(d[c])).sum() for c in d)),
                        'subject_id_overlap':int(d.iloc[:,0].astype(str).isin(bio.subject.astype(str)).sum())}
        report['dynamic_available']={k:dict(v) for k,v in dynamic.items()}
        report['dynamic_total_rows']=sum(len(d) for d in frames.values())
        report['dynamic_T2D_total_rows']=sum(len(d) for pid,d in frames.items() if status[pid.split(':')[1]]=='T2D')
        report['lipid_error_sentinels']={c:{'all':int((pd.to_numeric(bio[c],errors='coerce')==sentinel).sum()),
            'T2D':int(((pd.to_numeric(bio[c],errors='coerce')==sentinel)&(a>6.4)).sum())} for c,sentinel in [('LDL (Cal)',800),('VLDL (Cal)',400)]}
        report['fusion']={k:dict(v) for k,v in fusion.items()}
        report['interpolation_diagnostic']={device:{k:sum(d[k] for d in arr) for k in arr[0]} for device,arr in linear.items()}
        report['paired_device_difference_mg_dl']={'paired_minutes':len(pairdiff),'mean':float(np.mean(pairdiff)),
            'median':float(np.median(pairdiff)),'p05_p95':np.quantile(pairdiff,[.05,.95]).tolist()}
    safe_json(OUT/'static_cgmacros.json',static)
    return series,frames,report


def benchmark(cg_series):
    from tsfile import TsFileReader
    series=[]; assignments=[];report={};by_key=defaultdict(list)
    for path in sorted((ROOT/'glucofm_bench').glob('*.tsfile')):
        with TsFileReader(str(path)) as reader:
            schemas=reader.get_all_table_schemas()
            assert len(schemas)==1
            table=next(iter(schemas))
            with reader.query_table(table,['dataset','subject_id','split','bgvalue'],batch_size=100000) as query:
                while (batch:=query.read_arrow_batch()) is not None:
                    df=batch.to_pandas()
                    for key,g in df.groupby(['dataset','subject_id','split'],sort=False):by_key[key].append(g)
        report[path.name]={'sha256':hash_file(path)}
    overlap=defaultdict(set)
    for (ds,sid,split),parts in by_key.items():
        df=pd.concat(parts,ignore_index=True)
        # Actual TsFile physical time is milliseconds, verified from 300000 deltas.
        t,v,q=prepare(df.time.to_numpy(dtype='int64')*1_000_000,df.bgvalue,5)
        pid='GlucoFM:'+str(ds)+':'+str(sid)
        overlap[(ds,sid)].add(split)
        series.append(Series(pid,'GlucoFM_'+str(ds), 'T2D' if ds=='ShanghaiT2DM' else 'unknown',t,v,5,q))
        assignments.append({'participant_id':pid,'source_dataset':ds,'source_subject_id':str(sid),'split':split,
                            'rows':len(df),'time_min':int(df.time.min()),'time_max':int(df.time.max())})
    report['source_counts']={ds:{split:sum(k[0]==ds and k[2]==split for k in by_key) for split in ['train','test']} for ds in sorted({k[0] for k in by_key})}
    report['cross_split_subjects']=sum(len(v)>1 for v in overlap.values())
    split_details=[]
    for key,splits in overlap.items():
        if len(splits)<2:continue
        ds,sid=key
        a=pd.concat(by_key[(ds,sid,'train')],ignore_index=True)
        b=pd.concat(by_key[(ds,sid,'test')],ignore_index=True)
        shared=len(np.intersect1d(a.time,b.time))
        split_details.append({'participant_id':'GlucoFM:'+str(ds)+':'+str(sid),
            'shared_timestamps':shared,'test_starts_after_train':bool(b.time.min()>a.time.max())})
    report['cross_split_shared_timestamps']=sum(d['shared_timestamps'] for d in split_details)
    report['cross_split_chronological_subjects']=sum(d['test_starts_after_train'] for d in split_details)
    safe_json(OUT/'benchmark_cross_split.json',split_details)
    report['participant_series']=len(by_key);report['unique_source_subjects']=len(overlap)
    report['observations']=sum(s.quality['rows'] for s in series)
    report['physical_timestamp_unit']='milliseconds; all raw delta modes inspected'
    report['no_validation_split']=True
    # Coalesce supplied temporal partitions for audit counts only, never count a person twice.
    grouped=defaultdict(list)
    for s in series:grouped[s.participant_id].append(s)
    statuses={s.participant_id:s.diabetes_status for s in cg_series}
    merged=[]
    for pid,parts in grouped.items():
        first=parts[0]
        t,v,q=prepare(np.concatenate([s.time for s in parts]),np.concatenate([s.glucose for s in parts]),5)
        status=first.diabetes_status
        if first.dataset=='GlucoFM_CGMacros':
            match=re.search(r'(\d+)$',pid)
            if match: status=statuses.get('CGMacros:'+str(int(match.group(1))),'unknown')
        merged.append(Series(pid,first.dataset,status,t,v,5,q))
    series=merged
    safe_json(OUT/'benchmark_subjects.json',assignments)
    return series,report,assignments


def tooling(raw_series,bench_assignments):
    report={}; extracted=[];meta={};split_report={};provenance=[]
    raw={(s.dataset,s.participant_id):s for s in raw_series}
    with zipfile.ZipFile(ROOT/'Glucose-ML-Project-main.zip') as z:
        names=z.namelist()
        for n in names:
            if '/3_Glucose-ML-collection/' not in n or not n.endswith('-metadata.csv'):continue
            d=pd.read_csv(z.open(n),dtype={'person_id':str},keep_default_na=False);ds=n.split('/')[2];meta[ds]=d
            report[ds]={'metadata_rows':len(d),'unique_metadata_subjects':d.person_id.nunique(),
                'status_counts':d.diabetes_type.value_counts().to_dict(),
                'metadata_fields':d.columns.tolist(),'cgm_files':0,'glucose_observations':0,
                'cgm_devices':d.CGM_type.value_counts().to_dict() if 'CGM_type' in d else {},
                'metadata_missingness':{c:int((~present(d[c])).sum()) for c in d if c!='person_id'}}
            readme=next((name for name in names if name=='Glucose-ML-Project-main/3_Glucose-ML-collection/'+ds+'/README.md'),None)
            if readme:
                prose=z.read(readme).decode()
                report[ds]['source_urls']=re.findall(r'https://[^\s)<>]+',prose)
                start=prose.find('**License Details**');end=prose.find('**Publication Reference',start)
                report[ds]['license_note']=prose[start:end if end>start else None].replace('**License Details**','').strip() if start>=0 else 'not supplied'
        for n in names:
            if '/3_Glucose-ML-collection/' not in n or '-extracted-glucose-files/' not in n or not n.endswith('.csv'):continue
            ds=n.split('/')[2];sid=Path(n).stem;d=pd.read_csv(z.open(n),keep_default_na=False)
            if not {'timestamp','glucose_value_mg_dl'}.issubset(d.columns):raise ValueError('Unexpected tooling CGM schema')
            clock_only=bool(d.timestamp.astype(str).str.fullmatch(r'\d{1,2}:\d{2}:\d{2}').all())
            ambiguous=clock_only or ds=='Park_2025'
            # Clock-only days and repeated relative challenge clocks cannot be silently anchored.
            ts=np.full(len(d),np.iinfo(np.int64).min,dtype='int64') if ambiguous else times(d.timestamp)
            t,v,q=prepare(ts,pd.to_numeric(d.glucose_value_mg_dl,errors='coerce'),15 if ds.startswith('Shanghai') else 5)
            if ambiguous:
                report[ds]['timestamp_limitation']='clock-only without day chronology' if clock_only else 'relative minutes with repeated challenge sessions; session identity not exported'
                clock=pd.to_timedelta(d.timestamp.astype(str)).astype('int64').to_numpy() if clock_only else pd.to_numeric(d.timestamp,errors='coerce').to_numpy()*MINUTE
                qc=report[ds].setdefault('unresolved_clock_qc',Counter())
                qc['rows']+=len(clock);qc['repeated_clock_rows']+=len(clock)-len(np.unique(clock))
                qc['backward_clock_intervals']+=int((np.diff(clock)<0).sum())
            m=meta[ds];status=str(m.loc[m.person_id.astype(str)==sid,'diabetes_type'].iloc[0]) if (m.person_id.astype(str)==sid).any() else 'unknown'
            period=float(np.median(np.diff(t)/MINUTE)) if len(t)>1 else 5
            t,v,q=prepare(ts,pd.to_numeric(d.glucose_value_mg_dl,errors='coerce'),period)
            s=Series('GlucoseML:'+ds+':'+sid,'GlucoseML_'+ds,status,t,v,period,q)
            extracted.append(s);report[ds]['cgm_files']+=1;report[ds]['glucose_observations']+=q['rows']
            report[ds].setdefault('period_distribution',Counter())[str(period)]+=1
            raw_id=None
            if ds=='ShanghaiT2DM':raw_id='ShanghaiT2D:'+normalize_id(sid)
            elif ds=='ShanghaiT1DM':raw_id='ShanghaiT1D:'+normalize_id(sid)
            elif ds in ['CGMacros_Dexcom','CGMacros_Libre']:raw_id='CGMacros:'+str(int(sid.split('-')[-1]))
            if raw_id:
                source_ds={'ShanghaiT2DM':'ShanghaiT2D','ShanghaiT1DM':'ShanghaiT1D'}.get(ds,ds)
                reference=raw.get((source_ds,raw_id))
                match=None
                if reference is not None:
                    shared,ia,ib=np.intersect1d(reference.time,t,return_indices=True)
                    same=np.isclose(reference.glucose[ia],v[ib],atol=1e-4)
                    match={'shared_timestamps':len(shared),'equal_values':int(same.sum()),
                           'raw_observations':len(reference.time),'copy_observations':len(t),
                           'copy_exact_equal':len(shared)==len(t)==len(reference.time) and bool(same.all())}
                provenance.append({'raw_participant_id':raw_id,'harmonized_participant_id':s.participant_id,
                    'basis':'matching stable source ID; value/timestamp comparison if same device', 'comparison':match})
        n='Glucose-ML-Project-main/7_Open-ML-Resources/participant_splits.csv'
        splits=pd.read_csv(z.open(n),keep_default_na=False)
        split_report={'rows':len(splits),'columns':splits.columns.tolist(),
            'cross_split_subjects':int((splits.groupby(['dataset','person_id']).split_assignment.nunique()>1).sum()),
            'duplicate_subject_rows':int(splits.duplicated(['dataset','person_id']).sum()),
            'counts':{ds:g.split_assignment.value_counts().to_dict() for ds,g in splits.groupby('dataset')}}
        split_report['prepared_copy_csvs']=sum('/7_Open-ML-Resources/Open-ML-Ready-Datasets/' in n and n.endswith('.csv') for n in names)
        split_report['case_study_csvs']=sum('/6_Case-study/' in n and n.endswith('.csv') for n in names)
        split_report['nested_zip_archives']=sum(n.endswith('.zip') for n in names)
        # Nested archives are usually data copies; inventory each without running any code.
        split_report['nested_archives']=[]
        for n in names:
            if n.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(z.read(n))) as inner:
                    split_report['nested_archives'].append({'source_group':n.split('/')[2],
                        'file_count':len(inner.namelist()),'extensions':dict(Counter(Path(k).suffix for k in inner.namelist() if not k.endswith('/')))})
    # Benchmark IDs carry original source labels. Exact IDs matched privately; no cohort re-linking.
    source_map={'ShanghaiT2DM':'ShanghaiT2D','ShanghaiT1DM':'ShanghaiT1D','CGMacros':'CGMacros'}
    for item in bench_assignments:
        ds=item['source_dataset'];sid=item['source_subject_id'];raw_id=None
        if ds in ['ShanghaiT2DM','ShanghaiT1DM']:raw_id=source_map[ds]+':'+normalize_id(sid)
        elif ds=='CGMacros':
            match=re.search(r'(\d+)$',sid)
            if match:raw_id='CGMacros:'+str(int(match.group(1)))
        if raw_id:
            provenance.append({'raw_participant_id':raw_id,'harmonized_participant_id':item['participant_id'],
                'split':item['split'],'basis':'recoverable source ID; cohort duplicate regardless of altered values',
                'present_in_raw':raw_id in {s.participant_id for s in raw_series}})
    safe_json(OUT/'provenance_subject_map.json',provenance)
    safe_json(OUT/'tooling_splits.json',split_report)
    counts={'sources':report,'splits':split_report,
        'overlap':{'mapped_rows':len(provenance),
            'tooling_raw_map':dict(Counter(p['harmonized_participant_id'].split(':')[1] for p in provenance if p['harmonized_participant_id'].startswith('GlucoseML:'))),
            'tooling_exact_matches':sum(p.get('comparison') is not None and p['comparison']['copy_exact_equal'] for p in provenance),
            'benchmark_raw_map':dict(Counter(p['harmonized_participant_id'].split(':')[1] for p in provenance if p['harmonized_participant_id'].startswith('GlucoFM:'))),
            'benchmark_raw_match':sum(p.get('present_in_raw',False) for p in provenance)}}
    return extracted,counts


def meal_personalization(series,frames):
    rows=[]
    for s in series:
        d=frames.get(s.participant_id)
        if d is None:continue
        if s.dataset.startswith('Shanghai'):
            cols=[c for c in ['Dietary intake','饮食'] if c in d]
            meal=np.logical_or.reduce([present(d[c]).to_numpy() for c in cols]) if cols else np.zeros(len(d),bool)
            carb=macro=np.zeros(len(d),bool)
        else:
            meal=present(d['Meal Type']).to_numpy()
            carb=pd.to_numeric(d.Carbs,errors='coerce').notna().to_numpy()
            macro=np.logical_and.reduce([pd.to_numeric(d[c],errors='coerce').notna().to_numpy() for c in ['Carbs','Protein','Fat','Fiber','Calories']])
        mt=np.unique(d.loc[meal,'_time_ns'].to_numpy());mt=mt[mt!=np.iinfo(np.int64).min]
        history,masks,targets=window_masks(s,60)
        idx=np.searchsorted(mt,s.time,side='right')-1
        recent=(idx>=0)&((s.time-mt[np.maximum(idx,0)])<=120*MINUTE) if len(mt) else np.zeros(len(s.time),bool)
        mcarb=np.unique(d.loc[meal&carb,'_time_ns']);mmacro=np.unique(d.loc[meal&macro,'_time_ns'])
        recentcarb=np.zeros(len(s.time),bool);recentmacro=np.zeros(len(s.time),bool)
        for arr,mask in [(mcarb,recentcarb),(mmacro,recentmacro)]:
            if len(arr):
                pos=np.searchsorted(arr,s.time,side='right')-1
                mask[:]=(pos>=0)&((s.time-arr[np.maximum(pos,0)])<=120*MINUTE)
        complete_meals=0;pre_meals=0;responses=[]
        for meal_time in mt:
            i=np.searchsorted(s.time,meal_time,side='right')-1
            if i<0 or meal_time-s.time[i]>s.period*MINUTE/2:continue
            if history[i]:pre_meals+=1
            if history[i] and masks[120][i]:
                complete_meals+=1
                j=targets[120][i]
                responses.append((meal_time,int(s.time[j]),float(np.max(s.glucose[i+1:j+1])-s.glucose[i])))
        personalized=[]
        for time,end,response in responses:
            past=[v for tm,te,v in responses if te<=time and tm<time]
            if len(past)>=3:personalized.append(response>float(np.quantile(past,.9)))
        dates=s.time//(MINUTE*1440);hours=(s.time//(MINUTE*60))%24
        night_dates=np.unique(dates[hours<6])
        nights=sum(np.sum((dates==day)&(hours<6))>=4 for day in night_dates)
        repeated_hours=sum(len(np.unique(dates[hours==h]))>=3 for h in range(24))
        rows.append({'participant_id':s.participant_id,'dataset':s.dataset,'status':s.diabetes_status,
            'meal_events':len(mt),'meals_carbs':len(mcarb),'meals_complete_macros':len(mmacro),
            'recent_meal_landmarks':int((history&recent).sum()),
            'recent_meal_carbs_landmarks':int((history&recentcarb).sum()),
            'recent_meal_macros_landmarks':int((history&recentmacro).sum()),
            'recent_meal_complete_2h_landmarks':int((masks[120]&recent).sum()),
            'pre_meal_60m_history':pre_meals,'meal_complete_2h_trajectory':complete_meals,
            'night_baseline_days':int(nights),'hours_seen_on_3_days':int(repeated_hours),
            'personal_response_eligible_meals':len(personalized),'personal_response_positive_meals':sum(personalized)})
    safe_json(OUT/'participant_meal_personalization.json',rows)
    report={}
    for ds in sorted({r['dataset'] for r in rows}):
        report[ds]={}
        for pop in ['all','T2D']:
            rr=[r for r in rows if r['dataset']==ds and (pop=='all' or r['status']=='T2D')]
            fields=[k for k in rows[0] if k not in ['participant_id','dataset','status']]
            report[ds][pop]={k:sum(r[k] for r in rr) for k in fields}
            report[ds][pop].update(participants=len(rr),participants_with_meals=sum(r['meal_events']>0 for r in rr),
                participants_10_meals=sum(r['meal_events']>=10 for r in rr),
                participants_complete_2h_meal=sum(r['meal_complete_2h_trajectory']>0 for r in rr),
                meal_count_quantiles=np.quantile([r['meal_events'] for r in rr],[0,.25,.5,.75,1]).tolist() if rr else [],
                participants_3_nights=sum(r['night_baseline_days']>=3 for r in rr),
                participants_24_repeated_hours=sum(r['hours_seen_on_3_days']==24 for r in rr),
                personalized_positive_participants=sum(r['personal_response_positive_meals']>0 for r in rr))
    return report


def excursion(series):
    records=[]
    for s in series:
        if s.diabetes_status!='T2D':continue
        _,masks,targets=window_masks(s,120)
        if not masks:continue
        ids=np.flatnonzero(masks[120]);j=targets[120][ids]
        peak=np.array([np.max(s.glucose[i+1:end+1]) for i,end in zip(ids,j)])
        current=s.glucose[ids]
        sustained180=np.array([np.any((s.glucose[i+1:end]>180)&(s.glucose[i+2:end+1]>180)) for i,end in zip(ids,j)],bool)
        for name,outcome,predictor,eligible in [
            ('peak>180',peak>180,current>180,np.ones(len(ids),bool)),
            ('incident_peak>180',peak>180,current>180,current<=180),
            ('incident_two_readings>180',sustained180,current>180,current<=180),
            ('peak>250',peak>250,current>250,np.ones(len(ids),bool)),
            ('incident_peak>250',peak>250,current>250,current<=250),
            ('rise>=30',peak-current>=30,current>180,np.ones(len(ids),bool)),
            ('rise>=50',peak-current>=50,current>180,np.ones(len(ids),bool))]:
            y=outcome[eligible];p=predictor[eligible]
            records.append({'participant_id':s.participant_id,'dataset':s.dataset,'definition':name,
                'landmarks':len(y),'positive_landmarks':int(y.sum()),
                'tp':int((y&p).sum()),'fp':int((~y&p).sum()),'tn':int((~y&~p).sum()),'fn':int((y&~p).sum())})
    safe_json(OUT/'excursion_by_participant.json',records)
    report={}
    for ds in sorted({r['dataset'] for r in records}):
        report[ds]={}
        for name in sorted({r['definition'] for r in records}):
            rr=[r for r in records if r['dataset']==ds and r['definition']==name]
            d={k:sum(r[k] for r in rr) for k in ['landmarks','positive_landmarks','tp','fp','tn','fn']}
            d['positive_participants']=sum(r['positive_landmarks']>0 for r in rr)
            d['participants']=sum(r['landmarks']>0 for r in rr)
            d['prevalence']=d['positive_landmarks']/d['landmarks'] if d['landmarks'] else None
            d['diagnostic_only_current_threshold_agreement']=(d['tp']+d['tn'])/d['landmarks'] if d['landmarks'] else None
            report[ds][name]=d
    return report


def auxiliary():
    report={}
    with zipfile.ZipFile(ROOT/'aibhs_datasets-main.zip') as z:
        for group in ['pima_diabetes','diabetes_130_hospitals','brfss','nhanes']:
            entries=[]
            for n in z.namelist():
                if '/'+group+'/' not in n or n.endswith('/'):continue
                if n.endswith('.csv'):
                    d=pd.read_csv(z.open(n),low_memory=False)
                    if group=='pima_diabetes':
                        d=pd.read_csv(z.open(n),header=None,names=['pregnancies','glucose','blood_pressure','skin_thickness','insulin','BMI','pedigree','age','diabetes_outcome'])
                    cols=d.columns.tolist()
                    idcol=next((c for c in ['patient_nbr','SEQN','seqn','SEQNO'] if c in d),None)
                    entries.append({'format':'csv','rows':len(d),'columns':cols,
                        'distinct_participants':int(d[idcol].nunique()) if idcol else None,
                        'participant_identity':'source ID' if idcol else 'row identity only',
                        'diabetes_categories':d['DIABETE4'].value_counts().to_dict() if 'DIABETE4' in d else (d['diabetes_outcome'].value_counts().to_dict() if 'diabetes_outcome' in d else None),
                        'T2D_confirmed_count':int(((d.DIABETE4==1)&(d.DIABTYPE==2)).sum()) if {'DIABETE4','DIABTYPE'}.issubset(d.columns) else None,
                        'forecast_windows':0,'reason':'no timestamped CGM trajectory'})
                elif n.endswith('.xpt'):
                    d=pd.read_sas(io.BytesIO(z.read(n)),format='xport');entries.append({'format':'xpt','rows':len(d),'columns':d.columns.tolist(),
                        'distinct_participants':int(d.SEQN.nunique()) if 'SEQN' in d else None,'forecast_windows':0,'reason':'cross-sectional survey component'})
                elif n.endswith(('.data','.txt')):
                    lines=z.read(n).decode(errors='replace').splitlines()
                    entries.append({'format':Path(n).suffix,'nonempty_lines':sum(bool(l.strip()) for l in lines), 'forecast_windows':0})
            report[group]=entries
        # Other cardiac/static survey datasets with metabolic fields are inspected too.
        other={}
        for n in z.namelist():
            if not n.endswith('.csv') or any('/'+g+'/' in n for g in ['pima_diabetes','diabetes_130_hospitals','brfss','nhanes','eicu_demo','mimic_iv_demo']):continue
            d=pd.read_csv(z.open(n),low_memory=False)
            fields=[c for c in d if any(k in c.lower() for k in ['glucose','diabet','hba1c','insulin'])]
            if fields:other[n.split('/')[1]]={'rows':len(d),'metabolic_fields':fields,
                'nonmissing':{c:int(present(d[c]).sum()) for c in fields},'forecast_windows':0,'reason':'static cardiac/general-health table without CGM'}
        report['other_aibhs_metabolic']=other
        # Hospital demo resources may contain sparse glucose labs, but are unrelated cohorts.
        for group in ['eicu_demo','mimic_iv_demo']:
            mimic_items=set()
            if group=='mimic_iv_demo':
                itemname=next(n for n in z.namelist() if '/mimic_iv_demo/' in n and n.endswith('/d_labitems.csv.gz'))
                itemdf=pd.read_csv(io.BytesIO(z.read(itemname)),compression='gzip')
                mimic_items=set(itemdf.loc[itemdf.label.astype(str).str.contains('glucose',case=False,regex=False),'itemid'])
            glucose_rows=0;subjects=set();tables=[];patient_count=None;stay_person={}
            for n in z.namelist():
                if '/'+group+'/' not in n or not n.endswith('.csv.gz'):continue
                basename=Path(n).name
                if basename not in ['lab.csv.gz','labevents.csv.gz','patients.csv.gz','patient.csv.gz']:continue
                d=pd.read_csv(io.BytesIO(z.read(n)),compression='gzip',low_memory=False)
                tables.append({'table':basename,'rows':len(d),'fields':d.columns.tolist()})
                if basename=='patient.csv.gz':
                    patient_count=int(d.uniquepid.nunique());stay_person=dict(zip(d.patientunitstayid.astype(str),d.uniquepid.astype(str)))
                elif basename=='patients.csv.gz':patient_count=int(d.subject_id.nunique())
                if group=='eicu_demo' and 'labname' in d:
                    mask=d.labname.astype(str).str.contains('glucose',case=False,regex=False)
                    glucose_rows+=int(mask.sum());subjects.update(d.loc[mask,'patientunitstayid'].astype(str))
                elif group=='mimic_iv_demo' and 'itemid' in d:
                    mask=d.itemid.isin(mimic_items);glucose_rows+=int(mask.sum());subjects.update(d.loc[mask,'subject_id'].astype(str))
            report[group]={'tables':tables,'glucose_lab_rows':glucose_rows,
                'glucose_stays_or_subjects':len(subjects),'dataset_distinct_people':patient_count,
                'glucose_people':len({stay_person[s] for s in subjects if s in stay_person}) if group=='eicu_demo' else len(subjects),'forecast_windows':0,
                'reason':'ICU demo lab records; no verified T2D CGM path; excluded from primary forecasting'}
    with zipfile.ZipFile(ROOT/'medical.zip') as z:
        n=next(n for n in z.namelist() if n.endswith('/glucose.csv'));d=pd.read_csv(z.open(n))
        report['medical_glucose']={'rows':len(d),'fields':d.columns.tolist(),
            'missing_cells':int(d.isna().sum().sum()),'distinct_participants':int(d.user_id.nunique()),
            'glucose_unit':'mmol/L per supplied PerHeart descriptor; mg/dL = value * 18.0182','forecast_windows':0,
            'reason':'isolated measurements; source/diabetes classification not independently established; no dense CGM'}
        normalized=d.copy();normalized['participant_id']='PerHeart:'+d.user_id.astype(str)
        normalized['glucose_mg_dl']=pd.to_numeric(d.value,errors='coerce')*18.0182
        normalized['source_dataset']='PerHeart';normalized['availability_time']=pd.NaT
        p=OUT/'canonical'/'perheart_spot_glucose.parquet';p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        normalized.to_parquet(p,index=False);p.chmod(0o600)
    # Inspect prior safety-reviewed inventory for peripheral heart cohorts; no new cross-cohort joins.
    for path in [ROOT/'personal_questionnaires.csv',ROOT/'pone.0190323.s017.csv']:
        d=pd.read_csv(path,sep=';' if path.name.startswith('pone') else ',')
        fields=[c for c in d if any(k in c.lower() for k in ['glucose','diabet','hba1c','insulin'])]
        idcol=next((c for c in ['Patient_ID','user_id','ID'] if c in d),None)
        report[path.name]={'rows':len(d),'participants':int(d[idcol].nunique()) if idcol else None,'metabolic_fields':fields,
            'forecast_windows':0,'reason':'questionnaire/daily cardiac summaries without CGM'}
    with zipfile.ZipFile(next(ROOT.glob('hospitalized*.zip'))) as z:
        d=pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith('/dat.csv'))))
        fields=[c for c in d if any(k in c.lower() for k in ['glucose','diabet','hba1c','insulin'])]
        report['Zigong_HF']={'rows':len(d),'metabolic_fields':fields,
            'nonmissing':{c:int(present(d[c]).sum()) for c in fields},'forecast_windows':0,'reason':'HF admission records and spot labs, no CGM'}
    report['synthea']={'synthetic':True,'real_T2D_participants':0,'role':'FHIR integration only; never real-data validation'}
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
    print('Metabolic audit: reading raw Shanghai recordings (no identifiers logged).',flush=True)
    sh,shframes,shreport=shanghai()
    print('Metabolic audit: reading CGMacros published interpolated grids.',flush=True)
    cg,cgframes,cgreport=cgmacros()
    print('Metabolic audit: checking actual benchmark splits and provenance.',flush=True)
    bench,breport,assignments=benchmark(cg)
    tools,treport=tooling(sh+cg,assignments)
    aliases={'AZT1D':'AZT1D','BIG_IDEA_LAB':'BIGIDEAs','Bris-T1D Open':'Bris-T1D_Open',
        'CGMacros':'CGMacros_Dexcom','Colas2019':'Colas_2019','D1NAMO':'D1NAMO','HUPA-UCM':'HUPA-UCM',
        'Hall2018':'Hall_2018','ShanghaiT1DM':'ShanghaiT1DM','ShanghaiT2DM':'ShanghaiT2DM','T1DM-UOM':'T1D-UOM'}
    copies={(s.dataset.removeprefix('GlucoseML_'),s.participant_id.split(':',2)[2]):s for s in tools}
    source_copy_map=[]
    for s in bench:
        source=s.dataset.removeprefix('GlucoFM_');sid=s.participant_id.split(':',2)[2]
        project=aliases[source];reference=copies.get((project,sid))
        if source=='BIG_IDEA_LAB' and reference is None:
            # Converted IDs prepend a source label to the original three-character ID.
            reference=copies.get((project,sid.rsplit('_',1)[-1]))
        if source=='CGMacros' and reference is None:
            match=re.search(r'(\d+)$',sid)
            if match:
                reference=next((a for a in tools if a.dataset=='GlucoseML_CGMacros_Dexcom' and int(a.participant_id.split('-')[-1])==int(match.group(1))),None)
        if reference is not None:
            s.diabetes_status=reference.diabetes_status
            shared,ia,ib=np.intersect1d(s.time,reference.time,return_indices=True)
            source_copy_map.append({'benchmark_participant_id':s.participant_id,'tooling_participant_id':reference.participant_id,
                'source_group':project,'diabetes_status':s.diabetes_status,'shared_timestamps':len(shared),
                'equal_glucose_values':int(np.isclose(s.glucose[ia],reference.glucose[ib],atol=1e-3).sum()),
                'basis':'exact stable same-cohort ID; independent of interpolation/value changes'})
    safe_json(OUT/'tooling_benchmark_subject_map.json',source_copy_map)
    treport['overlap']['benchmark_tooling_people']=len(source_copy_map)
    treport['overlap']['benchmark_tooling_sources']=dict(Counter(a['source_group'] for a in source_copy_map))
    treport['overlap']['benchmark_mapped_disease_labels']=dict(Counter(a['diabetes_status'] for a in source_copy_map))
    series=sh+cg+bench+tools
    safe_json(OUT/'series_quality.json',[{'participant_id':s.participant_id,'dataset':s.dataset,'status':s.diabetes_status,'period':s.period,**s.quality} for s in series])
    groups=defaultdict(list)
    for s in series:groups[s.dataset].append(s)
    print('Metabolic audit: enumerating observed/published-grid windows and descriptive targets.',flush=True)
    feasibility={ds:summarize(ss) for ds,ss in groups.items()}
    report={'policy':{'histories_minutes':HISTORIES,'horizons_minutes':HORIZONS,
        'target_tolerance':'half of actual/audited sampling interval','max_gap':'1.5 audited periods',
        'interpolation':'none added by this audit','availability':'receipt absent: CGM observation-time availability is an offline assumption',
        'certification':'CGMacros and benchmark counts are published-grid bounds, not certified observed-only windows'},
        'input_hashes':{p.as_posix():hash_file(p) for p in ROOT.rglob('*') if p.is_file() and '.cache' not in p.parts},
        'shanghai':shreport,'cgmacros':cgreport,'benchmark':breport,'tooling':treport,
        'forecast':feasibility,'meals_personalization':meal_personalization(sh+cg,{**shframes,**cgframes}),
        'excursions':excursion(sh+cg), 'auxiliary':auxiliary()}
    safe_json(OUT/'aggregate_audit.json',report)
    safe_json(OUT/'canonical'/'source_units.json',json.loads(Path('src/metabolic/source_schema.json').read_text()))
    for p in (OUT/'canonical').rglob('*.parquet'):
        # Verify canonical schema and preserve raw fields without printing records.
        df=pd.read_parquet(p)
        assert 'participant_id' in df and 'availability_time' in df
    # Restrict private outputs, including nested directories.
    for p in OUT.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    print(json.dumps({'aggregate_report':'data/metabolic/aggregate_audit.json',
        'Shanghai_T2D_participants':feasibility['ShanghaiT2D']['T2D']['participants'],
        'CGMacros_T2D_participants':cgreport['diabetes_status_counts']['T2D'],
        'benchmark_cross_split_subjects':breport['cross_split_subjects'],'trained_models':0}),flush=True)

if __name__=='__main__':main()
