"""Read-only D3 Shadow assessment on the 64 D2 model-development people.

This deliberately does not call D2 stage loaders or alter its access ledger.
No calibration or holdout workbook/cache is opened. Unknown source receipts are
an explicit offline availability-at-observation assumption, not measured facts.
"""
import io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic.audit import times,present
from src.metabolic.windows import prepare,MINUTE
from src.metabolic_d2.protocol import ROOT as D2_ROOT,SOURCE,digest,write_json
from .reference import ROOT,register,assert_protected
from .buffer import Buffer
from .events import CGMAdapter,MealAdapter
from .shadow import Shadow

POLICY='Receipt unknown in source; offline assumption: available at observation. Source clock timezone unspecified.'

def development_inputs():
    register()
    manifest=json.loads((D2_ROOT/'participant_manifest.json').read_text())
    allowed=set(manifest['model_development'])
    denied=set(manifest['holdout'])|set(manifest['calibration'])
    if len(allowed)!=64 or allowed&denied:raise RuntimeError('Invalid development-only access scope')
    with np.load(D2_ROOT/'cache/development.npz',allow_pickle=False) as z:
        landmarks={key:z[key] for key in ['pid','time','warm','sequence','y']}
    if set(landmarks['pid'])!=allowed:raise RuntimeError('Development cache scope mismatch')
    by_id={};members=[]
    with zipfile.ZipFile(SOURCE) as archive:
        for name in archive.namelist():
            if not name.startswith('Shanghai_T2DM/') or not name.endswith(('.xls','.xlsx')):continue
            pid='ShanghaiT2D:'+Path(name).stem.split('_')[0]
            if pid not in allowed:continue
            if pid in denied:raise PermissionError('Non-development participant access')
            frame=pd.read_excel(io.BytesIO(archive.read(name)),keep_default_na=False)
            column=next(c for c in frame if c.strip() in ['CGM (mg / dl)','CGM'])
            time=times(frame['Date']);glucose=pd.to_numeric(frame[column],errors='coerce').to_numpy()
            meal_columns=[c for c in ['Dietary intake','饮食'] if c in frame]
            meal=np.logical_or.reduce([present(frame[c]).to_numpy() for c in meal_columns]) if meal_columns else np.zeros(len(frame),bool)
            by_id.setdefault(pid,[]).append((time,glucose,time[meal]));members.append(name)
    if set(by_id)!=allowed:raise RuntimeError('Missing development source recordings')
    write_json(ROOT/'development_access.json',{'purpose':'D3 Shadow descriptive quality; no D2 selection or fitting',
        'participant_ids':sorted(allowed),'members_read':members,'participants':64,
        'holdout_records_accessed':False,'calibration_records_accessed':False,
        'source_sha256':digest(SOURCE),'development_cache_sha256':digest(D2_ROOT/'cache/development.npz'),
        'D3_config_sha256':digest('configs/metabolic_d3.json'),'availability_policy':POLICY})
    return by_id,landmarks

def source_clock(ns):return str(np.datetime_as_string(np.datetime64(int(ns),'ns'),unit='ns'))

def assess():
    by_id,landmarks=development_inputs();shadow=Shadow();subjects=[];total=0;method_counts={};qc_totals={}
    for pid,parts in sorted(by_id.items()):
        t,g,qc=prepare(np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts]),15)
        for key in ['rows','unusable_rows','duplicate_extra_rows','conflicting_timestamps','out_of_40_400_rows']:
            qc_totals[key]=qc_totals.get(key,0)+qc[key]
        mt=np.unique(np.concatenate([p[2] for p in parts]));mt=mt[mt!=np.iinfo(np.int64).min]
        # Fixed hourly thinning by ordinal within each person's warm landmarks.
        indices=np.flatnonzero((landmarks['pid']==pid)&landmarks['warm'])[::4]
        buffer=Buffer(pid);position=0;meal_position=0;errors=[];median_errors=[];fallback_slots=0;supports=[]
        for index in indices:
            cutoff=int(landmarks['time'][index])
            while position<len(t) and t[position]<=cutoff:
                stamp=source_clock(t[position])
                buffer.ingest(CGMAdapter().adapt(event_id='development-cgm-'+str(position),patient_id=pid,source='Shanghai native source replay',
                    observed_time=stamp,available_time=stamp,value=float(g[position]),unit='mg/dL',synthetic=False,native_period_minutes=15,
                    context={'availability_policy':POLICY,'source_available_time':None}))
                position+=1
            while meal_position<len(mt) and mt[meal_position]<=cutoff:
                stamp=source_clock(mt[meal_position])
                buffer.ingest(MealAdapter().adapt(event_id='source-meal-'+str(meal_position),patient_id=pid,source='Shanghai dietary annotation',
                    observed_time=stamp,available_time=stamp,synthetic=False,
                    context={'meal_type':None,'availability_policy':POLICY,'source_available_time':None,'nutrition_quantities':None}))
                meal_position+=1
            expected=shadow.predict(buffer,cutoff)
            if not expected['available']:continue
            actual=np.r_[landmarks['sequence'][index,-1],landmarks['y'][index]]
            errors.append(np.abs(np.array(expected['trajectory'])-actual))
            median_errors.append(np.abs(expected['median']-actual))
            method_counts[expected['method']]=method_counts.get(expected['method'],0)+1
            fallback_slots+=sum(s['fallback_to_prefix_median'] for s in expected['support']['hour_slots'])
            supports.append(expected['support']['observations'])
        if errors:
            subjects.append({'count':len(errors),'error_sum':np.sum(errors,axis=0),'median_error_sum':np.sum(median_errors,axis=0),
                'MAE':np.mean(errors,axis=0),'median_MAE':np.mean(median_errors,axis=0),
                'fallback_slots':fallback_slots,'supports':supports})
            total+=len(errors)
        if len(subjects)%16==0:print(json.dumps({'development_participants_assessed':len(subjects),'windows':total}),flush=True)
    if len(subjects)!=64 or not total:raise RuntimeError('Incomplete development quality assessment')
    result={'scope':'DEVELOPMENT-ONLY descriptive expectation quality; not independent validation or D2 model performance',
        'participants':len(subjects),'landmarks':total,'sampling':'Every fourth warm D2 development landmark within each participant; fixed before assessment',
        'horizons_minutes':[0,30,60,90,120],'units':'mg/dL','availability_policy':POLICY,
        'Shadow_MAE_subject_macro':np.mean([s['MAE'] for s in subjects],axis=0).tolist(),
        'Shadow_MAE_pooled':(np.sum([s['error_sum'] for s in subjects],axis=0)/total).tolist(),
        'prefix_median_MAE_subject_macro':np.mean([s['median_MAE'] for s in subjects],axis=0).tolist(),
        'prefix_median_MAE_pooled':(np.sum([s['median_error_sum'] for s in subjects],axis=0)/total).tolist(),
        'methods':method_counts,'hour_slot_fallback_fraction':sum(s['fallback_slots'] for s in subjects)/(5*total),
        'prefix_observation_count_quantiles':np.quantile(np.concatenate([s['supports'] for s in subjects]),[0,.5,1]).tolist(),
        'native_source_QC':qc_totals,'holdout_records_accessed':False,'calibration_records_accessed':False,
        'source_sha256':digest(SOURCE),'config_sha256':digest('configs/metabolic_d3.json'),
        'assessment_code_sha256':digest(__file__),'access_record_sha256':digest(ROOT/'development_access.json')}
    result['shadow_code_sha256']=digest('src/metabolic_d3/shadow.py')
    write_json(ROOT/'shadow_quality.json',result);assert_protected()
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':assess()
