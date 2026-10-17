"""Raw-only native windows and causal prefixes; stage-specific once-built caches."""
import io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic.audit import times,present
from src.metabolic.windows import Series,prepare,window_masks,nearest_indices,MINUTE
from .protocol import ROOT,SOURCE,assert_stage,config,write_json,digest,record_access

HORIZONS=np.array([30,60,90,120])
CGM_NAMES=[f'glucose_lag_{v}m' for v in [120,105,90,75,60,45,30,15,0]]+[
    'slope_15m','slope_30m','slope_60m','slope_120m','recent_mean','recent_median',
    'recent_min','recent_max','recent_sd','acceleration','hour_sin','hour_cos']
PERSONAL_NAMES=['prefix_median','prefix_sd','hour_baseline','night_baseline','typical_slope',
    'past_high_glucose_fraction','current_minus_baseline','current_minus_hour','slope_minus_typical',
    'prefix_observation_hours','hour_baseline_available','night_baseline_available']
MEAL_NAMES=['minutes_since_meal','meal_count_last_120m']
STATIC_NAMES=['age_years','sex_female']


def cgm_features(seq,t):
    seq=np.asarray(seq,dtype=float)
    slopes=np.column_stack([(seq[:,-1]-seq[:,-1-k])/minutes for k,minutes in [(1,15),(2,30),(4,60),(8,120)]])
    summaries=np.column_stack([seq.mean(1),np.median(seq,axis=1),seq.min(1),seq.max(1),seq.std(1),
        (seq[:,-1]-2*seq[:,-2]+seq[:,-3])/225])
    phase=(t%(MINUTE*1440))/(MINUTE*1440)*2*np.pi
    return np.column_stack([seq,slopes,summaries,np.sin(phase),np.cos(phase)]).astype('float32')


def personal_features(t,g,cutoffs,current,slopes):
    """Prefix ends before current120min window. No future outcomes or meal labels."""
    c=config();result=[];eligible=[]
    for cutoff,now,slope in zip(cutoffs,current,slopes):
        end=np.searchsorted(t,cutoff-120*MINUTE,side='left')
        start=np.searchsorted(t,cutoff-c['personal_lookback_days']*1440*MINUTE,side='left')
        tt,v=t[start:end],g[start:end]
        warm=len(v)>=c['warm_prefix_equivalent_hours']*4 and cutoff-t[0]>=50*60*MINUTE
        eligible.append(warm)
        if not len(v):result.append([np.nan]*10+[0,0]);continue
        median=float(np.median(v));sd=float(np.std(v));hours=(tt//(60*MINUTE))%24
        hour=(cutoff//(60*MINUTE))%24
        hv=v[(hours==hour)&(tt<=cutoff-24*60*MINUTE)]
        nv=v[hours<6]
        hour_ok=len(hv)>=4;night_ok=len(nv)>=8
        hb=float(np.median(hv)) if hour_ok else np.nan
        nb=float(np.median(nv)) if night_ok else np.nan
        delta=np.diff(tt)/MINUTE;dg=np.diff(v)
        valid=(delta>=7.5)&(delta<=22.5)
        typical=float(np.median(dg[valid]/delta[valid])) if valid.any() else np.nan
        result.append([median,sd,hb,nb,typical,float(np.mean(v>180)),now-median,
            now-hb if hour_ok else np.nan,slope-typical,len(v)/4,int(hour_ok),int(night_ok)])
    return np.array(result,dtype='float32'),np.array(eligible,bool)


def meal_features(meal_times,cutoffs):
    """Annotation-time assumption; strictly no meal with event time later than T."""
    meal_times=np.unique(np.asarray(meal_times,dtype='int64'))
    if not len(meal_times):return np.full((len(cutoffs),2),np.nan,'float32'),np.zeros(len(cutoffs),bool)
    pos=np.searchsorted(meal_times,cutoffs,side='right')-1
    age=(cutoffs-meal_times[np.maximum(pos,0)])/MINUTE
    age[pos<0]=np.nan
    counts=np.searchsorted(meal_times,cutoffs,side='right')-np.searchsorted(meal_times,cutoffs-120*MINUTE,side='left')
    return np.column_stack([age,counts]).astype('float32'),(pos>=0)&(age<=180)


def labels(g,indices,future_indices,t):
    """Incident requires two adjacent observed readings afterT; no future input."""
    y=[];peaks=[];peak_time=[];spike=[];rise30=[];rise50=[]
    for i in indices:
        end=np.searchsorted(t,t[i]+120*MINUTE,side='right')
        future=g[i+1:end];ft=t[i+1:end]
        if not len(future):raise ValueError('No observed future within event horizon')
        peak=float(np.max(future));peaks.append(peak)
        peak_time.append(float((ft[np.argmax(future)]-t[i])/MINUTE))
        adjacent=(np.diff(ft)>=7.5*MINUTE)&(np.diff(ft)<=22.5*MINUTE)
        incident=bool(g[i]<=180 and np.any((future[:-1]>180)&(future[1:]>180)&adjacent))
        spike.append(incident);rise30.append(peak-g[i]>=30);rise50.append(peak-g[i]>=50)
    y=g[future_indices]
    return {'y':y.astype('float32'),'true_peak':np.array(peaks,'float32'),
        'true_peak_minutes':np.array(peak_time,'float32'),'spike':np.array(spike,bool),
        'rise30':np.array(rise30,bool),'rise50':np.array(rise50,bool)}


def load_static(z,wanted):
    # Read only stable demographic columns; no clinical lab/outcome columns are parsed.
    name=next(n for n in z.namelist() if '/' not in n and n.endswith('.xlsx') and 'T2DM' in n)
    df=pd.read_excel(io.BytesIO(z.read(name)),usecols=['Patient Number','Age (years)','Gender (Female=1, Male=2)'])
    output={}
    for _,row in df.iterrows():
        recording=str(row['Patient Number']);pid='ShanghaiT2D:'+recording.split('_')[0]
        if pid in wanted:
            age=pd.to_numeric(row['Age (years)'],errors='coerce')
            sex=pd.to_numeric(row['Gender (Female=1, Male=2)'],errors='coerce')
            output[recording]=[float(age),float(sex==1) if sex in [1,2] else np.nan]
    return output


def build_cache(stage):
    m=assert_stage(stage);wanted=m['model_development' if stage=='development' else stage]
    path=ROOT/'cache'/f'{stage}.npz';provenance=path.with_suffix('.json')
    if path.exists():
        p=json.loads(provenance.read_text())
        if p.get('builder_sha256')!=digest(__file__) or p['source_sha256']!=digest(SOURCE) or p['manifest_sha256']!=digest(ROOT/'participant_manifest.json') or p['cache_sha256']!=digest(path):
            raise RuntimeError('Cache provenance mismatch')
        return load_cache(stage)
    wanted=set(wanted);by_id={};recordings=0
    with zipfile.ZipFile(SOURCE) as z:
        static=load_static(z,wanted)
        for name in z.namelist():
            if not name.startswith('Shanghai_T2DM/') or not name.endswith(('.xls','.xlsx')):continue
            recording=Path(name).stem;pid='ShanghaiT2D:'+recording.split('_')[0]
            if pid not in wanted:continue
            df=pd.read_excel(io.BytesIO(z.read(name)),keep_default_na=False)
            df['_time']=times(df['Date']);col=next(c for c in df if c.strip() in ['CGM (mg / dl)','CGM'])
            df['_glucose']=pd.to_numeric(df[col],errors='coerce')
            df['_recording']=recording
            by_id.setdefault(pid,[]).append(df);recordings+=1
    accumulated={};audits=[]
    for pid,parts in sorted(by_id.items()):
        df=pd.concat(parts,ignore_index=True);t,g,q=prepare(df['_time'],df['_glucose'],15)
        s=Series(pid,'ShanghaiT2D','T2D',t,g,15,q)
        valid,masks,targets=window_masks(s,120)
        complete=np.logical_and.reduce([masks[h] for h in HORIZONS]);idx=np.flatnonzero(complete)
        # 9 actual observed values at native lag times; never select values after cutoff.
        seqidx=np.column_stack([nearest_indices(t,t[idx]-lag*MINUTE,7.5*MINUTE)[0] for lag in range(120,-1,-15)])
        lag_ok=np.column_stack([nearest_indices(t,t[idx]-lag*MINUTE,7.5*MINUTE)[1] for lag in range(120,-1,-15)])
        use=lag_ok.all(1)&(t[seqidx]<=t[idx,None]).all(1)&(np.diff(seqidx,axis=1)>0).all(1)
        idx=idx[use];seqidx=seqidx[use];seq=g[seqidx].astype('float32');cutoff=t[idx]
        future=np.column_stack([targets[h][idx] for h in HORIZONS]);x=cgm_features(seq,cutoff)
        personal,warm=personal_features(t,g,cutoff,seq[:,-1],x[:,10])
        mc=[c for c in ['Dietary intake','饮食'] if c in df]
        meal=np.logical_or.reduce([present(df[c]).to_numpy() for c in mc]) if mc else np.zeros(len(df),bool)
        mt=np.unique(df.loc[meal,'_time'].to_numpy());mt=mt[mt!=np.iinfo(np.int64).min]
        mealx,meal_ok=meal_features(mt,cutoff)
        # Associate static values with the recording covering this observed landmark.
        staticx=[];record_ids=[]
        for ti in cutoff:
            matching=df.loc[(df['_time']==ti)&pd.to_numeric(df['_glucose'],errors='coerce').notna(),'_recording']
            recording=str(matching.iloc[0]);record_ids.append(recording)
            staticx.append(static.get(recording,[np.nan,np.nan]))
        meal_episode=np.zeros(len(idx),bool)
        for tm in mt:
            k=np.searchsorted(cutoff,tm,side='right')-1
            if k>=0 and tm-cutoff[k]<=7.5*MINUTE:meal_episode[k]=True
        arrays={'pid':np.full(len(idx),pid),'recording':np.array(record_ids),'time':cutoff,'sequence':seq,'sequence_time':t[seqidx],
            'x':x,'personal':personal,'warm':warm,'meal':mealx,'meal_eligible':meal_ok,'meal_episode':meal_episode,
            'static':np.array(staticx,'float32'),'target_time':t[future],**labels(g,idx,future,t)}
        for key,value in arrays.items():accumulated.setdefault(key,[]).append(value)
        audits.append({'participant_id':pid,'windows':len(idx),'warm_windows':int(warm.sum()),'meal_windows':int(meal_ok.sum()),
            'static_windows':int(np.isfinite(arrays['static']).all(1).sum()),'native_QC':q,
            'history_only_skipped':int(valid.sum()-complete.sum()),'lag_grid_exclusions':int((~use).sum())})
    data={k:np.concatenate(v) for k,v in accumulated.items()}
    assert set(data['pid'])==wanted
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    np.savez_compressed(path,**data);path.chmod(0o600)
    write_json(provenance,{'stage':stage,'builder_sha256':digest(__file__),'source_sha256':digest(SOURCE),'manifest_sha256':digest(ROOT/'participant_manifest.json'),
        'cache_sha256':digest(path),'source':'raw native ShanghaiT2DM only','participants':len(wanted),'windows':len(data['y']),
        'features':{'cgm':CGM_NAMES,'personal':PERSONAL_NAMES,'meal':MEAL_NAMES,'static':STATIC_NAMES},'per_participant':audits})
    record_access(stage,len(wanted),recordings)
    print(json.dumps({'cache_built':stage,'participants':len(wanted),'windows':len(data['y']),'warm_windows':int(data['warm'].sum()),
        'meal_windows':int(data['meal_eligible'].sum()),'holdout_opened':stage=='holdout'}),flush=True)
    return data


def load_cache(stage):
    assert_stage(stage)
    with np.load(ROOT/'cache'/f'{stage}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}


def subset(data,mask):return {k:v[mask] for k,v in data.items()}

if __name__=='__main__':build_cache('development')
