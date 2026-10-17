"""Aggregate source summaries only; no real-patient pairing or event labels.

CHIRON units remain provisional and are never pooled with verified home units.
eICU inputs are the already isolated Phase45B development export, not a legacy
whole-cohort loader. All source files are read-only.
"""
from collections import defaultdict,Counter
import datetime as dt
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from src.data.io import file_hash,read_jsonl
from src.vitaldb_feasibility.source import private_json

ROOT=Path('data/synthetic_grounding')
HOME=['heart_rate','systolic_bp','diastolic_bp','spo2','weight','temperature']


def summary(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    if not len(a):return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'variance':float(a.var()),'sd':float(a.std()),
      'quantiles_01_05_25_50_75_95_99':np.quantile(a,[.01,.05,.25,.5,.75,.95,.99]).tolist()}


def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[m],b[m])[0,1]) if sum(m)>3 and np.std(a[m])>0 and np.std(b[m])>0 else None


def longitudinal(dataset):
    path=Path('data/processed')/dataset/'observations.jsonl';daily=defaultdict(list);measurements=defaultdict(list);all_dates=defaultdict(set);quality=Counter()
    for r in read_jsonl(path):
        concept=r['concept'];quality[(concept,r['value']['quality_status'])]+=1
        value=r['value']['usable_value'];time=r['observed_time']
        if value is None or time is None:continue
        if dataset=='chiron':day=int(time)
        else:day=dt.datetime.fromisoformat(time).date().toordinal()
        pid=r['patient_id'];daily[(pid,day,concept)].append(float(value));measurements[(pid,concept)].append((day,float(value)));all_dates[pid].add(day)
    collapsed={k:float(np.median(v)) for k,v in daily.items()};concepts=sorted({k[2] for k in collapsed});stats={};pmeans={};samples={}
    for concept in concepts:
        values=[];means=[];within=[];difference=[];slopes=[];lagpairs=[];coverage=[];frequencies=[]
        for pid in sorted(all_dates):
            observed=sorted((d,v) for (p,d,c),v in collapsed.items() if p==pid and c==concept)
            if not observed:continue
            days=np.array([d for d,v in observed]);x=np.array([v for d,v in observed]);values.extend(x);means.append(float(np.mean(x)));pmeans[(pid,concept)]=float(np.median(x))
            if len(x)>1:within.append(float(np.std(x)))
            for i in range(1,len(x)):
                elapsed=days[i]-days[i-1];slopes.append(float((x[i]-x[i-1])/elapsed))
                if elapsed==1:difference.append(float(x[i]-x[i-1]));lagpairs.append((x[i-1],x[i]))
            span=max(all_dates[pid])-min(all_dates[pid])+1;coverage.append(len(days)/span);frequencies.append(len(measurements[(pid,concept)])/span)
        samples[concept]=np.asarray(values,dtype=np.float64)
        stats[concept]={'daily_values':summary(values),'patient_means':summary(means),'within_patient_sd':summary(within),
          'consecutive_day_changes':summary(difference),'slope_per_day':summary(slopes),'lag1_daily_correlation':correlation(*zip(*lagpairs)) if lagpairs else None,
          'observed_day_fraction_over_medical_span':summary(coverage),'observations_per_day_over_medical_span':summary(frequencies),
          'quality_counts':{s:n for (c,s),n in quality.items() if c==concept}}
    keys=sorted({(p,d) for p,d,c in collapsed});matrix=np.array([[collapsed.get((p,d,c),np.nan) for c in concepts] for p,d in keys]);corr={}
    for a in range(len(concepts)):
        for b in range(a+1,len(concepts)):corr[concepts[a]+'__'+concepts[b]]=correlation(matrix[:,a],matrix[:,b])
    # Paired-source patient medians generate population covariance, not resampled real identities.
    ids=sorted(all_dates);baseline=np.array([[pmeans.get((p,c),np.nan) for c in HOME] for p in ids]);center=[float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else None for col in baseline.T]
    sd=np.array([float(np.std(col[np.isfinite(col)])) if np.isfinite(col).any() else 1 for col in baseline.T]);sd=np.where(sd>0,sd,1);covcorr=np.eye(len(HOME))
    for a in range(len(HOME)):
        for b in range(a+1,len(HOME)):
            c=correlation(baseline[:,a],baseline[:,b]);covcorr[a,b]=covcorr[b,a]=.4*c if c is not None else 0
    eigen,vectors=np.linalg.eigh(covcorr);covcorr=(vectors*np.maximum(eigen,.1))@vectors.T;d=np.sqrt(np.diag(covcorr));covcorr/=d[:,None]*d[None,:]
    marginal_levels=[0,.01,.05,.25,.5,.75,.95,.99,1]
    marginal_quantiles=[np.quantile(col[np.isfinite(col)],marginal_levels).tolist() if np.isfinite(col).any() else None for col in baseline.T]
    sample_path=ROOT/(dataset+'_daily_distributions.npz');np.savez_compressed(sample_path,**samples);sample_path.chmod(0o600)
    return {'distribution_cache':str(sample_path),'distribution_cache_sha256':file_hash(sample_path),'dataset':dataset,'evidence_label':'REAL DATA EXPLORATORY','subjects':len(ids),'signals':stats,'daily_correlations':corr,
      'units':'provisional CHIRON within-source only' if dataset=='chiron' else 'PerHeart descriptor-supported units',
      'sampling_caveat':'daily median view; missing days based on observed medical span, not verified enrollment/nonwear; no clinical stable-state labels',
      'population_baseline':{'signals':HOME,'marginal_quantile_levels':marginal_levels,'marginal_quantiles_patient_medians':marginal_quantiles,'median_of_patient_medians':center,'sd_patient_medians':sd.tolist(),'shrunken_correlation':covcorr.tolist()}}


def multivariate(path,signals):
    with np.load(path,allow_pickle=False) as z:array=z['sequence'][:,:,:6]
    flat=array.reshape(-1,6);corr={}
    for i,a in enumerate(signals):
        for j,b in enumerate(signals):
            if j>i:corr[a+'__'+b]=correlation(flat[:,i],flat[:,j])
    return {'signals':{c:summary(flat[:,i]) for i,c in enumerate(signals)},'correlations':corr,
      'evidence_label':'REAL DATA EXPLORATORY','caveat':'selected overlapping windows, dependent observations; different acquisition/setting from home; no labels opened'}


def synthea_library():
    base=Path('data/processed/synthea');resources=Counter();codes=defaultdict(Counter);examples={}
    for table in ['conditions','medications','encounters']:
        for r in read_jsonl(base/(table+'.jsonl')):
            kind=r['resource_type'];resources[kind]+=1
            for item in r.get('coding',[]):
                if item.get('display'):codes[table][(item.get('system'),item.get('code'),item.get('display'))]+=1
            if kind not in examples:
                raw=r['raw_fields'];examples[kind]={'fields':sorted(raw),'status':raw.get('status'),'category':raw.get('category')}
    return {'evidence_label':'NOT VALIDATED','source':'audited Synthea synthetic FHIR sample; structure and coding library only, no population clinical claim',
      'resources':dict(resources),'shapes':examples,'common_codes':{t:[{'system':k[0],'code':k[1],'display':k[2],'count':n} for k,n in c.most_common(40)] for t,c in codes.items()}}


def build():
    ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    sources=[Path('data/processed')/d/'manifest.json' for d in ['chiron','perheart','synthea']]
    sources += [Path('data/processed')/d/'observations.jsonl' for d in ['chiron','perheart']]
    sources += [Path('data/processed/synthea')/(t+'.jsonl') for t in ['conditions','medications','encounters']]
    sources += [Path('data/results/phase45b')/p for p in ['eicu_development.npz','vitaldb_60.npz','development_access.json','vitaldb_cohort.json']]
    result={'version':'synthetic_grounding_v1','evidence_label':'REAL DATA EXPLORATORY','sources_sha256':{str(p):file_hash(p) for p in sources},
      'chiron':longitudinal('chiron'),'perheart':longitudinal('perheart'),
      'eicu_development':multivariate(Path('data/results/phase45b/eicu_development.npz'),['map','sbp','dbp','hr','spo2','rr']),
      'vitaldb':multivariate(Path('data/results/phase45b/vitaldb_60.npz'),['map','sbp','dbp','hr','spo2','rr']),
      'synthea':synthea_library(),
      'unsupported_channels':{'hrv':'HR_SD is not interval-based HRV; simulator assumption','steps':'supplied wrist/activity records absent; assumption',
        'sleep':'lying duration is not sleep; supplied sleep records absent; assumption','respiration':'ICU/surgical dynamics only, home calibration unsupported'},
      'no_real_patient_linkage':True,'spent_test_clinical_access':False}
    private_json(ROOT/'grounding.json',result)
    print({'grounding_sources':['CHIRON','PerHeart','eICU development','VitalDB','Synthea'],'real_patient_linkage':False},flush=True)
    return result

if __name__=='__main__':build()
