"""Aggregate development evidence, paired uncertainty, and sensor perturbations.

Only Phase 4.5B exports/models are opened. This module never reads original
cohort labels, spent-test artifacts, or whole-cohort physiology arrays.
"""
import json
from collections import Counter
import numpy as np
import joblib
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.ssl_physiology.protocol import ROOT, SIGNALS
from src.ssl_physiology.cohort import tensor, save_npz
from src.ssl_physiology.pretrain import load, metric_channels
from src.predictive.metrics import basic, calibration_diagnostics, paired_bootstrap
from src.vitaldb_feasibility.source import private_json


def embedding_change(base, changed):
    denominator=np.linalg.norm(base,axis=1)*np.linalg.norm(changed,axis=1)
    cosine=np.divide(np.sum(base*changed,axis=1),denominator,out=np.zeros(len(base)),where=denominator>0)
    rms=np.sqrt(np.mean((base-changed)**2,axis=1))
    return {'rms_median':float(np.median(rms)),'rms_p95':float(np.quantile(rms,.95)),
            'cosine_median':float(np.median(cosine)),'cosine_p05':float(np.quantile(cosine,.05))}


def perturb(seq,condition,seed=2026):
    """Label-independent perturbation of observed predictors only; no filling."""
    result=seq.copy()
    if condition=='truncate60':return result[:,-12:]
    if condition.startswith('missing_'):
        j=SIGNALS.index(condition[8:]);result[:,:,j]=np.nan;result[:,:,6+j]=0;result[:,:,12+j]=1
        return result
    values=result[:,:,:6].copy();observed=result[:,:,6:12]>0
    if condition=='noise':
        # Measurement stress test in original units, not an asserted device model.
        noise=np.random.default_rng(seed).normal(size=values.shape)*np.array([3,5,3,5,1,2])
        values=np.where(observed,values+noise,np.nan)
    elif condition=='sparse_bp':
        keep=np.arange(seq.shape[1])%4==0;observed[:,:,0:3]&=keep[None,:,None]
    else:raise ValueError('Unknown sensor perturbation')
    for i in range(len(seq)):
        updated=tensor(values[i],observed[i]);result[i]=updated
        if condition=='noise':result[i,:,12:]=seq[i,:,12:] # masks/timing are unchanged
        else:result[i,:,15:]=seq[i,:,15:] # non-BP history is unchanged
    return result


def summarize():
    rows=json.loads((ROOT/'eicu_development_rows.json').read_text())
    oof=json.loads((ROOT/'development_oof.json').read_text());folds=json.loads((ROOT/'fold_metrics.json').read_text())
    if len(folds)!=10 or len(oof)!=2*len(rows):raise ValueError('Incomplete development cross-validation')
    models=sorted(oof[0]['raw']);y=np.array([r['label'] for r in rows]);raw={};cal={};results={}
    for name in models:
        raw[name]=np.zeros((2,len(rows)));cal[name]=np.zeros((2,len(rows)))
        for s,seed in enumerate([2026,2031]):
            subset=[r for r in oof if r['seed']==seed]
            if sorted(r['row_index'] for r in subset)!=list(range(len(rows))):raise ValueError('OOF rows repeated or absent')
            for r in subset:raw[name][s,r['row_index']]=r['raw'][name];cal[name][s,r['row_index']]=r['calibrated'][name]
        scores=np.array([f['models'][name]['raw']['auprc'] for f in folds]);gain=scores-np.array([f['models']['map_only']['raw']['auprc'] for f in folds])
        results[name]={'per_repeat_raw':[basic(y,p) for p in raw[name]],'per_repeat_calibrated':[basic(y,p) for p in cal[name]],
           'mean_repeat_raw':{k:float(np.mean([basic(y,p)[k] for p in raw[name]])) for k in basic(y,raw[name][0])},
           'mean_repeat_calibrated':{k:float(np.mean([basic(y,p)[k] for p in cal[name]])) for k in basic(y,cal[name][0])},
           'calibration_diagnostics':[calibration_diagnostics(y,p) for p in cal[name]],
           'fold_ap':{'mean':float(np.mean(scores)),'std':float(np.std(scores,ddof=1)),'min':float(min(scores)),'max':float(max(scores))},
           'fold_ap_gain_over_map':{'mean':float(np.mean(gain)),'positive_folds':int(sum(gain>0)),'total':10},
           'paired_ap_vs_map':None}
    # Compute after all matrices exist (alphabetical order is not a dependency).
    for name in models:
        results[name]['paired_ap_vs_map']=paired_bootstrap(rows,raw[name].mean(axis=0),raw['map_only'].mean(axis=0),n=1000)
    contrasts={}
    for left,right in [('latent_plus_map','map_logistic'),('latent_plus_map','map_only'),('latent_frozen_ehr','latent_frozen'),
      ('latent_finetuned_ehr','latent_finetuned'),('latent_delta','latent_recent'),('latent_delta_ehr','latent_recent_ehr'),
      ('latent_frozen','random_encoder'),('latent_finetuned','supervised_scratch')]:
        contrasts[left+'__vs__'+right]=paired_bootstrap(rows,raw[left].mean(axis=0),raw[right].mean(axis=0),n=1000)
    strata={}
    for name,mask in [('hard',np.array([r['audit']['hard'] for r in rows])),('normal80',np.array([r['audit']['map_stratum']=='clearly_normal' for r in rows]))]:
        indices=np.flatnonzero(mask);subset=[rows[i] for i in indices]
        strata[name]={'windows':len(subset),'positive_windows':int(sum(y[indices])),
          'patients':len({r['patient_id'] for r in subset}),'positive_patients':len({r['patient_id'] for r in subset if r['label']}),'models':{}}
        for model in models:
            p=raw[model].mean(axis=0)[indices];budget=max(1,int(np.ceil(.1*len(indices))));top=np.argsort(-p,kind='stable')[:budget]
            strata[name]['models'][model]={'mean_prediction_two_repeats':basic(y[indices],p),
               'per_repeat':[basic(y[indices],q[indices]) for q in raw[model]],
               'top10percent_rank_budget':budget,'positive_windows_in_top10percent':int(sum(y[indices][top])),
               'ranking_only_no_clinical_threshold':True}
    for name,mask in [('hard',np.array([r['audit']['hard'] for r in rows])),('normal80',np.array([r['audit']['map_stratum']=='clearly_normal' for r in rows]))]:
        subset=[r for r,m in zip(rows,mask) if m]
        strata[name]['paired_ap_vs_map']={model:paired_bootstrap(subset,raw[model].mean(axis=0)[mask],raw['map_only'].mean(axis=0)[mask],n=1000)
            for model in ['latent_frozen','latent_finetuned','latent_no_map','latent_no_bp','latent_plus_map']}
    with np.load(ROOT/'development_latents.npz') as z:available=z['reference_available']
    gate={name:bool(v['fold_ap_gain_over_map']['mean']>=.02 and v['fold_ap_gain_over_map']['positive_folds']>=8 and
        v['paired_ap_vs_map']['interval95'] is not None and v['paired_ap_vs_map']['interval95'][0]>0 and
        v['mean_repeat_calibrated']['brier']<=results['map_only']['mean_repeat_calibrated']['brier']+.005) for name,v in results.items() if name.startswith('latent')}
    output={'population':{'patients':len({r['patient_id'] for r in rows}),'positive_patients':len({r['patient_id'] for r in rows if r['label']}),
      'windows':len(rows),'positive_windows':int(sum(y))},'models':results,'contrasts':contrasts,'strata':strata,
      'personal_reference':{'available_windows':int(sum(available)),'total_windows':len(rows),'available_patients':len({r['patient_id'] for r,a in zip(rows,available) if a})},
      'primary_gate_numeric_components':gate,'gate_caveat':'Additionally requires credible conditional/noBP and hard-case evidence; bootstrap conditional on development selection, not external validation',
      'bootstrap_estimand':'paired patient clusters, mean of two held-out predictions for each window; repeats are not independent samples',
      'independent_validation_still_required':True,'test_evaluations':0}
    private_json(ROOT/'transfer_summary.json',output);save_npz(ROOT/'oof_prediction_matrices.npz',**{'raw__'+k:v for k,v in raw.items()},**{'cal__'+k:v for k,v in cal.items()})
    print({'development_analysis_complete':True,'any_numeric_gate_passed':any(gate.values())},flush=True)
    return output


def distribution(values):
    observed=values[np.isfinite(values)]
    return {'observed_points':len(observed),'missing_fraction':float(1-np.isfinite(values).mean()),
      'q05_q25_median_q75_q95':np.quantile(observed,[.05,.25,.5,.75,.95]).tolist() if len(observed) else None}


def domain_shift():
    vr,vs,vf,vt=load();er=json.loads((ROOT/'eicu_development_rows.json').read_text())
    with np.load(ROOT/'eicu_development.npz') as z:es=z['sequence']
    # Deduplicate the same closed grid observation appearing in overlapping windows.
    def unique_points(rows,seq,vital):
        seen=set();points=[]
        for r,s in zip(rows,seq):
            start=r['start_minutes'] if vital else r['t0']-5-360
            for t,value in enumerate(s):
                key=(r['case_id'] if vital else r['icu_stay_id'],start+5*t)
                if key not in seen:seen.add(key);points.append(value[:6])
        return np.asarray(points)
    vp=unique_points(vr,vs,True);ep=unique_points(er,es,False);signals={}
    for j,name in enumerate(SIGNALS):
        a=vp[:,j];b=ep[:,j];aa=a[np.isfinite(a)];bb=b[np.isfinite(b)];iqr=float(np.subtract(*np.quantile(aa,[.75,.25])))
        signals[name]={'vitaldb':distribution(a),'eicu_development':distribution(b),'ks_statistic':float(ks_2samp(aa,bb).statistic),
           'wasserstein_original_units':float(wasserstein_distance(aa,bb)),'wasserstein_vital_iqr':float(wasserstein_distance(aa,bb)/iqr) if iqr>0 else None}
    va={r['patient_id']:r['age'] for r in vr};ea={r['patient_id']:r['features'].get('ehr__age') for r in er}
    out={'signals':signals,'age_subject_weighted':{'vitaldb':distribution(np.array(list(va.values()),float)),
      'eicu_development':distribution(np.array([np.nan if x is None else x for x in ea.values()],float)),
      'eicu_caveat':'age above89 is topcoded in source and represented with a separate flag; VitalDB ages may differ across repeat cases'},
      'native_cadence':{'vitaldb_seconds_typical':2,'eicu_seconds_typical':300,'canonical_seconds':300,
        'source':'Phase45A measured native numeric interval; eICU vitalPeriodic observation offsets; downsampling does not equalize acquisition'},
      'settings':{'vitaldb':'single-center intraoperative surgical/anesthetic monitoring','eicu':'multicenter critical-care demo population'},
      'point_count':{'vitaldb':len(vp),'eicu_development':len(ep)},'sampling':'unique case/stay-grid points covered by eligible windows; not source-population prevalence or independent observations',
      'negative_transfer':'See paired frozen/random/scratch/MAP contrasts; surgical pretraining is not assumed beneficial',
      'test_clinical_data_used':False}
    private_json(ROOT/'domain_shift.json',out);return out


def robustness():
    conditions=['missing_spo2','missing_rr','missing_hr','noise','sparse_bp','truncate60']
    rows,seq,future,train=load();selected=json.loads((ROOT/'ssl_results.json').read_text())['selected_task'];bundle=joblib.load(ROOT/(selected+'__all.joblib'))
    held=seq[~train];norm=bundle['normalizer'];encoder=bundle['encoder'];base=encoder.embedding(norm.transform(held));vital={}
    truth=norm.targets(future[~train,-1]);base_future=encoder.forward(norm.transform(held))[:,-1]@encoder.params['out']+encoder.params['out_bias']
    for condition in conditions:
        changed=perturb(held,condition);x=norm.transform(changed);z=encoder.embedding(x)
        forecast=encoder.forward(x)[:,-1]@encoder.params['out']+encoder.params['out_bias']
        vital[condition]={'embedding':embedding_change(base,z),'future15_direct_head':metric_channels(truth,forecast,norm)}
    erows=json.loads((ROOT/'eicu_development_rows.json').read_text());oof=json.loads((ROOT/'development_oof.json').read_text())
    with np.load(ROOT/'eicu_development.npz') as z:eseq=z['sequence']
    raw_results={};models=['latent_frozen','latent_frozen_ehr','latent_finetuned','latent_finetuned_ehr']
    y=np.array([r['label'] for r in erows]);base_z=encoder.embedding(norm.transform(eseq))
    for condition in conditions:
        changed=perturb(eseq,condition);z=encoder.embedding(norm.transform(changed));output={name:np.zeros((2,len(erows))) for name in models}
        calibrated={name:np.zeros((2,len(erows))) for name in models}
        for s,seed in enumerate([2026,2031]):
            for fold in range(5):
                indices=np.array(sorted(r['row_index'] for r in oof if r['seed']==seed and r['fold']==fold));heldrows=[erows[i] for i in indices]
                for name in models:
                    saved=joblib.load(ROOT/'fold_models'/f'seed{seed}_fold{fold}'/(name+'.joblib'));model=saved['model']
                    if name=='latent_frozen':p=model.predict(z[indices])
                    elif name=='latent_frozen_ehr':head,ehr=model;p=head.predict(np.concatenate([z[indices],ehr.transform(heldrows)],axis=1))
                    else:head,ehr=model;p=head.predict(changed[indices],ehr.transform(heldrows) if ehr is not None else None)
                    output[name][s,indices]=p;calibrated[name][s,indices]=saved['calibrator'].predict(p)
        measured={}
        for name in models:
            baseline=np.zeros((2,len(erows)))
            for s,seed in enumerate([2026,2031]):
                for r in oof:
                    if r['seed']==seed:baseline[s,r['row_index']]=r['raw'][name]
            delta=abs(output[name]-baseline)
            measured[name]={'per_repeat_raw':[basic(y,p) for p in output[name]],'mean_ap':float(np.mean([basic(y,p)['auprc'] for p in output[name]])),
              'mean_ap_change':float(np.mean([basic(y,p)['auprc']-basic(y,b)['auprc'] for p,b in zip(output[name],baseline)])),
              'absolute_raw_risk_change_median':float(np.median(delta)),'absolute_raw_risk_change_p95':float(np.quantile(delta,.95)),
              'per_repeat_calibrated':[basic(y,p) for p in calibrated[name]]}
        raw_results[condition]={'embedding':embedding_change(base_z,z),'models':measured}
        save_npz(ROOT/('robustness_'+condition+'.npz'),**{'raw__'+name:v for name,v in output.items()},**{'cal__'+name:v for name,v in calibrated.items()})
        print({'robustness_condition':condition,'development_only':True},flush=True)
    out={'vitaldb_heldout':vital,'vitaldb_baseline_future15_direct_head':metric_channels(truth,base_future,norm),'eicu_development':raw_results,
      'perturbation_design':'whole missing channel; Gaussian observed-only noise SD MAP3/SBP5/DBP3 mmHg,HR5 bpm,SpO2 1percentagepoint,RR2/min; BP observations every fourth grid step; final60minutes only',
      'interpretation':'controlled missingness/noise stress tests, not validated wearable/device behavior; physiology-only perturbation leaves as-of EHR unchanged; raw risk differences reflect fitted heads; calibration remains frozen',
      'test_evaluations':0}
    private_json(ROOT/'robustness_results.json',out);return out


def run():
    with threadpool_limits(limits=1):
        summarize();domain_shift();robustness()

if __name__=='__main__':run()
