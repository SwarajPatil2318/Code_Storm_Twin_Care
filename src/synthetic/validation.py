"""Pilot realism, diversity, temporal, leakage and diagnostic difficulty gates.

Only simple diagnostic models are fitted on pilot TRAIN patients and evaluated
on pilot VALIDATION patients. No final Twin/Shadow/TDI/time-to-event model exists.
"""
import argparse,json,hashlib,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance,ks_2samp
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from src.synthetic.config import load,SIGNALS,EVIDENCE,STATUS
from src.synthetic.grounding import summary,correlation
from src.synthetic.generate import iter_rows,code_hash,simulate_one
from src.synthetic.outcomes import MODEL_FEATURES,EHR_FEATURES,features,labels
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
      'prevalence':float(np.mean(y)),'rows':len(y),'positive_rows':int(sum(y))} if len(set(y))==2 else {'auroc':None,'auprc':None,'prevalence':float(np.mean(y)),'rows':len(y),'positive_rows':int(sum(y))}


def pipeline(tree=False):
    estimator=DecisionTreeClassifier(max_depth=3,min_samples_leaf=.015,class_weight='balanced',random_state=2026) if tree else LogisticRegression(C=.1,max_iter=1500,class_weight='balanced',random_state=2026)
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),estimator)


def diagnostic_models(root,rows,outcomes,roles,ground_truth,config):
    names=MODEL_FEATURES;matrix=np.array([[np.nan if r['features'][n] is None else r['features'][n] for n in names] for r in rows]);y=np.array([outcomes[r['window_id']]['event_within_72h'] for r in rows])
    train=np.array([roles[r['patient_id']]=='train' for r in rows]);held=np.array([roles[r['patient_id']]=='validation' for r in rows]);result={};saved={}
    candidates={'map_only':[i for i,n in enumerate(names) if n.startswith('phys__map__')],
      'hr_only':[i for i,n in enumerate(names) if n.startswith('phys__heart_rate__')],
      'ehr_only':[i for i,n in enumerate(names) if n.startswith('ehr__')],
      'physiology_only':[i for i,n in enumerate(names) if n.startswith('phys__')],
      'logistic_regression':list(range(len(names))),'simple_tree':list(range(len(names)))}
    for name,columns in candidates.items():
        model=pipeline(name=='simple_tree').fit(matrix[train][:,columns],y[train]);p=model.predict_proba(matrix[held][:,columns])[:,1];result[name]=metrics(y[held],p);saved[name]=model
    single=[]
    for j,name in enumerate(names):
        if not name.startswith('phys__') or name.endswith('__coverage'):continue
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),DecisionTreeClassifier(max_depth=1,min_samples_leaf=.015,class_weight='balanced',random_state=2026)).fit(matrix[train,j:j+1],y[train])
        p=model.predict_proba(matrix[held,j:j+1])[:,1];single.append({'feature':name,**metrics(y[held],p)})
    result['single_threshold_max_validation_auroc']=max(v['auroc'] for v in single);result['single_threshold_results']=single
    coverage=[i for i,n in enumerate(names) if n.endswith('__coverage')]
    clocks=np.array([[r['landmark_hour']/24] for r in rows]);identity=np.array([[int(hashlib.sha256(r['patient_id'].encode()).hexdigest()[:8],16)/2**32] for r in rows])
    profiles=[ground_truth[r['patient_id']]['profile'] for r in rows];profile_encoder=OneHotEncoder(categories=[config['phenotype_order']],handle_unknown='ignore',sparse_output=False).fit(np.array(profiles)[train,None]);profile_matrix=profile_encoder.transform(np.array(profiles)[:,None])
    nuisance={}
    for name,x in [('timestamp_only',clocks),('missingness_only',matrix[:,coverage]),('patient_id_hash_only',identity),('phenotype_id_only',profile_matrix)]:
        nuisance[name]={}
        for kind in ['logistic','tree']:
            model=pipeline(kind=='tree').fit(x[train],y[train]);nuisance[name][kind]=metrics(y[held],model.predict_proba(x[held])[:,1])
    import joblib
    directory=root/'diagnostics';directory.mkdir(mode=0o700,exist_ok=True)
    for name,model in saved.items():path=directory/(name+'.joblib');joblib.dump(model,path);path.chmod(0o600)
    return {'models':result,'nuisance_attacks':nuisance,'train_patients':len({r['patient_id'] for r,t in zip(rows,train) if t}),
      'validation_patients':len({r['patient_id'] for r,h in zip(rows,held) if h}),'test_model_evaluations':0,
      'diagnostic_only':True,'not_final_twin':True,'train_only_imputation_and_scaling':True,'classifier_input_features':names,
      'interpretation':'Pilot diagnostic metrics selected for generator checks, not unbiased clinical performance; near-perfect accuracy from class imbalance is not a success metric.'}


def physiological_summary(root,manifest,ground_truth,grounding,config):
    patients=[];daily=[];true_daily=[];latent_daily=[];states=[];missing=[];correlation_pairs=defaultdict(lambda:[[],[]]);quality=Counter();opportunities=np.zeros(len(SIGNALS));observed=np.zeros(len(SIGNALS));bounds_ok=True;temporal_ok=True;pressure_ok=True
    clipping=np.zeros(len(SIGNALS));ctx={r['patient_id']:r['context'] for r in iter_rows(root/'ehr/records.jsonl.gz')}
    for name in manifest['batches']:
        with np.load(root/'observations'/name,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(root/'ground_truth'/name,allow_pickle=False) as z:truth={k:z[k] for k in z.files}
        assert np.array_equal(data['patient_ids'],truth['patient_ids'])
        patients.extend(data['patient_ids'].tolist());values=data['value'];raw=data['raw_value'];q=data['quality_status'];scheduled=data['scheduled'];mask=data['missingness_mask'];available=data['available_hour'];t=data['observed_hour'];true=truth['true_physiology']
        bounds=np.array([config['signal_bounds'][s] for s in SIGNALS]);bounds_ok &= bool(np.all(true>=bounds[:,0]-1e-4)&np.all(true<=bounds[:,1]+1e-4))
        pressure_ok &= bool(np.all(true[:,:,2]>true[:,:,3]) and np.allclose(true[:,:,4],(true[:,:,2]+2*true[:,:,3])/3,atol=2e-5))
        temporal_ok &= bool(np.all(available[np.isfinite(available)]>=np.broadcast_to(t[None,:,None],available.shape)[np.isfinite(available)]))
        assert np.array_equal(np.isnan(values),mask) and np.all(mask[np.isin(q,[1,3])])
        for code in range(4):quality[str(code)]+=int(np.sum(q==code))
        opportunities+=scheduled.sum(axis=(0,1));observed+=(~mask).sum(axis=(0,1));missing.extend((1-(~mask).sum(axis=1)/np.maximum(scheduled.sum(axis=1),1)).tolist())
        shape=(len(values),manifest['days'],24,len(SIGNALS));value_days=values.reshape(shape).copy()
        # HR reference comparison uses morning/evening snapshots, matching medical
        # home acquisition rather than pretending dense wrist cadence is observed.
        value_days[:,:,:,0][:,:,[h for h in range(24) if h not in [8,20]]]=np.nan
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);daily.extend(np.nanmedian(value_days,axis=2).tolist())
        true_day=true.reshape(shape).mean(axis=2);true_daily.extend(true_day.tolist());latent_daily.extend(truth['latent'].reshape(len(values),manifest['days'],24,4).mean(axis=2).tolist());states.extend(truth['state'].tolist())
        # Centre each patient to distinguish within-patient coupling from case mix.
        centered=true_day-true_day.mean(axis=1,keepdims=True);burden=truth['latent'].reshape(len(values),manifest['days'],24,4).mean(axis=2)
        for title,a,b in [('hr_activity',centered[:,:,0],centered[:,:,8]),('bp_hr',centered[:,:,2],centered[:,:,0]),
          ('spo2_respiration',centered[:,:,5],centered[:,:,6]),('sleep_hrv',centered[:,:,9],centered[:,:,1]),
          ('weight_congestion',centered[:,:,7],burden[:,:,1]),('activity_deterioration',centered[:,:,8],burden.sum(axis=2))]:
            correlation_pairs[title][0].extend(a.ravel());correlation_pairs[title][1].extend(b.ravel())
    for row in iter_rows(root/'quality/records.jsonl.gz'):clipping+=row['true_value_clipping_counts']
    daily=np.array(daily);true_daily=np.array(true_daily);states=np.array(states);latents=np.array(latent_daily);comparison={};daily_stats={}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);centered_observed=daily-np.nanmean(daily,axis=1,keepdims=True)
    observable_cross={}
    for title,a,b in [('hr_activity',centered_observed[:,:,0],centered_observed[:,:,8]),('bp_hr',centered_observed[:,:,2],centered_observed[:,:,0]),
      ('spo2_respiration',centered_observed[:,:,5],centered_observed[:,:,6]),('sleep_hrv',centered_observed[:,:,9],centered_observed[:,:,1]),
      ('weight_congestion',centered_observed[:,:,7],latents[:,:,1]),('activity_deterioration',centered_observed[:,:,8],latents.sum(axis=2))]:
        observable_cross[title]=correlation(a.ravel(),b.ravel())

    with np.load(grounding['perheart']['distribution_cache']) as real:
        for j,signal in enumerate(SIGNALS):
            a=daily[:,:,j];values=a[np.isfinite(a)];within=[];between=[];diff=[];slopes=[];lag=[]
            for patient_days in a:
                good=np.flatnonzero(np.isfinite(patient_days));v=patient_days[good]
                if len(v)>1:within.append(float(v.std()));between.append(float(v.mean()))
                for k in range(1,len(good)):
                    gap=good[k]-good[k-1];slopes.append(float((v[k]-v[k-1])/gap))
                    if gap==1:diff.append(float(v[k]-v[k-1]));lag.append((v[k-1],v[k]))
            ds={'daily_values':summary(values),'within_patient_sd':summary(within),'patient_means':summary(between),'consecutive_day_changes':summary(diff),
              'slope_per_day':summary(slopes),'lag1_daily_correlation':correlation(*zip(*lag)) if lag else None}
            daily_stats[signal]=ds
            if signal not in real:continue
            r=grounding['perheart']['signals'][signal];b=real[signal];sd=float(np.std(b));q=np.array(ds['daily_values']['quantiles_01_05_25_50_75_95_99']);rq=np.quantile(b,[.01,.05,.25,.5,.75,.95,.99])
            comparison[signal]={'real':r,'synthetic':ds,'wasserstein_original_units':float(wasserstein_distance(values,b)),
              'wasserstein_real_sd':float(wasserstein_distance(values,b)/sd),'ks_statistic':float(ks_2samp(values,b).statistic),
              'maximum_quantile_distance_real_sd':float(np.max(abs(q-rq))/sd),
              'within_sd_ratio':float(ds['within_patient_sd']['mean']/r['within_patient_sd']['mean']),
              'day_change_sd_ratio':float(ds['consecutive_day_changes']['sd']/r['consecutive_day_changes']['sd']),
              'variance_ratio':float(ds['daily_values']['variance']/r['daily_values']['variance'])}
    private_json(root/'realism_metrics.json',{'comparison':comparison,'synthetic_signals':daily_stats,'cross_signal_within_patient':observable_cross,'true_physiology_cross_signal':{k:correlation(*v) for k,v in correlation_pairs.items()},
      'eicu_vitaldb_context':'Separate distributions/correlations retained in grounding; surgical/ICU absolute ranges are not forced onto home physiology',
      'chiron_context':'Provisional-unit daily structure retained separately; HR_SD is not HRV and skin temperature is not core/home temperature',
      'evidence':EVIDENCE,'evidence_label':STATUS})
    np.savez_compressed(root/'validation_arrays.npz',patient_ids=np.array(patients),daily=daily,true_daily=true_daily,latent_daily=latents,states=states,missing=np.array(missing))
    (root/'validation_arrays.npz').chmod(0o600)
    return {'comparison':comparison,'daily_stats':daily_stats,'cross_signal':observable_cross,'true_physiology_cross_signal':{k:correlation(*v) for k,v in correlation_pairs.items()},
      'bounds_ok':bounds_ok,'causal_receipt':temporal_ok,'pressure_order':pressure_ok,'quality_counts':dict(quality),
      'missing_fraction_by_signal':dict(zip(SIGNALS,(1-observed/np.maximum(opportunities,1)).tolist())),
      'missing_fraction_between_patient_sd':dict(zip(SIGNALS,np.array(missing).std(axis=0).tolist())),
      'clipping_fraction_by_signal':dict(zip(SIGNALS,(clipping/(manifest['patients']*manifest['days']*24)).tolist())),
      'patients':patients,'stable_days':(states==0).sum(axis=1)/24,'daily':daily}


def leakage_contracts(root,manifest,rows,outcomes,roles,ground_truth,config,grounding):
    names=set(MODEL_FEATURES);valid_schema=all(set(r['features'])==names for r in rows)
    matches=all(r['window_id'] in outcomes and outcomes[r['window_id']]['patient_id']==r['patient_id'] for r in rows)
    labels_correct=True
    for r in rows:
        expected=labels(ground_truth[r['patient_id']]['event_time_hours'],r['landmark_hour'],manifest['days']*24);actual=outcomes[r['window_id']]
        labels_correct &= all(actual[k]==v for k,v in expected.items())
    p,b,s,phys,stream=simulate_one(0,config,grounding,manifest['seed']);landmark=360;before=features(p['context'],stream,landmark)
    altered={k:v.copy() for k,v in stream.items()};altered['value'][landmark+1:]=1e8;altered['available_hour'][landmark+1:]=0
    future_invariant=before==features(p['context'],altered,landmark)
    altered={k:v.copy() for k,v in stream.items()};delayed=(altered['available_hour']>landmark)&(np.arange(len(altered['value']))[:,None]<=landmark);altered['value'][delayed]=1e8
    receipt_invariant=before==features(p['context'],altered,landmark)
    poisoned={**p['context'],'latent':999,'event_time':370,'seed':999,'profile':'event_positive'}
    hidden_invariant=before==features(poisoned,stream,landmark)
    public_arrays_safe=True
    for name in manifest['batches']:
        with np.load(root/'observations'/name) as z:public_arrays_safe &= not bool(set(z.files)&{'state','latent','event_time','event_hazard','stable_daily','vulnerability'})
    repeated_counts=Counter(r['patient_id'] for r in rows);fixed_schedule=len(set(repeated_counts.values()))==1
    split_isolation=set(roles)==set(ground_truth) and all(role in {'train','validation','test'} for role in roles.values())
    rawlabelsseparate=all(not(set(r)&{'event_within_72h','time_to_event_hours','event_time'}) for r in rows)
    result={'strict_feature_allowlist':valid_schema,'outcome_linkage':matches,'labels_correct':labels_correct,'future_values_invariant':future_invariant,
      'late_receipt_invariant':receipt_invariant,'hidden_context_ignored':hidden_invariant,'observation_arrays_no_hidden_state':public_arrays_safe,
      'label_tables_separate':rawlabelsseparate,'patient_split_isolation':split_isolation,'fixed_landmark_counts_no_event_censor_sidechannel':fixed_schedule,
      'metadata_exclusion':'patient/window IDs, landmark time, source labels and simulator metadata are linkage/audit fields, never classifier inputs',
      'evidence':EVIDENCE,'evidence_label':STATUS}
    private_json(root/'leakage_audit.json',result);return result


def validate(root='data/synthetic/pilot',config_path=None):
    started=time.monotonic();root=Path(root);config=load(config_path) if config_path else load();manifest=json.loads((root/'manifest.json').read_text());config['days']=manifest['days']
    grounding=json.loads(Path(config['grounding_path']).read_text());ground_truth={r['patient_id']:r for r in iter_rows(root/'ground_truth/records.jsonl.gz')}
    rows=list(iter_rows(root/'model_features/records.jsonl.gz'));outcomes={r['window_id']:r for r in iter_rows(root/'outcomes/records.jsonl.gz')};roles=json.loads((root/'splits.json').read_text())['assignments']
    if manifest['patients']!=1000:raise ValueError('Primary pilot certificate requires exactly1000 patients')
    if manifest['config_sha256']!=config['_config_sha256'] or manifest['simulator_code_sha256']!=code_hash():raise ValueError('Pilot generated with stale simulator rules')
    for name,item in manifest['artifacts'].items():
        if file_hash(root/name)!=item['sha256']:raise ValueError('Pilot artifact integrity failure')
    physiology=physiological_summary(root,manifest,ground_truth,grounding,config)
    leakage=leakage_contracts(root,manifest,rows,outcomes,roles,ground_truth,config,grounding)
    diagnostics=diagnostic_models(root,rows,outcomes,roles,ground_truth,config);private_json(root/'difficulty_metrics.json',diagnostics)
    truth=list(ground_truth.values());profile_counts=Counter(r['profile'] for r in truth);paths=Counter(e['path'] for r in truth for e in r['episodes']);recovery=sum(e['recovered_without_event'] for r in truth for e in r['episodes']);rapid=sum(e['rapid'] for r in truth for e in r['episodes'])
    decoys=Counter(e['kind'] for r in truth for e in r['decoys']);vulnerability=np.array([ground_truth[p]['vulnerability'] for p in physiology['patients']]);high=np.quantile(vulnerability,.75)
    stable_high=sum(ground_truth[p]['event_time_hours'] is None and ground_truth[p]['vulnerability']>=high and days>=14 for p,days in zip(physiology['patients'],physiology['stable_days']))
    low=np.quantile(vulnerability,.25);abnormal_low=sum(r['event_time_hours'] is None and r['vulnerability']<=low and any(d['kind'] in ['tachycardia','exercise','bp_change','mild_illness'] for d in r['decoys']) for r in truth)
    event_hours=[r['event_time_hours'] for r in truth if r['event_time_hours'] is not None];base=np.array([r['stable_baseline_mean'] for r in truth]);adherence=[r['adherence'] for r in truth]
    diversity={'profiles':dict(profile_counts),'paths':dict(paths),'event_time_hours':summary(event_hours),'episode_durations_hours':summary([e['planned_duration_hours'] for r in truth for e in r['episodes']]),
      'rapid_episodes':rapid,'recovered_without_event':recovery,'decoys':dict(decoys),'stable_high_risk_non_event_patients':int(stable_high),'low_risk_abnormal_non_event_patients':int(abnormal_low),
      'baseline_unique_count':len({tuple(x) for x in base}),'baseline_sd_by_signal':dict(zip(SIGNALS,base.std(axis=0).tolist())),
      'adherence':summary(adherence),'comorbidity_profiles':len({tuple(r['context'][k] for k in ['hypertension','diabetes','ckd']) for r in iter_rows(root/'ehr/records.jsonl.gz')})}
    private_json(root/'diversity_metrics.json',diversity)
    v=config['validation'];realism_by_signal={}
    for signal,r in physiology['comparison'].items():
        realism_by_signal[signal]=bool(r['wasserstein_real_sd']<=v['realism_max_wasserstein_real_sd'] and r['maximum_quantile_distance_real_sd']<=v['realism_max_quantile_distance_real_sd'] and
          v['within_sd_ratio_limits'][0]<=r['within_sd_ratio']<=v['within_sd_ratio_limits'][1] and v['day_change_sd_ratio_limits'][0]<=r['day_change_sd_ratio']<=v['day_change_sd_ratio_limits'][1])
    maximum_single=max(diagnostics['models']['single_threshold_max_validation_auroc'],diagnostics['models']['map_only']['auroc'],diagnostics['models']['hr_only']['auroc'])
    nuisance_max=max(x['auroc'] for family in diagnostics['nuisance_attacks'].values() for x in family.values());joint=diagnostics['models']['logistic_regression'];cross=physiology['cross_signal']
    from src.synthetic.state_model import event_hazard
    context_matters=event_hazard(.8,.8,72,config)>event_hazard(.8,.2,72,config)
    # Reproduction/test evidence is supplied by the mandatory verification command;
    # provisional validation cannot authorize scaling until those gates are proved.
    gates={'realism_supported_home_signals':all(realism_by_signal.values()),'physiological_bounds':physiology['bounds_ok'],
      'pressure_order_and_derived_map':physiology['pressure_order'],'causal_receipt':physiology['causal_receipt'],
      'heterogeneous_unique_baselines':diversity['baseline_unique_count']==1000,
      'all_profiles_and_paths':len(profile_counts)==7 and all(paths.get(p,0)>=v['minimum_each_path_patients'] for p in config['paths']),
      'event_timing_varies':len(event_hours)>20 and np.std(event_hours)>72,
      'event_prevalence_in_declared_band':config['states']['event_prevalence_acceptance'][0]<=manifest['event_patient_prevalence']<=config['states']['event_prevalence_acceptance'][1],
      'recovery_and_decoys':recovery>=v['minimum_non_event_recoveries'] and len(decoys)==8 and rapid>=20 and abnormal_low>=30,
      'stable_high_risk_controls':stable_high>=v['minimum_stable_high_risk'],
      'temporal_correlation':all(physiology['daily_stats'][s]['lag1_daily_correlation']>=v['daily_lag1_min'] for s in ['heart_rate','systolic_bp','diastolic_bp','spo2','weight','temperature']),
      'cross_signal_plausibility':cross['hr_activity']>.03 and cross['spo2_respiration']<-.03 and cross['weight_congestion']>.03 and cross['sleep_hrv']>.03 and cross['activity_deterioration']<-.03,
      'single_variable_nontrivial':maximum_single<v['single_variable_max_auroc'],
      'nuisance_nontrivial':nuisance_max<v['nuisance_max_auroc'],
      'multivariate_learnable_not_near_perfect':v['multivariate_min_auroc']<=joint['auroc']<v['diagnostic_max_auroc'] and joint['auprc']>=v['multivariate_min_ap_over_prevalence']*joint['prevalence'],
      'no_hidden_or_future_leakage':all(value for key,value in leakage.items() if isinstance(value,bool)),
      'ehr_context_influences_latent_hazard':context_matters,'different_adherence':np.std(adherence)>.05,
      'bounded_missingness_at_opportunities':all(v['sensor_missing_fraction_range'][0]<=x<=v['sensor_missing_fraction_range'][1] for x in physiology['missing_fraction_by_signal'].values()),
      'limited_physiology_clipping':all(x<=v['max_truth_clipping_fraction'] for x in physiology['clipping_fraction_by_signal'].values()),
      'reproducibility':False,'regression_tests':False,'fhir_canonical_linkage':False,'ground_truth_privacy':False}
    gates={key:bool(value) for key,value in gates.items()}
    result={'patients':1000,'seed':manifest['seed'],'days':manifest['days'],'variant':manifest['variant'],'gates':gates,'all_gates_passed':False,
      'realism_by_signal':realism_by_signal,'maximum_single_variable_auroc':maximum_single,'maximum_nuisance_auroc':nuisance_max,
      'physiology_quality':{k:physiology[k] for k in ['quality_counts','missing_fraction_by_signal','missing_fraction_between_patient_sd','clipping_fraction_by_signal']},
      'config_sha256':manifest['config_sha256'],'simulator_code_sha256':code_hash(),'grounding_sha256':manifest['grounding_sha256'],
      'pilot_artifact_sha256':{name:item['sha256'] for name,item in manifest['artifacts'].items()},
      'elapsed_seconds':time.monotonic()-started,'evidence':EVIDENCE,'evidence_label':STATUS,'not_clinical_validation':True,'final_predictor_trained':False}
    private_json(root/'validation.json',result)
    print({'pilot_validation_computed':True,'pending_or_failed_gates':[k for k,v in gates.items() if not v],'joint_diagnostic_auroc':round(joint['auroc'],4),'maximum_single_variable_auroc':round(maximum_single,4)},flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/synthetic/pilot');parser.add_argument('--config');args=parser.parse_args()
    with threadpool_limits(limits=1):validate(args.root,args.config)
