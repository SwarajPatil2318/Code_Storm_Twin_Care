"""Streaming integrity/label audit of the fresh large synthetic cohort.

No classifier is fitted or evaluated here. The large test split remains reserved
for Phase 6; checks concern simulation contracts, not prediction performance.
"""
import argparse,json,stat,time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np
from src.synthetic.config import load,SIGNALS
from src.synthetic.generate import iter_rows,code_hash,verify_pilot
from src.synthetic.outcomes import MODEL_FEATURES,labels
from src.synthetic.validation import physiological_summary
from src.synthetic.reader import canonical_observations,canonical_contexts
from src.data.io import file_hash
from src.data.schema import record
from src.vitaldb_feasibility.source import private_json


def audit(root='data/synthetic/cohort10000'):
    started=time.monotonic();root=Path(root);m=json.loads((root/'manifest.json').read_text());c=load();c['days']=m['days']
    gates={};gates['validated_pilot_authorization']=verify_pilot(c,m['seed'],m['days'])==m['pilot_gate_sha256']
    gates['unchanged_generation_rules']=m['config_sha256']==c['_config_sha256'] and m['simulator_code_sha256']==code_hash()
    gates['artifact_integrity']=all((root/name).stat().st_size==item['bytes'] and file_hash(root/name)==item['sha256'] for name,item in m['artifacts'].items())
    grounding=json.loads(Path(c['grounding_path']).read_text());gates['grounding_integrity']=file_hash(c['grounding_path'])==m['grounding_sha256'] and all(file_hash(p)==h for p,h in grounding['sources_sha256'].items())
    truth={r['patient_id']:r for r in iter_rows(root/'ground_truth/records.jsonl.gz')};patients={r['patient_id']:r for r in iter_rows(root/'patients/records.jsonl.gz')};roles=json.loads((root/'splits.json').read_text());counts=Counter(roles['assignments'].values());pilot=set(json.loads(Path('data/synthetic/pilot/splits.json').read_text())['assignments'])
    gates['unique_namespaced_patients']=len(patients)==len(truth)==m['patients'] and all(p.startswith('synthetic:') for p in patients)
    gates['fresh_population_disjoint_from_pilot']=not(set(patients)&pilot) and m['patient_index_offset']==1000
    gates['patient_split_isolation']=set(roles['assignments'])==set(patients) and roles['generated_before_split'] and dict(counts)=={'train':int(.7*m['patients']),'validation':int(.85*m['patients'])-int(.7*m['patients']),'test':m['patients']-int(.85*m['patients'])}
    linkage=True;schema=True;label_ok=True;window_counts=Counter();windows=0;positive=Counter()
    expected_hours=set(range(c['landmarks']['history_hours'],m['days']*24-c['landmarks']['horizon_hours']+1,c['landmarks']['stride_hours']));seen_hours={p:set() for p in patients}
    for f,o in zip_longest(iter_rows(root/'model_features/records.jsonl.gz'),iter_rows(root/'outcomes/records.jsonl.gz')):
        if f is None or o is None:linkage=False;break
        pid=f['patient_id'];hour=f['landmark_hour'];windows+=1;window_counts[pid]+=1
        linkage &= pid in truth and all(f[k]==o[k] for k in ['patient_id','window_id','landmark_hour'])
        schema &= set(f['features'])==set(MODEL_FEATURES) and set(f)=={'patient_id','window_id','landmark_hour','features','evidence','evidence_label'}
        schema &= all(v is None or isinstance(v,(int,float)) and np.isfinite(v) for v in f['features'].values())
        if pid not in truth:label_ok=False;continue
        seen_hours[pid].add(hour);expected=labels(truth[pid]['event_time_hours'],hour,m['days']*24);label_ok &= all(o[k]==v for k,v in expected.items())
        for key in ['event_0_24h','event_24_48h','event_48_72h','event_within_72h']:positive[key]+=o[key]
    gates['all_feature_label_linkage']=bool(linkage and windows==m['landmarks']);gates['strict_public_feature_allowlist']=bool(schema);gates['all_future_labels_correct']=bool(label_ok)
    gates['fixed_landmarks_no_outcome_censor']=all(seen_hours[p]==expected_hours and window_counts[p]==len(expected_hours) for p in patients)
    fhir_ok=True;ehrids=set();resources=0
    for r in iter_rows(root/'ehr/records.jsonl.gz'):
        pid=r['patient_id'];ehrids.add(pid);ref='Patient/'+pid.split(':')[1];entries=r['fhir']['entry'];identities={e['resource']['resourceType']+'/'+e['resource']['id'] for e in entries}
        fhir_ok &= pid in patients and ref in identities and r['available_hour']<0
        for e in entries:
            a=e['resource'];resources+=1;fhir_ok &= e['fullUrl'].startswith(('https://','urn:')) and (a['resourceType']=='Patient' or a['subject']['reference']==ref)
            if a['resourceType']=='Observation':fhir_ok &= a['issued']>=a['effectiveDateTime'] and a['valueQuantity']['system']=='http://unitsofmeasure.org'
    gates['ehr_fhir_linkage']=bool(fhir_ok and ehrids==set(patients));contexts=0
    for ctx in canonical_contexts(root):json.dumps(record(ctx),allow_nan=False);contexts+=1
    for obs in canonical_observations(root,limit=200):json.dumps(record(obs),allow_nan=False)
    gates['canonical_schema']=contexts==m['patients']
    phys=physiological_summary(root,m,truth,grounding,c);v=c['validation'];realism={}
    for signal,r in phys['comparison'].items():
        realism[signal]=bool(r['wasserstein_real_sd']<=v['realism_max_wasserstein_real_sd'] and r['maximum_quantile_distance_real_sd']<=v['realism_max_quantile_distance_real_sd'] and v['within_sd_ratio_limits'][0]<=r['within_sd_ratio']<=v['within_sd_ratio_limits'][1] and v['day_change_sd_ratio_limits'][0]<=r['day_change_sd_ratio']<=v['day_change_sd_ratio_limits'][1])
    gates['home_realism_same_declared_gates']=all(realism.values());gates['physiological_bounds']=phys['bounds_ok'];gates['pressure_order_and_map']=phys['pressure_order'];gates['causal_receipt_and_quality_masks']=phys['causal_receipt']
    gates['temporal_correlation']=all(phys['daily_stats'][s]['lag1_daily_correlation']>=v['daily_lag1_min'] for s in realism)
    cross=phys['cross_signal'];gates['cross_signal_plausibility']=cross['hr_activity']>.03 and cross['spo2_respiration']<-.03 and cross['weight_congestion']>.03 and cross['sleep_hrv']>.03 and cross['activity_deterioration']<-.03
    profiles=Counter(r['profile'] for r in truth.values());paths=Counter(e['path'] for r in truth.values() for e in r['episodes']);decoys=Counter(e['kind'] for r in truth.values() for e in r['decoys'])
    recovery=sum(e['recovered_without_event'] for r in truth.values() for e in r['episodes']);rapid=sum(e['rapid'] for r in truth.values() for e in r['episodes']);actual_events=sum(r['event_time_hours'] is not None for r in truth.values())
    gates['diverse_profiles_paths_decoys']=len(profiles)==7 and len(paths)==5 and len(decoys)==8 and recovery>0 and rapid>0
    gates['unique_personal_baselines']=len({tuple(r['stable_baseline_mean']) for r in truth.values()})==m['patients']
    gates['event_counts_and_prevalence']=actual_events==m['event_patients'] and c['states']['event_prevalence_acceptance'][0]<=actual_events/m['patients']<=c['states']['event_prevalence_acceptance'][1]
    gates['missingness_and_limited_clipping']=all(v['sensor_missing_fraction_range'][0]<=x<=v['sensor_missing_fraction_range'][1] for x in phys['missing_fraction_by_signal'].values()) and all(x<=v['max_truth_clipping_fraction'] for x in phys['clipping_fraction_by_signal'].values())
    gates['private_permissions']=all(stat.S_IMODE(p.stat().st_mode)==(0o700 if p.is_dir() else 0o600) for p in [root]+list(root.rglob('*')))
    import subprocess,shutil,tempfile
    with tempfile.TemporaryDirectory() as directory:
        shutil.copyfile('.gitignore',Path(directory)/'.gitignore');subprocess.run(['git','init','-q',directory],check=True)
        ownfiles=sorted(str(p) for p in root.rglob('*') if p.is_file());ignored=subprocess.run(['git','-C',directory,'check-ignore','--stdin'],input='\n'.join(ownfiles)+'\n',text=True,capture_output=True)
        gates['all_private_files_git_excluded']=set(ignored.stdout.splitlines())==set(ownfiles)
    old=json.loads(Path('data/synthetic_grounding/prior_preservation.json').read_text());gates['earlier_phases_preserved']=all(file_hash(p)==h for p,h in old['public_sha256'].items()) and all(Path(p).stat().st_size==s['bytes'] and Path(p).stat().st_mtime_ns==s['mtime_ns'] for p,s in old['private_size_mtime'].items())
    gates['synthetic_evidence_and_stop_boundary']=m['evidence']=='SYNTHETIC POC' and m['not_clinical_validation'] and not m['final_predictor_trained']
    result={'patients':m['patients'],'days':m['days'],'seed':m['seed'],'gates':{k:bool(x) for k,x in gates.items()},'all_gates_passed':bool(all(gates.values())),'event_patients':actual_events,'event_patient_prevalence':actual_events/m['patients'],'landmarks_checked':windows,'positive_landmarks':dict(positive),'split_counts':dict(counts),'profiles':dict(profiles),'paths':dict(paths),'decoys':dict(decoys),'rapid_episodes':rapid,'recovered_without_event':recovery,'realism_by_signal':realism,'missing_fraction_by_signal':phys['missing_fraction_by_signal'],'fhir_resources_checked':resources,'canonical_contexts_checked':contexts,'large_test_model_evaluations':0,'audit_seconds':time.monotonic()-started,'config_sha256':m['config_sha256'],'simulator_code_sha256':m['simulator_code_sha256'],'grounding_sha256':m['grounding_sha256'],'manifest_sha256':file_hash(root/'manifest.json'),'evidence':'SYNTHETIC POC','evidence_label':'SYNTHETIC POC VALIDATED' if all(gates.values()) else 'NOT VALIDATED','validation_scope':'Simulation integrity and declared realism gates; diagnostic difficulty established on pilot only. No clinical or model-performance validation.'}
    private_json(root/'large_validation.json',result);print({'large_cohort_audit_passed':result['all_gates_passed'],'failed_gates':[k for k,x in gates.items() if not x],'patients':m['patients'],'labels_checked':windows},flush=True)
    return result

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/synthetic/cohort10000');a=p.parse_args()
    with threadpool_limits(limits=1):audit(a.root)
