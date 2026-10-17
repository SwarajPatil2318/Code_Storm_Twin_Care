"""Pre-test checks and causal pipeline reproduction; no test or oracle reads."""
import json,subprocess,tempfile,shutil,stat
from pathlib import Path
import numpy as np
import joblib
from src.twin.protocol import load,code_hash,EVIDENCE,ALLOWLIST
from src.twin.cache import read_cache,validation_groups
from src.twin.access import Cohort
from src.twin.features import patient_history
from src.twin.train import view,predict
from src.twin.models import cumulative
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def run(config=None):
    c=config or load();out=Path(c['output']);cohort=Cohort(c['cohort'],out,c);gates={};development=json.loads((out/'development_results.json').read_text());uncertainty=json.loads((out/'uncertainty_fit.json').read_text())
    from src.twin.robustness import register as register_robustness
    from src.twin.report import register as register_decision
    register_robustness(c);register_decision(c)
    gates['required_model_ablations']=set(development['models'])==set(c['model_variants']);gates['uncertainty_models_ready']=set(uncertainty['models'])=={'full_twin',development['primary_model']}
    gates['test_never_decoded']=not (out/'features/test').exists() and not (out/'features/consumer_test').exists() and not (out/'evaluation_started.json').exists()
    for log in out.glob('access_*.json'):
        data=json.loads(log.read_text());gates['test_never_decoded'] &= all(r['role']!='test' and not r['oracle'] for r in data['clinical_rows_decoded'])
    train=read_cache('train',c);val=read_cache('validation',c);groups,val_roles=validation_groups(val['patient_ids'],c);training_ids=set(train['patient_ids']);validation_ids=set(val['patient_ids'])
    gates['patient_partitions_isolated']=not(training_ids&validation_ids) and all(cohort.roles[p]=='train' for p in training_ids) and all(cohort.roles[p]=='validation' for p in validation_ids)
    gates['validation_subroles_isolated']=sum(int(mask.sum()) for mask in groups.values())==len(val['labels']) and all(sum(mask[i] for mask in groups.values())==1 for i in range(len(val['labels'])))
    ids,counts=np.unique(train['patient_ids'],return_counts=True)
    gates['fixed_temporal_landmarks']=bool(np.all(counts==51) and np.array_equal(train['landmark_hours'].reshape(-1,51),np.broadcast_to(np.arange(168,1369,24),(len(ids),51))))
    engines=joblib.load(out/'shadow_engines.joblib');gates['shadow_train_only']=all(set(engine.fit_patients)<=training_ids for engine in engines.values());allowed=set(ALLOWLIST)
    selected=groups['selection'];cal_ids=set(val['patient_ids'][groups['calibration']]);op_ids=set(val['patient_ids'][groups['operating']]);gates['feature_allowlist']=True;gates['train_only_models']=True;gates['calibration_validation_only']=True;gates['threshold_validation_only']=True;gates['cumulative_risk_consistency']=True;optimizers={}
    for name in c['model_variants']:
        directory=out/'models'/name;model=joblib.load(directory/'selected.joblib');cal=joblib.load(directory/'calibration.joblib');gates['feature_allowlist'] &= set(model.names)<=allowed;gates['train_only_models'] &= set(model.fit_patients)<=training_ids;gates['calibration_validation_only'] &= set(cal.fit_patients)<=cal_ids
        optimizers[name]=[{'iterations':int(np.atleast_1d(h.n_iter_)[0]),'limit':h.max_iter,'hit_limit':bool(np.atleast_1d(h.n_iter_)[0]>=h.max_iter)} for h in getattr(model,'heads',[]) if model.spec['kind']=='logistic' and hasattr(h,'n_iter_')]
        threshold=json.loads((directory/'thresholds.json').read_text());gates['threshold_validation_only'] &= not threshold['test_used'] and threshold['validation_patients']==len(op_ids) and set(threshold['fit_patient_ids'])==op_ids
        probability=cumulative(np.load(directory/'validation_hazards.npy'));gates['cumulative_risk_consistency'] &= bool(np.all(np.isfinite(probability)) and np.all((probability>=0)&(probability<=1)) and np.all(np.diff(probability,axis=1)>=-1e-12))
    # Re-extract an actual TRAIN patient twice from public signals; compare cached
    # arrays and apply future/late-receipt poison at an intermediate cutoff.
    first_id=str(train['patient_ids'][0]);context=next(r['context'] for r in cohort.rows('ehr','train') if r['patient_id']==first_id);pid,stream=next((p,s) for p,s in cohort.streams('train') if p==first_id)
    a=list(patient_history(context,stream,c));b=list(patient_history(context,stream,c));gates['causal_feature_reproduction']=all(np.array_equal(a[i][1][k],b[i][1][k],equal_nan=True) for i in range(51) for k in a[i][1])
    lookup=np.flatnonzero(train['patient_ids']==pid);gates['cache_matches_source']=all(np.allclose(train[k][lookup[i]],a[i][1][k],equal_nan=True) for i in range(51) for k in a[i][1])
    altered={k:v.copy() for k,v in stream.items()};altered['value'][361:]=1e9;altered['available_hour'][361:]=0;before=[r for r in a if r[0]<=360];after=[r for r in patient_history(context,altered,c) if r[0]<=360]
    gates['future_poison_invariant']=all(np.array_equal(before[i][1][k],after[i][1][k],equal_nan=True) for i in range(len(before)) for k in before[i][1])
    tests=subprocess.run(['.venv/bin/python','-m','unittest','discover','-s','tests','-q'],capture_output=True,text=True);log=out/'regression_tests.log';log.write_text(tests.stdout+tests.stderr);log.chmod(0o600);gates['all_regressions_passed']=tests.returncode==0
    registration=json.loads((out/'registration.json').read_text());gates['previous_phases_preserved']=all(file_hash(p)==h for p,h in registration['previous_public_sha256'].items()) and file_hash('data/synthetic/phase5_complete.json')==registration['phase5_completion_sha256']
    gates['phase5_certificate_identity']=file_hash(Path(c['cohort'])/'large_validation.json')==registration['cohort_certificate_sha256'] and cohort.metadata_sha==registration['cohort_manifest_sha256'] and file_hash(Path(c['cohort'])/'splits.json')==registration['split_sha256']
    prior=json.loads(Path('data/synthetic_grounding/prior_preservation.json').read_text())
    gates['prior_private_metadata_preserved']=all(Path(p).stat().st_size==s['bytes'] and Path(p).stat().st_mtime_ns==s['mtime_ns'] for p,s in prior['private_size_mtime'].items())
    gates['phase5_original_artifact_hashes']=all(file_hash(Path(c['cohort'])/p)==m['sha256'] for p,m in cohort.manifest['artifacts'].items())
    for p in out.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    files=sorted(str(p) for p in out.rglob('*') if p.is_file())
    with tempfile.TemporaryDirectory() as temporary:
        shutil.copyfile('.gitignore',Path(temporary)/'.gitignore');subprocess.run(['git','init','-q',temporary],check=True);ignored=subprocess.run(['git','-C',temporary,'check-ignore','--stdin'],input='\n'.join(files)+'\n',text=True,capture_output=True);gates['private_git_exclusion']=set(files)==set(ignored.stdout.splitlines())
    result={'gates':{k:bool(v) for k,v in gates.items()},'all_passed':bool(all(gates.values())),'selected_optimizer_diagnostics':optimizers,'optimizer_scope':'Fixed300iteration logistic candidate budget; any hit-limit selected approximation is retained as a stated limitation, not retuned usingTEST','model_code_sha256':code_hash(),'config_sha256':c['_sha256'],'test_clinical_access':False,**EVIDENCE};private_json(out/'preflight.json',result)
    print({'phase6_preflight_passed':result['all_passed'],'failed_gates':[k for k,v in gates.items() if not v]},flush=True);return result

if __name__=='__main__':run()
