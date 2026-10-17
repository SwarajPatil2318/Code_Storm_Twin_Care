"""Complete pilot acceptance with reproduction, regression, schemas and privacy."""
import argparse,hashlib,json,stat,subprocess,tempfile,shutil,time
from pathlib import Path
import numpy as np
from src.synthetic.config import load
from src.synthetic.generate import generate,iter_rows,code_hash
from src.synthetic.reader import canonical_observations,canonical_contexts
from src.data.schema import record
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def verify(root='data/synthetic/pilot',config_path=None):
    root=Path(root);validation=json.loads((root/'validation.json').read_text());manifest=json.loads((root/'manifest.json').read_text());config=load(config_path) if config_path else load()
    if validation['simulator_code_sha256']!=code_hash() or validation['config_sha256']!=config['_config_sha256']:raise ValueError('Stale pilot validation')
    preserved=json.loads(Path('data/synthetic_grounding/prior_preservation.json').read_text());unchanged_public=all(Path(p).exists() and file_hash(p)==h for p,h in preserved['public_sha256'].items())
    unchanged_private=all(Path(p).exists() and Path(p).stat().st_size==m['bytes'] and Path(p).stat().st_mtime_ns==m['mtime_ns'] for p,m in preserved['private_size_mtime'].items())
    if not unchanged_public or not unchanged_private:raise ValueError('Earlier phase files changed')
    grounding=json.loads(Path(config['grounding_path']).read_text())
    for path,expected in grounding['sources_sha256'].items():
        if file_hash(path)!=expected:raise ValueError('Measured grounding source changed')
    for dataset in ['chiron','perheart']:
        if file_hash(grounding[dataset]['distribution_cache'])!=grounding[dataset]['distribution_cache_sha256']:raise ValueError('Grounding distribution cache changed')
    # Repeat a sample from the actual measured grounding/config twice, not a toy fixture.
    with tempfile.TemporaryDirectory(prefix='twin_synthetic_reproduction_') as directory:
        a=Path(directory)/'a';b=Path(directory)/'b';first=generate(24,manifest['days'],manifest['seed'],a,config_path);second=generate(24,manifest['days'],manifest['seed'],b,config_path)
        byte_same=first['artifacts']==second['artifacts']
        # Also compare the prefix to the full pilot using array/record equivalence;
        # batch partitioning differs between sample24 and cohort1000.
        with np.load(a/'observations/batch_00000.npz') as sample,np.load(root/'observations/batch_00000.npz') as pilot:
            prefix_same=all(np.array_equal(sample[k],pilot[k][:24],equal_nan=True) if sample[k].ndim==3 else True for k in ['raw_value','value','quality_status','missingness_mask','scheduled','available_hour','reason_code'])
        prefix_ehr=list(iter_rows(a/'ehr/records.jsonl.gz'))==list(iter_rows(root/'ehr/records.jsonl.gz'))[:24]
    fhir_ok=True;resources=0;demographics={r['patient_id']:r for r in iter_rows(root/'patients/records.jsonl.gz')};ehrids=set()
    for row in iter_rows(root/'ehr/records.jsonl.gz'):
        pid=row['patient_id'];ehrids.add(pid);entries=row['fhir']['entry'];identities={e['resource']['resourceType']+'/'+e['resource']['id'] for e in entries};patient_ref='Patient/'+pid.split(':')[1]
        fhir_ok &= patient_ref in identities and pid in demographics and row['available_hour']<0
        for entry in entries:
            r=entry['resource'];resources+=1;fhir_ok &= entry['fullUrl'].startswith(('https://','urn:'))
            if r['resourceType']!='Patient':fhir_ok &= r['subject']['reference']==patient_ref
            if r['resourceType']=='Observation':fhir_ok &= r['issued']>=r['effectiveDateTime'] and r['status']=='final' and r['valueQuantity']['system']=='http://unitsofmeasure.org'
    fhir_ok &= ehrids==set(demographics)
    canonical=list(canonical_observations(root,limit=200));canonical_ok=all(o.patient_id in demographics and o.dataset=='synthetic' and (o.value.usable_value is None or isinstance(o.value.usable_value,float)) for o in canonical)
    for obs in canonical:json.dumps(record(obs),allow_nan=False)
    contexts=list(canonical_contexts(root));canonical_ok &= len(contexts)==manifest['patients'] and all(c.patient_id in demographics and c.available_time>=c.observed_time for c in contexts)
    for context in contexts:json.dumps(record(context),allow_nan=False)
    # Existing older tests use their synthetic fixtures and do not evaluate the
    # spent real test partition. Keep the complete test output in a private artifact.
    tests=subprocess.run(['.venv/bin/python','-m','unittest','discover','-s','tests','-q'],capture_output=True,text=True)
    log=root/'regression_tests.log';log.write_text(tests.stdout+tests.stderr);log.chmod(0o600)
    if tests.returncode:print({'regression_tests_failed':True,'private_log':str(log)},flush=True)
    for p in root.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    paths=sorted(p for p in root.rglob('*') if p.is_file());permissions=all(stat.S_IMODE(p.stat().st_mode)==(0o700 if p.is_dir() else 0o600) for p in [root]+list(root.rglob('*')))
    with tempfile.TemporaryDirectory() as directory:
        shutil.copyfile('.gitignore',Path(directory)/'.gitignore');subprocess.run(['git','init','-q',directory],check=True)
        ignored=subprocess.run(['git','-C',directory,'check-ignore','--stdin'],input='\n'.join(map(str,paths))+'\n',text=True,capture_output=True)
        all_ignored=set(ignored.stdout.splitlines())==set(map(str,paths))
    # Runtime/metadata are excluded from canonical record reproduction; arrays,
    # FHIR, observations, features, labels, truths and split files are exact.
    validation['gates'].update(reproducibility=bool(byte_same and prefix_same and prefix_ehr),regression_tests=tests.returncode==0,
      fhir_canonical_linkage=bool(fhir_ok and canonical_ok),ground_truth_privacy=bool(permissions and all_ignored))
    validation['all_gates_passed']=all(validation['gates'].values());validation['evidence_label']='SYNTHETIC POC VALIDATED' if validation['all_gates_passed'] else 'NOT VALIDATED'
    validation['verification']={'sample_patients_regenerated_twice':24,'artifact_files_byte_identical':len(first['artifacts']) if byte_same else 0,'pilot_prefix_arrays_identical':bool(prefix_same),
      'pilot_prefix_ehr_records_identical':prefix_ehr,'fhir_resources_checked':resources,'canonical_observations_checked':len(canonical),'canonical_contexts_checked':len(contexts),'private_files_git_excluded':len(paths),
      'previous_public_files_unchanged':len(preserved['public_sha256']),'previous_private_metadata_unchanged':len(preserved['private_size_mtime']),
      'old_private_verification':'size/mtime only; no spent-test clinical files hashed or decoded',
      'regression_tests_exit_code':tests.returncode,'not_clinical_validation':True,'runtime_metadata_not_byte_compared':True}
    private_json(root/'validation.json',validation)
    private_json(root/'reproducibility.json',{'seed':manifest['seed'],'config_sha256':manifest['config_sha256'],'simulator_code_sha256':code_hash(),
      'source_sha256':{str(p):file_hash(p) for p in sorted(Path('src/synthetic').glob('*.py'))},'grounding_sha256':manifest['grounding_sha256'],
      'numpy_version':np.__version__,'all_gates_passed':validation['all_gates_passed'],'sample_byte_reproduction':byte_same,'evidence':'SYNTHETIC POC','evidence_label':validation['evidence_label']})
    print({'pilot_all_gates_passed':validation['all_gates_passed'],'remaining_failed_gates':[k for k,v in validation['gates'].items() if not v],
      'sample_exact_reproduction':byte_same,'regressions_passed':tests.returncode==0,'fhir_resources_checked':resources},flush=True)
    return validation

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/synthetic/pilot');parser.add_argument('--config');args=parser.parse_args()
    with threadpool_limits(limits=1):verify(args.root,args.config)
