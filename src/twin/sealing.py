"""Freeze architecture/calibration/thresholds before one joint test transaction."""
import datetime as dt,json,uuid,os
from pathlib import Path
import joblib
from src.twin.protocol import load,EVIDENCE,code_hash
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json

REGISTRY=Path('data/results/phase6_test_registry')

def freeze(config=None):
    c=config or load();out=Path(c['output']);path=out/'selection_seal.json'
    if path.exists():return verify(c)
    development=json.loads((out/'development_results.json').read_text());uncertainty=json.loads((out/'uncertainty_fit.json').read_text())
    if set(development['models'])!=set(c['model_variants']):raise ValueError('Required comparisons incomplete')
    required=[out/'shadow_engines.joblib',out/'registration.json',out/'development_results.json',out/'uncertainty_fit.json',out/'robustness_registration.json',out/'decision_registration.json']
    for role in ['train','validation','consumer_train','consumer_validation']:required.append(out/'features'/role/'manifest.json')
    for name in c['model_variants']:
        directory=out/'models'/name
        required.extend(directory/n for n in ['selected.joblib','selection.json','calibration.joblib','thresholds.json'])
    for name in uncertainty['models']:
        required.extend(out/'models'/name/f'bootstrap_{member}.joblib' for member in range(c['uncertainty']['bootstrap_members']))
        required.extend(out/'models'/name/f'bootstrap_{member}.json' for member in range(c['uncertainty']['bootstrap_members']))
    if not (out/'preflight.json').exists():raise ValueError('Complete pre-test preflight required')
    preflight=json.loads((out/'preflight.json').read_text())
    if not preflight['all_passed'] or preflight['model_code_sha256']!=code_hash():raise ValueError('Stale or failed preflight')
    required.append(out/'preflight.json');hashes={str(p.relative_to(out)):file_hash(p) for p in required}
    seal={'status':'FROZEN','config_sha256':c['_sha256'],'model_code_sha256':code_hash(),'cohort_manifest_sha256':file_hash(Path(c['cohort'])/'manifest.json'),'artifacts':hashes,
      'feature_set':development['models']['full_twin']['selection']['input_features'],'challenge_definition':c['challenge'],'primary_model':development['primary_model'],'frozen_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'all_choices_frozen':['architectures','hyperparameters','features','Shadow','TDI','calibration','thresholds','challenge','uncertainty','robustness','comparators'],'test_opened':False,**EVIDENCE}
    private_json(path,seal);print({'phase6_model_selection_frozen':True,'test_opened':False,'primary_model':seal['primary_model']},flush=True);return seal

def verify(config=None):
    c=config or load();out=Path(c['output']);seal=json.loads((out/'selection_seal.json').read_text())
    if seal['status']!='FROZEN' or seal['config_sha256']!=c['_sha256'] or seal['model_code_sha256']!=code_hash():raise ValueError('Frozen study source/config changed')
    for name,h in seal['artifacts'].items():
        if file_hash(out/name)!=h:raise ValueError('Frozen study artifact changed')
    return seal

def begin(config=None):
    c=config or load();out=Path(c['output']);seal=verify(c);REGISTRY.mkdir(parents=True,exist_ok=True,mode=0o700);REGISTRY.chmod(0o700);registry=REGISTRY/(seal['cohort_manifest_sha256']+'.json')
    if (out/'evaluation_complete.json').exists():return json.loads((out/'evaluation_complete.json').read_text())
    if registry.exists():
        transaction=json.loads(registry.read_text())
        if transaction['output_root']!=str(out.resolve()) or transaction['selection_seal_sha256']!=file_hash(out/'selection_seal.json'):raise ValueError('Synthetic test already reserved/evaluated by another frozen study')
        if not (out/'evaluation_started.json').exists():private_json(out/'evaluation_started.json',transaction)
        return transaction
    transaction={'event_id':str(uuid.uuid4()),'status':'STARTED','output_root':str(out.resolve()),'selection_seal_sha256':file_hash(out/'selection_seal.json'),'cohort_manifest_sha256':seal['cohort_manifest_sha256'],
      'model_code_sha256':seal['model_code_sha256'],'config_sha256':c['_sha256'],'model_artifact_sha256':{k:v for k,v in seal['artifacts'].items() if k.endswith('.joblib')},
      'evaluation_started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'joint_frozen_evaluation':True,'optimization_against_test_allowed':False,**EVIDENCE}
    descriptor=os.open(registry,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,'w') as f:json.dump(transaction,f,sort_keys=True,indent=2);f.write('\n')
    private_json(out/'evaluation_started.json',transaction);return transaction

def complete(results,config=None):
    c=config or load();out=Path(c['output']);transaction=json.loads((out/'evaluation_started.json').read_text());transaction.update(status='COMPLETE',completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      results_sha256=file_hash(out/'test_results.json'),test_patients=1500,test_landmarks=76500,logical_test_evaluations=1)
    private_json(out/'evaluation_complete.json',transaction);private_json(REGISTRY/(transaction['cohort_manifest_sha256']+'.json'),transaction);return transaction

if __name__=='__main__':freeze()
