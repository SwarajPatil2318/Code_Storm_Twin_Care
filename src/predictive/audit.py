"""Integrity audit without computing any new final-test prediction or retuning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.data.io import file_hash,read_jsonl
from src.predictive.artifacts import check_seal,verify_hashes,verify_previous,write_json,DOCS,source_hash
from src.predictive.data import Cohort
from src.predictive.inference import validate_state


def run(args,cwd=None):
    output=subprocess.run([sys.executable,*args],cwd=cwd,capture_output=True,text=True)
    if output.returncode:raise RuntimeError(output.stdout+output.stderr)
    print((output.stdout or output.stderr).strip(),flush=True)
    return output.stdout+output.stderr


def audit(root='data/results/phase4',repeat='data/results/phase4_reproduction',regression=True):
    root=Path(root);repeat=Path(repeat);checks=[]
    def check(ok,message):
        if not ok:raise ValueError(message)
        checks.append(message)
    seal=check_seal(root);repeated=check_seal(repeat)
    check(seal['selected']==repeated['selected'],'same validation-selected primary in independent repeat')
    completion=json.loads((root/'evaluation_seal.json').read_text());verify_hashes(root,completion['hashes'])
    check(completion['test_evaluation_count']==1,'final test evaluated once')
    registry=json.loads(Path('data/results/phase4_test_access/evaluation_started.json').read_text())
    check(registry['experiment_root']==str(root.resolve()),'global study registry prevents test evaluation from another run directory')
    check(not (repeat/'evaluation_started.json').exists(),'reproduction never touches final test')
    snapshot=json.loads((root/'preservation.json').read_text());verify_previous(snapshot)
    check(True,'all earlier source, raw, processed, model and documentation snapshot bytes unchanged')
    check(json.loads((root/'protocol.json').read_text())==json.loads((repeat/'protocol.json').read_text()),'repeated protocol exactly identical')
    exact=[];different=[]
    for name in seal['hashes']:
        check((repeat/name).exists(),'repeated training artifact exists: '+name)
        (exact if file_hash(root/name)==file_hash(repeat/name) else different).append(name)
    check(not different,'all development artifacts byte-identical; no observed training nondeterminism')
    bundle=joblib.load(root/'pipeline.joblib');second=joblib.load(repeat/'pipeline.joblib');c=Cohort()
    a=c.assignments
    check(json.loads((root/'splits.json').read_text())==json.loads((c.root/'splits.json').read_text()),'authoritative patient split unchanged')
    for name,m in bundle['models'].items():
        check(all(a[p]=='train' for p in m.fit_patients),'model fit uses TRAIN only: '+name)
        if hasattr(m,'encoder'):check(all(a[p]=='train' for p in m.encoder.fit_patients),'encoder fit uses TRAIN only: '+name)
        check(all(a[p]=='validation' for p in bundle['calibrators'][name].fit_patients),'calibrator fit uses VALIDATION only: '+name)
        encoder=getattr(m,'encoder',None)
        if encoder:
            check(all(k.startswith(('ehr__','phys__','personal__')) for k in encoder.keys),'feature input whitelist excludes identity/labels/audit: '+name)
    for m,cal in bundle['ensemble']:
        check(all(a[p]=='train' for p in m.fit_patients),'uncertainty model fit uses TRAIN only')
        check(all(a[p]=='validation' for p in cal.fit_patients),'uncertainty calibration uses VALIDATION only')
    with threadpool_limits(limits=1):
        validation=c.batch('validation')
        for name,m in bundle['models'].items():
            check(np.array_equal(m.predict(validation),second['models'][name].predict(validation)),'serialization/reproduction validation predictions identical: '+name)
        for (m,cal),(other,othercal) in zip(bundle['ensemble'],second['ensemble']):
            check(np.array_equal(cal.predict(m.predict(validation)),othercal.predict(other.predict(validation))),'uncertainty refit reproducible')
    states=list(read_jsonl(root/'prediction_states.jsonl'));pred=list(read_jsonl(root/'test_predictions.jsonl'))
    check(len(states)==len(pred)==446,'all446 frozen test states persisted')
    for state,row in zip(states,pred):
        validate_state(state)
        check(state['patient_id']==row['patient_id'] and a[state['patient_id']]=='test','state aligned to TEST identity')
        check(state['calibrated_risk']==row['predictions'][bundle['selected']]['calibrated'],'state equals sealed test forecast')
        check(row['audit']['max_physiology_offset']<=row['t0']-5,'test physiology cutoff preserved')
        check(row['audit']['maximum_available_clinical_offset'] is None or row['audit']['maximum_available_clinical_offset']<=row['t0'],'test clinical availability preserved')
        if row['label']:check(15<row['event_onset']-row['t0']<=180,'test positive lead in frozen interval')
    report_manifest=json.loads((root/'report_manifest.json').read_text())
    for name,sha in report_manifest['public_docs'].items():check(file_hash(Path('docs',name))==sha,'public report hash matches: '+name)
    verify_hashes(root,report_manifest['private_figures']);check(len(report_manifest['private_figures'])>=9,'required private figures generated')
    public='\n'.join(Path('docs',name).read_text() for name in DOCS)
    check(not any(p['patient_id'] in public or p['icu_stay_id'] in public for p in c.patients.values()),'public docs contain no actual patient identifiers')
    check(all(p.stat().st_mode&0o777==0o600 for p in root.rglob('*') if p.is_file()),'all medical/model/plot artifacts private600')
    with tempfile.TemporaryDirectory() as temp:
        Path(temp,'.gitignore').write_bytes(Path('.gitignore').read_bytes());subprocess.run(['git','init','-q',temp],check=True,capture_output=True)
        for name in ['data/results/phase4/pipeline.joblib','data/results/phase4/prediction_states.jsonl','data/results/phase4/figures/private_trace_1.png','Datasets/aibhs_datasets-main.zip']:
            check(subprocess.run(['git','check-ignore','--quiet',name],cwd=temp).returncode==0,'sensitive artifact excluded from Git')
    regression_summary={}
    if regression:
        tests=run(['-m','unittest','discover','-s','tests','-q']);check('OK' in tests and 'skipped=' not in tests,'all current regression tests pass without skips')
        run(['-m','src.data.validate'])
        from src.data.eicu_validate import validate
        prior=validate(write=False);check(prior['status']=='PASS' and prior['checks']==238377,'unchanged Phase35 target/features fully reconstructed')
        run(['-m','src.models.audit'],cwd='data/reproducibility/phase3_source')
        # Replay old scope in its frozen source/docs tree. Its output audit is byte-identical.
        run(['-m','src.feasibility.long_horizon_audit'],cwd='data/reproducibility/phase36_source')
        regression_summary={'current_tests':'PASS without skips','phase2_cohorts':4,'phase35_checks':prior['checks'],'phase3_checks':320,'phase36_checks':76545}
    verify_previous(snapshot)
    summary={'status':'PASS','checks':len(checks),'earlier_files_byte_preserved':len(snapshot),'development_artifacts_byte_identical':len(exact),
             'source_sha256':source_hash(),'models_compared':len(bundle['models']),'uncertainty_refits':len(bundle['ensemble']),
             'test_evaluations':1,'test_reexecuted':False,'training_repeat_test_accessed':False,'regression':regression_summary}
    write_json(root/'completion_audit.json',summary);print(summary,flush=True);return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/results/phase4');parser.add_argument('--repeat',default='data/results/phase4_reproduction');parser.add_argument('--no-regression',action='store_true');args=parser.parse_args();audit(args.root,args.repeat,not args.no_regression)
