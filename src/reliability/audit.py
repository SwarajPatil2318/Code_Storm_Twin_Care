"""Final requirement evidence; metadata/hashes only, no fresh model evaluation."""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from src.data.io import file_hash
from src.reliability.access import load_config,root,save_json,timed,PHASE6
from src.reliability.sealing import verify


def regressions(config=None):
    c=config or load_config();out=root(c);marker=out/'final_regression_tests.json'
    if marker.exists():return json.loads(marker.read_text())
    start=time.monotonic()
    result=subprocess.run(['.venv/bin/python','-m','unittest','discover','-s','tests','-q'],capture_output=True,text=True)
    path=out/'regression_tests.log';path.write_text(result.stdout+result.stderr);path.chmod(0o600)
    import re
    match=re.search(r'Ran (\d+) tests',result.stdout+result.stderr)
    record={'exit_code':result.returncode,'test_count':int(match.group(1)) if match else None,
        'full_suite_runs_this_phase':1,'log_sha256':file_hash(path)}
    save_json(marker,record);timed(c,'final_regression_tests_once',start)
    print({'phase65_regressions_passed':result.returncode==0,'test_count':record['test_count']},flush=True)
    if result.returncode:raise ValueError('Final regressions failed; inspect private log without dumping records')
    return record


def audit(config=None):
    c=config or load_config();out=root(c);frozen=verify(c)
    complete=json.loads((out/'replication_evaluation_complete.json').read_text())
    result=json.loads((out/'replication_results.json').read_text())
    report=json.loads((out/'report_manifest.json').read_text())
    gate=json.loads((Path(c['replication'])/'replication_audit.json').read_text())
    tests=json.loads((out/'final_regression_tests.json').read_text())
    old=json.loads((PHASE6/'phase6_complete.json').read_text())
    prior_docs=old['documentation_sha256']
    docs_preserved=True
    for name,h in prior_docs.items():
        path=Path(name)
        if not path.exists():path=Path('docs')/name
        docs_preserved &= path.exists() and file_hash(path)==h
    fields=['reliability','adaptive_sensing','calibration','risk','observed_state','expected_shadow_state']
    states=json.loads((out/'phase7_twin_states.json').read_text())['states']
    figures=json.loads((out/'figure_manifest.json').read_text())
    decision=json.loads((out/'decision.json').read_text())
    checks={'frozen_source_models_and_configuration':True,
        'exactly_one_replication_evaluation':complete['status']=='COMPLETE' and complete['logical_replication_evaluations']==result['logical_replication_evaluations']==1 and complete['event_id']==result['event_id'],
        'replication_result_checksum':complete['results_sha256']==file_hash(out/'replication_results.json'),
        'independent_seed2027_cohort_audited':gate['all_passed'] and result['patients']==2500 and result['landmarks']==127500,
        'no_spent_test_use_recorded':result['original_phase6_test_used'] is False and frozen['original_test_clinical_access_permitted'] is False,
        'phase6_reports_preserved':bool(docs_preserved),
        'six_documentation_artifacts':len(report['documentation_sha256'])==6 and all(file_hash(Path('docs')/name)==h for name,h in report['documentation_sha256'].items()),
        'six_necessary_figures':figures['count']==6 and all((out/'figures'/(name+'.png')).exists() for name in figures['figures']),
        'headline_patient_cluster_intervals':all(v['requested']==1000 for system in result['intervals'].values() for horizon in system.values() for v in horizon.values()),
        'phase7_structured_outputs':len(states)>=3 and all(set(fields)<=set(s) for s in states),
        'negative_demo_results_retained':(out/'phase7_demo_cases.json').exists(),
        'explicit_synthetic_evidence':all(s['evidence_label']=='SYNTHETIC POC VALIDATION' and s['clinical_validation']=='NOT CLINICALLY VALIDATED' for s in states),
        'full_regression_suite_once':tests['exit_code']==0 and tests['full_suite_runs_this_phase']==1 and tests['test_count']>=219,
        'decision_exact':decision['decision'] in {'GO','PARTIAL GO','NO-GO'},
        'private_permissions':all(stat.S_IMODE(p.stat().st_mode)==(0o700 if p.is_dir() else 0o600) for p in [out,Path(c['replication'])]+list(out.rglob('*'))+list(Path(c['replication']).rglob('*')))}
    save_json(out/'final_audit.json',{'all_passed':all(checks.values()),'gates':checks,
        'scope':'Persisted evidence and checksums only; no test clinical decode or new prediction',
        'replication_evaluation_event_id':result['event_id']})
    print({'phase65_final_audit_passed':all(checks.values()),'failed':[k for k,v in checks.items() if not v]},flush=True)
    if not all(checks.values()):raise ValueError('Final completeness audit failed')
    save_json(out/'phase6_5_complete.json',{'status':'COMPLETE','decision':decision['decision'],
        'replication_evaluations':1,'replication_evaluation_event_id':result['event_id'],
        'frozen_manifest_sha256':file_hash(out/'PHASE6_5_FROZEN_MANIFEST.json'),
        'artifacts_sha256':{str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='phase6_5_complete.json'},
        'documentation_sha256':report['documentation_sha256']})
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tests',action='store_true');args=parser.parse_args()
    if args.tests:regressions()
    else:audit()
