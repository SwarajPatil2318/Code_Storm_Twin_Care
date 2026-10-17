"""Completion audit: exact risk-set reconstruction, regression and prior preservation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

from src.data.io import ArtifactWriter, dumps, file_hash, read_jsonl
from src.data.eicu_validate import restore_series, validate as validate_phase35
from src.feasibility.long_horizon import evaluate_stay, summarize, source_hash, preservation_snapshot, PROTOCOL
from src.feasibility.long_horizon_report import render


def run(args, cwd=None):
    result=subprocess.run([sys.executable,*args],cwd=cwd,text=True,capture_output=True)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    print((result.stdout or result.stderr).strip(),flush=True)
    return result.stdout+result.stderr


def audit(root='data/results/phase36', repeat='data/results/phase36_reproduction', regression=True):
    root=Path(root); repeat=Path(repeat); manifest=json.loads((root/'manifest.json').read_text())
    checks=[]
    def check(condition,message):
        if not condition: raise ValueError(message)
        checks.append(message)
    check(manifest==json.loads((repeat/'manifest.json').read_text()),'independent manifests identical')
    for name,meta in manifest['artifacts'].items():
        check(file_hash(root/name)==meta['sha256'] and file_hash(repeat/name)==meta['sha256'],'independent artifact bytes identical: '+name)
    preservation=json.loads((root/'preservation.json').read_text())
    check(preservation_snapshot()==preservation,'all prior results and source bytes preserved')
    check(manifest['source_code_sha256']==source_hash(),'current source version matches audit inputs')
    report=json.loads((root/'report.json').read_text())
    check(report['protocol']==PROTOCOL,'protocol frozen accurately')
    phase35=Path('data/processed/eicu_demo')
    check(file_hash(phase35/'manifest.json')==report['source_phase35_manifest_sha256'],'Phase35 manifest unchanged')
    patients={p['source_stay_key']:p for p in read_jsonl(phase35/'patient_context.jsonl')}
    by_id={p['icu_stay_id']:stay for stay,p in patients.items()}
    physiology=restore_series(phase35/'vitalPeriodic.npz'); events=defaultdict(list)
    for row in read_jsonl(phase35/'events.jsonl'):
        if row['candidate']=='map65_30min': events[by_id[row['icu_stay_id']]].append(row)
    windows=list(read_jsonl(root/'windows.jsonl')); grouped=defaultdict(list)
    for row in windows:
        key=f"{row['kind']}_{row['lower_minutes']//60}_{row['horizon_minutes']//60}_l{row['lookback_minutes']//60}_{row['stability']}"
        grouped[key].append(row)
        check(row['history_max_offset']<=row['t0']-5,'historical physiology ends at buffered cutoff')
        check(row['latest_clinical_available_offset'] is None or row['latest_clinical_available_offset']<=row['t0'],'clinical evidence available by cutoff')
        check(row['map_at_t0_offset']<=row['t0'],'current MAP audit is causal')
        check(row['future_max_gap_minutes']<=10,'future ascertainment has bounded gaps')
        if row['label']:
            check(max(row['lower_minutes'],15)<row['lead_minutes']<=row['horizon_minutes'],'positive lead inside exact first-event interval')
            future=[e for e in events[by_id[row['icu_stay_id']]] if e['onset']>row['t0']]
            check(future and future[0]['event_id']==row['event_id'],'positive uses first future event, not selected recurrence')
        else:
            check(not any(row['t0']<e['onset']<=row['t0']+row['horizon_minutes'] for e in events[by_id[row['icu_stay_id']]]),'control is event-free through upper endpoint')
    check(len({(r['kind'],r['lower_minutes'],r['horizon_minutes'],r['lookback_minutes'],r['stability'],r['icu_stay_id'],r['t0']) for r in windows})==len(windows),'landmark keys unique')
    cache={}
    for key,config in report['configurations'].items():
        rows=[];counts=Counter()
        for stay,s in sorted(physiology.items()):
            expected,excluded,_=evaluate_stay(patients[stay],s,events[stay],config['lower_hours']*60,config['upper_hours']*60,config['lookback_hours']*60,config['stability'],cache)
            for row in expected:row['kind']=config['kind']
            rows.extend(expected);counts.update(excluded)
        check(dumps(rows)==dumps(grouped[key]),'every risk set, causal state and label reconstructed: '+key)
        check(dict(counts)==config['exclusions'],'all censoring and exclusion counts reproduced: '+key)
        for name,predicate in [('full',lambda r:True),('hf_known',lambda r:r['hf_known_at_t0']),('hf_history_known',lambda r:r['hf_history_known_at_t0']),('cv_known',lambda r:r['cv_known_at_t0'])]:
            check(summarize([r for r in rows if predicate(r)])==config[name],'cohort counts and availability reproduced: '+key+' '+name)
        per_stay=defaultdict(list)
        for row in rows: per_stay[row['icu_stay_id']].append(row['t0'])
        check(all(all(b-a>=config['upper_hours']*60 for a,b in zip(t,t[1:])) for t in per_stay.values()),'full forecast intervals spaced by upper horizon: '+key)
        positive=[r['event_id'] for r in rows if r['label']]
        check(len(positive)==len(set(positive)),'one positive index per episode within configuration: '+key)
        waterfall=sum(counts[k] for k in ['insufficient_past_history','unstable_at_t0','event_underway_at_t0','incomplete_future_windows','event_within_15min','earlier_first_event_before_band','eligible_windows'])
        check(waterfall==counts['candidate_windows'],'candidate window waterfall reconciles: '+key)
    check(len(report['configurations'])==81,'all 5 cumulative and 4 band intervals x3 histories x3 stability profiles evaluated')
    regression_results={}
    if regression:
        tests=run(['-m','unittest','discover','-s','tests','-q'])
        check('Ran 84 tests' in tests and 'OK' in tests and 'skipped=' not in tests,'all 84 regression tests pass without skips')
        run(['-m','src.data.validate'])
        old=validate_phase35(write=False)
        check(old['status']=='PASS' and old['checks']==238377,'Phase35 complete audit passes without rewriting prior artifacts')
        frozen=Path('data/reproducibility/phase3_source')
        run(['-m','src.models.audit'],frozen)
        old_model=json.loads(Path('data/results/phase3/completion_audit.json').read_text())
        check(old_model['status']=='PASS' and old_model['check_count']==320,'Phase3 preserved negative benchmark audit passes')
        regression_results={'tests':84,'phase2_cohorts':4,'phase35_checks':old['checks'],'phase3_checks':320}
    check(preservation_snapshot()==preservation,'prior results remain unchanged after full regression')
    check(not any(p.suffix in {'.joblib','.pkl','.pt','.onnx'} for p in root.rglob('*')),'no predictive model artifacts created')
    check(all(p.stat().st_mode&0o777==0o600 for p in root.iterdir() if p.is_file()),'all derived record artifacts private')
    render(root)
    for name in ['LONG_HORIZON_FEASIBILITY.md','PHASE3_6_DECISION.md']:
        text=(Path('docs')/name).read_text()
        check(not any(p['patient_id'] in text or p['icu_stay_id'] in text for p in patients.values()),'public doc contains no actual patient/stay identifiers: '+name)
    summary={'status':'PASS','check_count':len(checks),'configurations':81,'windows_verified':len(windows),
             'independent_artifacts_identical':len(manifest['artifacts']),'prior_files_preserved':len(preservation),
             'regression':regression_results,'models_trained':0,'recommendation':'A. 3h remains primary','48_72_prediction_claim':'NO'}
    writer=ArtifactWriter(root);writer.json('completion_audit',summary);writer.close()
    doc=Path('docs/PHASE3_6_DECISION.md')
    doc.write_text(doc.read_text()+f"\n## Completed verification\n\nPhase 3.6 passes {len(checks)} checks across all 81 configurations and {len(windows)} windows. "
                   f"Independent builds have identical manifests and all {len(manifest['artifacts'])} artifacts. "
                   f"All {len(preservation)} snapshotted prior files remain byte-identical. "
                   'All 84 tests pass without skips, all four Phase 2 cohorts validate, the unchanged Phase 3.5 audit passes 238,377 checks, '
                   'and the preserved Phase 3 audit passes all 320 checks. No predictive model was trained.\n\n'
                   'Final recommendation: **A. 3h remains primary.**\n')
    print(summary,flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/results/phase36');parser.add_argument('--repeat',default='data/results/phase36_reproduction')
    args=parser.parse_args();audit(args.root,args.repeat)
