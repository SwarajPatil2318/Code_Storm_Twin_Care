"""Read-only completion checks. Never loads predictors, fits models, or evaluates eICU."""
import argparse,gzip,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from src.vitaldb_feasibility.source import ROOT,sha,private_json,cases,tracks,decode
from src.vitaldb_feasibility.retrieve import TRACKS
from src.vitaldb_feasibility.windows import episodes,counts,causal_features
DOCS=['VITALDB_DATA_AUDIT.md','VITALDB_SIGNAL_MAP.md','VITALDB_HYPOTENSION_FEASIBILITY.md','VITALDB_TRANSFER_PLAN.md','PHASE4_5A_DECISION.md']

def check(condition,message):
    if not condition:raise AssertionError(message)

def volume():
    metadata=json.loads((ROOT/'metadata/retrieval_manifest.json').read_text());network=sum(v['bytes'] for v in metadata.values())
    calls=len(metadata);explicit_hashes={};extra=0
    # Initial bounded probes retained before the canonical metadata fetch also consumed bytes.
    for name in ['metadata/cases.csv','metadata/api_index.html']:
        p=ROOT/name
        if p.exists():extra+=p.stat().st_size;calls+=1
    network+=extra
    for p in ROOT.rglob('*.provenance.json'):
        response=p.with_suffix('').with_suffix('')
        # Metadata responses are already counted in their consolidated manifest.
        if response.parent==ROOT/'metadata':continue
        record=json.loads(p.read_text());network+=record['bytes'];calls+=1
    histories=list(ROOT.glob('retrieval_history_*.json'))
    numeric=list((ROOT/'numeric').rglob('*.json'))
    completed_run_bytes=sum(r['new_network_body_bytes'] for p in histories for r in json.loads(p.read_text()))
    successful_numeric_bytes=sum(json.loads(p.read_text())['body_bytes'] for p in numeric)
    network+=max(completed_run_bytes,successful_numeric_bytes)
    calls+=len(numeric)
    failed_attempts=sum(len(r['failures']) for p in histories for r in json.loads(p.read_text()))
    files=[p for p in ROOT.rglob('*') if p.is_file()]
    return {'known_response_body_bytes':network,'known_response_body_MiB':network/2**20,
            'cache_bytes':sum(p.stat().st_size for p in files),'cache_MiB':sum(p.stat().st_size for p in files)/2**20,
            'cached_files':len(files),'successful_gets_estimate':calls,'failed_get_attempts':failed_attempts,
            'failed_response_body_bytes_unknown':True if failed_attempts else False,
            'volume_excludes_http_tls_headers':True,'full_vital_or_parquet_files_downloaded':0,
            'numeric_minute_cache_bytes':sum(p.stat().st_size for p in (ROOT/'numeric').rglob('*.npz')),
            'initial_probe_body_bytes':extra}

def run():
    snapshot=json.loads((ROOT/'prior_preservation.json').read_text())
    for name,digest in snapshot.items():
        path=Path(name);check(path.is_file() and sha(path.read_bytes())==digest,'Earlier phase artifact changed')
    private_paths=[p for p in ROOT.rglob('*') if p.is_file()]
    for path in private_paths:check(path.stat().st_mode&0o777==0o600,'Private file permission')
    for path in [ROOT,*[p for p in ROOT.rglob('*') if p.is_dir()]]:check(path.stat().st_mode&0o777==0o700,'Private directory permission')
    with tempfile.TemporaryDirectory() as name:
        temp=Path(name);(temp/'.gitignore').write_text(Path('.gitignore').read_text());subprocess.run(['git','init','-q',name],check=True,capture_output=True)
        result=subprocess.run(['git','-C',name,'check-ignore','--stdin'],input='\n'.join(str(p) for p in private_paths)+'\n',text=True,capture_output=True,check=True)
        check(len(result.stdout.splitlines())==len(private_paths),'All record-bearing artifacts Git ignored')
    cohort=cases();index={}
    for row in tracks():index[(row['caseid'],row['tname'])]=row
    map_expected={c['caseid'] for c in cohort.values() if any((c['caseid'],n) in index for n in TRACKS['map'])}
    check(len(list((ROOT/'numeric/map').glob('*.json')))==len(map_expected),'Exhaustive MAP union retrieved')
    selection=set(json.loads((ROOT/'multisignal_selection.json').read_text())['case_ids']);stream_count=0
    for concept in TRACKS:
        expected=map_expected if concept=='map' else {i for i in selection if any((i,n) in index for n in TRACKS[concept])}
        found={p.stem for p in (ROOT/'numeric'/concept).glob('*.json')}
        check(found==expected,'Selected concept coverage complete')
        for caseid in found:
            path=ROOT/'numeric'/concept/(caseid+'.json');record=json.loads(path.read_text());cache=path.with_suffix('.npz')
            check(sha(cache.read_bytes())==record['cache_sha256'],'Numeric cache checksum')
            check(record['patient_id']=='vitaldb:subject:'+cohort[caseid]['subjectid'],'Patient namespace')
            check(record['track'] in TRACKS[concept],'Measured discovered source track')
            with np.load(cache,allow_pickle=False) as z:
                finite=np.isfinite(z['latest_raw_seconds']);check(np.all(z['latest_raw_seconds'][finite]<=z['time_minutes'][finite]*60),'Closed minute bin causality')
            for event in record['events']:
                check(event['confirmation_seconds']-event['onset_seconds']>=1800,'Native event confirmation >=30minutes')
                check(event['last_low_seconds']>=event['confirmation_seconds'],'Native confirmation within run')
                check(event['low_observations']>=181,'Dense event minimum support')
            stream_count+=1
    for path in ROOT.rglob('*.provenance.json'):
        raw=path.with_suffix('').with_suffix('');record=json.loads(path.read_text())
        check(sha(raw.read_bytes())==record['sha256'],'Raw bounded response checksum')
        check(sha(decode(raw.read_bytes()))==record['decoded_sha256'],'Decoded response checksum')
    report=json.loads((ROOT/'feasibility_report.json').read_text());rows=[json.loads(line) for line in (ROOT/'risk_windows.jsonl').read_text().splitlines()]
    native= json.loads((ROOT/'native_check_report.json').read_text())
    check(native['status']=='PASS' and native['engineering_cases']==3,'Independent native source/prefix checks passed')
    tests=json.loads((ROOT/'tests.json').read_text())
    check(tests['successful'] and not tests['skipped'],'Full project tests passed without skips')
    reanalysis=json.loads((ROOT/'reanalysis_report.json').read_text())
    check(reanalysis['status']=='PASS' and not reanalysis['changed_outputs'],'Cache-only reanalysis byte-identical')
    for row in rows:
        state=row['audit_stability_state'];check(row['causal_features']==causal_features(state),'T0 audit values excluded from buffered feature fields')
        check(row['causal_features']['feature_cutoff']==row['t0_minutes']-5,'Input buffer')
        check(state[row['stability']],'Prespecified stratum satisfied')
        if row['label']:
            check(15<row['lead_minutes']<=row['horizon_minutes'],'Target lead horizon')
            if row['lead_band']:check(row['lead_minutes']>max(15,row['horizon_minutes']-30),'Terminal band is genuinely early')
    for key,value in report['risk_sets'].items():
        h,hor,band,stability=key.split('_',3);h=int(h[7:]);hor=int(hor[7:])
        selected=[r for r in rows if r['history_minutes']==h and r['horizon_minutes']==hor and r['lead_band']==(band=='band') and r['stability']==stability]
        check(counts(selected)==value,'Case/window counts reproduce stored report')
    proof=json.loads((ROOT/'replay_proof.json').read_text());canonical=ROOT/'pilot/canonical_replay_observations.jsonl.gz'
    check(sha(canonical.read_bytes())==proof['cache_sha256'],'Canonical replay cache integrity')
    check(sha(gzip.decompress(canonical.read_bytes()))==proof['canonical_sha256'],'Canonical replay integrity')
    check(not report['models_trained'] and not report['eicu_test_accessed'],'No model training / eICU holdout use')
    for name in DOCS:
        p=Path('docs')/name;check(p.exists(),'Required public aggregate document exists')
        content=p.read_text();check('vitaldb:subject:' not in content and 'Patient/' not in content,'Public document no patient references')
    private_json(ROOT/'volume_report.json',volume())
    source_files=sorted(Path('src/vitaldb_feasibility').glob('*.py'))+[Path('tests/test_vitaldb.py')]
    result={'status':'PASS','previous_files_byte_preserved':len(snapshot),'numeric_streams_verified':stream_count,
            'risk_windows_verified':len(rows),'private_files_verified':len(private_paths),'required_documents':len(DOCS),
            'source_sha256':sha(''.join(str(p)+sha(p.read_bytes()) for p in source_files).encode()),
            'source_files':{str(p):sha(p.read_bytes()) for p in source_files},'models_trained':0,'eicu_test_accessed':False,
            'tests':tests,'cache_only_reanalysis':reanalysis,'independent_native_events_checked':native['events_checked'],'volumes':volume()}
    private_json(ROOT/'verification_report.json',result);print({k:result[k] for k in ['status','previous_files_byte_preserved','numeric_streams_verified','risk_windows_verified','required_documents','models_trained','eicu_test_accessed']})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--volume-only',action='store_true');args=parser.parse_args()
    if args.volume_only:private_json(ROOT/'volume_report.json',volume());print(volume())
    else:run()
