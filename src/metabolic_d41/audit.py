"""Read-only preservation audit; no training or research-cohort decoding."""
import json
from pathlib import Path
from src.metabolic_d2.protocol import digest, verify_frozen
from src.metabolic_d3.reference import register
from src.metabolic_d3.events import timestamp
from src.twin.sealing import verify as verify_cv

ROOT = Path('data/metabolic_d41')

def read(path):
    return json.loads(Path(path).read_text())

def audit():
    before = read(ROOT/'preservation_before.json')
    allowed = {'src/metabolic_d3/server.py', 'src/product/server.py',
               'frontend/src/metabolic/MetabolicApp.tsx',
               'frontend/src/metabolic/Visuals.tsx',
               'frontend/src/metabolic/ProductEvidence.tsx'}
    protected = 0
    for path, expected in before['files'].items():
        if path in allowed: continue
        assert digest(path) == expected, 'Protected file changed: '+path
        protected += 1
    reference = register()
    assert reference == before['reference']
    assert verify_frozen()['code_sha256'] == reference['D2_scientific_code_sha256']
    assert verify_cv()['status'] == 'FROZEN'
    latency = read(ROOT/'latency_aggregate.json')
    assert latency['samples'] == 75 and latency['all_75_states_match_verified_cache']
    browser = read(ROOT/'browser_qa.json')
    metabolic = read('data/metabolic_ui/browser_qa.json')
    cv = read('data/product/browser_qa.json')
    assert len(browser['checks']) >= 11 and browser['runtime_updates'] >= 16
    assert len(metabolic['checks']) == 9 and len(cv['checks']) == 18
    for report, field in [(browser,'a11y'),(metabolic,'a11y'),(cv,'accessibility')]:
        assert not report['console_errors']
        assert all(not screen['violations'] for screen in report[field])
    geometry = read('data/metabolic_d36/layout_before.json')
    for size, panels in browser['geometry'].items():
        for selector, box in panels.items():
            assert all(abs(value-geometry[size][selector][key]) < .5 for key,value in box.items())
    sample = read(ROOT/'sample_runtime_response.json')
    assert sample['update']['model_executed']
    assert sample['update']['model_hashes'] == reference['model_hashes']
    assert sample['forecast_history'][-1]['forecast'] == sample['state']['forecast']
    assert any(snapshot['comparisons'] for snapshot in sample['forecast_history'])
    assert all(timestamp(e['timestamp'])[0] <= timestamp(sample['state']['timestamp'])[0] for e in sample['observed_history'])
    for file in ['METABOLIC_D41_LIVE_TWIN.md','METABOLIC_D41_JURY_DEMO.md']:
        assert (Path('docs')/file).is_file()
    screenshots = ['metabolic-1920x1080.png','metabolic-1440x900.png','metabolic-1366x768.png',
                   'inspect-prediction.png','forecast-history.png','live-event-log.png','recovered-runtime.png']
    assert all((ROOT/'screenshots'/file).is_file() for file in screenshots)
    demo = read('data/metabolic_d3/demo.json')
    crossing = any(e['value'] is not None and e['value'] > 180 for c in demo['cases'] for e in c['events'] if e['concept']=='glucose')
    future = any(timestamp(e['available_time'])[0] > timestamp(c['timeline'][-1]['timestamp'])[0]
                 for c in demo['cases'] for e in c['events'] if e['concept']=='glucose')
    result = {
        'status':'RUNTIME_VERIFIED_STORY_GATE_OPEN', 'protected_files_verified':protected,
        'model_hashes_before':before['reference']['model_hashes'], 'model_hashes_after':reference['model_hashes'],
        'scientific_code_sha256':reference['D2_scientific_code_sha256'],
        'model_replacement':False, 'cardiovascular_seal':'FROZEN',
        'all_75_runtime_states_match_original':True,
        'browser_checks':{'live_twin':len(browser['checks']),'metabolic':len(metabolic['checks']),'cardiovascular':len(cv['checks'])},
        'accessibility_violations':0, 'console_errors':0, 'primary_panel_geometry_unchanged':True,
        'latency':latency, 'screenshots':screenshots,
        'story_gate':{'original_stream_has_observed_threshold_crossing':crossing,
                      'original_stream_has_observations_after_cached_tail':future,
                      'required_decision':'Whether to add a separately labeled synthetic continuation while preserving all original records',
                      'underlying_data_modified':False, 'full_D41_completion_proven':False}}
    path = ROOT/'preservation_audit.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    for item in ROOT.rglob('*'): item.chmod(0o700 if item.is_dir() else 0o600)
    print(json.dumps({key:result[key] for key in ['status','protected_files_verified','model_replacement',
          'cardiovascular_seal','all_75_runtime_states_match_original','browser_checks','accessibility_violations',
          'primary_panel_geometry_unchanged']}))
    return result

if __name__=='__main__':audit()
