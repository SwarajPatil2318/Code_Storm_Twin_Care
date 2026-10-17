"""Aggregate D4.2 preservation/verification audit; never publishes records."""
import json
from pathlib import Path
from src.metabolic_d2.protocol import digest
from src.metabolic_d3.reference import register
from src.twin.sealing import verify

ROOT=Path('data/metabolic_d42')
def read(path):return json.loads(Path(path).read_text())
def audit():
    before=read(ROOT/'preservation_before.json')
    for path in ['src/metabolic_d3/engine.py','src/metabolic_d3/shadow.py','data/metabolic_d3/demo.json']:
        assert digest(path)==before['files'][path], 'Protected scientific/data file changed'
    d41_before=read('data/metabolic_d41/preservation_before.json')
    reference=register()
    assert reference==d41_before['reference']
    allowed={'src/metabolic_d3/server.py','src/product/server.py','frontend/src/metabolic/MetabolicApp.tsx','frontend/src/metabolic/Visuals.tsx','frontend/src/metabolic/ProductEvidence.tsx'}
    protected=0
    for path,expected in d41_before['files'].items():
        if path not in allowed:
            assert digest(path)==expected,'Protected source/artifact changed: '+path
            protected+=1
    assert verify()['status']=='FROZEN'
    latency=read('data/metabolic_d41/latency_aggregate.json')
    assert latency['samples']==75 and latency['all_75_states_match_verified_cache']
    browser=read(ROOT/'browser_qa.json')
    assert browser['status']=='passed' and 60<=browser['jury_duration_seconds']<=90
    assert browser['layout_preserved'] and not browser['console_errors'] and not browser['accessibility_violations']
    assert set(browser['highlighted_sources'])=={'CGM','Smartwatch','Meal logger'}
    runtime=read('data/metabolic_d41/browser_qa.json')
    assert len(runtime['checks'])==11 and not runtime['console_errors']
    cv=read('data/product/browser_qa.json')
    assert len(cv['checks'])==18 and not cv['console_errors']
    result={'status':'VERIFIED','protected_files':protected,'model_hashes_before':d41_before['reference']['model_hashes'],
            'model_hashes_after':reference['model_hashes'],'original_demo_data_unchanged':True,
            'shadow_engine_unchanged':True,'all_75_TwinStates_unchanged':True,'cardiovascular_seal':'FROZEN',
            'jury_duration_seconds':browser['jury_duration_seconds'],'source_highlights':browser['highlighted_sources'],
            'LLM_used':False,'wearable_model_input':False,'layout_preserved':True,
            'runtime_browser_groups':11,'cardiovascular_browser_checks':18,
            'limitations':['Original meal arrives at end; no meal-first rise claimed','Original meal quantities absent','Activity is independent simulated presentation context','No later observed threshold crossing in original streams','Research PoC; no clinical validation']}
    (ROOT/'preservation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for item in ROOT.rglob('*'):item.chmod(0o700 if item.is_dir() else 0o600)
    print(json.dumps({key:result[key] for key in ['status','protected_files','all_75_TwinStates_unchanged','jury_duration_seconds','layout_preserved','LLM_used']}))
if __name__=='__main__':audit()
