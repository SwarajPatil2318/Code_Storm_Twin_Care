"""Read-only completion audit; never train models or decode patient cohorts."""
import json
from pathlib import Path

from src.metabolic_d2.protocol import digest, verify_frozen
from src.metabolic_d3.reference import register
from src.twin.sealing import verify as verify_cardiovascular


ROOT = Path('data/metabolic_d36')


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    before = read(ROOT / 'preservation_before.json')
    allowed = {'src/metabolic_d3/server.py',
               'frontend/src/metabolic/MetabolicApp.tsx',
               'frontend/metabolic-qa.mjs'}
    protected = 0
    for path, expected in before['files'].items():
        if path in allowed or path.startswith('frontend/dist/'):
            continue
        assert digest(path) == expected, 'Protected file changed: ' + path
        protected += 1
    assert register() == before['reference']
    assert verify_frozen()['code_sha256'] == before['scientific_code']
    assert digest('data/metabolic_d3/demo.json') == before['demo_sha256']
    assert verify_cardiovascular()['status'] == 'FROZEN'
    assert (ROOT / 'api_cases_before.json').read_bytes() == (ROOT / 'api_cases_after.json').read_bytes()
    evidence = read('artifacts/metabolic_d36/evidence.json')
    assert not evidence['production_model_changed']
    assert evidence['fusion']['decision'] == 'NO MODEL REPLACEMENT'
    assert not evidence['fusion']['fusion_supported']
    assert all(low < 0 < high for low, high in evidence['fusion']['bootstrap_ci95'])
    browser = read(ROOT / 'browser_qa.json')
    metabolic = read('data/metabolic_ui/browser_qa.json')
    cardiovascular = read('data/product/browser_qa.json')
    assert browser['accessibility_violations'] == 0
    for report in [browser, metabolic, cardiovascular]:
        assert not report['console_errors']
    assert all(not view['violations'] for view in metabolic['a11y'])
    assert all(not view['violations'] for view in cardiovascular['accessibility'])
    screenshots = ['after-1920x1080.png', 'after-1440x900.png',
                   'after-1366x768.png', 'technical-evidence.png',
                   'how-twin-works.png', 'data-lineage.png']
    for name in screenshots:
        assert (ROOT / 'screenshots' / name).is_file()
    result = {
        'status': 'COMPLETE', 'protected_files_verified': protected,
        'frozen_reference_unchanged': True, 'cardiovascular_seal': 'FROZEN',
        'demo_and_case_api_bytes_unchanged': True,
        'model_replacement': False, 'experimental_models_loaded': False,
        'browser_check_groups': {
            'evidence_integration': len(browser['checks']),
            'metabolic': len(metabolic['checks']),
            'cardiovascular': len(cardiovascular['checks'])},
        'audited_accessibility_violations': 0, 'console_errors': 0,
        'evidence_open_ms': browser['evidence_ms'],
        'visual_comparison': read(ROOT / 'visual_comparison.json'),
        'screenshots': screenshots,
    }
    path = ROOT / 'completion_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    path.chmod(0o600)
    print(json.dumps({k: result[k] for k in [
        'status', 'protected_files_verified', 'frozen_reference_unchanged',
        'cardiovascular_seal', 'model_replacement', 'browser_check_groups',
        'audited_accessibility_violations', 'console_errors']}))
    return result


if __name__ == '__main__':
    verify()
