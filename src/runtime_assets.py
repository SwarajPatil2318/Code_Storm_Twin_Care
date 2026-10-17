"""Deployment-only asset verification; research seals and training remain unchanged."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'data/runtime/manifest.json'
# Explicit deployment boundary. Never copy research cohorts or patient databases.
ASSETS = (
    'data/metabolic_d2/models/boosting_population.pkl',
    'data/metabolic_d2/models/boosting_personal.pkl',
    'data/metabolic_d2/models/risk.pkl',
    'data/metabolic_risk_v1/model.pkl',
    'data/metabolic_d3/demo.json',
    'data/product/demo.json',
    'data/product/build_audit.json',
    'artifacts/productization_p6/model_contract.json',
    'artifacts/productization_p7/alert_policy.json',
    'artifacts/productization_p8/model_evidence.json',
    'artifacts/metabolic_risk_v1/manifest.json',
    'artifacts/metabolic_d36/evidence.json',
)
OPTIONAL_ASSETS = ('data/productization_p9/selected_inputs.json',)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_asset(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or name not in (*ASSETS, *OPTIONAL_ASSETS):
        raise ValueError('Asset outside deployment allowlist')
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Asset leaves deployment root')
    return result


def verify(root=ROOT, manifest=None):
    root = Path(root)
    path = Path(manifest) if manifest else root / MANIFEST
    value = json.loads(path.read_text())
    if value.get('schema_version') != 'twincare-runtime/1':
        raise ValueError('Unsupported runtime bundle')
    expected = value.get('assets', {})
    if not set(ASSETS) <= set(expected) or set(expected) - set((*ASSETS, *OPTIONAL_ASSETS)):
        raise ValueError('Incomplete runtime bundle')
    for name, digest in expected.items():
        target = safe_asset(root, name)
        if not target.is_file() or sha(target) != digest:
            raise ValueError('Runtime asset missing or modified: ' + name)
    return value


def portable_reference():
    configured = os.environ.get('TWINCARE_RUNTIME_MANIFEST')
    if not configured:
        return None  # Deployment must opt in; research verification stays unchanged.
    path = Path(configured)
    value = verify(ROOT, path)
    from src.metabolic_d2.protocol import code_hash
    if code_hash() != value['reference']['D2_scientific_code_sha256']:
        raise RuntimeError('Frozen scientific code changed')
    record = value['reference']
    if sha(ROOT / 'configs/metabolic_d3.json') != record['D3_config_sha256']:
        raise RuntimeError('Frozen runtime configuration changed')
    if set(record['paths']) != {'cold', 'warm', 'risk'}:
        raise ValueError('Invalid forecast model roles')
    for role, name in record['paths'].items():
        if name not in ASSETS[:3] or value['assets'][name] != record['model_hashes'][role]:
            raise ValueError('Forecast artifact identity mismatch')
    contract = json.loads((ROOT / 'artifacts/productization_p6/model_contract.json').read_text())
    if contract['forecast']['sha256'] != record['model_hashes']:
        raise ValueError('Runtime forecast contract mismatch')
    return record
