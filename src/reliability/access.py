"""Fail-closed spent-test guard and provenance-checked development-only reuse.

Routing metadata and compressed discarded bytes are not clinical deserialization.
Mixed-source reads are confined to readers that filter patient slabs/JSON first.
"""
import contextvars
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

from src.data.io import file_hash
from src.twin.access import metadata_patient, selective_npz, OBS_KEYS
from src.twin.protocol import load as phase6_config, feature_code_hash, code_hash, EVIDENCE
from src.vitaldb_feasibility.source import private_json

BASE = Path(__file__).resolve().parents[2]
PHASE6 = BASE / 'data/results/phase6'
ORIGINAL = BASE / 'data/synthetic/cohort10000'
REPLICATION = BASE / 'data/synthetic/phase6_5_replication_2027'
FROZEN = BASE / 'data/results/phase6_5/PHASE6_5_FROZEN_MANIFEST.json'
_ROUTED = contextvars.ContextVar('phase65_approved_mixed_read', default=None)
_INSTALLED = False


def _permitted(path):
    """Only specific development artifacts; no broad Phase 6 output allowance."""
    path = Path(path).resolve()
    if path == REPLICATION or REPLICATION in path.parents:
        return FROZEN.exists()
    if path == ORIGINAL or ORIGINAL in path.parents:
        rel = path.relative_to(ORIGINAL).as_posix()
        return rel in {'manifest.json', 'splits.json', 'large_validation.json'} or path == _ROUTED.get()
    if path == PHASE6 or PHASE6 in path.parents:
        rel = path.relative_to(PHASE6).as_posix()
        if rel in {'selection_seal.json', 'features/train/manifest.json', 'features/validation/manifest.json', 'shadow_engines.joblib', 'phase6_complete.json'}:
            return True
        if any(rel.startswith('features/' + role + '/') and rel.endswith('.npy') for role in ['train', 'validation']):
            return True
        if rel.startswith('models/personalized_fusion/'):
            return Path(rel).name in {'selected.joblib', 'calibration.joblib', 'thresholds.json', 'validation_raw_hazards.npy', 'validation_hazards.npy', 'selection.json'}
        return False
    return True


def install_guard():
    global _INSTALLED
    if _INSTALLED:
        return
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            name = args[0].decode() if isinstance(args[0], bytes) else args[0]
            if not _permitted(name):
                raise PermissionError('Phase 6.5 forbids spent TEST and unapproved Phase 6 clinical artifacts')
    sys.addaudithook(audit)
    _INSTALLED = True


def load_config(path='configs/phase6_5_v1.json'):
    p = Path(path)
    c = json.loads(p.read_text())
    c['_sha256'] = file_hash(p)
    return c


def root(config):
    p = Path(config['output'])
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.chmod(0o700)
    return p


def save_json(path, value):
    private_json(path, {**EVIDENCE, **value})


def save_arrays(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez_compressed(path, **values)
    path.chmod(0o600)


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    joblib.dump(model, path)
    path.chmod(0o600)


def timed(config, stage, start, details=None):
    path = root(config) / 'compute_log.json'
    d = json.loads(path.read_text()) if path.exists() else {**EVIDENCE, 'stages': []}
    d['stages'].append({'stage': stage, 'seconds': time.monotonic() - start, **(details or {})})
    private_json(path, d)


def phase6_identity():
    install_guard()
    c = phase6_config()
    seal = json.loads((PHASE6 / 'selection_seal.json').read_text())
    if seal['model_code_sha256'] != code_hash() or seal['config_sha256'] != c['_sha256']:
        raise ValueError('Original Phase 6 source/configuration changed')
    if seal['primary_model'] != 'personalized_fusion':
        raise ValueError('Expected frozen personalized reference')
    return c, seal


def reference_artifact(name):
    _, seal = phase6_identity()
    p = PHASE6 / 'models/personalized_fusion' / name
    key = str(p.relative_to(PHASE6))
    expected = seal['artifacts'].get(key)
    # Validation hazards are cached development output; their generating raw
    # hazards/calibrator are sealed, and equivalence is verified on reuse.
    if expected and file_hash(p) != expected:
        raise ValueError('Frozen Phase 6 reference artifact changed')
    if not expected and name not in {'validation_raw_hazards.npy', 'validation_hazards.npy'}:
        raise ValueError('Reference artifact lacks frozen provenance')
    return p


def development_cache(role):
    if role not in {'train', 'validation'}:
        raise PermissionError('Spent TEST cache access is forbidden')
    c, _ = phase6_identity()
    p = PHASE6 / 'features' / role
    m = json.loads((p / 'manifest.json').read_text())
    if m['role'] != role or m['config_sha256'] != c['_sha256'] or m['feature_code_sha256'] != feature_code_hash():
        raise ValueError('Development cache provenance mismatch')
    if m['cohort_manifest_sha256'] != file_hash(ORIGINAL / 'manifest.json'):
        raise ValueError('Development cohort identity mismatch')
    d = {}
    for name, expected in m['artifacts'].items():
        if '/' in name or not name.endswith('.npy'):
            raise ValueError('Unsafe cache artifact name')
        if file_hash(p / name) != expected:
            raise ValueError('Development cache changed')
        d[name[:-4]] = np.load(p / name, mmap_mode='r', allow_pickle=False)
    roles = json.loads((ORIGINAL / 'splits.json').read_text())['assignments']
    if any(roles.get(str(pid)) != role for pid in np.unique(d['patient_ids'])):
        raise ValueError('Development patient split violation')
    return d, m


def routed_rows(table, allowed):
    """Evaluation-only annotations or historical EHR, never test clinical decode."""
    if table not in {'ground_truth', 'ehr'}:
        raise PermissionError('Unapproved development table')
    phase6_identity()
    roles = json.loads((ORIGINAL / 'splits.json').read_text())['assignments']
    if not allowed or any(roles.get(str(p)) not in {'train', 'validation'} for p in allowed):
        raise PermissionError('Mixed-table routing must contain development patients only')
    path = ORIGINAL / table / 'records.jsonl.gz'
    token = _ROUTED.set(path.resolve())
    try:
        manifest = json.loads((ORIGINAL / 'manifest.json').read_text())
        if file_hash(path) != manifest['artifacts'][str(path.relative_to(ORIGINAL))]['sha256']:
            raise ValueError('Original source changed')
        with gzip.open(path, 'rb') as f:
            for line in f:
                if metadata_patient(line) in allowed:
                    yield json.loads(line)
    finally:
        _ROUTED.reset(token)


def routed_streams(allowed):
    phase6_identity()
    roles = json.loads((ORIGINAL / 'splits.json').read_text())['assignments']
    if not allowed or any(roles.get(str(p)) not in {'train', 'validation'} for p in allowed):
        raise PermissionError('Observation slabs must contain development patients only')
    manifest = json.loads((ORIGINAL / 'manifest.json').read_text())
    for name in manifest['batches']:
        path = ORIGINAL / 'observations' / name
        token = _ROUTED.set(path.resolve())
        try:
            if file_hash(path) != manifest['artifacts']['observations/' + name]['sha256']:
                raise ValueError('Original observation source changed')
            with np.load(path, allow_pickle=False) as archive:
                ids = archive['patient_ids'].tolist()
            indices = [i for i, p in enumerate(ids) if p in allowed]
            if not indices:
                continue
            d = selective_npz(path, indices, OBS_KEYS)
            for k, i in enumerate(indices):
                yield ids[i], {name: values[k] for name, values in d.items()}
        finally:
            _ROUTED.reset(token)


def deterministic_patients(ids, count, seed, purpose):
    unique = sorted(set(np.asarray(ids).astype(str)), key=lambda p: hashlib.sha256(f'{seed}:{purpose}:{p}'.encode()).digest())
    if count > len(unique):
        raise ValueError('Subset exceeds available patients')
    return unique[:count]
