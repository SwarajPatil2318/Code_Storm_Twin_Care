"""Private experiment artifacts and train-role guards."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import importlib.metadata
import joblib
from src.data.io import dumps, file_hash, read_jsonl
from src.data.validate import validate_dataset

SEED = 2026
FEATURE_VERSION = 'phase3_causal_v1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(dumps(value)+'\n')
    path.chmod(0o600)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('w') as f:
        for row in rows:
            f.write(dumps(row)+'\n')
    path.chmod(0o600)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_model(path, model):
    joblib.dump(model, path)
    Path(path).chmod(0o600)


def code_hash():
    h = hashlib.sha256()
    for p in sorted(Path('src').rglob('*.py')):
        h.update(str(p).encode()); h.update(p.read_bytes())
    h.update(Path('requirements-models.txt').read_bytes())
    return h.hexdigest()


def metadata(root, dataset):
    validate_dataset(Path(root)/dataset)
    return {'dataset': dataset, 'dataset_manifest_sha256': file_hash(Path(root)/dataset/'manifest.json'),
            'dataset_version': load_json(Path(root)/dataset/'manifest.json'),
            'feature_version': FEATURE_VERSION, 'seed': SEED, 'code_sha256': code_hash(),
            'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'python': platform.python_version(),
            'dependencies': {p: importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','joblib','threadpoolctl']}}


def require_roles(ids, assignments, allowed):
    if not ids or len(ids) != len(set(ids)) or any(assignments.get(p) not in allowed for p in ids):
        raise ValueError('Fit role or patient identity violation')


def verify_inputs(root, info):
    if file_hash(Path(root)/info['dataset']/'manifest.json') != info['dataset_manifest_sha256']:
        raise ValueError('Processed data changed since training')
    validate_dataset(Path(root)/info['dataset'])


def sealed_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Experiment already exists; use a new private run directory')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def artifact_hashes(directory, exclude=()):
    return {str(p.relative_to(directory)):file_hash(p) for p in sorted(Path(directory).rglob('*')) if p.is_file() and p.name not in exclude}


def verify_hashes(directory, hashes):
    for name, sha in hashes.items():
        if file_hash(Path(directory)/name) != sha:
            raise ValueError('Experiment artifact changed after selection')
