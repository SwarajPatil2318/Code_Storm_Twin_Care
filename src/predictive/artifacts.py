"""Versioned experiment seals and preservation of all earlier evidence."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import json
from src.data.io import file_hash
from src.models.common import write_json,write_rows,save_model,artifact_hashes,sealed_directory,verify_hashes

DOCS=['PHASE4_MODEL_CARD.md','MULTIMODAL_FUSION_RESULTS.md','PERSONALIZATION_ABLATION.md','TEMPORAL_MODEL_RESULTS.md',
      'HARD_CASE_ANALYSIS.md','HF_SUBGROUP_ANALYSIS.md','PHASE4_DECISION.md','PHASE4_PROTOCOL.json']


def source_hash():
    digest=hashlib.sha256()
    paths=sorted(Path('src/predictive').glob('*.py'))+[Path('tests/test_predictive.py'),Path('requirements-models.txt')]
    for p in paths:digest.update(str(p).encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def versions():
    return {'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','joblib','threadpoolctl']}}


def previous_files():
    paths=set()
    for pattern in ['data/processed/eicu_demo*','data/results/phase3','data/results/phase36*','data/processed/zigong','data/processed/chiron','data/processed/perheart','data/processed/synthea','data/audit']:
        for root in Path('.').glob(pattern):paths.update(p for p in root.rglob('*') if p.is_file())
    paths.update(p for p in Path('data/raw/phase3_5').rglob('*') if p.is_file())
    paths.update(p for p in Path('docs').glob('*') if p.is_file() and p.name not in DOCS)
    paths.update(p for p in Path('src').rglob('*') if p.suffix in ['.py','.json'] and 'predictive' not in p.parts)
    paths.update(p for p in Path('src/feasibility').glob('*.py'))
    paths.update(p for p in Path('tests').glob('*.py') if p.name!='test_predictive.py')
    paths.update(Path(p) for p in ['Datasets/aibhs_datasets-main.zip','AGENTS.md','.gitignore','README.md','data/README.md','requirements-models.txt'])
    return {str(p):file_hash(p) for p in sorted(paths) if p.is_file()}


def verify_previous(snapshot):
    for name,sha in snapshot.items():
        if not Path(name).is_file() or file_hash(name)!=sha:raise ValueError('Earlier evidence changed: '+name)


def freeze(root,extra):
    root=Path(root)
    payload={'status':'FROZEN','source_code_sha256':source_hash(),
             'dataset_manifest_sha256':file_hash('data/processed/eicu_demo/manifest.json'),
             'hashes':artifact_hashes(root,exclude=['selection_seal.json']),**extra}
    write_json(root/'selection_seal.json',payload);return payload


def check_seal(root):
    root=Path(root);seal=json.loads((root/'selection_seal.json').read_text())
    verify_hashes(root,seal['hashes'])
    if seal['source_code_sha256']!=source_hash():raise ValueError('Code changed after modeling pipeline froze')
    if seal['dataset_manifest_sha256']!=file_hash('data/processed/eicu_demo/manifest.json'):raise ValueError('Source changed after selection')
    return seal
