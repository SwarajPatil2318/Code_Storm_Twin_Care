"""Pre-development metadata, source preservation and study registration."""
import json
from pathlib import Path
from collections import Counter
from src.twin.protocol import load,EVIDENCE,feature_code_hash
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def register(config=None):
    c=config or load();root=Path(c['cohort']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True,mode=0o700);out.chmod(0o700)
    if (out/'registration.json').exists():
        old=json.loads((out/'registration.json').read_text())
        if old['config_sha256']!=c['_sha256']:raise ValueError('Registered study config cannot be changed silently')
        return old
    m=json.loads((root/'manifest.json').read_text());gate=json.loads((root/'large_validation.json').read_text());roles=json.loads((root/'splits.json').read_text())['assignments']
    if not gate['all_gates_passed'] or Counter(roles.values())!={'train':7000,'validation':1500,'test':1500}:raise ValueError('Authoritative Phase 5 cohort/partitions required')
    # Metadata and hashes may cover original bytes, but no clinical test decoding.
    previous={str(p):file_hash(p) for top in ['src','docs','tests','configs'] for p in Path(top).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not (top=='src' and 'twin' in p.parts and p.name!='__init__.py') and p.name!='test_twin.py' and p.name!='phase6_v1.json'}
    phase5=json.loads(Path('data/synthetic/phase5_complete.json').read_text())
    for mapping in ['source_sha256','documentation_sha256']:
        for name,h in phase5[mapping].items():
            if file_hash(name)!=h:raise ValueError('Phase 5 public artifact changed')
    result={'version':c['version'],'config':{k:v for k,v in c.items() if not k.startswith('_')},'config_sha256':c['_sha256'],'cohort_manifest_sha256':file_hash(root/'manifest.json'),'cohort_certificate_sha256':file_hash(root/'large_validation.json'),
      'phase5_completion_sha256':file_hash('data/synthetic/phase5_complete.json'),'split_counts':dict(Counter(roles.values())),'split_sha256':file_hash(root/'splits.json'),'previous_public_sha256':previous,
      'test_clinical_rows_decoded':0,'test_metrics_available_to_selection':False,'feature_construction_code_sha256':feature_code_hash(),**EVIDENCE}
    private_json(out/'registration.json',result);print({'phase6_study_registered':True,'train_patients':7000,'validation_patients':1500,'test_patients_reserved':1500,'test_clinical_access':False},flush=True)
    return result

if __name__=='__main__':register()
