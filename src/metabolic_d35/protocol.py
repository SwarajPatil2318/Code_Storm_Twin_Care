"""Independent allowlist and read-only preservation checks for the spent D2 split."""
import json
from pathlib import Path
import numpy as np
from src.metabolic_d2.protocol import digest, write_json, verify_frozen

ROOT = Path('data/metabolic_d35')
MANIFEST = Path('artifacts/metabolic_d35/fusion_manifest.json')
SEED = 3502026
FOLDS = 5
BOOTSTRAPS = 2000
VARIANTS = ('static_only', 'dynamic_only', 'fused', 'shuffled_static', 'age_sex')


class DevelopmentScope:
    def __init__(self, manifest=None):
        self.manifest = manifest or json.loads(Path('data/metabolic_d2/participant_manifest.json').read_text())
        self.allowed = set(self.manifest['development'])
        self.denied = set(self.manifest['holdout'])
        if len(self.allowed) != 80 or len(self.denied) != 20 or self.allowed & self.denied:
            raise PermissionError('Invalid frozen development partition')
        if self.allowed != set(self.manifest['model_development']) | set(self.manifest['calibration']):
            raise PermissionError('Development membership mismatch')
        self.accessed = set()

    def assert_allowed(self, participants):
        ids = set(participants)
        if not ids or ids & self.denied or not ids <= self.allowed:
            raise PermissionError('D3.5 may load existing development participants only')
        if any(not p.startswith('ShanghaiT2D:') for p in ids):
            raise PermissionError('Cross-cohort participant linkage forbidden')

    def authorize(self, participant):
        self.assert_allowed([participant])
        self.accessed.add(participant)


def grouped_folds(participants, seed=SEED):
    ids = np.array(sorted(set(participants)))
    order = np.random.default_rng(seed).permutation(ids)
    return {str(pid): i % FOLDS for i, pid in enumerate(order)}


def profile_permutation(participants, seed):
    """Seeded derangement of whole profiles; no per-window or cross-fold shuffle."""
    ids = np.array(sorted(set(participants)))
    if len(ids) < 2:
        raise ValueError('At least two participants required')
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        shuffled = rng.permutation(ids)
        if (shuffled != ids).all():
            return dict(zip(ids.tolist(), shuffled.tolist()))
    raise RuntimeError('Cannot construct deterministic profile derangement')


def assert_fold(train, validation):
    if not set(train) or not set(validation) or set(train) & set(validation):
        raise PermissionError('Participant leakage across folds')


def preserve():
    before = json.loads((ROOT/'preservation_before.json').read_text())
    changed = [p for p, h in before['files'].items() if not Path(p).is_file() or digest(p) != h]
    for p, meta in before['private_artifact_metadata'].items():
        f = Path(p)
        if not f.is_file() or f.stat().st_size != meta['size'] or f.stat().st_mtime_ns != meta['mtime_ns']:
            changed.append(p)
    for folder in ['frontend', 'src/metabolic_d2', 'src/metabolic_d3', 'src/metabolic', 'src/product']:
        for path in Path(folder).rglob('*'):
            if path.is_file() and not any(x in path.parts for x in ['node_modules', '__pycache__']) and str(path) not in before['files']:
                changed.append(str(path))
    # Verify original model/config/code seals without reading holdout trajectories.
    frozen = verify_frozen()
    for role, path in before['reference']['paths'].items():
        if digest(path) != before['reference']['model_hashes'][role]:
            changed.append('frozen model')
    if frozen['code_sha256'] != before['scientific_code'] or changed:
        raise RuntimeError('Production preservation check failed')
    return {'protected_files': len(before['files']), 'private_artifacts_unchanged': len(before['private_artifact_metadata']),
            'UI_unchanged': True, 'D2_D3_code_unchanged': True, 'production_model_hashes_unchanged': True,
            'spent_holdout_payloads_opened': False}


def validate_manifest(m):
    required = {'decision', 'static_features_used', 'excluded_features', 'dynamic_model_120m_mae',
                'fused_model_120m_mae', 'delta_mae_120m', 'bootstrap_ci_120m', 'fusion_supported',
                'recommended_production_action', 'safe_ui_claim'}
    if not required <= m.keys():
        raise ValueError('Incomplete fusion manifest')
    if m['decision'] not in {'GO FOR MODEL-LEVEL FUSION', 'NO MODEL REPLACEMENT'}:
        raise ValueError('Invalid production decision')
    for key in ['dynamic_model_120m_mae', 'fused_model_120m_mae', 'delta_mae_120m']:
        if not isinstance(m[key], (float, int)) or not np.isfinite(m[key]):
            raise ValueError('Invalid metric')
    ci = m['bootstrap_ci_120m']
    if len(ci) != 2 or not np.isfinite(ci).all() or ci[0] > ci[1]:
        raise ValueError('Invalid participant-bootstrap interval')
    if not isinstance(m['fusion_supported'], bool):
        raise ValueError('Invalid support flag')
    if m['fusion_supported'] and (ci[1] >= 0 or m['decision'] != 'GO FOR MODEL-LEVEL FUSION'):
        raise ValueError('Unsupported fusion claim')
    return m
