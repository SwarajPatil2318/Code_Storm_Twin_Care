"""Immutable study freeze precedes all seed-2027 generation and evaluation."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from src.data.io import file_hash
from src.synthetic.generate import code_hash as simulator_hash
from src.synthetic.config import load as simulator_config
from src.twin.protocol import code_hash as phase6_hash
from src.reliability.access import load_config,root,save_json,install_guard
from src.reliability.features import NAMES


def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path('src/reliability').glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def freeze(config=None):
    c=config or load_config();out=root(c);path=out/'PHASE6_5_FROZEN_MANIFEST.json'
    if path.exists():return verify(c)
    if Path(c['replication']).exists():raise ValueError('Replication generation must follow the freeze')
    d=json.loads((out/'full_development_complete.json').read_text())
    if d['config_sha256']!=c['_sha256'] or d['logical_full_development_runs']!=1:raise ValueError('Full development evidence inconsistent')
    names=['selected_pipeline.joblib','sensing_utility.joblib','fast_selection_final.json','development_subset.json',
        'full_development_complete.json','development_cache_provenance.json','full_sensing_cases.json']
    artifacts={name:file_hash(out/name) for name in names}
    models=json.loads((out/'fast_selection_final.json').read_text())
    record={'status':'FROZEN','frozen_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_code_sha256':source_hash(),'phase6_source_sha256':phase6_hash(),
        'simulator_code_sha256':simulator_hash(),'simulator_version':simulator_config()['version'],
        'simulator_config_sha256':simulator_config()['_config_sha256'],
        'config_sha256':c['_sha256'],'recovery_addendum_sha256':file_hash(Path('configs/phase6_5_recovery_rule.json')),
        'artifacts':artifacts,'reliability_features':NAMES,'winner':models['winner'],
        'calibration_methods':models['calibration_methods'],'thresholds_artifact':'selected_pipeline.joblib',
        'sensing_candidates':c['sensing_candidates'],'burden_weights':c['burden_weights'],
        'sensing_utility_logic':'TRAIN Ridge predicts hide/reveal Brier gain + .001 decision correction - relative burden; positive expected utility; ambiguity/staleness gate; one request with 72h cooldown',
        'ambiguity_rules':{k:c[k] for k in ['ambiguity_relative_margin','ambiguity_absolute_margin','sensor_stale_hours','request_cooldown_hours']},
        'random_seeds':{'subset':c['subset_seed'],'replication':2027},'replication_patients':2500,'days':60,
        'decision_criteria':{'recovery_relative_reduction':.10,'maximum_sensitivity_loss':.03,
            'calibration':'Mean Brier improves and at least two horizons reduce absolute intercept plus absolute slope-minus-one without AUPRC dropping below95% of reference',
            'sensing':'Observed selected hide/reveal Brier/decision gain minus burden >0, corrected decisions exceed harmed decisions, and at least one decision changes',
            'GO':'All recovery, sensitivity, calibration and sensing gates pass on frozen replication',
            'PARTIAL GO':'At least one reliability component passes; major component failures retained',
            'NO-GO':'None of recovery, calibration or sensing gates pass'},
        'environment':{'python':platform.python_version(),'numpy':np.__version__},
        'original_test_spent':True,'original_test_clinical_access_permitted':False,
        'replication_evaluation_budget':1,'post_replication_tuning_permitted':False}
    save_json(path,record)
    print({'phase65_frozen':True,'replication_not_generated':True,'winner':models['winner']},flush=True)
    return record


def verify(config=None):
    install_guard()
    c=config or load_config();out=root(c);path=out/'PHASE6_5_FROZEN_MANIFEST.json'
    if not path.exists():raise PermissionError('Study must be frozen before replication access')
    m=json.loads(path.read_text())
    if m['status']!='FROZEN' or m['source_code_sha256']!=source_hash() or m['config_sha256']!=c['_sha256']:
        raise ValueError('Frozen Phase 6.5 source/config changed')
    if m['phase6_source_sha256']!=phase6_hash() or m['simulator_code_sha256']!=simulator_hash() or m['simulator_config_sha256']!=simulator_config()['_config_sha256']:
        raise ValueError('Frozen reference/simulator changed')
    if any(file_hash(out/name)!=expected for name,expected in m['artifacts'].items()):raise ValueError('Frozen selected artifact changed')
    return m


if __name__=='__main__':freeze()
