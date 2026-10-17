"""Independent evidence verification/reanalysis, without model refitting."""
import io
import json
import pickle
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic.audit import times
from src.metabolic.windows import prepare, Series, window_masks, nearest_indices, MINUTE
from src.metabolic_d2.data import cgm_features, personal_features, subset
from src.metabolic_d2.protocol import digest, write_json
from .data import load_inputs, recording_pid, SOURCE
from .protocol import ROOT, SEED, DevelopmentScope, preserve, assert_fold, VARIANTS
from .experiment import participant_errors, paired_interval, decision
from .report import report


def same(actual, expected, approximate=False):
    equal = np.allclose(actual, expected, rtol=0, atol=0, equal_nan=True) if approximate else np.array_equal(actual, expected)
    if not equal: raise AssertionError('Private raw/model parity mismatch; values suppressed')


def raw_parity(data, scope):
    """Check four landmarks/person against guarded source Excel and D2 functions."""
    frames = {}; members = []
    with zipfile.ZipFile(SOURCE) as archive:
        for name in archive.namelist():
            if not name.startswith('Shanghai_T2DM/') or not name.endswith(('.xls','.xlsx')): continue
            pid = recording_pid(Path(name).stem)
            if pid not in scope.allowed: continue
            scope.authorize(pid)  # Authorization precedes opening the workbook.
            frame = pd.read_excel(io.BytesIO(archive.read(name)), keep_default_na=False)
            frames.setdefault(pid, []).append(frame); members.append(name)
    checked = 0
    for pid, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        col = next(c for c in frame if c.strip() in ['CGM (mg / dl)', 'CGM'])
        t, g, q = prepare(times(frame['Date']), pd.to_numeric(frame[col], errors='coerce'), 15)
        series = Series(pid, 'ShanghaiT2D', 'T2D', t, g, 15, q)
        _, masks, target = window_masks(series, 120)
        rows = np.flatnonzero(data['pid'] == pid)
        selection = rows[np.unique(np.linspace(0, len(rows)-1, 4).astype(int))]
        cutoffs = data['time'][selection]; indices = np.searchsorted(t, cutoffs)
        if not np.array_equal(t[indices], cutoffs): raise AssertionError('Raw cutoff mismatch')
        sequence_indices = np.column_stack([nearest_indices(t, cutoffs-lag*MINUTE, 7.5*MINUTE)[0] for lag in range(120,-1,-15)])
        same(t[sequence_indices], data['sequence_time'][selection])
        same(g[sequence_indices].astype('float32'), data['sequence'][selection])
        expected_targets = np.column_stack([target[h][indices] for h in [30,60,90,120]])
        if not all(masks[h][indices].all() for h in [30,60,90,120]): raise AssertionError('D2 exclusions changed')
        same(t[expected_targets], data['target_time'][selection])
        same(g[expected_targets].astype('float32'), data['y'][selection])
        x = cgm_features(data['sequence'][selection], cutoffs)
        same(x, data['x'][selection], approximate=True)
        personal, warm = personal_features(t, g, cutoffs, data['sequence'][selection,-1], x[:,10])
        same(personal, data['personal'][selection], approximate=True)
        same(warm, data['warm'][selection])
        checked += len(selection)
    if set(frames) != scope.allowed: raise AssertionError('Raw development membership mismatch')
    write_json(ROOT/'raw_access_audit.json', {'participants': sorted(frames), 'members_read': members,
               'holdout_feature_records_loaded': False, 'landmarks_independently_checked': checked})
    return {'participants': len(frames), 'recordings': len(members), 'landmarks': checked,
            'native_history_targets_and_dynamic_prefix_exact': True}


def verify():
    begin = time.perf_counter(); scope = DevelopmentScope(); preserve()
    data, profiles, audit = load_inputs(scope)
    raw = raw_parity(data, scope)
    results = json.loads((ROOT/'results.json').read_text())
    frozen = json.loads((ROOT/'protocol_frozen.json').read_text())
    if frozen['config_sha256'] != digest('configs/metabolic_d35.json'): raise RuntimeError('Fixed protocol changed')
    original_models = {str(ROOT/'models'/f'{v}_fold{f}.pkl'): results['fold_results'][f]['variants'][v]['model_sha256'] for f in range(5) for v in VARIANTS}
    with np.load(ROOT/'out_of_fold_predictions.npz', allow_pickle=False) as archive:
        scope.assert_allowed(archive['participant_id'])
        same(archive['participant_id'], data['pid'])
        same(archive['y'], data['y'])
        same(archive['time'], data['time'])
        if digest(ROOT/'out_of_fold_predictions.npz') != results['prediction_sha256']: raise RuntimeError('Predictions changed')
        fold = archive['fold']; predictions = {v:archive[v] for v in VARIANTS}
        importance = {k: archive[k+'_permuted'] for k in ['static','dynamic']}
    for f in range(5):
        valid = subset(data, fold == f); train_ids = set(data['pid'][fold != f])
        assert_fold(train_ids, valid['pid'])
        donors = json.loads((ROOT/'folds'/f'{f}_shuffle.json').read_text())
        if set(donors) != train_ids or set(donors.values()) != train_ids or any(k == v for k,v in donors.items()):
            raise AssertionError('Training participant-profile derangement violated')
        for variant in VARIANTS:
            path = ROOT/'models'/f'{variant}_fold{f}.pkl'
            if digest(path) != original_models[str(path)]: raise RuntimeError('Experimental model changed')
            # Local experimental pickle only, verified by the recorded model hash.
            with path.open('rb') as stream: model = pickle.load(stream)
            if set(model.training_participants) != train_ids: raise AssertionError('Model fit scope mismatch')
            if model.dynamic_preprocessor and set(model.dynamic_preprocessor.training_participants) != train_ids: raise AssertionError('Dynamic preprocessing leakage')
            if model.static_encoder and set(model.static_encoder.training_participants) != train_ids: raise AssertionError('Static preprocessing leakage')
            same(model.predict(valid,profiles), predictions[variant][fold == f])
    # Pure numeric precision correction to published inference evidence; no refit.
    per = {v: participant_errors(data['y'], p, data['pid']) for v,p in predictions.items()}
    for name, base, new in [('fused_minus_dynamic','dynamic_only','fused'), ('fused_minus_shuffled','shuffled_static','fused'),
                            ('age_sex_minus_dynamic','dynamic_only','age_sex'), ('fused_minus_age_sex','age_sex','fused'),
                            ('shuffled_minus_dynamic','dynamic_only','shuffled_static')]:
        results[name] = paired_interval(per[base],per[new])
    for group, p in importance.items():
        results['modality_importance'][group]['mae_degradation'] = (participant_errors(data['y'],p,data['pid'])-per['fused']).mean(0).tolist()
    results['fusion_supported'], results['outcome'] = decision(results['fused_minus_dynamic'],results['fused_minus_shuffled'])
    results['preservation'] = preserve()
    write_json(ROOT/'results.json', results)
    np.savez_compressed(ROOT/'per_participant_errors.npz', participant_id=np.unique(data['pid']), **per,
                        fused_minus_dynamic=per['fused']-per['dynamic_only'])
    (ROOT/'per_participant_errors.npz').chmod(0o600)
    report(results, audit)
    output = {'raw_parity': raw, 'all_25_models_reproduce_OOF_exactly': True,
              'original_predictions_unchanged': True, 'participant_fit_scopes_verified': True,
              'model_hashes_unchanged': True, 'preservation': preserve(), 'seconds': time.perf_counter()-begin,
              'current_code_hashes': {str(p):digest(p) for p in Path('src/metabolic_d35').glob('*.py')}}
    write_json(ROOT/'verification.json', output)
    print(json.dumps({k:v for k,v in output.items() if k != 'current_code_hashes'}), flush=True)
    return output


if __name__ == '__main__': verify()
