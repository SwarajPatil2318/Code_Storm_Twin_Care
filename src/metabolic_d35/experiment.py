"""Run once: fixed five-fold ablation, private OOF/models, aggregate evidence."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.metabolic_d2.data import subset
from src.metabolic_d2.metrics import forecast_metrics
from src.metabolic_d2.risk import trajectory_score
from src.metabolic_d2.protocol import digest, write_json
from .data import load_inputs, dynamic_features, SAFE
from .models import FusionModel
from .protocol import ROOT, SEED, FOLDS, BOOTSTRAPS, VARIANTS, DevelopmentScope, grouped_folds, profile_permutation, assert_fold, preserve, validate_manifest


def participant_errors(y, predictions, ids):
    y = np.asarray(y, dtype=float); predictions = np.asarray(predictions, dtype=float)
    return np.array([np.abs(predictions[ids == pid]-y[ids == pid]).mean(0) for pid in np.unique(ids)])


def paired_interval(per_base, per_new, seed=SEED, repeats=BOOTSTRAPS):
    difference = per_new-per_base
    draw = np.random.default_rng(seed).integers(0, len(difference), size=(repeats, len(difference)))
    return {'delta_mae': difference.mean(0).tolist(),
            'ci95': np.quantile(difference[draw].mean(1), [.025, .975], axis=0).T.tolist(),
            'participants_improved': (difference < 0).sum(0).tolist(), 'bootstrap_unit': 'participant'}


def decision(primary, shuffled):
    supported = all(d < 0 for d in primary['delta_mae']) and all(ci[1] < 0 for ci in primary['ci95']) and shuffled['ci95'][3][1] < 0
    return supported, 'supported' if supported else 'hurts' if primary['ci95'][3][0] > 0 else 'inconclusive'


def dynamic_permutation(data, seed):
    rng = np.random.default_rng(seed); order = np.arange(len(data['pid']))
    for pid in np.unique(data['pid']):
        rows = np.flatnonzero(data['pid'] == pid); order[rows] = rng.permutation(rows)
    return dynamic_features(data)[order], data['sequence'][order, -1]


def run():
    started = time.perf_counter(); scope = DevelopmentScope(); preserve()
    # A completed run is immutable; do not silently launch optimization/retraining.
    if (ROOT/'results.json').exists(): raise FileExistsError('D3.5 already complete; verify existing evidence rather than refitting')
    data, profiles, audit = load_inputs(scope)
    folds = grouped_folds(data['pid']); row_fold = np.array([folds[p] for p in data['pid']])
    write_json(ROOT/'protocol_frozen.json', {'config_sha256': digest('configs/metabolic_d35.json'),
               'D2_config_sha256': digest('configs/metabolic_d2.json'), 'seed': SEED, 'folds': folds,
               'source_sha256': audit['source_sha256'], 'model_variants': VARIANTS,
               'code_hashes': {str(p): digest(p) for p in Path('src/metabolic_d35').glob('*.py')}})
    predictions = {variant: np.zeros_like(data['y']) for variant in VARIANTS}
    importance = {modality: np.zeros_like(data['y']) for modality in ['static', 'dynamic']}
    fold_results = []; training_records = []
    for fold in range(FOLDS):
        train = subset(data, row_fold != fold); valid = subset(data, row_fold == fold)
        train_ids = set(train['pid']); valid_ids = set(valid['pid'])
        assert_fold(train_ids, valid_ids); scope.assert_allowed(train_ids | valid_ids)
        donors = profile_permutation(train_ids, SEED+fold)
        write_json(ROOT/'folds'/f'{fold}_shuffle.json', donors)
        record = {'fold': fold, 'training_participants': len(train_ids), 'validation_participants': len(valid_ids), 'variants': {}}
        for variant in VARIANTS:
            begin = time.perf_counter()
            model = FusionModel(variant).fit(train, profiles, train_ids, SEED+fold,
                                            donors if variant == 'shuffled_static' else None)
            p = model.predict(valid, profiles); predictions[variant][row_fold == fold] = p
            model_path = ROOT/'models'/f'{variant}_fold{fold}.pkl'; model.save(model_path)
            record['variants'][variant] = {**forecast_metrics(valid['y'], p, valid['pid']), 'seconds': time.perf_counter()-begin, 'model_sha256': digest(model_path)}
            training_records.append({'fold': fold, 'variant': variant, 'participants': model.training_participants,
                                     'dynamic_preprocessing_participants': model.dynamic_preprocessor.training_participants if model.dynamic_preprocessor else [],
                                     'static_preprocessing_participants': model.static_encoder.training_participants if model.static_encoder else [],
                                     'validation_participants': sorted(valid_ids), 'static_features': model.static_encoder.names if model.static_encoder else []})
            if variant == 'fused':
                shuffled_valid = profile_permutation(valid_ids, SEED+100+fold)
                importance['static'][row_fold == fold] = model.predict(valid, profiles, donors=shuffled_valid)
                dynamic, current = dynamic_permutation(valid, SEED+200+fold)
                importance['dynamic'][row_fold == fold] = model.predict(valid, profiles, dynamic=dynamic, current=current)
            print(json.dumps({'fold': fold+1, 'variant': variant, 'training_participants': len(train_ids),
                              'validation_participants': len(valid_ids), 'seconds': round(time.perf_counter()-begin, 2)}), flush=True)
        fold_results.append(record)
        write_json(ROOT/'folds'/f'{fold}_metrics.json', record)
    per = {v: participant_errors(data['y'], p, data['pid']) for v, p in predictions.items()}
    primary = paired_interval(per['dynamic_only'], per['fused'])
    control = paired_interval(per['shuffled_static'], per['fused'])
    supported, outcome = decision(primary, control)
    contribution = {}
    for modality, p in importance.items():
        pe = participant_errors(data['y'], p, data['pid'])
        contribution[modality] = {'mae_degradation': (pe-per['fused']).mean(0).tolist(), 'method': 'validation modality permutation, participant macro; predictive reliance, not causation'}
    eligible = data['sequence'][:, -1] <= 180
    spike = {}
    for variant in ['dynamic_only', 'fused']:
        target = data['spike'][eligible]; score = trajectory_score(predictions[variant][eligible])
        spike[variant] = {'AUPRC': float(average_precision_score(target, score)), 'AUROC': float(roc_auc_score(target, score)),
                          'eligible_windows': int(eligible.sum()), 'positive_windows': int(target.sum()),
                          'scope': 'Uncalibrated forecast-derived trajectory ranking on current<=180 landmarks; unchanged D2 incident target'}
    results = {'participants': len(profiles), 'windows': len(data['y']), 'folds': FOLDS, 'seed': SEED,
               'horizons': [30,60,90,120], 'metrics': {v: forecast_metrics(data['y'], p, data['pid']) for v,p in predictions.items()},
               'fused_minus_dynamic': primary, 'fused_minus_shuffled': control,
               'age_sex_minus_dynamic': paired_interval(per['dynamic_only'],per['age_sex']),
               'fused_minus_age_sex': paired_interval(per['age_sex'],per['fused']),
               'shuffled_minus_dynamic': paired_interval(per['dynamic_only'],per['shuffled_static']),
               'fold_results': fold_results, 'modality_importance': contribution, 'spike': spike,
               'fusion_supported': supported, 'outcome': outcome, 'preservation': preserve(),
               'runtime_seconds': time.perf_counter()-started, 'bootstrap_replicates': BOOTSTRAPS}
    path = ROOT/'out_of_fold_predictions.npz'
    np.savez_compressed(path, participant_id=data['pid'], time=data['time'], y=data['y'], fold=row_fold, **predictions,
                        static_permuted=importance['static'], dynamic_permuted=importance['dynamic'])
    path.chmod(0o600)
    np.savez_compressed(ROOT/'per_participant_errors.npz', participant_id=np.unique(data['pid']), **per,
                        fused_minus_dynamic=per['fused']-per['dynamic_only'])
    (ROOT/'per_participant_errors.npz').chmod(0o600)
    write_json(ROOT/'training_records.json', training_records)
    results['prediction_sha256'] = digest(path)
    write_json(ROOT/'results.json', results)
    # Reports and aggregate integration manifest never contain participant identities.
    from .report import report
    report(results, audit)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    run()
