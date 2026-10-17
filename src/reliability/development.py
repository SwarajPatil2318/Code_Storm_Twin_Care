"""Cached development preparation and one bounded fast-subset experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.io import file_hash
from src.twin.cache import validation_groups
from src.twin.models import cumulative
from src.twin.protocol import ALLOWLIST, EHR_COLUMNS, ABS_FEATURES, PERSONAL_FEATURES
from src.twin.metrics import basic, calibration_diagnostics
from src.reliability.access import (development_cache, phase6_identity, reference_artifact,
    load_config, root, save_json, save_arrays, save_model, timed, deterministic_patients, routed_rows)
from src.reliability.features import recovery_features, NAMES
from src.reliability.models import OffsetReliability, HorizonCalibration, select_calibration, temporal_recovery_rule


def reference_matrix(data):
    prefix = np.concatenate([data['ehr'], data['absolute'], data['personal']], axis=1)
    return np.pad(prefix, ((0,0),(0,len(ALLOWLIST)-prefix.shape[1])), constant_values=0)


def reference_predict(data, model, calibrator):
    return calibrator.transform(model.predict_hazards(reference_matrix(data)))


def subset_manifest(config, train, validation):
    out = root(config)
    path = out / 'development_subset.json'
    c, _ = phase6_identity()
    _, vr = validation_groups(validation['patient_ids'], c)
    chosen_train = deterministic_patients(train['patient_ids'], config['subset_train_patients'], config['subset_seed'], 'train')
    chosen_val = []
    counts = {'selection': 240, 'calibration': 240, 'operating': 120}
    if config['subset_validation_patients'] != sum(counts.values()):
        raise ValueError('Expected one fixed 600-patient validation subset')
    for role, n in counts.items():
        candidates = [p for p, r in vr.items() if r == role]
        chosen_val.extend(deterministic_patients(candidates, n, config['subset_seed'], role))
    payload = {'seed': config['subset_seed'], 'config_sha256': config['_sha256'],
        'train_patients': chosen_train, 'validation_patients': chosen_val,
        'validation_subroles': {p: vr[p] for p in chosen_val},
        'counts': {'train': len(chosen_train), 'validation': len(chosen_val), **counts},
        'sampling': 'One deterministic patient-grouped draw; no label-based favorable resampling'}
    if path.exists():
        stored = json.loads(path.read_text())
        if any(stored.get(k) != v for k, v in payload.items()):
            raise ValueError('Persisted development subset cannot be resampled')
    else:
        save_json(path, payload)
    return payload


def evaluation_annotations(data):
    """Post-prediction episode annotations for development evaluation only."""
    ids = np.asarray(data['patient_ids']).astype(str)
    hours = np.asarray(data['landmark_hours'])
    flags = {k: np.zeros(len(ids), bool) for k in ['recovery', 'deterioration', 'decoy', 'ordinary_stable']}
    allowed = set(ids)
    unique, inverse = np.unique(ids, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(np.bincount(inverse))]
    for row in routed_rows('ground_truth', allowed):
        k = np.searchsorted(unique, row['patient_id'])
        ix = order[starts[k]:starts[k+1]]
        h = hours[ix]
        active = np.zeros(len(ix), bool)
        recovery = np.zeros(len(ix), bool)
        deterioration = np.zeros(len(ix), bool)
        decoy = np.zeros(len(ix), bool)
        for e in row['episodes']:
            active |= (h >= e['onset_hour']) & (h < e['recovery_end_hour'])
            recovery |= (h >= e['recovery_hour']) & (h < e['recovery_end_hour'])
            deterioration |= (h >= e['onset_hour']) & (h < e['recovery_hour'])
        for e in row['decoys']:
            decoy |= (e['start_hour'] <= h) & (e['end_hour'] > h-48)
        flags['recovery'][ix] = recovery
        flags['deterioration'][ix] = deterioration
        flags['decoy'][ix] = decoy
        flags['ordinary_stable'][ix] = ~active & ~decoy
    return flags


def prepare_development(config=None):
    c = config or load_config()
    start = time.monotonic()
    out = root(c)
    if (out/'PHASE6_5_FROZEN_MANIFEST.json').exists():
        raise PermissionError('Development is closed after freeze')
    train, tm = development_cache('train')
    validation, vm = development_cache('validation')
    subset = subset_manifest(c, train, validation)
    model = joblib.load(reference_artifact('selected.joblib'))
    calibrator = joblib.load(reference_artifact('calibration.joblib'))
    expected_cols = EHR_COLUMNS + ABS_FEATURES + PERSONAL_FEATURES
    if model.names != expected_cols:
        raise ValueError('Personalized reference unexpectedly depends on Shadow/TDI')
    provenance = {'phase6_model_sha256': file_hash(reference_artifact('selected.joblib')),
        'phase6_calibrator_sha256': file_hash(reference_artifact('calibration.joblib')),
        'reliability_feature_names': NAMES, 'feature_source_sha256': file_hash(Path(__file__).with_name('features.py')),
        'phase65_config_sha256': c['_sha256'], 'clinical_test_access': False,
        'cache_manifests_sha256': {'train': file_hash(Path('data/results/phase6/features/train/manifest.json')),
                                    'validation': file_hash(Path('data/results/phase6/features/validation/manifest.json'))}}
    path = out/'development_cache.npz'
    proof_path = out/'development_cache_provenance.json'
    if path.exists():
        proof = json.loads(proof_path.read_text())
        if any(proof.get(k) != v for k,v in provenance.items()) or proof['artifact_sha256'] != file_hash(path):
            raise ValueError('Phase 6.5 development cache is stale; refuse silent recomputation')
        with np.load(path, allow_pickle=False) as z:
            data = {k:z[k] for k in z.files}
        timed(c, 'cached_artifact_loading', start, {'reused_phase65_cache': True})
        return data, subset
    blocks = []
    for role, source in [('train',train), ('validation',validation)]:
        f = recovery_features(source)
        if role == 'validation':
            raw = np.load(reference_artifact('validation_raw_hazards.npy'), allow_pickle=False)
            hazards = np.load(reference_artifact('validation_hazards.npy'), allow_pickle=False)
            if not np.allclose(hazards, calibrator.transform(raw), rtol=1e-12, atol=1e-12):
                raise ValueError('Cached validation calibration equivalence failed')
            # Verify row order and model provenance with a fixed evenly spaced sample.
            sample = np.linspace(0,len(raw)-1,128,dtype=int)
            selected = {k:source[k][sample] for k in ['ehr','absolute','personal']}
            if not np.allclose(raw[sample], model.predict_hazards(reference_matrix(selected)), atol=1e-12):
                raise ValueError('Cached validation prediction provenance failed')
        else:
            hazards = np.empty((len(source['labels']),3))
            for a in range(0,len(hazards),8192):
                sl = slice(a,a+8192)
                values = {k:source[k][sl] for k in ['ehr','absolute','personal']}
                hazards[sl] = reference_predict(values, model, calibrator)
        # Annotations are read only after these predictions/features exist.
        annotations = evaluation_annotations(source)
        ids = np.asarray(source['patient_ids']).astype(str)
        if role == 'train':
            subrole = np.full(len(ids),'train',dtype='<U11')
            fast = np.isin(ids,subset['train_patients'])
        else:
            _, vr = validation_groups(ids, phase6_identity()[0])
            subrole = np.array([vr[p] for p in ids])
            fast = np.isin(ids,subset['validation_patients'])
        blocks.append({'hazards':hazards,'features':f,'labels':np.asarray(source['labels']),
            'patient_ids':ids,'landmark_hours':np.asarray(source['landmark_hours']),
            'roles':subrole,'fast_subset':fast,**annotations})
        print({'phase65_development_cached_role':role,'patients':len(np.unique(ids)),'rows':len(ids),'spent_test_used':False},flush=True)
    data = {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    save_arrays(path, **data)
    save_json(proof_path, {**provenance,'artifact_sha256':file_hash(path),
        'phase6_feature_arrays_reused':len(tm['artifacts'])+len(vm['artifacts']),
        'phase6_feature_arrays_recomputed':0,'phase6_core_models_retrained':0,
        'validation_predictions_reused':len(validation['labels']), 'train_predictions_computed_once':len(train['labels']),
        'fast_event_patients':len(np.unique(data['patient_ids'][data['fast_subset'] & (data['labels'].sum(1)>0)])),
        'fast_recovery_negative_patients':len(np.unique(data['patient_ids'][data['fast_subset'] & data['recovery'] & (data['labels'].sum(1)==0)]))})
    timed(c,'cached_artifact_loading_and_one_time_reference_scoring',start,
        {'patients':8500,'rows':len(data['labels']),'feature_arrays_recomputed':0})
    return data,subset


def thresholds_at_reference_sensitivity(prob, reference, labels, mask, reference_thresholds):
    result = {}
    truth = np.cumsum(labels[mask],axis=1)
    for j in range(3):
        key = str((j+1)*24)
        t = reference_thresholds[key]['balanced']
        y = truth[:,j]
        target_tp = int(np.sum(y * (reference[mask,j]>=t)))
        positives = np.sort(prob[mask,j][y>0])[::-1]
        result[key] = {'balanced':float(positives[min(target_tp,len(positives))-1]) if target_tp else float(np.nextafter(prob[mask,j].max(),np.inf))}
    return result


def summary(data, probability, thresholds, mask):
    y = np.cumsum(data['labels'][mask],axis=1)
    result = {}
    for j in range(3):
        key = str((j+1)*24)
        p = probability[mask,j]
        t = thresholds[key]['balanced']
        result[key] = {'metrics':basic(y[:,j],p,t),'calibration':calibration_diagnostics(y[:,j],p)}
    negative = data['labels'].sum(1)==0
    result['subgroups72'] = {}
    for group in ['deterioration','recovery','decoy','ordinary_stable']:
        m = mask & data[group]
        if group != 'deterioration':
            m &= negative
        result['subgroups72'][group] = {'rows':int(m.sum()),'patients':len(np.unique(data['patient_ids'][m])),
            'alert_fraction':float(np.mean(probability[m,2]>=thresholds['72']['balanced'])) if m.any() else None,
            'metrics':basic(data['labels'][m].sum(1),probability[m,2],thresholds['72']['balanced']) if m.any() else None}
    return result


def fast_experiment(config=None):
    c = config or load_config()
    out = root(c)
    if (out/'fast_selection.json').exists():
        stored=json.loads((out/'fast_selection.json').read_text())
        if stored['config_sha256']!=c['_sha256']:
            raise ValueError('Fast selection already fixed with a different config')
        return stored
    data,_ = prepare_development(c)
    start=time.monotonic()
    fast=data['fast_subset']
    train=fast & (data['roles']=='train')
    selection=fast & (data['roles']=='selection')
    cal=fast & (data['roles']=='calibration')
    operating=fast & (data['roles']=='operating')
    h=data['hazards']
    ref=cumulative(h)
    thresholds=json.loads(reference_artifact('thresholds.json').read_text())['thresholds']
    base=summary(data,ref,thresholds,selection)
    model=OffsetReliability(c['reliability_l2']).fit(h[train],data['features'][train],data['labels'][train],['train'])
    adjusted=model.predict(h,data['features'])
    methods,trials=select_calibration(adjusted,data['labels'],cal,selection)
    candidate_cal=HorizonCalibration().fit(adjusted[cal],data['labels'][cal],['calibration'],methods)
    p=cumulative(candidate_cal.transform(adjusted))
    candidate_thresholds=thresholds_at_reference_sensitivity(p,ref,data['labels'],operating,thresholds)
    candidate=summary(data,p,candidate_thresholds,selection)
    old=base['subgroups72']['recovery']['alert_fraction']
    new=candidate['subgroups72']['recovery']['alert_fraction']
    sens_loss=base['72']['metrics']['sensitivity']-candidate['72']['metrics']['sensitivity']
    ap_retention=candidate['72']['metrics']['auprc']/base['72']['metrics']['auprc']
    gates={'recovery_reduction':old is not None and old>0 and new<=old*(1-c['minimum_recovery_relative_reduction']),
        'sensitivity_preserved':sens_loss<=c['maximum_sensitivity_loss'],
        'ranking_retained':ap_retention>=c['minimum_ranking_retention']}
    winner='offset_logistic' if all(gates.values()) else 'reference'
    if winner=='reference':
        methods,trials=select_calibration(h,data['labels'],cal,selection)
    chosen_h=adjusted if winner=='offset_logistic' else h
    selected_cal=HorizonCalibration().fit(chosen_h[cal],data['labels'][cal],['calibration'],methods)
    cp=cumulative(selected_cal.transform(chosen_h))
    ct=thresholds_at_reference_sensitivity(cp,ref,data['labels'],operating,thresholds)
    selected=summary(data,cp,ct,selection)
    save_model(out/'fast_reliability.joblib',model)
    result={'config_sha256':c['_sha256'],'winner':winner,'calibration_methods':methods,
        'calibration_trials':trials,'selection_gates':gates,'sensitivity_loss_candidate':sens_loss,
        'ap_retention_candidate':ap_retention,'reference':base,'candidate':candidate,'selected':selected,
        'candidate_count':1,'feature_names':NAMES,'subset_patients':1800,
        'threshold_selection':'Disjoint operating patients, match frozen reference sensitivity',
        'test_used':False,'replication_used':False,'exploratory_bootstraps':0}
    save_json(out/'fast_selection.json',result)
    timed(c,'fast_subset_experiments',start,{'patients':1800,'candidate_count':1,'bootstraps':0})
    print({'phase65_fast_selection':winner,'gates':gates,'calibration_methods':methods,'spent_test_used':False},flush=True)
    return result


def recovery_rule_experiment(config=None):
    """Second and final candidate, justified by the failed first recovery gate."""
    c=config or load_config();out=root(c)
    path=out/'fast_selection_final.json'
    if path.exists():
        d=json.loads(path.read_text())
        if d['config_sha256']!=c['_sha256']:raise ValueError('Final subset selection config changed')
        return d
    if (out/'PHASE6_5_FROZEN_MANIFEST.json').exists():raise PermissionError('Selection closed')
    # Keep the initial result as evidence. The config's candidate list is the
    # only expansion: no new patient draw, model architecture or strength sweep.
    initial=json.loads((out/'fast_selection.json').read_text())
    if initial['selection_gates']['recovery_reduction']:raise ValueError('Second candidate not justified')
    with np.load(out/'development_cache.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    start=time.monotonic();fast=data['fast_subset'];selection=fast&(data['roles']=='selection');cal=fast&(data['roles']=='calibration');op=fast&(data['roles']=='operating')
    h=data['hazards'];ref=cumulative(h);thresholds=json.loads(reference_artifact('thresholds.json').read_text())['thresholds']
    attenuation=.65
    adjusted=temporal_recovery_rule(h,data['features'],attenuation)
    methods,trials=select_calibration(adjusted,data['labels'],cal,selection)
    model=HorizonCalibration().fit(adjusted[cal],data['labels'][cal],['calibration'],methods)
    p=cumulative(model.transform(adjusted));ct=thresholds_at_reference_sensitivity(p,ref,data['labels'],op,thresholds)
    base=summary(data,ref,thresholds,selection);candidate=summary(data,p,ct,selection)
    old=base['subgroups72']['recovery']['alert_fraction'];new=candidate['subgroups72']['recovery']['alert_fraction']
    loss=base['72']['metrics']['sensitivity']-candidate['72']['metrics']['sensitivity'];retention=candidate['72']['metrics']['auprc']/base['72']['metrics']['auprc']
    gates={'recovery_reduction':old is not None and old>0 and new<=old*(1-c['minimum_recovery_relative_reduction']),
        'sensitivity_preserved':loss<=c['maximum_sensitivity_loss'],'ranking_retained':retention>=c['minimum_ranking_retention']}
    winner='temporal_recovery_rule' if all(gates.values()) else initial['winner']
    if winner!='temporal_recovery_rule':methods=initial['calibration_methods']
    final={**initial,'config_sha256':c['_sha256'],'winner':winner,'calibration_methods':methods,
        'candidate_count':2,'initial_experiment_sha256':file_hash(out/'fast_selection.json'),
        'second_candidate_justification':'First candidate failed the recovery false-alert reduction gate',
        'temporal_rule':{'attenuation':attenuation,'gates':gates,'results':candidate,'calibration_trials':trials,
            'sensitivity_loss':loss,'ap_retention':retention},'initial_selection_preserved':True}
    save_json(path,final);timed(c,'second_and_final_recovery_candidate',start,{'patients':1800,'total_candidates':2,'bootstraps':0})
    print({'phase65_final_subset_winner':winner,'temporal_rule_gates':gates},flush=True)
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--recovery-rule',action='store_true')
    args=p.parse_args()
    with threadpool_limits(limits=1):
        recovery_rule_experiment() if args.recovery_rule else prepare_development() if args.prepare_only else fast_experiment()
