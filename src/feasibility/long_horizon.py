"""Phase 3.6: cumulative forecasts and genuine first-event early-warning bands."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from src.data.io import ArtifactWriter, dumps, file_hash, read_jsonl
from src.data.hemodynamic import asof_context, covered, summarize_windows
from src.data.eicu_validate import restore_series
from src.data.eicu_spec import PROTOCOL as PHASE35_PROTOCOL

PROTOCOL = {
    'version': 'phase3_6_v1', 'event': 'map65_30min',
    'cumulative_hours': [6, 12, 24, 48, 72],
    'bands_hours': [[6, 12], [12, 24], [24, 48], [48, 72]],
    'lookback_hours': [6, 12, 24], 'stability': ['phase35_washout', 'stable65_2h', 'stable70_6h'],
    'feature_buffer_minutes': 5, 'confirmation_minutes': 30, 'min_lead_minutes': 15,
    'history_coverage': .8, 'future_coverage': .9, 'max_gap_minutes': 10,
    'landmark_start': 'first whole ICU-relative hour >= lookback + 5 minutes',
    'landmark_stride': 'upper forecast endpoint; no overlapping full future intervals except confirmation tail',
    'band_rule': 'first subsequent episode onset in (lower, upper]; earlier first event excluded, never a control',
    'stable65_2h': 'Phase35 washout + 90% recent 2h coverage, all recent MAP >=65, fresh current MAP >=65',
    'stable70_6h': 'stable65 + 90% recent 6h coverage, all recent MAP >=70, current MAP >=70, 6h slope >=-1 mmHg/h, last 2h median minus preceding 2h median >=-5 mmHg',
    'current_map': 'last usable entry <=T0, age <=10min; eligibility/audit only, predictor histories end T0-5',
    'future_observation': 'complete ICU follow-up to upper endpoint +30min, >=90% MAP support, no gap >10min, no interpolation',
    'split': 'Phase35 patient assignments preserved; no new train/validation/test assignment or test model selection',
    'support_gates': 'Phase35 broad-pilot support thresholds, engineering gates only; not proof of predictability',
    'no_model': True,
}


def source_hash():
    digest = hashlib.sha256()
    paths = sorted(Path('src/feasibility').glob('*.py')) + [Path('tests/test_long_horizon.py')]
    for path in paths:
        digest.update(str(path).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def preservation_snapshot():
    paths = set()
    for root in Path('data/processed').glob('eicu_demo*'):
        paths.update(p for p in root.rglob('*') if p.is_file())
    paths.update(Path('data/raw/phase3_5').rglob('*'))
    paths.update(Path('data/results/phase3').rglob('*'))
    paths.update(Path('src/data').rglob('*.py'))
    paths.update(Path('src/data').rglob('*.json'))
    paths.update(Path('tests').glob('test_*.py'))
    paths.discard(Path('tests/test_long_horizon.py'))
    paths.update(Path('docs').glob('*'))
    paths.difference_update({Path('docs/LONG_HORIZON_FEASIBILITY.md'), Path('docs/PHASE3_6_DECISION.md')})
    paths.add(Path('Datasets/aibhs_datasets-main.zip'))
    paths.add(Path('scripts/audit_phase35_completion.py'))
    paths.update(Path(p) for p in ['README.md','data/README.md','.gitignore','AGENTS.md','requirements-models.txt'])
    return {str(p): file_hash(p) for p in sorted(paths) if p.is_file()}


def distribution(values):
    if not values: return None
    return dict(zip(['min', 'p05', 'median', 'p95', 'max'], np.quantile(values, [0, .05, .5, .95, 1]).tolist()))


def slice_valid(series, start, end):
    times = series['offset']; values = series['systemicmean']
    a = np.searchsorted(times, start, 'left'); b = np.searchsorted(times, end, 'right')
    valid = np.isfinite(values[a:b])
    return times[a:b][valid], values[a:b][valid]


def causal_state(patient, series, t0, lookback):
    """All baselines/conditions use <=T0; historical inputs use <=T0-5."""
    end = t0 - 5
    times, values = slice_valid(series, end - lookback, end)
    recent_t, recent_v = slice_valid(series, t0 - 120, end)
    six_t, six_v = slice_valid(series, end - 360, end)
    last_t, last_v = slice_valid(series, 0, t0)
    current = float(last_v[-1]) if len(last_v) else None
    current_time = float(last_t[-1]) if len(last_t) else None
    freshness = t0 - current_time if current_time is not None else None
    previous_t, previous_v = slice_valid(series, end - 240, end - 120 - 1)
    recent2_t, recent2_v = slice_valid(series, end - 120, end)
    slope = float(np.polyfit((six_t - six_t[0]) / 60, six_v, 1)[0]) if len(six_v) > 1 else None
    median_drop = float(np.median(recent2_v) - np.median(previous_v)) if len(recent2_v) and len(previous_v) else None
    flags, _, labs = asof_context(patient, t0, patient['labs'])
    past_t, _ = slice_valid(series, 0, end)
    diffs = np.diff(past_t)
    historical_diff = np.diff(times)
    base = not np.any(recent_v < 65)
    strict65 = base and freshness is not None and freshness <= 10 and current >= 65 and len(recent_v) > 0 and np.min(recent_v) >= 65 and covered(series['offset'], series['systemicmean'], t0 - 120, end, .9, 10)
    strict70 = strict65 and current >= 70 and len(six_v) > 0 and np.min(six_v) >= 70 and covered(series['offset'], series['systemicmean'], end - 360, end, .9, 10) and slope is not None and slope >= -1 and median_drop is not None and median_drop >= -5
    return {
        'history_supported': covered(series['offset'], series['systemicmean'], end - lookback, end, .8, 10),
        'stable': {'phase35_washout': bool(base), 'stable65_2h': bool(strict65), 'stable70_6h': bool(strict70)},
        'map_at_t0': current, 'map_at_t0_offset': current_time, 'map_age_minutes': freshness,
        'recent_2h_map_min': float(min(recent_v)) if len(recent_v) else None,
        'recent_2h_map_median': float(np.median(recent_v)) if len(recent_v) else None,
        'recent_6h_map_min': float(min(six_v)) if len(six_v) else None,
        'recent_6h_slope_per_hour': slope, 'recent_2h_median_change': median_drop,
        'history_max_offset': float(times[-1]) if len(times) else None,
        'history_n': len(times), 'history_observed_fraction': min(1., len(times) / (lookback / 5 + 1)),
        'history_connected_hours': float(sum(historical_diff[historical_diff <= 10])) / 60,
        'available_past_map_span_hours': float(past_t[-1] - past_t[0]) / 60 if len(past_t) else 0.,
        'available_past_map_connected_hours': float(sum(diffs[diffs <= 10])) / 60,
        'hf_known_at_t0': flags['hf'] == 1, 'hf_history_known_at_t0': flags['hf_history'] == 1,
        'cv_known_at_t0': flags['cardiovascular'] == 1,
        'clinical_fusion': any(v == 1 for v in flags.values()) or bool(labs),
        'available_lab_concepts': sorted(labs),
        'latest_clinical_available_offset': max([r['available_offset'] for r in patient['conditions'] if r['available_offset'] <= t0 and r['observed_offset'] <= t0] + [r['available_offset'] for r in labs.values()], default=None),
    }


def observation_support(patient, series, t0, upper):
    end = t0 + upper + 30
    t, values = slice_valid(series, t0, end)
    all_t, _ = slice_valid(series, 0, patient['end'])
    last = float(all_t[-1]) if len(all_t) else -1
    discharge = patient['end'] < end
    recording_end = last < end
    edges = np.r_[t0, t, end]
    max_gap = float(max(np.diff(edges)))
    fraction = min(1., len(t) / ((upper + 30) / 5 + 1))
    support = covered(series['offset'], series['systemicmean'], t0, end, .9, 10)
    return {'complete': not discharge and not recording_end and support,
            'discharge_censored': discharge, 'recording_end_censored': recording_end,
            'missing_period_censored': not discharge and not recording_end and not support,
            'future_max_gap_minutes': max_gap, 'future_observed_fraction': fraction,
            'required_followup_end': end,
            'death_before_completion': discharge and patient['raw'].get('unitdischargestatus') == 'Expired',
            'censored_discharge_location': patient['raw'].get('unitdischargelocation', '') if discharge else None}


def evaluate_stay(patient, series, events, lower, upper, lookback, stability, cache=None):
    """Risk sets use spaced forward landmarks, not event-centered retrospective indices."""
    counts = Counter(); diagnostics = defaultdict(list); rows = []
    if not patient['adult']:
        return rows, {'not_adult_stays': 1}, dict(diagnostics)
    start = math.ceil((lookback + 5) / 60) * 60
    for t0 in range(start, int(patient['end']) + 1, upper):
        counts['candidate_windows'] += 1
        key = (patient['icu_stay_id'], t0, lookback)
        if cache is not None and key in cache: state = cache[key]
        else:
            state = causal_state(patient, series, t0, lookback)
            if cache is not None: cache[key] = state
        if not state['history_supported']:
            counts['insufficient_past_history'] += 1; continue
        counts['history_supported_windows'] += 1
        for k in ['map_at_t0', 'map_age_minutes', 'recent_2h_map_min', 'recent_2h_map_median', 'recent_6h_slope_per_hour']:
            if state[k] is not None: diagnostics['history_supported_' + k].append(state[k])
        if not state['stable'][stability]:
            counts['unstable_at_t0'] += 1; continue
        # Uses event labels only for retrospective underway exclusion, never baselines.
        if any(e['onset'] <= t0 <= e['last_low'] for e in events):
            counts['event_underway_at_t0'] += 1; continue
        counts['stable_at_risk_windows'] += 1
        support = observation_support(patient, series, t0, upper)
        if not support['complete']:
            counts['incomplete_future_windows'] += 1
            for k in ['discharge_censored', 'recording_end_censored', 'missing_period_censored', 'death_before_completion']:
                counts[k] += bool(support[k])
            if support['censored_discharge_location'] is not None:
                counts['discharge_location:' + support['censored_discharge_location']] += 1
            diagnostics['censored_future_max_gap_minutes'].append(support['future_max_gap_minutes'])
            diagnostics['censored_future_observed_fraction'].append(support['future_observed_fraction'])
            continue
        counts['complete_future_windows'] += 1
        future = [e for e in events if e['onset'] > t0]
        event = future[0] if future else None
        lead = event['onset'] - t0 if event else None
        if lead is not None and lead <= 15:
            counts['event_within_15min'] += 1; continue
        if lower and lead is not None and lead <= lower:
            counts['earlier_first_event_before_band'] += 1; continue
        positive = event is not None and lead <= upper and lead > lower
        row = {'patient_id': patient['patient_id'], 'icu_stay_id': patient['icu_stay_id'],
               't0': t0, 'lower_minutes': lower, 'horizon_minutes': upper, 'lookback_minutes': lookback,
               'stability': stability, 'label': int(positive), 'event_onset': event['onset'] if positive else None,
               'event_id': event['event_id'] if positive else None, 'lead_minutes': lead if positive else None,
               **{k: v for k, v in state.items() if k not in ['stable', 'history_supported']},
               'future_observed_fraction': support['future_observed_fraction'],
               'future_max_gap_minutes': support['future_max_gap_minutes']}
        rows.append(row); counts['eligible_windows'] += 1
    return rows, dict(counts), dict(diagnostics)


def summarize(rows):
    result = summarize_windows(rows)
    positive = [r for r in rows if r['label']]
    result.update(unique_positive_events=len({r['event_id'] for r in positive}),
                  time_to_event_hours=distribution([r['lead_minutes'] / 60 for r in positive]),
                  available_history_span_hours=distribution([r['available_past_map_span_hours'] for r in rows]),
                  available_history_connected_hours=distribution([r['available_past_map_connected_hours'] for r in rows]),
                  requested_history_connected_hours=distribution([r['history_connected_hours'] for r in rows]),
                  requested_history_observed_fraction=distribution([r['history_observed_fraction'] for r in rows]),
                  map_at_t0=distribution([r['map_at_t0'] for r in rows if r['map_at_t0'] is not None]),
                  recent_2h_map_min=distribution([r['recent_2h_map_min'] for r in rows if r['recent_2h_map_min'] is not None]),
                  recent_2h_map_median=distribution([r['recent_2h_map_median'] for r in rows if r['recent_2h_map_median'] is not None]),
                  positive_map_at_t0=distribution([r['map_at_t0'] for r in positive if r['map_at_t0'] is not None]),
                  positive_recent_2h_map_min=distribution([r['recent_2h_map_min'] for r in positive if r['recent_2h_map_min'] is not None]),
                  clinical_fusion_patients=len({r['patient_id'] for r in rows if r['clinical_fusion']}),
                  clinical_fusion_stays=len({r['icu_stay_id'] for r in rows if r['clinical_fusion']}),
                  clinical_fusion_windows=sum(r['clinical_fusion'] for r in rows))
    return result


def build(output='data/results/phase36', source='data/processed/eicu_demo'):
    output = Path(output); source = Path(source)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a fresh Phase 3.6 destination')
    preservation = preservation_snapshot(); version = source_hash()
    manifest = json.loads((source / 'manifest.json').read_text())
    for name, meta in manifest['artifacts'].items():
        if file_hash(source / name) != meta['sha256']: raise ValueError('Phase 3.5 artifact changed')
    patients = {p['source_stay_key']: p for p in read_jsonl(source / 'patient_context.jsonl')}
    series = restore_series(source / 'vitalPeriodic.npz')
    by_id = {p['icu_stay_id']: k for k, p in patients.items()}
    events = defaultdict(list)
    for event in read_jsonl(source / 'events.jsonl'):
        if event['candidate'] == 'map65_30min': events[by_id[event['icu_stay_id']]].append(event)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.phase36-stage-', dir=output.parent)); stage.chmod(0o700)
    writer = ArtifactWriter(stage)
    try:
        summaries = {}; cache = {}
        targets = [('cumulative', 0, h * 60) for h in PROTOCOL['cumulative_hours']] + [('band', a * 60, b * 60) for a, b in PROTOCOL['bands_hours']]
        for kind, lower, upper in targets:
            for history in PROTOCOL['lookback_hours']:
                for stable in PROTOCOL['stability']:
                    rows = []; counts = Counter(); diagnostic = defaultdict(list)
                    for stay, physiology in sorted(series.items()):
                        cohort, excluded, diag = evaluate_stay(patients[stay], physiology, events[stay], lower, upper, history * 60, stable, cache)
                        for row in cohort: row['kind'] = kind
                        rows.extend(cohort); counts.update(excluded)
                        for key, values in diag.items(): diagnostic[key].extend(values)
                    key = f'{kind}_{lower//60}_{upper//60}_l{history}_{stable}'
                    for row in rows: writer.emit('windows', row)
                    summaries[key] = {'kind': kind, 'lower_hours': lower // 60, 'upper_hours': upper // 60,
                                      'lookback_hours': history, 'stability': stable, 'full': summarize(rows),
                                      'hf_known': summarize([r for r in rows if r['hf_known_at_t0']]),
                                      'hf_history_known': summarize([r for r in rows if r['hf_history_known_at_t0']]),
                                      'cv_known': summarize([r for r in rows if r['cv_known_at_t0']]),
                                      'exclusions': dict(counts), 'diagnostics': {k: distribution(v) for k, v in diagnostic.items()}}
                    print({'configuration': key, 'patients': summaries[key]['full']['patients'], 'positive_patients': summaries[key]['full']['positive_patients']}, flush=True)
        writer.json('report', {'protocol': PROTOCOL, 'source_phase35_manifest_sha256': file_hash(source / 'manifest.json'),
                               'source_phase35_protocol': PHASE35_PROTOCOL, 'configurations': summaries,
                               'recommendation': 'A. 3h remains primary', 'judges_48_72_prediction_claim': 'NO',
                               'evidence': 'cohort/label feasibility only; no predictive performance or early-warning skill tested'})
        writer.json('preservation', preservation)
        writer.ensure_tables(['windows']); writer.close()
        if preservation_snapshot() != preservation: raise ValueError('Prior results changed during Phase 3.6')
        if source_hash() != version: raise ValueError('Phase 3.6 source changed during build')
        writer.manifest({'phase35_manifest_sha256': file_hash(source / 'manifest.json')}, 2026, PROTOCOL, version)
        if output.exists(): output.rmdir()
        stage.rename(output)
        return {'configurations': len(summaries), 'models_trained': 0, 'output': str(output)}
    except BaseException:
        writer.close(); shutil.rmtree(stage); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/results/phase36')
    parser.add_argument('--source', default='data/processed/eicu_demo')
    args = parser.parse_args(); print(build(args.output, args.source))
