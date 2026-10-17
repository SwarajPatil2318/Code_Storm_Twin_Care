"""Build three curated synthetic replays without training or reopening Phase 6 TEST.

Only the independent, already evaluated Phase 6.5 replication is read. Outcome
annotations select demonstration stories offline; no annotations enter the API.
"""
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.data.schema import Observation, Provenance, SourceValue, record
from src.synthetic.config import SIGNALS, UNITS
from src.synthetic.ehr_generator import bundle, instant
from src.synthetic.sensors import QUALITY, REASONS, SOURCES
from src.twin.inference import finite, quality_uncertainty, twin_state
from src.twin.protocol import PRIMARY_SIGNALS, code_hash

ROOT = Path('data/results/phase6')
REP = Path('data/results/phase6_5')
COHORT = Path('data/synthetic/phase6_5_replication_2027')
OUTPUT = Path('data/product')
FIELDS = ('absolute', 'personal', 'ehr', 'recent', 'recent_count', 'reference',
          'prior_recent', 'scale', 'persistence', 'baseline_count', 'confidence',
          'patient_ids', 'landmark_hours')
SOURCES_EVIDENCE = [
    {'name': 'CHIRON', 'role': 'Longitudinal HF physiology grounding', 'label': 'REAL-DATA RESEARCH'},
    {'name': 'PerHeart', 'role': 'HF home-monitoring and device-data grounding', 'label': 'REAL-DATA RESEARCH'},
    {'name': 'eICU', 'role': 'Critical-care physiology and deterioration research', 'label': 'REAL-DATA RESEARCH'},
    {'name': 'VitalDB', 'role': 'Physiological relationship research', 'label': 'REAL-DATA RESEARCH'},
    {'name': 'Synthea', 'role': 'Synthetic EHR/FHIR resource and coding shapes', 'label': 'EXPLORATORY'},
    {'name': 'Synthetic Twin Cohort', 'role': 'Paired EHR, wearable and outcome POC training/evaluation', 'label': 'SYNTHETIC POC VALIDATED'},
]
LIMITATIONS = [
    'Synthetic end-to-end training and evaluation; no prospective clinical validation.',
    'No treatment recommendations or clinical deployment readiness.',
    'TDI is descriptive; incremental risk-prediction benefit was not demonstrated.',
    'Incremental predictive benefit from EHR context was not established.',
    'Expected bands and data quality are not individual confidence guarantees.',
    'Recovery false alerts remain. Phase 6.5 reliability and Adaptive Sensing were NO-GO.',
    'The population-normal challenge did not establish a statistically conclusive advantage.',
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference():
    seal = json.loads((ROOT / 'selection_seal.json').read_text())
    if code_hash() != seal['model_code_sha256']:
        raise ValueError('Frozen Phase 6 source changed')
    approved = ['shadow_engines.joblib'] + [f'models/personalized_fusion/{f}' for f in
                                          ('selected.joblib', 'calibration.joblib', 'thresholds.json')]
    for path in approved:
        if sha(ROOT / path) != seal['artifacts'][path]:
            raise ValueError('Frozen Phase 6 artifact changed')
    manifest = json.loads((REP / 'replication_features/manifest.json').read_text())
    for field in FIELDS:
        filename = field + '.npy'
        expected = manifest['artifacts'][filename]
        if isinstance(expected, dict):
            expected = expected['sha256']
        if sha(REP / 'replication_features' / filename) != expected:
            raise ValueError('Replication feature integrity failure')
    sidecar = json.loads((REP / 'array_artifact_labels.json').read_text())
    if sha(REP / 'replication_predictions.npz') != sidecar['artifacts']['replication_predictions.npz']['sha256']:
        raise ValueError('Original replication prediction cache changed')
    return seal


def select_replays(data, risk):
    """Post-evaluation illustrative selection, not a new clinical experiment."""
    recent, ref, scale = data['recent'], data['reference'], data['scale']
    z = (recent - ref) / np.maximum(scale, 1e-6)
    dist = np.nanmean(np.minimum(abs(z), 8), axis=1)
    n = len(risk)
    ends = np.arange(7, n)
    valid = data['patient_ids'][ends] == data['patient_ids'][ends - 7]
    ends = ends[valid]
    deterioration = np.load(REP / 'replication_features/deterioration.npy', mmap_mode='r')
    recovery = np.load(REP / 'replication_features/recovery.npy', mmap_mode='r')
    ordinary = np.load(REP / 'replication_features/ordinary_stable.npy', mmap_mode='r')
    hr, hrv, steps = [PRIMARY_SIGNALS.index(s) for s in ('heart_rate', 'hrv_rmssd', 'steps')]
    core = [('heart_rate', 50, 100), ('spo2', 94, 100), ('systolic_bp', 90, 140), ('diastolic_bp', 55, 90)]
    population = np.ones(n, dtype=bool)
    for s, lo, hi in core:
        v = recent[:, PRIMARY_SIGNALS.index(s)]
        population &= np.isfinite(v) & (v >= lo) & (v <= hi)
    masks = [deterioration[ends] & population[ends] & (risk[ends, 2] > .083) &
             (risk[ends-7, 2] < .014) & (dist[ends-7] < .75) &
             (z[ends, hrv] < -1) & (recent[ends, steps] / np.maximum(ref[ends, steps], 1e-6) < .8)]
    # Prefer a clear, reproducible mechanism story without fabricating a smooth curve.
    candidates = ends[masks[0]]
    if not len(candidates):
        raise ValueError('No deterioration demonstration available')
    a = int(candidates[np.argmax(risk[candidates, 2] - risk[candidates-7, 2] - .05*dist[candidates-7])])
    candidates = ends[recovery[ends-3] & (dist[ends] < dist[ends-3]) &
                      (risk[ends-3, 2] > risk[ends-7, 2]) &
                      (risk[ends-3, 2] > risk[ends, 2]) &
                      (data['patient_ids'][ends] != data['patient_ids'][a])]
    if not len(candidates):
        raise ValueError('No transient/recovery demonstration available')
    b = int(candidates[np.argmax(dist[candidates-3] - dist[candidates])])
    candidates = ends[ordinary[ends] & ordinary[ends-7] & (dist[ends] < .65) &
                      (dist[ends-7] < .65) & (risk[ends, 2] < .014) &
                      (risk[ends-7, 2] < .014)]
    candidates = [i for i in candidates if max(risk[i-7:i+1, 2]) < .014 and
                  data['patient_ids'][i] not in [data['patient_ids'][a], data['patient_ids'][b]]]
    if not len(candidates):
        raise ValueError('No stable demonstration available')
    c = int(min(candidates, key=lambda i: np.mean(dist[i-7:i+1])))
    return [np.arange(end-7, end+1) for end in (a, b, c)]


def read_rows(folder):
    with gzip.open(COHORT / folder / 'records.jsonl.gz', 'rt') as handle:
        return {row['patient_id']: row for row in map(json.loads, handle)}


def observations(private_id, patient, first_hour, last_hour):
    statuses = {v:k for k,v in QUALITY.items()}
    reasons = {v:k for k,v in REASONS.items()}
    result = []
    for file in sorted((COHORT / 'observations').glob('batch_*.npz')):
        with np.load(file, allow_pickle=False) as arr:
            matches = np.flatnonzero(arr['patient_ids'] == private_id)
            if not len(matches):
                continue
            p = int(matches[0])
            file_sha = sha(file)
            for t in range(int(first_hour), int(last_hour)+1):
                for j, s in enumerate(SIGNALS):
                    if s == 'map' or not arr['scheduled'][p,t,j]:
                        continue
                    av = arr['available_hour'][p,t,j]
                    if np.isfinite(av) and av > last_hour:
                        continue
                    raw, value = arr['raw_value'][p,t,j], arr['value'][p,t,j]
                    prov = Provenance('synthetic-demo', 'synthetic-demo-stream', str(t-first_hour),
                                      s, s, 'identity; source units and QC retained', file_sha, UNITS[j], 'synthetic_scenario')
                    sv = SourceValue(finite(raw), finite(value), statuses[int(arr['quality_status'][p,t,j])],
                                     [reasons[int(arr['reason_code'][p,t,j])]], UNITS[j], prov)
                    obs = Observation(f"{patient['patient_id']}:{t-first_hour}:{s}", patient['patient_id'],
                                      'synthetic-demo', s, instant(patient,t) if np.isfinite(raw) else None,
                                      'absolute_utc', instant(patient,av) if np.isfinite(av) else None,
                                      'simulated_receipt', sv, metadata={'synthetic': True, 'sensor_source': SOURCES[j],
                                                                      'scheduled_opportunity_time': instant(patient,t)})
                    result.append(record(obs))
            break
    result.sort(key=lambda x: x['available_time'] or x['metadata']['scheduled_opportunity_time'])
    return result


def build():
    seal = verify_reference()
    data = {k: np.load(REP / 'replication_features' / (k+'.npy'), mmap_mode='r') for k in FIELDS}
    with np.load(REP / 'replication_predictions.npz', allow_pickle=False) as arr:
        risk = arr['raw_risk'].copy()  # Original Phase 6 model AND original calibration.
    sequences = select_replays(data, risk)
    labels = np.load(REP / 'replication_features/labels.npy', mmap_mode='r')
    if labels[sequences[1]].any() or labels[sequences[2]].any():
        raise ValueError('Non-event demo annotation mismatch')
    chosen = np.concatenate(sequences)
    small = {k: v[chosen] for k,v in data.items()}
    engines = joblib.load(ROOT / 'shadow_engines.joblib')
    forecast = engines['shadow'].predict(small)
    divergence = engines['divergence'].transform(small, forecast)
    quality = quality_uncertainty(small)
    thresholds = json.loads((ROOT / 'models/personalized_fusion/thresholds.json').read_text())['thresholds']
    r = risk[chosen]
    hazards = np.c_[r[:,0], (r[:,1]-r[:,0])/np.maximum(1-r[:,0], 1e-8),
                    (r[:,2]-r[:,1])/np.maximum(1-r[:,1], 1e-8)]
    patients, ehr = read_rows('patients'), read_rows('ehr')
    stories = [('anaya', 'Anaya Rao', 'Personal drift', 'Observed personal deviations and increasing estimated risk.'),
               ('vikram', 'Vikram Sen', 'Transient disturbance', 'Signals return toward baseline; recovery ambiguity and false alerts remain.'),
               ('meera', 'Meera Shah', 'Stable comparison', 'Low estimated risk with ordinary personal variability.')]
    output = []
    for k, (alias, name, story, note) in enumerate(stories):
        private = str(small['patient_ids'][k*8])
        person = {'patient_id': 'demo:'+alias, 'start_time': patients[private]['observed_time'],
                  'context': ehr[private]['context']}
        fhir = bundle(person, None)
        c = person['context']
        patient = {'id': person['patient_id'], 'name': name, 'age': int(round(c['age'])),
                   'synthetic': True, 'label': 'SYNTHETIC DEMO PATIENT', 'story': story,
                   'context': 'Heart failure · home monitoring', 'note': note}
        context = {'demographics': {'age': patient['age'], 'gender': 'female' if c['female'] else 'male'},
                   'diagnoses': [x for x in ('hf','hypertension','diabetes','ckd') if x=='hf' or c[x]],
                   'medications': [x for x in ('beta_blocker','ace_arb','diuretic','mra','sglt2') if c[x]],
                   'labs': [{'name': x, 'value': c[x], 'unit': u, 'observed_time': instant(person,-504),
                             'available_time': instant(person,ehr[private]['available_hour'])}
                            for x,u in [('creatinine','mg/dL'),('sodium','mmol/L'),('hemoglobin','g/dL'),('bnp','pg/mL')] if c[x] is not None],
                   'fhir': fhir, 'scope': 'Simulated historical context; EHR incremental predictive benefit unproved.'}
        frames = []
        for j in range(k*8, (k+1)*8):
            timestamp = instant(person,small['landmark_hours'][j])
            original = twin_state(j, small, divergence, hazards, np.zeros_like(r), quality, thresholds, timestamp)
            bands = {}
            for h in (24,48,72):
                p = float(r[j,[24,48,72].index(h)]); cut = thresholds[str(h)]
                bands[str(h)] = 'Elevated' if p >= cut['balanced'] else 'Moderate' if p >= cut['high_sensitivity'] else 'Low'
            state = {'patient': patient, 'timestamp': timestamp, 'replay_day': j-k*8-7,
                     'risk': {**{f'next_{h}h': float(r[j,i]) for i,h in enumerate((24,48,72))},
                              'likely_window': 'Horizon estimates; no exact event-time prediction', 'bands': bands,
                              'source': 'Frozen Phase 6 personalized_fusion + original calibration'},
                     'divergence': {'descriptive_tdi': original['divergence']['tdi'],
                                    'status': 'Aligned' if original['divergence']['tdi'] < .35 else 'Diverging' if original['divergence']['tdi'] < .65 else 'Marked divergence',
                                    'components': original['divergence']['components']},
                     'status': 'Needs attention' if bands['72']=='Elevated' else 'Watch' if bands['72']=='Moderate' else 'Stable',
                     'actual_state': original['observed_state'], 'shadow_state': original['expected_shadow_state'],
                     'personal_baseline': original['personalization']['baseline_summary'],
                     'why_now': {'personal_deviations': original['why_now']['personal_deviations'],
                                 'temporal_changes': original['why_now']['temporal_changes'],
                                 'missing_data': original['why_now']['missing_data'],
                                 'interpretation': original['why_now']['interpretation']},
                     'data_quality': {'level': 'Limited' if quality[j]>.5 else 'Moderate' if quality[j]>.25 else 'Good',
                                      'availability_score': float(quality[j]), 'missing': original['why_now']['missing_data'],
                                      'scope': 'Availability and baseline-history heuristic, not calibrated certainty'},
                     'signal_units': original['signal_units'], 'state_aggregation': original['state_aggregation'],
                     'ehr_context': {key:value for key,value in context.items() if key!='fhir'},
                     'evidence': {'risk': 'SYNTHETIC POC VALIDATED', 'divergence': 'EXPLORATORY',
                                  'clinical': 'NOT CLINICALLY VALIDATED', 'range': 'ESTIMATED'},
                     'population_context': {'definition': 'Illustrative observed core bands: HR 50–100 bpm, SpO2 94–100%, SBP 90–140 and DBP 55–90 mmHg. Not universal clinical normality.',
                                            'broadly_in_core_bands': all(original['observed_state'][s] is not None and lo<=original['observed_state'][s]<=hi for s,lo,hi in [('heart_rate',50,100),('spo2',94,100),('systolic_bp',90,140),('diastolic_bp',55,90)])}}
            frames.append(state)
        obs = observations(private, person, small['landmark_hours'][k*8], small['landmark_hours'][k*8+7])
        output.append({'patient': patient, 'timeline': frames, 'ehr': context, 'observations': obs})
    metadata = {'version': 'phase7-v1', 'reference': 'Phase 6 personalized_fusion',
                'model_code_sha256': seal['model_code_sha256'], 'config_sha256': seal['config_sha256'],
                'model_sha256': seal['artifacts']['models/personalized_fusion/selected.joblib'],
                'calibration_sha256': seal['artifacts']['models/personalized_fusion/calibration.joblib'],
                'prediction_cache_sha256': sha(REP/'replication_predictions.npz'),
                'sources': SOURCES_EVIDENCE, 'limitations': LIMITATIONS,
                'cohort_separation': 'Datasets were used as independent evidence modules and cohort-level grounding sources.',
                'adaptive_sensing': 'Omitted; decision benefit not validated in Phase 6.5.',
                'selection': 'Post-evaluation curated stories from the independent replication; not representative performance evidence.',
                'display_bands': 'Frozen validation high-sensitivity/balanced thresholds; research display labels, not clinical urgency.'}
    OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    OUTPUT.chmod(0o700)
    path = OUTPUT/'demo.json'
    path.write_text(json.dumps({'metadata':metadata,'patients':output},allow_nan=False,separators=(',',':')))
    path.chmod(0o600)
    audit = {'demo_patients':3,'frames':24,'observations':sum(len(x['observations']) for x in output),
             'frozen_reference_verified':True, 'risk_source':'original Phase 6 raw_risk',
             'api_cache_sha256':sha(path), 'selection':metadata['selection'],
             'stories': [{'story':x['patient']['story'], 'risk72':[round(t['risk']['next_72h'],4) for t in x['timeline']],
                          'tdi':[round(t['divergence']['descriptive_tdi'],3) for t in x['timeline']]} for x in output]}
    (OUTPUT/'build_audit.json').write_text(json.dumps(audit,indent=2))
    (OUTPUT/'build_audit.json').chmod(0o600)
    print(json.dumps({k:v for k,v in audit.items() if k!='stories'}))

if __name__ == '__main__':
    build()
