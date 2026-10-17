"""Select development identities before decoding clinical cells or CGM arrays."""
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic_d2.data import CGM_NAMES, PERSONAL_NAMES
from src.metabolic_d2.protocol import digest, write_json
from .protocol import ROOT, DevelopmentScope

SOURCE = Path('Datasets/diabetes_datasets.zip')
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
NUMERIC = ['age_years', 'diabetes_duration_years', 'smoking_pack_years']
CATEGORICAL = ['sex', 'alcohol_history', 'medication_context', 'other_medication_context']
SAFE = NUMERIC + CATEGORICAL
FIELD_MAP = {'Age (years)': 'age_years', 'Gender (Female=1, Male=2)': 'sex',
             'Duration of diabetes (years)': 'diabetes_duration_years',
             'Smoking History (pack year)': 'smoking_pack_years',
             'Alcohol Drinking History (drinker/non-drinker)': 'alcohol_history',
             'Hypoglycemic Agents': 'medication_context', 'Other Agents': 'other_medication_context'}
MISSING = {'', '/', '-', '--', 'nan', 'none', 'na', 'n/a'}


def recording_pid(recording):
    return 'ShanghaiT2D:' + str(recording).split('_')[0]


def clinical_rows(content, scope, decoded=None):
    """Single-sheet source XLSX: IDs decoded first; denied clinical cells skipped.

    The shared workbook necessarily contains the whole cohort. Its physical XML
    is scanned, but feature values are decoded only after the ID allowlist gate.
    No pandas whole-cohort frame, D1 static cache, or holdout clinical table exists.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as book:
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            strings = list(ET.fromstring(book.read('xl/sharedStrings.xml')))
        def value(cell):
            if cell is None: return None
            if cell.attrib.get('t') == 'inlineStr':
                return ''.join(cell.find('s:is', NS).itertext())
            v = cell.find('s:v', NS)
            if v is None: return None
            return ''.join(strings[int(v.text)].itertext()) if cell.attrib.get('t') == 's' else v.text
        header = None
        with book.open('xl/worksheets/sheet1.xml') as stream:
            for _, row in ET.iterparse(stream, events=['end']):
                if row.tag != '{'+NS['s']+'}row': continue
                cells = {re.sub(r'\d+', '', c.attrib['r']): c for c in row}
                if header is None:
                    header = {col: value(cell) for col, cell in cells.items()}
                    if header.get('A') != 'Patient Number': raise ValueError('Unexpected static identity column')
                    yield None, list(header.values())
                else:
                    recording = value(cells.get('A'))
                    if recording:
                        pid = recording_pid(recording)
                        if pid in scope.allowed:
                            scope.authorize(pid)
                            if decoded is not None: decoded.append(pid)
                            yield recording, {name: value(cells.get(col)) for col, name in header.items() if col != 'A'}
                row.clear()


def available(value):
    return value is not None and str(value).strip().lower() not in MISSING


def medication_category(value):
    # Conservative source-text categories; do not infer doses or oral drug classes.
    if not available(value): return None
    text = str(value).lower()
    if re.search(r'insulin|lispro|aspart|glargine|detemir|degludec|novolin|humalog|lantus|胰岛素', text):
        return 'explicit_insulin_term'
    return 'recorded_without_explicit_insulin_term'


def numeric(value):
    if not available(value): return np.nan
    try: return float(value)
    except (ValueError, TypeError): return np.nan


def profile_table(rows, data, scope):
    """One profile from the earliest recording; never merge later-visit updates."""
    first = {}
    for rec, cutoff in zip(data['recording'], data['sequence_time'][:, 0]):
        first[str(rec)] = min(first.get(str(rec), int(cutoff)), int(cutoff))
    by_pid = {}
    for recording, raw in rows:
        if recording not in first: raise ValueError('Clinical recording has no matching development CGM')
        pid = recording_pid(recording); scope.authorize(pid)
        if pid not in by_pid or first[recording] < by_pid[pid]['available_at_assumed_ns']:
            profile = {'participant_id': pid, 'source_recording': recording,
                       'available_at_assumed_ns': first[recording], 'observation_time': None,
                       'availability_time': None, 'availability_basis': 'Historical onboarding assumption; actual receipt absent',
                       'source': 'ShanghaiT2DM recording-linked clinical summary', 'original_fields': raw}
            for original, name in FIELD_MAP.items():
                v = raw.get(original)
                if name in NUMERIC:
                    number = numeric(v)
                    # Quarantine invalid values; preserve the original source field.
                    good = np.isfinite(number) and (18 <= number <= 120 if name == 'age_years' else number >= 0)
                    profile[name] = number if good else None
                elif name == 'sex':
                    profile[name] = {1: 'female', 2: 'male'}.get(numeric(v))
                elif 'medication' in name:
                    profile[name] = medication_category(v)
                else: profile[name] = str(v).strip().lower() if available(v) else None
            by_pid[pid] = profile
    scope.assert_allowed(by_pid)
    if set(by_pid) != scope.allowed: raise ValueError('Incomplete same-participant static linkage')
    # Persist raw fields/provenance only privately, separate from model input table.
    write_json(ROOT/'static_provenance.json', list(by_pid.values()))
    frame = pd.DataFrame([{k: p[k] for k in ['participant_id', 'source_recording', 'available_at_assumed_ns']+SAFE}
                          for p in by_pid.values()]).sort_values('participant_id').reset_index(drop=True)
    frame.to_parquet(ROOT/'participant_static.parquet', index=False)
    (ROOT/'participant_static.parquet').chmod(0o600)
    return frame


def load_dynamic(scope):
    """Reuse exact sealed D2 development and calibration windows, never its loader."""
    chunks = []
    keys = ['pid', 'recording', 'time', 'sequence', 'sequence_time', 'x', 'personal', 'warm', 'y', 'target_time', 'spike']
    for stage, expected in [('development', 'model_development'), ('calibration', 'calibration')]:
        path = Path('data/metabolic_d2/cache')/(stage+'.npz')
        provenance = json.loads(path.with_suffix('.json').read_text())
        if provenance['cache_sha256'] != digest(path): raise RuntimeError('D2 development cache integrity mismatch')
        with np.load(path, allow_pickle=False) as cache:
            ids = cache['pid']
            scope.assert_allowed(ids)  # Gate before decoding glucose, targets or other arrays.
            if set(ids) != set(scope.manifest[expected]): raise PermissionError('Unexpected cached cohort')
            chunks.append({k: cache[k] for k in keys})
    data = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    validate_temporal(data)
    return data


def validate_temporal(data):
    if not (data['sequence_time'] <= data['time'][:, None]).all(): raise ValueError('Future glucose input')
    offsets = (data['target_time']-data['time'][:, None])/60_000_000_000
    if not (np.abs(offsets-np.array([30,60,90,120])) <= 7.5).all(): raise ValueError('D2 target tolerance changed')
    if not (np.diff(data['sequence_time'], axis=1) > 0).all(): raise ValueError('Invalid native history')


def dynamic_features(data):
    personal = data['personal'].copy()
    personal[~data['warm'], :] = np.nan
    personal[~data['warm'], -2:] = 0
    return np.column_stack([data['x'], personal]).astype('float32')


def load_inputs(scope=None):
    scope = scope or DevelopmentScope()
    data = load_dynamic(scope)
    with zipfile.ZipFile(SOURCE) as archive:
        name = next(n for n in archive.namelist() if '/' not in n and n.endswith('.xlsx') and 'T2DM' in n)
        stream = clinical_rows(archive.read(name), scope)
        _, headers = next(stream)
        rows = list(stream)
    profiles = profile_table(rows, data, scope)
    mapping = profiles.set_index('participant_id')
    for pid, cut in zip(data['pid'], data['time']):
        if mapping.loc[pid, 'available_at_assumed_ns'] > cut: raise ValueError('Static future-visit leakage')
    missing = {h: sum(not available(row.get(h)) for _, row in rows) for h in headers if h != 'Patient Number'}
    audit = {'development_participants': len(profiles), 'clinical_recordings': len(rows), 'windows': len(data['y']),
             'header_fields': headers, 'recording_missingness': missing,
             'profile_missingness': {k: int(profiles[k].isna().sum()) for k in SAFE},
             'profile_category_counts': {k: int(profiles[k].nunique(dropna=True)) for k in CATEGORICAL},
             'fields_used': SAFE, 'dynamic_features': CGM_NAMES+PERSONAL_NAMES,
             'source_sha256': digest(SOURCE), 'holdout_feature_rows_decoded': 0}
    write_json(ROOT/'input_audit.json', audit)
    return data, profiles, audit
