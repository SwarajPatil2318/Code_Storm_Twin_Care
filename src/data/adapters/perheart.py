"""Irregular home measurements with deterministic duplicate lineage."""
import zipfile
from zoneinfo import ZoneInfo
import datetime as dt

from .base import Adapter, SOURCE_SCHEMAS
from ..io import read_table
from ..quality import SchemaError, classify, normalize_id, parse_number
from ..schema import Observation, Outcome

MEASUREMENTS = {
    'medical/blood_pressure.csv': {'bp_sys':('systolic_bp','mmHg'), 'bp_dia':('diastolic_bp','mmHg'), 'hr':('heart_rate','bpm')},
    'medical/body_mass.csv': {'value':('weight','kg')},
    'medical/glucose.csv': {'value':('glucose','mmol/L')},
    'medical/oxidation.csv': {'hr':('heart_rate','bpm'), 'perf':('perfusion_index','1'), 'sat':('spo2','%')},
    'medical/temperature.csv': {'value':('temperature','degC')},
}


class PerheartAdapter(Adapter):
    def __init__(self, root, writer):
        super().__init__(root, writer, 'perheart')

    def build(self):
        self.register_input('personal_questionnaires.csv')
        self.register_input('medical.zip')
        self.register_input('perheart_dataset_descriptor.pdf')
        rows = read_table((self.root/'personal_questionnaires.csv').read_text(encoding='utf-8-sig'), SOURCE_SCHEMAS['perheart_questionnaires'])
        for lineno, row in rows:
            patient = self.add_patient(row['user_id'], ['home_medical_measurements','limited_demographics'])
            self.preserve('personal_questionnaires.csv', lineno, row, 'questionnaire')
            attributes = {}
            for field, raw in row.items():
                if field == 'user_id':
                    continue
                prov = self.provenance('personal_questionnaires.csv', lineno, field, field,
                                       'preserved_questionnaire_value')
                if field.startswith('recent_hf_'):
                    value = classify(raw, field, prov, numeric=False, blocked_reason='unadjudicated_questionnaire_not_event_label')
                    self.outcome(Outcome(patient, self.dataset, field, value, metadata={'not_supervised_event_label':True}))
                else:
                    attributes[field] = classify(raw, field, prov, numeric=False,
                                                 blocked_reason=None if field in {'age','sex','start_date'} else 'mixed_pre_post_pilot_capture_unknown')
            self.context(patient, attributes)
        retained, duplicate_count = 0, 0
        with zipfile.ZipFile(self.root/'medical.zip') as archive:
            csv_members = {name for name in archive.namelist() if name.endswith('.csv')}
            if csv_members != set(MEASUREMENTS):
                raise SchemaError('Unexpected PerHeart medical archive schema')
            for member in sorted(MEASUREMENTS):
                rows = read_table(archive.read(member).decode('utf-8-sig'), SOURCE_SCHEMAS['perheart_medical'][member])
                seen = {}
                timestamp_keys = {}
                for _, candidate in rows:
                    pkey = (normalize_id(candidate['user_id'], self.dataset), parse_number(candidate['ts']))
                    values = tuple((k,candidate[k]) for k in sorted(candidate) if k not in {'user_id','ts'})
                    timestamp_keys.setdefault(pkey, set()).add(values)
                for lineno, row in rows:
                    patient = normalize_id(row['user_id'], self.dataset)
                    if patient not in self.patients:
                        raise SchemaError('PerHeart measurement references an unknown patient')
                    self.preserve(member, lineno, row, 'medical_measurement')
                    epoch = parse_number(row['ts'])
                    try:
                        instant = dt.datetime.fromtimestamp(epoch, dt.timezone.utc) if epoch is not None else None
                    except (OverflowError, OSError, ValueError):
                        instant = None
                    timestamp = instant.isoformat() if instant else None
                    # Identity/timestamp lexical normalization is documented; measurements remain literal.
                    key = (patient, epoch, tuple((k,row[k]) for k in sorted(row) if k not in {'user_id','ts'}))
                    duplicate = key in seen
                    original_line = seen.get(key)
                    seen.setdefault(key, lineno)
                    conflict = len(timestamp_keys[(patient, epoch)]) > 1
                    if duplicate:
                        duplicate_count += 1
                        self.writer.emit('duplicates', {'patient_id':patient, 'dataset':self.dataset,
                                                        'source_file':member, 'duplicate_row':lineno,
                                                        'retained_row':original_line, 'policy':'first_source_row', 'raw_record':row})
                    else:
                        retained += 1
                    sys, dia = parse_number(row.get('bp_sys')), parse_number(row.get('bp_dia'))
                    bp_conflict = sys is not None and dia is not None and sys <= dia
                    for field, (concept, unit) in MEASUREMENTS[member].items():
                        reasons = []
                        if timestamp is None: reasons.append('missing_or_invalid_timestamp')
                        if conflict: reasons.append('conflicting_same_timestamp_measurement')
                        if bp_conflict and concept in {'systolic_bp','diastolic_bp'}: reasons.append('systolic_not_above_diastolic')
                        prov = self.provenance(member, lineno, field, concept, 'numeric_parse_no_unit_conversion',
                                               unit=unit, support='supplied_descriptor', archive='medical.zip')
                        value = classify(row[field], concept, prov, unit, duplicate=duplicate, extra_reasons=reasons)
                        self.observation(Observation(f'perheart:{member}:{lineno}:{field}',patient,self.dataset,concept,
                                                     timestamp,'absolute_utc',None,'unknown_source_availability',value,
                                                     metadata={'source_timezone':'Europe/Warsaw',
                                                               'local_date':instant.astimezone(ZoneInfo('Europe/Warsaw')).date().isoformat() if instant else None,
                                                               'device_context':{k:v for k,v in row.items() if k not in {'user_id','ts',*MEASUREMENTS[member]}},
                                                               'duplicate_of_source_row':original_line,
                                                               'retrospective_feature_policy':'assume_reading_available_at_measurement_time'}))
                self.stats[member + '_source_rows'] = len(rows)
        self.stats.update(questionnaire_rows_retained=len(self.patients),
                          medical_rows_retained=retained, duplicate_medical_rows=duplicate_count)
        self.notes += ['Exactly identical repeated measurements use first source-row retention, with removed-row lineage.',
                       'Duplicate observations remain in the canonical table with no usable value, but are absent from timelines/features.',
                       'Wrist files absent; no waveform/activity/sleep values generated.',
                       'No confirmed supervised deterioration label; retrospective capture-time availability assumption is explicit.']
