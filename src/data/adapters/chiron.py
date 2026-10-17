"""Ordered daily summaries; future-normalized source features are quarantined."""
from collections import Counter
import math

from .base import Adapter, SOURCE_SCHEMAS
from ..io import read_table
from ..quality import SchemaError, classify, normalize_id, parse_number
from ..schema import Observation, Outcome

FILENAME = 'pone.0190323.s017.csv'
# Units are explicitly provisional conventions; no cross-source pooling is enabled.
MEASUREMENTS = {
    'HR_avg': ('heart_rate','bpm'), 'HR_SD': ('heart_rate_sd','bpm'),
    'SystolicBP': ('systolic_bp','mmHg'), 'DiastolicBP': ('diastolic_bp','mmHg'),
    'Weight': ('weight','kg'), 'SpO2': ('spo2','%'),
    'TempSkin_avg': ('skin_temperature','degC'), 'TempAmbient': ('ambient_temperature','degC'),
}


def blocked_feature_reason(field):
    if '_pers' in field:
        return 'future_leakage_full_study_personalization'
    if 'Chg' in field:
        return 'supplied_change_requires_causal_reconstruction'
    return 'unapproved_derived_feature_or_unresolved_unit'


class ChironAdapter(Adapter):
    def __init__(self, root, writer):
        super().__init__(root, writer, 'chiron')

    def build(self):
        self.register_input(FILENAME)
        rows = read_table((self.root/FILENAME).read_text(encoding='utf-8-sig'), SOURCE_SCHEMAS['chiron'], ';')
        ids = sorted({normalize_id(row['Patient_ID'], self.dataset) for _, row in rows})
        for patient in ids:
            self.add_patient(patient.split(':',1)[1], ['relative_daily_summaries','subjective_feel_labels'])
        seen = set()
        labels = Counter()
        self.feel_labels = []
        for lineno, row in rows:
            patient = normalize_id(row['Patient_ID'], self.dataset)
            day = parse_number(row['Day'])
            if day is not None and not day.is_integer():
                raise SchemaError('CHIRON Day must be an integer source-relative day')
            self.preserve(FILENAME, lineno, row, 'daily_summary')
            duplicate = (patient, day) in seen
            if duplicate:
                raise SchemaError('Repeated CHIRON patient/day requires explicit conflict adjudication')
            seen.add((patient, day))
            if day is None:
                self.issue('missing_timestamp', patient_id=patient, source_record=str(lineno))
            sys, dia = parse_number(row['SystolicBP']), parse_number(row['DiastolicBP'])
            bp_conflict = sys is not None and dia is not None and sys <= dia
            for field, (concept, unit) in MEASUREMENTS.items():
                prov = self.provenance(FILENAME, lineno, field, concept,
                                       'decimal_comma_parse; provisional_unit_no_conversion', support='provisional_source_convention')
                reasons = ['missing_timestamp'] if day is None else []
                if bp_conflict and concept in {'systolic_bp','diastolic_bp'}:
                    reasons.append('systolic_not_above_diastolic')
                value = classify(row[field], concept, prov, unit, decimal_comma=True,
                                 provisional_unit=True, extra_reasons=reasons)
                obs = Observation(f'chiron:{lineno}:{field}', patient, self.dataset, concept,
                                  day, 'relative_source_day', None, 'unknown_source_availability', value,
                                  metadata={'time_precision':'source_day_summary',
                                            'retrospective_feature_policy':'assume_daily_summary_available_at_end_of_recorded_day'})
                self.observation(obs)
            features = {}
            for field, raw in row.items():
                if field in MEASUREMENTS or field in {'Patient_ID','Day','Feel'} or field.startswith('_unnamed_'):
                    continue
                value = classify(raw, field, self.provenance(FILENAME, lineno, field, field, 'preserved_not_used'),
                                 decimal_comma=True, blocked_reason=blocked_feature_reason(field))
                features[field] = value
                self.check_value(patient, f'{lineno}:{field}', value, 'unapproved_source_feature')
            self.writer.emit('source_features', {'patient_id':patient, 'dataset':self.dataset,
                                                 'observed_time':day, 'features':{k: v.__dict__ | {'provenance':v.provenance.__dict__} for k,v in features.items()}})
            raw = row['Feel']
            prov = self.provenance(FILENAME, lineno, 'Feel', 'subjective_feel', 'ordinal_source_label')
            value = classify(raw, 'subjective_feel', prov, 'ordinal', blocked_reason=None if raw in {'1','2','3','4','5','?'} else 'invalid_feel_label')
            self.outcome(Outcome(patient, self.dataset, 'subjective_feel', value, day, 'relative_source_day',
                                 time_origin='comparison_to_previous_day', metadata={'not_hf_decompensation':True}))
            labels[raw] += 1
            self.feel_labels.append((patient,raw))
        self.stats['daily_rows_retained'] = len(rows)
        self.extra_report['class_distribution'] = dict(sorted(labels.items()))
        self.notes += ['All 25 observed IDs retained; publication discrepancy unresolved.',
                       'Eight raw summary channels use provisional source units; not certified for cross-cohort pooling.',
                       'Full-study personalized, supplied-change and unapproved derived features have no usable value.',
                       'Temporal features use strictly prior days under an explicit retrospective end-of-day availability assumption.']
