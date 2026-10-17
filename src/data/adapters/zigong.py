"""Conservative, separately labelled retrospective readmission cohort."""
from collections import Counter
import io
import zipfile

from .base import Adapter, SOURCE_SCHEMAS
from ..audit import ZIGONG
from ..io import read_table
from ..quality import SchemaError, classify, missing, normalize_id, parse_number
from ..schema import Outcome

TARGET = 're.admission.within.28.days'
APPROVED_FEATURES = ('gender', 'ageCat', 'admission.ward', 'admission.way',
                     'visit.times', 'congestive.heart.failure')
SUPPORTED_REASONS = {
    'gender': 'Recorded static demographic category; no dynamic capture time asserted.',
    'ageCat': 'Recorded age interval at index hospitalization; no exact age fabricated.',
    'admission.ward': 'Dictionary explicitly defines first admission ward.',
    'admission.way': 'Dictionary explicitly defines admission pathway.',
    'visit.times': 'Dictionary explicitly defines admissions before this hospitalization.',
    'congestive.heart.failure': 'Dictionary explicitly distinguishes prior HF history on admission from first index presentation.',
}
DISCHARGE = {'DestinationDischarge', 'discharge.department', 'dischargeDay'}
VITALS = {'body.temperature':('temperature','degC'), 'pulse':('heart_rate','bpm'),
          'respiration':('respiration','/min'), 'systolic.blood.pressure':('systolic_bp','mmHg'),
          'diastolic.blood.pressure':('diastolic_bp','mmHg'), 'weight':('weight','kg'),
          'height':('height','m'), 'BMI':('bmi','kg/m2'), 'LVEF':('lvef','%'),
          'oxygen.saturation':('spo2','%')}
EVENT_TIMES = {'time.of.death..days.from.admission.', 're.admission.time..days.from.admission.',
               'time.to.emergency.department.within.6.months'}

ADMISSION_EVENT_TIMES = EVENT_TIMES - {'time.to.emergency.department.within.6.months'}


def is_outcome(field):
    return (field.startswith(('death.within.', 're.admission.within.')) or field in EVENT_TIMES
            or field in {'outcome.during.hospitalization', 'return.to.emergency.department.within.6.months'})


INDEX_SEMANTIC_ANCHORS = {
    'gender': ('gender',), 'ageCat': ('age', 'categor'),
    'admission.ward': ('first', 'admission'), 'admission.way': ('admission',),
    'visit.times': ('before this hospitalization',),
    'congestive.heart.failure': ('past history', 'on admission'),
}


def verify_index_semantics(dictionary):
    for field, anchors in INDEX_SEMANTIC_ANCHORS.items():
        definition = dictionary.get(field, '').lower()
        if not all(anchor in definition for anchor in anchors):
            raise SchemaError('Source dictionary no longer supports the approved index-time contract')


def availability_contract(fields):
    result = []
    for field in fields:
        approved = field in APPROVED_FEATURES
        if approved:
            reason = SUPPORTED_REASONS[field]
        elif field == 'inpatient.number' or field.startswith('_unnamed_'):
            reason = 'Identifier/export index; never a predictor.'
        elif is_outcome(field):
            reason = 'Outcome or event-time information; label table only.'
        elif field in DISCHARGE:
            reason = 'Known after admission/discharge decision; future information at index.'
        else:
            reason = 'Aggregate hospitalization variable without defensible index-time capture/availability.'
        result.append({'feature': field, 'source': 'zigong/dat.csv',
                       'available_at_index_time': True if approved else False if is_outcome(field) or field in DISCHARGE else None,
                       'reason': reason, 'include_in_baseline': approved})
    result.append({'feature': 'Drug_name', 'source': 'zigong/dat_md.csv',
                   'available_at_index_time': None, 'include_in_baseline': False,
                   'reason': 'Untimed whole-stay medication exposure; dose/capture/administration time unknown.'})
    return result


def validate_feature_matrix(header, contract):
    approved = {item['feature'] for item in contract if item['include_in_baseline']}
    if header != ['patient_id', *APPROVED_FEATURES] or set(header[1:]) != approved:
        raise SchemaError('Predictor matrix violates the explicit feature allowlist')
    if any(is_outcome(field) for field in header):
        raise SchemaError('Outcome leaked into predictor matrix')


class ZigongAdapter(Adapter):
    def __init__(self, root, writer):
        super().__init__(root, writer, 'zigong')
        self.labels = {}

    def build(self):
        self.register_input(ZIGONG)
        with zipfile.ZipFile(self.root / ZIGONG) as archive:
            def member(filename):
                matches = [name for name in archive.namelist() if name.endswith('/' + filename)]
                if len(matches) != 1:
                    raise SchemaError('Expected exactly one Zigong archive member')
                return matches[0]
            source = member('dat.csv')
            rows = read_table(archive.read(source).decode('utf-8-sig'), SOURCE_SCHEMAS['zigong'])
            dictionary_rows = read_table(archive.read(member('dataDictionary.csv')).decode('utf-8-sig'),
                                         ['_unnamed_0', 'Variables', 'Description'])
            dictionary = {row['Variables']: row['Description'] for _, row in dictionary_rows}
            expected=set(SOURCE_SCHEMAS['zigong'])-{'_unnamed_0'}
            if len(dictionary)!=len(dictionary_rows) or set(dictionary)!=expected:
                raise SchemaError('Zigong dictionary variables changed or repeated')
            self.writer.json('source_dictionary',dictionary)
            verify_index_semantics(dictionary)
            contract = availability_contract(SOURCE_SCHEMAS['zigong'])
            predictors, labels = [], []
            for lineno, row in rows:
                patient = self.add_patient(row['inpatient.number'], ['clinical_context', 'recorded_readmission_labels'])
                self.preserve(source, lineno, row, 'index_hospitalization')
                attrs = {}
                for field, raw in row.items():
                    if field.startswith('_unnamed_') or field == 'inpatient.number':
                        continue
                    prov = self.provenance(source, lineno, field, field, archive=ZIGONG)
                    if is_outcome(field):
                        binary = field not in EVENT_TIMES and field != 'outcome.during.hospitalization'
                        extra = ['invalid_binary_label'] if binary and raw not in {'0', '1', 'NA'} else []
                        value = classify(raw, field, prov, unit='day' if field in ADMISSION_EVENT_TIMES else None,
                                         numeric=field in EVENT_TIMES, extra_reasons=extra)
                        horizon = {'value': 28, 'unit': 'day'} if '28.days' in field else {'value': 3 if '3.months' in field else 6, 'unit': 'month'} if 'months' in field else None
                        self.outcome(Outcome(patient, self.dataset, field, value, horizon=horizon,
                                             time_basis='days_from_admission' if field in ADMISSION_EVENT_TIMES else 'unknown',
                                             time_origin='admission' if field in ADMISSION_EVENT_TIMES else 'unconfirmed',
                                             event_time=parse_number(raw) if field in ADMISSION_EVENT_TIMES else None,
                                             metadata={'benchmark_name': 'retrospective 28-day readmission benchmark' if field == TARGET else None,
                                                       'dictionary_description': dictionary.get(field)}))
                        self.issue('leakage_prone_outcome_field', status='SUSPECT', patient_id=patient, original_field=field)
                    else:
                        blocked = None if field in APPROVED_FEATURES else 'not_available_at_index_time' if field in DISCHARGE else 'unknown_index_time_availability'
                        concept, unit = VITALS.get(field, (field, None))
                        if field in VITALS:
                            prov = self.provenance(source, lineno, field, concept, 'numeric_parse_no_unit_conversion', unit, 'supplied_dictionary', ZIGONG)
                        extra = ['invalid_binary_history'] if field=='congestive.heart.failure' and raw not in {'0','1','NA'} else []
                        attrs[field] = classify(raw, concept, prov, unit=unit, numeric=field in VITALS, blocked_reason=blocked, extra_reasons=extra)
                self.context(patient, attrs)
                if row[TARGET] not in {'0', '1'}:
                    raise SchemaError('Primary label must be observed binary; no automatic relabeling')
                self.labels[patient] = int(row[TARGET])
                predictors.append([patient] + [attrs[field].usable_value for field in APPROVED_FEATURES])
                labels.append([patient, int(row[TARGET])])
                self.check_outcomes(patient, row)
            med_source = member('dat_md.csv')
            meds = read_table(archive.read(med_source).decode('utf-8-sig'), ['_unnamed_0','inpatient.number','Drug_name'])
            for lineno, row in meds:
                patient = normalize_id(row['inpatient.number'], self.dataset)
                if patient not in self.patients:
                    raise SchemaError('Medication refers to an unknown patient')
                self.preserve(med_source, lineno, row, 'untimed_medication_exposure')
                value = classify(row['Drug_name'], 'medication', self.provenance(med_source, lineno, 'Drug_name', 'medication', archive=ZIGONG),
                                 numeric=False, blocked_reason='untimed_medication_not_approved_predictor')
                self.writer.emit('medications', {'patient_id': patient, 'dataset': self.dataset, 'value': value.__dict__ | {'provenance': value.provenance.__dict__},
                                                  'observed_time': None, 'available_time': None, 'evidence_type':'untimed_exposure'})
                self.check_value(patient, str(lineno), value, 'medication')
        header = ['patient_id', *APPROVED_FEATURES]
        validate_feature_matrix(header, contract)
        self.writer.csv('predictors', header, sorted(predictors))
        self.writer.csv('labels', ['patient_id', TARGET], sorted(labels))
        self.writer.json('feature_availability', contract)
        self.available_features.update(APPROVED_FEATURES)
        self.stats.update(clinical_rows_retained=len(rows), medication_rows_retained=len(meds))
        self.extra_report['class_distribution'] = dict(sorted(Counter(str(value) for value in self.labels.values()).items()))
        self.notes += ['Endpoint clock remains unconfirmed; no post-discharge claim.',
                       'Six semantically index-supported attributes; no verified capture timestamps.',
                       'All original stays retained, including discharge-status contradictions; no silent cohort exclusions.']

    def check_outcomes(self, patient, row):
        for prefix in ('death', 're.admission'):
            labels = [row[f'{prefix}.within.{h}'] for h in ('28.days','3.months','6.months')]
            if labels[0] > labels[1] or labels[1] > labels[2]:
                self.issue('nonmonotone_outcome_flags', patient_id=patient)
            field = 'time.of.death..days.from.admission.' if prefix == 'death' else 're.admission.time..days.from.admission.'
            event = parse_number(row[field])
            if labels[2] == '1' and event is None:
                self.issue('positive_6m_missing_event_day', patient_id=patient, original_field=field)
            discharge_day = parse_number(row['dischargeDay'])
            if event is not None and discharge_day is not None and event < discharge_day:
                self.issue('event_before_discharge', patient_id=patient, original_field=field)
            for label, horizon in zip(labels, (28,90,180)):
                if event is not None and (label == '1') != (0 <= event <= horizon):
                    self.issue('event_day_flag_disagreement_' + str(horizon), patient_id=patient, original_field=field)
        systolic, diastolic = parse_number(row['systolic.blood.pressure']), parse_number(row['diastolic.blood.pressure'])
        if systolic is not None and diastolic is not None and systolic <= diastolic:
            self.issue('systolic_not_above_diastolic', patient_id=patient)
        if (row['DestinationDischarge'] == 'Died') != (row['outcome.during.hospitalization'] == 'Dead'):
            self.issue('discharge_death_status_conflict', patient_id=patient)
