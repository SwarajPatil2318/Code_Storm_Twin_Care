"""Shared adapter bookkeeping and quality reporting, without patient logging."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from ..io import dumps, file_hash
from ..quality import SchemaError, normalize_id
from ..schema import SCHEMA_VERSION, PatientClinicalContext, Provenance, record

SOURCE_SCHEMAS = json.loads((Path(__file__).parents[1] / 'source_schemas.json').read_text())


class Adapter:
    def __init__(self, root, writer, dataset):
        self.root, self.writer, self.dataset = Path(root), writer, dataset
        self.patients = set()
        self.observations = []
        self.input_hashes = {}
        self.stats = Counter()
        self.reasons = Counter()
        self.states = Counter()
        self.field_states = defaultdict(Counter)
        self.available_features = set()
        self.coverage = []
        self.observation_keys = {}
        self.notes = []
        self.extra_report = {}

    def register_input(self, filename):
        digest = file_hash(self.root / filename)
        self.input_hashes[filename] = digest
        return digest

    def provenance(self, file, row, original, transformed, transformation='identity',
                   unit=None, support='unknown', archive=None):
        key = archive or file
        return Provenance(self.dataset, file, str(row), original, transformed,
                          transformation, self.input_hashes[key], unit, support)

    def add_patient(self, raw_id, capabilities, metadata=None):
        patient_id = normalize_id(raw_id, self.dataset)
        if patient_id in self.patients:
            raise SchemaError('Repeated patient identity in patient table')
        self.patients.add(patient_id)
        self.writer.emit('patients', {'schema_version': SCHEMA_VERSION, 'patient_id': patient_id,
                                     'dataset': self.dataset, 'synthetic': self.dataset == 'synthea',
                                     'capabilities': capabilities, 'metadata': metadata or {}})
        return patient_id

    def preserve(self, file, row, raw, row_type):
        self.writer.emit('source_records', {'dataset': self.dataset, 'source_file': file,
                                          'source_record': str(row), 'record_type': row_type,
                                          'raw_record': raw})
        self.stats['source_records'] += 1

    def check_value(self, patient, location, value, kind):
        self.states[value.quality_status] += 1
        field = value.provenance.original_field
        self.field_states[field][value.quality_status] += 1
        self.reasons.update(value.reasons)
        if value.quality_status != 'VALID' or value.reasons:
            self.writer.emit('quality', {'patient_id': patient, 'dataset': self.dataset,
                                        'record_id': location, 'record_kind': kind,
                                        'original_field': field, 'quality_status': value.quality_status,
                                        'reasons': value.reasons,
                                        'usable': value.usable_value is not None})

    def observation(self, observation):
        if observation.patient_id not in self.patients:
            raise SchemaError('Observation patient reference is inconsistent')
        signature = (observation.patient_id, observation.concept, observation.observed_time,
                     observation.value.unit, dumps(observation.value.raw_value))
        if observation.value.quality_status != 'DUPLICATE' and observation.observed_time is not None:
            if signature in self.observation_keys:
                self.issue('potential_duplicate_observation_retained', status='SUSPECT',
                           patient_id=observation.patient_id, record_id=observation.observation_id,
                           potential_duplicate_of=self.observation_keys[signature])
            else:
                self.observation_keys[signature] = observation.observation_id
        self.observations.append(observation)
        self.writer.emit('observations', observation)
        self.check_value(observation.patient_id, observation.observation_id, observation.value, 'observation')
        if observation.value.usable_value is not None:
            self.available_features.add(observation.concept)
        if observation.value.usable_value is None and observation.value.quality_status not in {'MISSING', 'DUPLICATE'}:
            self.writer.emit('quarantine', {'record_kind': 'observation', **record(observation)})
        if observation.observed_time is not None:
            self.coverage.append(observation.observed_time)

    def context(self, patient, attributes, context_type='aggregate_snapshot', observed=None, available=None):
        item = PatientClinicalContext(patient, self.dataset, attributes, observed, available, context_type)
        self.writer.emit('clinical_context', item)
        for field, value in attributes.items():
            self.check_value(patient, f'{context_type}:{field}', value, 'clinical_context')

    def outcome(self, outcome):
        self.writer.emit('outcomes', outcome)
        self.check_value(outcome.patient_id, outcome.endpoint, outcome.value, 'outcome')

    def issue(self, reason, status='INVALID', **details):
        self.reasons[reason] += 1
        self.writer.emit('quality', {'dataset': self.dataset, 'quality_status': status,
                                    'reasons': [reason], **details})

    def report(self):
        observations = Counter(o.value.quality_status for o in self.observations)
        concepts = defaultdict(Counter)
        for o in self.observations:
            concepts[o.concept][o.value.quality_status] += 1
        observed_patients = defaultdict(set)
        usable_patients = defaultdict(set)
        unusable_counts = Counter()
        for o in self.observations:
            if o.value.quality_status != 'MISSING': observed_patients[o.concept].add(o.patient_id)
            if o.value.usable_value is not None: usable_patients[o.concept].add(o.patient_id)
            else: unusable_counts[o.concept] += 1
        missingness = {'source_missing_assertions':self.states.get('MISSING',0),
                       'missing_observations':observations.get('MISSING',0),
                       'by_concept':{concept:{'source_missing_fraction':counts.get('MISSING',0)/sum(counts.values()),
                                             'unusable_fraction':unusable_counts[concept]/sum(counts.values()),
                                             'patients_without_modality':len(self.patients)-len(observed_patients[concept]),
                                             'patients_without_usable_modality':len(self.patients)-len(usable_patients[concept])}
                                     for concept,counts in sorted(concepts.items())},
                       'scheduled_time_missingness':'unknown_no_schedule_supplied'}
        return {'missingness':missingness,'dataset': self.dataset, 'patients_retained': len(self.patients),
                'source_counts': dict(sorted(self.stats.items())),
                'canonical_observations': len(self.observations),
                'usable_observations': sum(o.value.usable_value is not None for o in self.observations),
                'quarantined_observations': sum(o.value.usable_value is None and o.value.quality_status not in {'MISSING','DUPLICATE'} for o in self.observations),
                'observation_quality': dict(sorted(observations.items())),
                'quality_states_all_assertions': dict(sorted(self.states.items())),
                'quality_reasons': dict(sorted(self.reasons.items())),
                'field_quality': {k: dict(sorted(v.items())) for k, v in sorted(self.field_states.items())},
                'concept_quality': {k: dict(sorted(v.items())) for k, v in sorted(concepts.items())},
                'available_features': sorted(self.available_features),
                'temporal_coverage': [min(self.coverage), max(self.coverage)] if self.coverage else None,
                'notes': self.notes, **self.extra_report}
