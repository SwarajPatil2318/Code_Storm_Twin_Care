"""Read existing synthetic source observations, never cached predictions."""
from abc import ABC, abstractmethod
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from .contracts import CanonicalEvent, clock

SCENARIOS = {'rising': 'predicted_spike', 'stable': 'normal_response', 'personal_deviation': 'personally_atypical'}
REPLAY = Path(__file__).resolve().parents[1] / 'data/metabolic_d3/demo.json'


class SourceAdapter(ABC):
    """Provider lifecycle and canonical normalization boundary."""
    @abstractmethod
    def health_check(self): pass
    @abstractmethod
    def sync(self, patient_id, connection, received_at): pass
    @abstractmethod
    def normalize(self, patient_id, raw, received_at): pass
    def connect(self, patient_id, config):
        self.health_check()
        return {'source_case': SCENARIOS[config['scenario']], 'synthetic': True}
    def disconnect(self, patient_id): return {'stopped': True}
    def get_status(self, connection): return connection['status']
    def ingest(self, patient_id, raw, received_at): return self.normalize(patient_id, raw, received_at)


class ReplayAdapter(SourceAdapter):
    source_type = 'CGM'
    provider_type = 'replay_adapter'
    def __init__(self, path=REPLAY): self.path = Path(path)
    def health_check(self):
        if not self.path.is_file(): raise ValueError('Adapter unavailable')
        data = json.loads(self.path.read_text())
        if set(SCENARIOS.values()) - {c['case'] for c in data['cases']}: raise ValueError('Replay cases unavailable')
        return True
    def source(self, connection):
        self.health_check()
        case = next(c for c in json.loads(self.path.read_text())['cases'] if c['case'] == SCENARIOS[connection['metadata']['scenario']])
        return case
    def normalize(self, patient_id, raw, received_at):
        return CanonicalEvent(patient_id=patient_id, source_type=self.source_type, provider_type=self.provider_type,
            event_type='CGM_READING', observed_at=raw['observed_time'], received_at=received_at,
            sequence_id=raw['event_id'], raw_reference='metabolic_d3/demo.json#'+raw['event_id'],
            payload={'glucose_mg_dl': raw['value'], 'timestamp': raw['observed_time']},
            quality={'status': 'MISSING' if raw['value'] is None else 'GOOD', 'reasons': []},
            provenance={'origin':'existing_synthetic_replay', 'synthetic':True, 'source_observed_at':raw['observed_time'],
                        'source_available_at':raw['available_time'], 'original_value':raw['original_value'],
                        'original_unit':raw['original_unit'], 'native_period_minutes':raw['native_period_minutes'],
                        'dataset':'metabolic_d3_synthetic_replay', 'patient_mapping':'explicit_synthetic_import; no cohort identity link'}).view()
    def sync(self, patient_id, connection, received_at):
        case = self.source(connection)
        events = [e for e in case['events'] if e['concept'] == 'glucose']
        # Start with the last native two-hour window and then advance through the existing visible replay.
        start = clock(case['timeline'][0]['timestamp']) - timedelta(hours=2)
        events = [e for e in events if clock(e['observed_time']) >= start]
        offset = connection['metadata'].get('cursor', 0)
        batch = events[offset:offset+9]
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = [self.normalize(patient_id,e,received_at) for e in batch]
        for e in result: e['provenance'].update({'source_case':case['case'], 'source_file_sha256':digest})
        return result, offset+len(batch), offset+len(batch)>=len(events)


class WearableAdapter(ReplayAdapter):
    source_type = 'WEARABLE'
    provider_type = 'simulated_wearable'
    def normalize(self, patient_id, raw, received_at):
        kind = 'HEART_RATE' if raw['concept']=='heart_rate' else 'ACTIVE_MINUTES'
        key = 'bpm' if kind=='HEART_RATE' else 'minutes'
        return CanonicalEvent(patient_id=patient_id, source_type=self.source_type, provider_type=self.provider_type,
            event_type=kind, observed_at=raw['observed_time'], received_at=received_at, sequence_id=raw['event_id'],
            payload={key:raw['value'], **({'level':raw['level'],'window_minutes':90} if kind=='ACTIVE_MINUTES' else {})},
            quality={'status':'GOOD','reasons':[]}, provenance={'origin':raw.get('origin','existing_synthetic_replay'),
            'synthetic':True,'source_observed_at':raw['observed_time'],'source_available_at':raw['available_time'],
            'original_unit':raw['unit'],'dataset':'metabolic_d3_synthetic_replay','patient_mapping':'explicit_synthetic_import; no cohort identity link'}).view()
    def sync(self, patient_id, connection, received_at):
        case=self.source(connection)
        start=clock(case['timeline'][0]['timestamp']);end=clock(case['timeline'][-1]['timestamp'])
        events=[e for e in case['events'] if e['concept']=='heart_rate' and start-timedelta(hours=2)<=clock(e['observed_time'])<=end]
        # Exact existing D4.2 presentation schedule; no frontend-generated measurements.
        for minute,level,active in [(0,'Light',4),(120,'Moderate',18),(240,'Light',6),(360,'Moderate',16)]:
            observed=start+timedelta(minutes=minute)
            if observed<=end:
                events.append({'concept':'activity','event_id':'activity-schedule-'+str(minute),'observed_time':observed.isoformat(),
                    'available_time':observed.isoformat(),'value':active,'unit':'active minutes / 90 min','level':level,
                    'origin':'existing_D4.2_independent_presentation_schedule'})
        events.sort(key=lambda e:(clock(e['observed_time']),e['event_id']))
        offset=connection['metadata'].get('cursor',0);batch=events[offset:offset+6]
        result=[self.normalize(patient_id,e,received_at) for e in batch]
        for e in result:e['provenance'].update({'source_case':case['case'],'source_file_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()})
        return result,offset+len(batch),offset+len(batch)>=len(events)
