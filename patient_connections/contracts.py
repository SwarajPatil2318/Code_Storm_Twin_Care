"""Validated canonical observations with explicit two-clock provenance."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from uuid import uuid5, NAMESPACE_URL
from src.patient_domain.models import validate_id, json_safe

SOURCES = ('EHR', 'CGM', 'WEARABLE', 'MEAL_LOGGER')
STATES = ('NOT_CONNECTED', 'CONNECTING', 'CONNECTED', 'SYNCING', 'STALE', 'ERROR', 'DISCONNECTED')
TYPES = {'CGM_READING': ('CGM', 'glucose_mg_dl', 'mg/dL'),
         'HEART_RATE': ('WEARABLE', 'bpm', 'bpm'),
         'STEP_COUNT': ('WEARABLE', 'count', 'count'),
         'ACTIVITY': ('WEARABLE', 'level', None),
         'ACTIVE_MINUTES': ('WEARABLE', 'minutes', 'min'),
         'SLEEP_SUMMARY': ('WEARABLE', 'hours', 'h'),
         'DEVICE_STATUS': (None, 'status', None)}


def clock(value):
    if not isinstance(value, str): raise ValueError('Timezone-aware timestamp required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.utcoffset() is None: raise ValueError('Timezone-aware timestamp required')
    return parsed.astimezone(timezone.utc)


def iso(value):
    return clock(value).isoformat()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_owner(value, pid):
    if isinstance(value, dict):
        if 'patient_id' in value and value['patient_id'] != pid: raise ValueError('Ownership mismatch')
        if any(k.lower() in ('password', 'token', 'secret', 'access_token', 'api_key') for k in value):
            raise ValueError('Credentials are not accepted')
        for v in value.values(): check_owner(v, pid)
    elif isinstance(value, list):
        for v in value: check_owner(v, pid)


@dataclass(frozen=True, kw_only=True)
class CanonicalEvent:
    patient_id: str
    source_type: str
    provider_type: str
    event_type: str
    observed_at: str
    received_at: str
    payload: dict
    quality: dict
    provenance: dict
    schema_version: str = 'twincare-event/1'
    event_id: str | None = None
    device_id: str | None = None
    sequence_id: str | None = None
    raw_reference: str | None = None
    ingestion_trace_id: str | None = None

    def view(self, as_of=None):
        validate_id(self.patient_id)
        if self.source_type not in SOURCES or self.event_type not in TYPES or self.schema_version != 'twincare-event/1':
            raise ValueError('Unsupported event contract')
        source, key, unit = TYPES[self.event_type]
        if source and source != self.source_type: raise ValueError('Event/source mismatch')
        if not isinstance(self.provider_type, str) or not 1 <= len(self.provider_type) <= 80: raise ValueError('Provider required')
        observed, received = clock(self.observed_at), clock(self.received_at)
        wall = as_of or datetime.now(timezone.utc)
        if received > wall + timedelta(minutes=5) or observed > wall + timedelta(minutes=5) or observed > received + timedelta(minutes=5):
            raise ValueError('Future timestamp exceeds five-minute tolerance')
        if not isinstance(self.payload, dict) or key not in self.payload: raise ValueError('Payload required')
        allowed={key}
        if self.event_type=='CGM_READING':allowed|={'timestamp','trend','source_status'}
        if self.event_type=='ACTIVE_MINUTES':allowed|={'level','window_minutes'}
        if set(self.payload)-allowed:raise ValueError('Unsupported payload field')
        value = self.payload[key]
        if key in ('status', 'level'):
            if not isinstance(value, str) or not value.strip() or len(value) > 80: raise ValueError('Invalid contextual value')
        elif value is not None:
            if not finite(value) or value < 0 or (key in ('glucose_mg_dl', 'bpm') and value == 0): raise ValueError('Invalid observation value')
            if key == 'count' and type(value) is not int: raise ValueError('Steps require an integer count')
            if key == 'hours' and value > 24: raise ValueError('Invalid sleep duration')
        if self.event_type == 'CGM_READING' and clock(self.payload.get('timestamp')) != observed: raise ValueError('CGM clock mismatch')
        if not isinstance(self.quality, dict) or self.quality.get('status') not in ('GOOD', 'MISSING', 'SUSPECT'):
            raise ValueError('Explicit quality required')
        reasons=self.quality.get('reasons',[])
        if not isinstance(reasons,list) or any(not isinstance(r,str) for r in reasons):raise ValueError('Invalid quality reasons')
        for key in ('device_id','sequence_id','raw_reference','ingestion_trace_id'):
            v=getattr(self,key)
            if v is not None and (not isinstance(v,str) or len(v)>500):raise ValueError('Invalid optional event metadata')
        if value is None and self.quality['status'] != 'MISSING': raise ValueError('Missingness must be explicit')
        if not isinstance(self.provenance, dict) or not self.provenance.get('origin') or type(self.provenance.get('synthetic')) is not bool:
            raise ValueError('Source provenance required')
        result = asdict(self)
        result['observed_at'], result['received_at'] = observed.isoformat(), received.isoformat()
        if self.event_type=='CGM_READING':result['payload']={**self.payload,'timestamp':observed.isoformat()}
        check_owner(result, self.patient_id)
        # Receipt time is excluded: replaying the same sample does not create a new observation.
        identity = [self.patient_id, self.source_type, self.provider_type, self.event_type,
                    result['observed_at'], self.sequence_id, None if self.sequence_id else result['payload']]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        expected = 'twincare-entity:' + str(uuid5(NAMESPACE_URL, 'twincare-event:' + digest))
        if self.event_id and self.event_id != expected: raise ValueError('Event identity mismatch')
        result['event_id'] = expected
        result['active_model_input'] = False
        json_safe(result)
        return result
