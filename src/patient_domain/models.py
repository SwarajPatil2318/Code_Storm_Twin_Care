"""Canonical patient-owned entities; eligibility describes a contract, not activation."""
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
from typing import Any
from uuid import UUID, uuid4

MODEL_ELIGIBLE = frozenset(('age_years', 'diabetes_duration_years', 'smoking_pack_years',
                           'sex', 'alcohol_history', 'medication_context', 'other_medication_context'))
PROFILE_FIELDS = MODEL_ELIGIBLE | frozenset(('diagnoses', 'medications', 'smoking_history', 'labs', 'notes'))


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(kind):
    return f'twincare-{kind}:{uuid4()}'


def validate_id(value, kind='patient'):
    prefix = f'twincare-{kind}:'
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError('Invalid namespaced identifier')
    UUID(value[len(prefix):])


def timestamp(value):
    if not isinstance(value, str) or datetime.fromisoformat(value.replace('Z', '+00:00')).utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')


def json_safe(value):
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode()) > 262144:
        raise ValueError('Record too large')
    return encoded


@dataclass(frozen=True, kw_only=True)
class Patient:
    patient_id: str
    display_name: str
    sex: str = 'unknown'
    date_of_birth: str | None = None
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    status: str = 'active'
    synthetic: bool = False

    def validate(self):
        validate_id(self.patient_id)
        if not isinstance(self.display_name, str) or not self.display_name.strip() or len(self.display_name) > 120:
            raise ValueError('Display name required (maximum 120 characters)')
        if self.sex not in ('female', 'male', 'other', 'unknown') or self.status not in ('active', 'inactive', 'archived'):
            raise ValueError('Invalid patient enum')
        if type(self.synthetic) is not bool:
            raise ValueError('Synthetic must be boolean')
        if self.date_of_birth is not None:
            born = date.fromisoformat(self.date_of_birth)
            if born > date.today():
                raise ValueError('Birth date cannot be in the future')
        timestamp(self.created_at)
        timestamp(self.updated_at)

    def view(self, as_of=None):
        as_of = as_of or datetime.now(timezone.utc).date()
        result = asdict(self)
        born = date.fromisoformat(self.date_of_birth) if self.date_of_birth else None
        result['age_years'] = (as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))) if born else None
        result['age_as_of'] = as_of.isoformat()
        return result


@dataclass(frozen=True, kw_only=True)
class ClinicalField:
    patient_id: str
    name: str
    value: Any = None
    unit: str | None = None
    observed_time: str | None = None
    available_time: str | None = None
    source_provenance: dict = field(default_factory=dict)
    source_document_id: str | None = None
    review_status: str = 'unreviewed'

    def view(self):
        validate_id(self.patient_id)
        if self.name not in PROFILE_FIELDS:
            raise ValueError('Unknown clinical field')
        for value in (self.observed_time, self.available_time):
            if value is not None:
                timestamp(value)
        if self.review_status not in ('unreviewed', 'confirmed', 'conflicted'):
            raise ValueError('Invalid review status')
        def check_owner(value):
            if isinstance(value, dict):
                if 'patient_id' in value and value['patient_id'] != self.patient_id:
                    raise ValueError('Clinical provenance ownership mismatch')
                for nested in value.values():
                    check_owner(nested)
            elif isinstance(value, list):
                for nested in value:
                    check_owner(nested)
        check_owner(asdict(self))
        json_safe(asdict(self))
        return {**asdict(self), 'model_input_eligible': self.name in MODEL_ELIGIBLE,
                'active_model_input': False}


@dataclass(frozen=True, kw_only=True)
class ClinicalProfile:
    patient_id: str
    fields: dict[str, ClinicalField] = field(default_factory=dict)
    updated_at: str = field(default_factory=now)

    def view(self):
        validate_id(self.patient_id)
        timestamp(self.updated_at)
        result = {}
        for name, value in self.fields.items():
            if value.patient_id != self.patient_id or value.name != name:
                raise ValueError('Clinical field ownership mismatch')
            result[name] = value.view()
        return {'patient_id': self.patient_id, 'fields': result, 'updated_at': self.updated_at,
                'active_model_input': False}


@dataclass(frozen=True, kw_only=True)
class Owned:
    patient_id: str
    entity_id: str
    created_at: str = field(default_factory=now)
    synthetic: bool = False


@dataclass(frozen=True, kw_only=True)
class UploadedDocument(Owned):
    document_type: str
    document_date: str
    filename: str
    media_type: str
    storage_path: str
    sha256: str
    metadata: dict = field(default_factory=dict)
    extraction_status: str = 'pending'


@dataclass(frozen=True, kw_only=True)
class DeviceConnection(Owned):
    device_type: str
    source: str
    status: str = 'not_connected'
    connected_at: str | None = None
    last_seen: str | None = None


@dataclass(frozen=True, kw_only=True)
class Observation(Owned):
    concept: str
    raw_value: Any
    original_unit: str | None
    observed_time: str
    available_time: str
    source_provenance: dict
    usable_value: Any = None
    canonical_unit: str | None = None
    quality_status: str = 'unreviewed'
    quality_reasons: list = field(default_factory=list)
    time_basis: str = 'source_recorded'


@dataclass(frozen=True, kw_only=True)
class MealEvent(Owned):
    observed_time: str
    available_time: str
    description: str
    source_provenance: dict
    meal_type: str = 'unspecified'


@dataclass(frozen=True, kw_only=True)
class TwinState(Owned):
    cutoff_time: str
    module: str
    schema_version: str
    payload: dict
    source_provenance: dict


@dataclass(frozen=True, kw_only=True)
class Prediction(Owned):
    prediction_time: str
    horizon: str
    prediction_kind: str
    model_version: str
    output: dict
    source_provenance: dict
    twin_state_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class Alert(Owned):
    observed_time: str
    severity: str
    message: str
    status: str = 'open'
    prediction_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class AuditEvent(Owned):
    action: str
    entity_type: str
    target_id: str
    actor: str = 'local_service'
    details: dict = field(default_factory=dict)


ENTITY_TYPES = {'documents': UploadedDocument, 'device-connections': DeviceConnection,
                'observations': Observation, 'meal-events': MealEvent, 'twin-states': TwinState,
                'predictions': Prediction, 'alerts': Alert, 'audit-events': AuditEvent}


def owned_view(entity):
    validate_id(entity.patient_id)
    validate_id(entity.entity_id, 'entity')
    timestamp(entity.created_at)
    result = asdict(entity)
    if type(entity.synthetic) is not bool:
        raise ValueError('Synthetic must be boolean')
    for key in ('observed_time', 'available_time', 'cutoff_time', 'prediction_time', 'connected_at', 'last_seen'):
        if result.get(key) is not None:
            timestamp(result[key])
    def check_nested(value):
        if isinstance(value, dict):
            if 'patient_id' in value and value['patient_id'] != entity.patient_id:
                raise ValueError('Nested ownership mismatch')
            for item in value.values():
                check_nested(item)
        elif isinstance(value, list):
            for item in value:
                check_nested(item)
    check_nested(result)
    if isinstance(entity, UploadedDocument):
        date.fromisoformat(entity.document_date)
        if entity.extraction_status != 'pending':
            raise ValueError('Extraction is not implemented in P1')
        if len(entity.sha256) != 64 or any(c not in '0123456789abcdef' for c in entity.sha256):
            raise ValueError('Invalid document digest')
    json_safe(result)
    return result
