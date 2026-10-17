"""Versioned, serializable contracts shared by independent source adapters."""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = '2.0.0'
DATASETS = ('zigong', 'chiron', 'perheart', 'synthea')


class QualityStatus(str, Enum):
    VALID = 'VALID'
    MISSING = 'MISSING'
    SUSPECT = 'SUSPECT'
    INVALID = 'INVALID'
    DUPLICATE = 'DUPLICATE'


@dataclass(frozen=True)
class Provenance:
    dataset: str
    source_file: str
    source_record: str
    original_field: str
    transformed_field: str
    transformation: str
    source_sha256: str
    original_unit: str | None = None
    unit_support: str = 'unknown'


@dataclass(frozen=True)
class SourceValue:
    raw_value: Any
    usable_value: Any
    quality_status: str
    reasons: list[str]
    unit: str | None
    provenance: Provenance


@dataclass(frozen=True)
class Observation:
    observation_id: str
    patient_id: str
    dataset: str
    concept: str
    observed_time: str | float | None
    time_basis: str
    available_time: str | float | None
    availability_basis: str
    value: SourceValue
    coding: list[dict] = field(default_factory=list)
    unit_code: str | None = None
    unit_system: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PatientClinicalContext:
    patient_id: str
    dataset: str
    attributes: dict[str, SourceValue]
    observed_time: str | float | None = None
    available_time: str | float | None = None
    context_type: str = 'aggregate_snapshot'


@dataclass(frozen=True)
class Outcome:
    patient_id: str
    dataset: str
    endpoint: str
    value: SourceValue
    observed_time: str | float | None = None
    time_basis: str = 'unknown'
    horizon: dict | None = None
    time_origin: str = 'unconfirmed'
    event_time: Any = None
    censoring_status: str = 'unknown'
    metadata: dict = field(default_factory=dict)


def record(value):
    return asdict(value) if hasattr(value, '__dataclass_fields__') else value
