"""D1 canonical contracts. Unknown availability is preserved, never fabricated."""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParticipantStatic:
    participant_id: str
    source_dataset: str
    diabetes_status: str
    recording_id: str | None = None
    age: float | None = None
    sex: str | None = None
    BMI: float | None = None
    HbA1c: float | None = None
    diabetes_duration: float | None = None
    fasting_glucose: float | None = None
    fasting_insulin: float | None = None
    medications: str | None = None
    comorbidities: str | None = None
    observation_time: str | None = None
    availability_time: str | None = None
    timing_basis: str = 'unknown'
    original_fields: dict[str, Any] = field(default_factory=dict)
    original_units: dict[str, str] = field(default_factory=dict)
    data_quality: list[str] = field(default_factory=list)

@dataclass
class MetabolicObservation:
    participant_id: str
    source_dataset: str
    timestamp: str | None
    availability_time: str | None
    glucose_mg_dl: float | None
    glucose_source: str
    HR: float | None = None
    steps: float | None = None
    activity: float | None = None
    sleep: float | None = None
    SpO2: float | None = None
    meal_event: str | None = None
    carbs_g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    fiber_g: float | None = None
    calories: float | None = None
    insulin_event: str | None = None
    medication_event: str | None = None
    data_quality: list[str] = field(default_factory=list)
    original_fields: dict[str, Any] = field(default_factory=dict)
    original_units: dict[str, str] = field(default_factory=dict)
    source_file: str | None = None
    recording_id: str | None = None
    timezone: str | None = None
    availability_basis: str = 'unknown'
    is_observed_glucose: bool | None = None


def glucose_to_mg_dl(value, unit):
    if value is None: return None
    if unit == 'mg/dL': return float(value)
    if unit == 'mmol/L': return float(value)*18.0182
    raise ValueError('Ambiguous glucose unit; normalization refused')
