"""Quality classification; hard-invalid data never have a usable value."""
import math
from decimal import Decimal, InvalidOperation

from .schema import QualityStatus, SourceValue

MISSING_TOKENS = {'', '?', 'NA', 'NaN', 'nan', 'null', 'None'}
# Screening bands are deliberately broad and are not diagnostic thresholds.
SCREEN_BOUNDS = {
    'heart_rate': (20, 250), 'heart_rate_sd': (0, 150),
    'systolic_bp': (50, 300), 'diastolic_bp': (20, 180),
    'weight': (20, 300), 'temperature': (30, 43),
    'skin_temperature': (20, 43), 'ambient_temperature': (-40, 60),
    'height': (1, 2.5), 'bmi': (10, 80), 'lvef': (0, 100), 'respiration': (5, 70),
    'glucose': (1, 40), 'perfusion_index': (0, 100),
}


class SchemaError(ValueError):
    """A source cannot safely be interpreted under the current schema contract."""


def missing(value):
    return value is None or (isinstance(value, str) and value.strip() in MISSING_TOKENS)


def parse_number(value, decimal_comma=False):
    if missing(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        if decimal_comma:
            text = text.replace(',', '.')
        result = float(text)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def normalize_id(value, dataset):
    if missing(value):
        raise SchemaError('Missing patient ID')
    text = str(value).strip()
    if dataset != 'synthea':
        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise SchemaError('Invalid numeric patient ID') from exc
        if not number.is_finite() or number != number.to_integral_value() or number <= 0:
            raise SchemaError('Patient ID must be a positive integer')
        text = str(int(number))
    if not text or any(c in text for c in '/\n\r\x00'):
        raise SchemaError('Invalid patient ID syntax')
    return f'{dataset}:{text}'


def classify(raw, concept, provenance, unit=None, numeric=True,
             decimal_comma=False, blocked_reason=None, provisional_unit=False,
             duplicate=False, extra_reasons=(), pediatric=False):
    """Soft suspect values may remain usable; explicit blocking/hard errors do not."""
    reasons = list(extra_reasons)
    value = parse_number(raw, decimal_comma) if numeric else raw
    if missing(raw):
        status = QualityStatus.MISSING
        value = None
        reasons.append('source_missing')
    elif numeric and value is None:
        status = QualityStatus.INVALID
        reasons.append('invalid_numeric_value')
    else:
        status = QualityStatus.VALID
        if numeric and concept == 'spo2' and not 0 <= value <= 100:
            status = QualityStatus.INVALID
            value = None
            reasons.append('spo2_outside_0_100')
        elif numeric and concept in SCREEN_BOUNDS and not (pediatric and concept == 'weight'):
            lo, hi = SCREEN_BOUNDS[concept]
            if not lo <= value <= hi:
                status = QualityStatus.SUSPECT
                value = None
                reasons.append('outside_plausibility_screen')
        if numeric and concept in {'weight', 'heart_rate', 'systolic_bp', 'diastolic_bp'} and parse_number(raw, decimal_comma) is not None and parse_number(raw, decimal_comma) <= 0:
            status = QualityStatus.INVALID
            value = None
            reasons.append('nonpositive_physiological_value')
    if numeric and unit is None:
        reasons.append('unknown_unit')
        if status == QualityStatus.VALID:
            status = QualityStatus.SUSPECT
        value = None
    if provisional_unit:
        reasons.append('provisional_unit_within_source_only')
        if status == QualityStatus.VALID:
            status = QualityStatus.SUSPECT
    if blocked_reason:
        reasons.append(blocked_reason)
        value = None
        if status == QualityStatus.VALID:
            status = QualityStatus.SUSPECT
    if extra_reasons:
        value = None
        if status == QualityStatus.VALID:
            status = QualityStatus.SUSPECT
    if duplicate:
        status = QualityStatus.DUPLICATE
        value = None
        reasons.append('exact_duplicate_first_source_row_retained')
    return SourceValue(raw, value, status.value, sorted(set(reasons)), unit, provenance)
