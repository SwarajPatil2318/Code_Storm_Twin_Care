"""Causal presentation metadata. Never changes features, forecasts or TwinState.

The optional activity schedule is explicitly simulated demonstration context,
independent of CGM/model outputs and excluded from the engine buffer.
"""
from datetime import datetime, timezone
from src.metabolic.windows import MINUTE


def iso(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat().replace('+00:00', 'Z')


def present(buffer, state, start_ns, previous_ns=None, previous_state=None):
    cutoff = buffer.as_of
    events = sorted((e for e in buffer.events.values()
                     if e.observed_ns <= cutoff and e.available_ns <= cutoff),
                    key=lambda e: (e.available_ns, e.observed_ns, e.event_id))
    received = []
    for e in events:
        if e.available_ns < start_ns:
            continue
        source = {'glucose': 'cgm', 'meal': 'meal_logger', 'heart_rate': 'smartwatch',
                  'activity': 'smartwatch', 'steps': 'smartwatch', 'ehr': 'ehr'}[e.concept]
        if source == 'ehr':
            continue  # Historical context is synced, never advertised as live.
        context = e.context or {}
        received.append({'source': source, 'concept': e.concept,
                         'timestamp': e.observed_time, 'receipt_time': e.available_time,
                         'value': e.value, 'unit': e.unit, 'synthetic': e.synthetic,
                         'meal_type': context.get('meal_type'),
                         'carbohydrate_g': context.get('carbohydrate_g'),
                         'origin': 'original_demo_event', 'model_input': e.concept == 'glucose'})
    activity = None
    # Explicit overlay allowed by D4.2 when original streams have no activity.
    for minutes, level, active in [(0, 'Light', 4), (120, 'Moderate', 18),
                                    (240, 'Light', 6), (360, 'Moderate', 16)]:
        when = start_ns + minutes * MINUTE
        if when > cutoff:
            break
        activity = {'level': level, 'active_minutes': active, 'window_minutes': 90,
                    'timestamp': iso(when), 'source': 'smartwatch',
                    'label': 'SIMULATED DEMO ACTIVITY', 'model_input': False,
                    'origin': 'independent_presentation_schedule'}
        received.append({'source': 'smartwatch', 'concept': 'activity',
                         'timestamp': iso(when), 'receipt_time': iso(when),
                         'value': active, 'unit': 'active minutes / 90 min',
                         'activity_level': level, 'synthetic': True,
                         'origin': 'independent_presentation_schedule', 'model_input': False})
    received.sort(key=lambda e: (e['receipt_time'], e['source']))
    new = [] if previous_ns is None else [e for e in received
          if previous_ns < datetime.fromisoformat(e['receipt_time'].replace('Z', '+00:00')).timestamp() * 1e9 <= cutoff]
    evidence = []
    glucose = buffer.glucose()[2]  # Honor original conflict/missingness handling.
    meals = [e for e in events if e.concept == 'meal']
    meal = meals[-1] if meals else None
    if meal and 0 <= cutoff - meal.observed_ns <= 120 * MINUTE:
        before = [e for e in glucose if e.observed_ns <= meal.observed_ns]
        anchor = before[-1] if before else None
        change = None if anchor is None else state['current_glucose'] - anchor.value
        evidence.append({'type': 'recent_meal', 'sources': ['meal_logger', 'cgm'],
                         'timestamp': meal.observed_time, 'classification': 'CONTEXTUAL',
                         'evidence_level': 'contextual', 'causal_claim_allowed': False,
                         'meal_age_minutes': (cutoff - meal.observed_ns) / MINUTE,
                         'meal_type': (meal.context or {}).get('meal_type'),
                         'carbohydrate_g': (meal.context or {}).get('carbohydrate_g'),
                         'glucose_change_since_meal': change,
                         'comparison_timestamp': anchor.observed_time if anchor else None,
                         'scientific_signal': 'received_meal_timing; association only'})
    current, expected = state['current_glucose'], state['shadow']['current']
    if current is not None and expected is not None:
        evidence.append({'type': 'personal_divergence', 'sources': ['cgm', 'shadow_twin'],
                         'timestamp': state['timestamp'], 'classification': 'MODEL-DERIVED',
                         'evidence_level': 'direct_comparison', 'causal_claim_allowed': False,
                         'current_glucose': current, 'expected_glucose': expected,
                         'difference_mg_dl': current - expected,
                         'scientific_signal': 'current glucose minus causal D3 personal Shadow'})
    if len(glucose) >= 2:
        first, last = glucose[-2:]
        evidence.append({'type': 'recent_change', 'sources': ['cgm'],
                         'timestamp': last.observed_time, 'classification': 'OBSERVED',
                         'evidence_level': 'observed', 'causal_claim_allowed': False,
                         'change_mg_dl': last.value - first.value,
                         'window_minutes': (last.observed_ns-first.observed_ns)/MINUTE,
                         'scientific_signal': 'difference between last two received CGM observations'})
    if activity:
        evidence.append({'type': 'recent_activity', 'sources': ['smartwatch'],
                         'timestamp': activity['timestamp'], 'classification': 'CONTEXTUAL',
                         'evidence_level': 'contextual', 'causal_claim_allowed': False,
                         'activity_level': activity['level'], 'active_minutes': activity['active_minutes'],
                         'window_minutes': activity['window_minutes'],
                         'label': activity['label'], 'model_input': False,
                         'scientific_signal': 'independent simulated context; no predictive or causal attribution'})
    if previous_state is not None:
        evidence.append({'type': 'forecast_update', 'sources': ['cgm', 'd2_model'],
                         'timestamp': state['timestamp'], 'classification': 'MODEL-DERIVED',
                         'evidence_level': 'direct_comparison', 'causal_claim_allowed': False,
                         'previous_prediction_time': previous_state['timestamp'],
                         'forecast_changes': {key: state['forecast'][key] - previous_state['forecast'][key]
                                              for key in state['forecast']},
                         'scientific_signal': 'successive real forecasts; horizons are relative to different cutoffs, not identical target times'})
    return {'evidence': evidence, 'source_events': received[-256:], 'new_events': new,
            'activity': activity, 'ehr_status': 'Synced', 'llm_used': False}
