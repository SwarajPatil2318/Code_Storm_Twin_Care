"""Generated device observations obey the existing P3 adapter/event contract."""
from datetime import datetime, timezone
from patient_connections.contracts import CanonicalEvent, clock

PROVIDERS = {'CGM': 'synthetic_live_cgm', 'WEARABLE': 'synthetic_live_wearable'}


def events(pid, session, source, received_at=None, prior=False):
    state = session['physiology']; at = clock(state['time'])
    received = received_at or at
    provenance = {'origin': 'Synthetic physiology generator',
                  'synthetic': True, 'simulation_id': session['simulation_id'],
                  'seed': session['seed'], 'profile_id': session['profile_id'],
                  'native_period_minutes': 15, 'prior_synthetic_history': prior,
                  'time_basis': 'controlled_simulation_clock', 'commercial_provider_connected': False}
    provenance['wall_clock_generated_at'] = datetime.now(timezone.utc).isoformat()
    values = [('CGM_READING', {'glucose_mg_dl': state['glucose'], 'timestamp': state['time'],
                              'source_status': 'synthetic'})] if source == 'CGM' else [
        ('HEART_RATE', {'bpm': state['heart_rate']}),
        ('ACTIVE_MINUTES', {'minutes': 15 if state['activity'] == 'Walking' else 0,
                            'level': state['activity'], 'window_minutes': 15}),
    ]
    # HR and activity are live each tick. Steps update hourly so the unchanged
    # P6 24-hour contextual snapshot stays below its existing 256 KiB record cap.
    # All native CGM ticks remain present; no model input is sampled or discarded.
    if source == 'WEARABLE' and state['step'] % 4 == 0:
        values.append(('STEP_COUNT', {'count': state['steps_today']}))
    return [CanonicalEvent(patient_id=pid, source_type=source, provider_type=PROVIDERS[source],
                           event_type=kind, observed_at=at.isoformat(), received_at=received.isoformat(),
                           payload=payload, quality={'status': 'GOOD', 'reasons': []},
                           provenance={**provenance,'native_period_minutes':60 if kind=='STEP_COUNT' else 15},
                           device_id=session['simulation_id'], sequence_id=f"{session['simulation_id']}:{state['step']}:{kind}").view(as_of=received)
            for kind, payload in values]
