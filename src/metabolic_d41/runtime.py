"""Isolated, bounded synthetic sessions; never train or open research cohorts."""
from collections import deque
import json
import secrets
import threading
import time

import numpy as np
from src.metabolic.windows import MINUTE
from src.metabolic_d3.buffer import Buffer
from src.metabolic_d3.engine import TwinEngine
from src.metabolic_d3.events import Event, CGMAdapter, timestamp
from src.metabolic_d3.reference import Reference, NAME
from src.metabolic_d3.shadow import Shadow
from src.metabolic_d41.presentation import present


class RuntimeUnavailable(RuntimeError):
    pass


class RevisionConflict(ValueError):
    pass


class MeasuredBuffer(Buffer):
    def features(self, cutoff=None):
        start = time.perf_counter()
        result = super().features(cutoff)
        self.feature_ms = (time.perf_counter() - start) * 1000
        self.feature_schema = None if result[0] is None else {key: list(np.asarray(value).shape) for key, value in result[0].items()}
        return result


class MeasuredShadow(Shadow):
    def predict(self, buffer, cutoff=None):
        start = time.perf_counter()
        result = super().predict(buffer, cutoff)
        self.elapsed_ms = (time.perf_counter() - start) * 1000
        return result


class MeasuredReference:
    def __init__(self, reference, lock):
        self.reference = reference
        self.registration = reference.registration
        self.risk = reference.risk
        self.lock = lock
        self.calls = 0
        self.elapsed_ms = 0

    def predict(self, data):
        # A shared immutable model, serialized calls, separate per-session timers.
        with self.lock:
            start = time.perf_counter()
            result = self.reference.predict(data)
            self.elapsed_ms = (time.perf_counter() - start) * 1000
        self.calls += 1
        return result


class Session:
    def __init__(self, case, reference, model_lock, index=0, mode='LIVE TWIN'):
        self.case_id = case['case']
        self.mode = mode
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        self.revision = 0
        self.snapshots = deque(maxlen=128)
        self.logs = deque(maxlen=256)
        self.engine = TwinEngine(case['patient'], MeasuredReference(reference, model_lock))
        self.engine.buffer = MeasuredBuffer(case['patient']['id'])
        self.engine.shadow = MeasuredShadow()
        self.start_index = index
        self.index = index
        self.frame_times = [timestamp(f['timestamp'])[0] for f in case['timeline']]
        self.presentation_cutoff = None
        cutoff = self.frame_times[index]
        # The producer's future queue is NOT a member of the engine/buffer.
        # Initialize only received prefix; includes causal warm/Shadow history.
        validation_started = time.perf_counter()
        for raw in case['events']:
            event = Event(**raw)
            if event.available_ns <= cutoff and event.observed_ns <= cutoff:
                if not event.synthetic:
                    raise ValueError('Synthetic demo initialization requires synthetic source provenance')
                self.engine.buffer.ingest(event)
        self.engine.buffer.as_of = cutoff
        validation_ms = (time.perf_counter() - validation_started) * 1000
        self.latest = self._emit(validation_ms, validation_started, initialized=True)

    def _emit(self, validation_ms, started, initialized=False):
        previous_state = self.latest['state'] if hasattr(self, 'latest') else None
        before = self.engine.reference.calls
        state = self.engine.snapshot()
        if state['forecast']['plus_30m'] is None or self.engine.reference.calls != before + 1:
            raise RuntimeUnavailable('Complete native CGM history required for live inference')
        elapsed = (time.perf_counter() - started) * 1000
        self.revision += 1
        trace = 'MT-' + secrets.token_hex(8).upper()
        buffer = self.engine.buffer
        t, g, events, _ = buffer.glucose()
        cutoff = buffer.as_of
        recent = [e for e in events if e.observed_ns >= cutoff - 120 * MINUTE]
        history = [{'timestamp': e.observed_time, 'glucose_mg_dl': e.value} for e in recent]
        prep = buffer.feature_ms
        shadow = self.engine.shadow.elapsed_ms
        inference = self.engine.reference.elapsed_ms
        assembly = max(0, elapsed - validation_ms - prep - shadow - inference)
        timing = {'event_validation_ms': validation_ms, 'feature_preparation_ms': prep,
                  'model_inference_ms': inference, 'shadow_update_ms': shadow,
                  'state_assembly_ms': assembly, 'twin_update_ms': elapsed,
                  'serialization_ms': 0.0, 'endpoint_ms': 0.0}
        update = {'trace_id': trace, 'revision': self.revision, 'mode': self.mode,
                  'model_version': NAME, 'model_hashes': state['evidence']['model_hashes'],
                  'inference_timestamp': state['timestamp'], 'timings_ms': timing,
                  'model_executed': True, 'feature_schema': buffer.feature_schema,
                  'input': {'current_glucose': state['current_glucose'],
                            'usable_cgm_observations': len(recent), 'history_window_minutes': 120,
                            'personal_prefix_observations': len(t) - len(recent),
                            'personal_baseline_available': state['shadow']['available'],
                            'recent_slope': next((w['value'] for w in state['why_now'] if w['signal'] == 'recent_slope'), None),
                            'meal_context': state['recent_context']['meal'],
                            'static_clinical_context_available': state['recent_context']['ehr'] is not None,
                            'event_time': events[-1].observed_time,
                            'receipt_time': events[-1].available_time},
                  'pipeline': ['Canonical CGM / two-clock validation', 'Native 120-minute feature preparation',
                               NAME, 'D3 causal Shadow', 'Divergence / deterministic WhyNow', 'MetabolicTwinState'],
                  'stream_label': 'SIMULATED CGM STREAM', 'initialized': initialized}
        snapshot = {'prediction_time': state['timestamp'], 'model_version': NAME,
                    'forecast': dict(state['forecast']), 'spike_probability': state['spike']['probability'],
                    'peak': dict(state['predicted_peak']), 'trace_id': trace, 'comparisons': []}
        self.snapshots.append(snapshot)
        # Mature predictions only against observations actually received by now.
        observed = {e.observed_ns: e.value for e in events}
        for past in self.snapshots:
            origin = timestamp(past['prediction_time'])[0]
            past['comparisons'] = []
            for h in [30, 60, 90, 120]:
                target = origin + h * MINUTE
                if target <= cutoff and target in observed:
                    prediction = past['forecast'][f'plus_{h}m']
                    actual = observed[target]
                    past['comparisons'].append({'horizon_minutes': h, 'predicted': prediction,
                                                'observed': actual, 'absolute_error_mg_dl': abs(prediction - actual)})
        self.logs.append({'trace_id': trace, 'timestamp': state['timestamp'],
                          'glucose_mg_dl': state['current_glucose'], 'stages': update['pipeline'],
                          'timings_ms': timing})
        self.engine.last_state = state
        self.engine.timings.clear()  # Existing engine profiling list otherwise grows forever.
        self.touched = time.monotonic()
        self.latest = {'state': state, 'update': update, 'observed_history': history,
                       'forecast_history': list(self.snapshots), 'event_log': list(self.logs),
                       'index': self.index, 'complete': cutoff >= self.frame_times[-1]}
        self.latest['presentation'] = present(buffer, state, self.frame_times[0], self.presentation_cutoff, previous_state)
        self.presentation_cutoff = cutoff
        return self.latest

    def ingest(self, event, contexts=(), simulated_time=None, validation_started=None):
        started = time.perf_counter() if validation_started is None else validation_started
        cutoff = event.available_ns if simulated_time is None else timestamp(simulated_time)[0]
        if event.observed_ns > cutoff or event.available_ns > cutoff:
            raise ValueError('Future observation or receipt is not available at simulated time')
        if cutoff != event.available_ns:
            raise ValueError('CGM update cutoff must equal its receipt time')
        buffer = self.engine.buffer
        if buffer.as_of is not None and event.available_ns <= buffer.as_of:
            raise ValueError('CGM observations must advance the session clock')
        if event.patient_id != buffer.patient_id or not event.synthetic:
            raise ValueError('Only same-session synthetic demo events accepted')
        for item in contexts:
            if item.patient_id != buffer.patient_id or not item.synthetic:
                raise ValueError('Context must have the same synthetic session identity')
            if item.available_ns > cutoff or item.observed_ns > cutoff:
                raise ValueError('Future context cannot enter the session')
        # Validate on a temporary buffer first so rejected/inference-failed events
        # do not corrupt the last successfully emitted session state.
        candidate = MeasuredBuffer(buffer.patient_id, buffer.clock_offset_minutes)
        candidate.events = dict(buffer.events)
        candidate.as_of = buffer.as_of
        candidate.time_basis = buffer.time_basis
        candidate.cgm_source = buffer.cgm_source
        for item in contexts:
            candidate.ingest(item)
        candidate.ingest(event)
        data, _ = candidate.features(cutoff)
        if data is None:
            raise ValueError('CGM event does not preserve the frozen native 120-minute input schema')
        validation_ms = (time.perf_counter() - started) * 1000
        old_index = self.index
        self.index = min(len(self.frame_times)-1, sum(t <= cutoff for t in self.frame_times)-1)
        self.engine.buffer = candidate
        try:
            return self._emit(validation_ms, started)
        except Exception:
            self.engine.buffer = buffer
            self.index = old_index
            raise


class RuntimeStore:
    MAX_SESSIONS = 8
    TTL_SECONDS = 1800

    def __init__(self, demo, reference=None):
        self.cases = {case['case']: case for case in demo['cases']}
        self.sessions = {}
        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.latencies = deque(maxlen=2048)
        started = time.perf_counter()
        self.reference = reference
        self.load_error = False
        try:
            self.reference = reference or Reference()
        except Exception:
            self.load_error = True  # No model paths, identities or private data in logs.
        self.startup_ms = (time.perf_counter() - started) * 1000

    def create(self, payload):
        if self.load_error:
            raise RuntimeUnavailable('Frozen live inference unavailable')
        case_id = payload.get('case_id', 'predicted_spike')
        if case_id not in self.cases:
            raise ValueError('Unknown synthetic demo case')
        index = payload.get('start_index', 0)
        if type(index) is not int or not 0 <= index < len(self.cases[case_id]['timeline']):
            raise ValueError('Invalid start index')
        mode = payload.get('mode', 'LIVE TWIN')
        if mode not in ['LIVE TWIN', 'HISTORICAL / DEMO REPLAY']:
            raise ValueError('Invalid demo mode')
        with self.lock:
            now = time.monotonic()
            self.sessions = {key: value for key, value in self.sessions.items() if now - value.touched < self.TTL_SECONDS}
            if len(self.sessions) >= self.MAX_SESSIONS:
                raise RuntimeUnavailable('Live session capacity reached')
            session = Session(self.cases[case_id], self.reference, self.model_lock, index, mode)
            session_id = secrets.token_urlsafe(24)
            self.sessions[session_id] = session
        return {'session_id': session_id, **session.latest}

    def get(self, payload):
        session_id = payload.get('session_id')
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None or time.monotonic() - session.touched >= self.TTL_SECONDS:
            raise ValueError('Session missing or expired; reset the demo')
        return session_id, session

    def update(self, payload, direct=False):
        session_id, session = self.get(payload)
        with session.lock:
            if payload.get('expected_revision') != session.revision:
                raise RevisionConflict('Update revision changed; reset the demo')
            if direct:
                if payload.get('source') != 'cgm':
                    raise ValueError('This endpoint accepts CGM observations only')
                if payload.get('case_id', session.case_id) != session.case_id:
                    raise ValueError('Cross-case event')
                validation_started=time.perf_counter()
                if type(payload.get('glucose_mg_dl')) not in [int,float]:
                    raise ValueError('CGM glucose must be a numeric mg/dL value')
                event = CGMAdapter().adapt(event_id='runtime-' + secrets.token_hex(12),
                    patient_id=session.engine.patient['id'], source=session.engine.buffer.cgm_source,
                    observed_time=payload.get('event_time'), available_time=payload.get('receipt_time'),
                    value=payload.get('glucose_mg_dl'), unit='mg/dL', synthetic=True,
                    native_period_minutes=15)
                if event.value is None:
                    raise ValueError('Live event requires an explicit CGM value')
                result = session.ingest(event, simulated_time=payload.get('simulated_time'), validation_started=validation_started)
            else:
                # Isolated producer queue releases one next received CGM. Future
                # replay records stay outside the TwinEngine until this boundary.
                cutoff = session.engine.buffer.as_of
                queue = self.cases[session.case_id]['events']
                glucose = [e for e in queue if e['concept'] == 'glucose' and timestamp(e['available_time'])[0] > cutoff]
                if not glucose:
                    return {'session_id': session_id, **session.latest, 'complete': True}
                raw = min(glucose, key=lambda e: timestamp(e['available_time'])[0])
                validation_started=time.perf_counter()
                event = Event(**raw)
                contexts = [Event(**e) for e in queue if e['concept'] != 'glucose'
                            and cutoff < timestamp(e['available_time'])[0] <= event.available_ns]
                result = session.ingest(event, contexts, validation_started=validation_started)
            return {'session_id': session_id, **result}

    def delete(self, payload):
        with self.lock:
            self.sessions.pop(payload.get('session_id'), None)
        return {'released': True}

    def route(self, path, payload):
        if path.endswith('/session'):
            return self.create(payload)
        if path.endswith('/advance'):
            return self.update(payload)
        if path.endswith('/event'):
            return self.update(payload, direct=True)
        if path.endswith('/close'):
            return self.delete(payload)
        raise KeyError('Unknown live resource')

    def record_latency(self, response, endpoint_started):
        # Serialize once for a genuine measurement; final encoding adds these
        # measurements. endpoint_ms measures work before the socket write.
        started = time.perf_counter()
        json.dumps(response, allow_nan=False)
        serialization_ms = (time.perf_counter() - started) * 1000
        if 'update' in response:
            timings = response['update']['timings_ms']
            timings['serialization_ms'] = serialization_ms
            timings['endpoint_ms'] = (time.perf_counter() - endpoint_started) * 1000
            self.latencies.append(dict(timings))

    def performance(self):
        values = list(self.latencies)
        return {'startup_model_load_ms': self.startup_ms, 'samples': len(values),
                'scope': 'Local runtime measurements; endpoint excludes socket transfer and browser rendering',
                'metrics_ms': {key: {'median': float(np.median([v[key] for v in values])),
                                    'p95': float(np.quantile([v[key] for v in values], .95))}
                               for key in values[0]} if values else {}}
