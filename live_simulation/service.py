"""Persisted patient-scoped sessions over normal ingestion/readiness/inference."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import json
import secrets
import threading
import time
from patient_connections.contracts import clock, check_owner
from patient_connections.service import ConnectionError
from patient_inference.connections import InferenceConnections
from src.patient_domain.models import identifier, json_safe, DeviceConnection, owned_view
from judge_scenarios.clock import ReplayClock
from .engine import initial, evolve, register_meals, PRESETS
from .adapters import PROVIDERS, events

SPEEDS = {'1x': 1., '4x': 4., '15x': 15., '60x': 60., 'DEMO': 180.}


class LiveConnections(InferenceConnections):
    def __init__(self, inference):
        super().__init__(inference)
        self.simulation = None

    def connect(self,pid,source,config):
        with self.repo.connection() as db:
            previous=self._get(db,pid,source)
            if previous and previous['provider_type'] in PROVIDERS.values():
                raise ConnectionError('This patient uses a live synthetic source. Reconnect it through Simulated Live CGM or Wearable.')
        return super().connect(pid,source,config)

    def disconnect(self, pid, source):
        if self.simulation:
            with self.simulation.lock:
                result = super().disconnect(pid, source)
                if result['provider_type'] in PROVIDERS.values():
                    self.simulation.source_disconnected(pid, source)
                return result
        return super().disconnect(pid, source)

    def sync(self, pid, source):
        value = self.status(pid, source)
        if value['provider_type'] in PROVIDERS.values():
            return {'events_received': 0, 'connection': value, 'continuous': True}
        return super().sync(pid, source)


class Simulation:
    def __init__(self, inference, connections, simulation_clock=None):
        self.runtime = inference; self.repo = inference.repo; self.connections = connections
        self.clock = simulation_clock or ReplayClock()
        self.lock = threading.RLock(); self.wakeup = threading.Event(); self.closed = threading.Event()
        self.worker = None; self.sessions = {}
        with self.repo.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS simulation_sessions(
                patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id) ON DELETE CASCADE,
                simulation_id TEXT NOT NULL UNIQUE, data TEXT NOT NULL CHECK(json_valid(data)
                AND json_extract(data,'$.patient_id')=patient_id
                AND json_extract(data,'$.simulation_id')=simulation_id))''')
            for row in db.execute('SELECT data FROM simulation_sessions'):
                value = json.loads(row[0])
                if value['status'] == 'RUNNING':
                    value.update(status='PAUSED', next_due_wall=None,
                                 restart_notice='Paused after server restart. Resume to continue from saved physiology.')
                self.sessions[value['patient_id']] = value
                self._persist(db, value)
        connections.simulation = self
        # Recover a committed reading whose inference previously failed to persist.
        # Reuse the saved cutoff and normal P5/P6/P7 path; never rewrite inputs.
        for pid,session in list(self.sessions.items()):
            if (session.get('last_inference') or {}).get('status')=='INFERENCE_UNAVAILABLE':
                try:
                    with self.context(pid):
                        result=self.runtime.run_twin_inference(pid,session['simulation_time'])
                    session['last_inference']={k:result.get(k) for k in ('status','prediction_id','reference_time')}
                    self._save(session)
                except Exception:
                    # Keep the recoverable failure for the next restart/tick.
                    pass

    def _persist(self, db, value):
        check_owner(value, value['patient_id'])
        db.execute('INSERT INTO simulation_sessions VALUES(?,?,?) ON CONFLICT(patient_id) DO UPDATE SET data=excluded.data',
                   (value['patient_id'], value['simulation_id'], json_safe(value)))

    def _save(self, value):
        value['updated_at'] = datetime.now(timezone.utc).isoformat()
        with self.repo.connection() as db:
            self.repo._require(db, value['patient_id']); self._persist(db, value)
        self.sessions[value['patient_id']] = value

    def reference(self, pid):
        session = self.sessions.get(pid)
        if session:
            ref = clock(session['simulation_time'])
            if session.get('disconnected_at'):
                ref += max(timedelta(), datetime.now(timezone.utc) - clock(session['disconnected_at']))
            return ref
        return datetime.now(timezone.utc)

    @contextmanager
    def context(self, pid):
        # All ordinary patient actions and scheduler steps see one coherent clock.
        with self.lock, self.runtime._lock, self.clock.at(self.reference(pid)):
            yield

    def create(self, pid, config=None):
        config = config or {}
        if set(config) - {'seed', 'profile_id', 'speed'}:
            raise ValueError('Unsupported simulation configuration')
        patient = self.repo.get_patient(pid)
        if not patient['synthetic']:
            raise ConnectionError('Simulated live sources are available only for explicitly synthetic/example patients.', 403)
        with self.lock:
            if pid in self.sessions:
                if config:
                    raise ConnectionError('This patient already has a simulation. Use its existing controls.')
                return self.status(pid)
            profile = config.get('profile_id', 'moderate')
            if profile not in PRESETS or config.get('speed', 'DEMO') not in SPEEDS:
                raise ValueError('Choose an available synthetic preset and speed')
            seed = config.get('seed', secrets.randbits(32))
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError('Seed must be a nonnegative integer below 2^63')
            # A whole-second synthetic anchor gives simple reproducible native
            # tick timestamps. Round forward explicitly so existing clinical
            # confirmations remain available at the anchor.
            at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=1)
            value = {'simulation_id': identifier('entity'), 'patient_id': pid, 'status': 'STOPPED',
                     'simulation_time': at.isoformat(), 'wall_clock_started_at': None,
                     'speed': config.get('speed', 'DEMO'), 'speed_multiplier': SPEEDS[config.get('speed', 'DEMO')],
                     'seed': seed, 'profile_id': profile, 'physiology': initial(at, profile),
                     'created_at': at.isoformat(), 'updated_at': at.isoformat(), 'next_due_wall': None,
                     'history_loaded': False, 'error': None, 'disconnected_at': None, 'last_inference': None}
            self._save(value)
            return self.status(pid)

    def connect(self, pid, source, config):
        if source not in PROVIDERS or set(config) - {'acknowledge_synthetic', 'load_prior_history', 'profile_id', 'seed'}:
            raise ValueError('Live source configuration required')
        if config.get('acknowledge_synthetic') is not True:
            raise ValueError('Acknowledge the synthetic source before connecting')
        if type(config.get('load_prior_history', False)) is not bool:
            raise ValueError('Prior history choice must be boolean')
        with self.lock:
            with self.repo.connection() as db:
                patient=self.repo._require(db,pid)
                if not patient['synthetic']:raise ConnectionError('Synthetic patient required',403)
                previous=self.connections._get(db,pid,source)
                if previous and previous['provider_type']!=PROVIDERS[source]:
                    raise ConnectionError('This patient already uses a historical source. Create a separate synthetic patient for live simulation.')
            if pid not in self.sessions:
                self.create(pid, {k: v for k, v in config.items() if k in ('seed', 'profile_id')})
            else:
                session=self.sessions[pid]
                changed={k:v for k,v in config.items() if k in ('seed','profile_id') and v!=session[k]}
                if changed:
                    if session['physiology']['step'] or self.connections.observations(pid,source='CGM')['count']:
                        raise ConnectionError('The simulation preset and seed are fixed once CGM history begins. Reconnect the same preset.')
                    if changed.get('profile_id',session['profile_id']) not in PRESETS:raise ValueError('Unknown preset')
                    seed=changed.get('seed',session['seed'])
                    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Invalid seed')
                    session.update(changed);session['physiology']=initial(clock(session['simulation_time']),session['profile_id'])
                    self._save(session)
            with self.context(pid), self.repo.connection() as db:
                patient = self.repo._require(db, pid)
                if not patient['synthetic']:
                    raise ConnectionError('Synthetic patient required', 403)
                previous = self.connections._get(db, pid, source)
                if previous and previous['provider_type'] != PROVIDERS[source]:
                    raise ConnectionError('This patient already uses a historical source. Create a separate synthetic patient for live simulation.')
                base = owned_view(DeviceConnection(patient_id=pid, entity_id=previous['entity_id'] if previous else identifier('entity'),
                                                   device_type=source.lower(), source='Synthetic physiology generator', synthetic=True))
                value = {**self.connections._empty(pid, source), **base, **(previous or {}),
                         'connection_id': base['entity_id'], 'source_type': source, 'connection_schema_version': 3,
                         'provider_type': PROVIDERS[source], 'status': 'CONNECTED',
                         'connected_at': (previous or {}).get('connected_at') or self.clock.stamp(),
                         'metadata': {'mode': 'simulated_live', 'simulation_id': self.sessions[pid]['simulation_id'],
                                      'acknowledge_synthetic': True, 'native_period_minutes': 15},
                         'provenance': {'origin': 'Synthetic physiology generator; not a physical medical device',
                                        'synthetic': True, 'commercial_provider_connected': False}, 'error': None}
                self.connections._save(db, value); self.connections._audit(db, pid, source, 'SOURCE_CONNECTED')
            self.sessions[pid]['disconnected_at'] = None
            self._save(self.sessions[pid])
            if source == 'CGM' and config.get('load_prior_history'):
                self.load_history(pid)
            elif source == 'WEARABLE':
                with self.context(pid):
                    current=events(pid,self.sessions[pid],source)
                    # Reconnection cannot rewrite a sample already received at
                    # this tick, even if the next activity control has changed.
                    with self.repo.connection() as db:
                        seen={r[0] for r in db.execute('SELECT entity_id FROM observations WHERE patient_id=?',(pid,))}
                    current=[e for e in current if e['event_id'] not in seen]
                    if current:self._ingest(pid,source,current)
            return self.status(pid)

    def _ingest(self, pid, source, values):
        result = self.connections.ingest(pid, source, values)
        with self.repo.connection() as db:
            connection = self.connections._get(db, pid, source)
            connection['last_sync_at'] = self.clock.stamp()
            self.connections._save(db, connection)
        return result

    def load_history(self, pid):
        with self.context(pid):
            session = self.sessions[pid]
            if session['history_loaded']:
                return self.status(pid)
            if session['status'] != 'STOPPED' or session['physiology']['step'] != 0:
                raise ConnectionError('Choose prior history before starting this simulation.')
            source = self.connections.status(pid, 'CGM')
            if source['provider_type'] != PROVIDERS['CGM'] or source['status'] not in ('CONNECTED', 'STALE'):
                raise ConnectionError('Connect Simulated Live CGM before loading prior history.')
            end = clock(session['simulation_time'])
            # 55 hours at native cadence. Receipt is the current import time;
            # P5 decides whether this actual causal history supports personalization.
            physiology = initial(end - timedelta(hours=55), session['profile_id'])
            batch = []
            for i in range(221):
                if i:
                    physiology = evolve(physiology, session['profile_id'], session['seed'])
                frame = {**session, 'physiology': physiology}
                batch.extend(events(pid, frame, 'CGM', received_at=end, prior=True))
                if len(batch) == 100 or i == 220:
                    self._ingest(pid, 'CGM', batch); batch = []
            session['physiology'] = physiology; session['history_loaded'] = True
            self._save(session)
            return self.status(pid)

    def _connected(self, pid, source):
        value = self.connections.status(pid, source)
        return value['provider_type'] == PROVIDERS[source] and value['status'] in ('CONNECTED', 'STALE')

    def control(self, pid, action, value=None):
        with self.context(pid):
            if pid not in self.sessions:
                raise ConnectionError('Connect a simulated live source first.')
            session = self.sessions[pid]
            if action in ('start', 'resume'):
                if not self._connected(pid, 'CGM'):
                    raise ConnectionError('Connect Simulated Live CGM before starting.')
                if session['status'] == 'RUNNING':
                    return self.status(pid)
                session.update(status='RUNNING', error=None, restart_notice=None, disconnected_at=None,
                               wall_clock_started_at=datetime.now(timezone.utc).isoformat())
                # First Start receives a real initial reading; readiness is never forced.
                if session['physiology']['step'] == 0:
                    self._ingest(pid, 'CGM', events(pid, session, 'CGM'))
                session['next_due_wall'] = time.time() + 900 / session['speed_multiplier']
            elif action in ('pause', 'stop'):
                session.update(status='PAUSED' if action == 'pause' else 'STOPPED', next_due_wall=None)
            elif action == 'speed':
                if value not in SPEEDS:
                    raise ValueError('Choose 1x, 4x, 15x, 60x or DEMO')
                old = session['speed_multiplier']
                remaining = max(0., session['next_due_wall'] - time.time()) * old if session['next_due_wall'] else None
                session.update(speed=value, speed_multiplier=SPEEDS[value])
                if remaining is not None:
                    session['next_due_wall'] = time.time() + remaining / session['speed_multiplier']
            elif action == 'activity':
                if value not in ('Resting', 'Walking'):
                    raise ValueError('Choose Resting or Walking')
                session['physiology']['activity'] = value
            else:
                raise ValueError('Unknown simulation control')
            self._save(session); self.wakeup.set()
            return self.status(pid)

    def source_disconnected(self, pid, source):
        session = self.sessions.get(pid)
        if not session:
            return
        if source == 'CGM':
            session.update(status='STOPPED', next_due_wall=None, disconnected_at=datetime.now(timezone.utc).isoformat())
            self._save(session); self.wakeup.set()

    def step(self, pid):
        """Backend-only native tick. Scheduler and engineering tests share this path."""
        with self.lock, self.runtime._lock:
            session = self.sessions[pid]
            if session['status'] != 'RUNNING':
                raise ConnectionError('Start or resume the simulation before advancing.')
            with self.clock.at(clock(session['simulation_time'])):
                if not self._connected(pid, 'CGM'):
                    self.source_disconnected(pid, 'CGM'); return self.status(pid)
            # Inputs available by the new sensor time may affect that new
            # observation, never the already persisted preceding observation.
            next_reference=clock(session['simulation_time'])+timedelta(minutes=15)
            with self.clock.at(next_reference):
                meals = self.runtime.meals.recent_context(pid,next_reference.isoformat(),lookback_minutes=480)['meals']
                register_meals(session['physiology'],meals,next_reference)
                next_state = evolve(session['physiology'], session['profile_id'], session['seed'])
            frame = {**session, 'physiology': next_state, 'simulation_time': next_state['time']}
            with self.clock.at(clock(frame['simulation_time'])):
                if self._connected(pid, 'WEARABLE'):
                    self._ingest(pid, 'WEARABLE', events(pid, frame, 'WEARABLE'))
                result = self._ingest(pid, 'CGM', events(pid, frame, 'CGM'))
                # Crash-retry duplicate readings still reconcile the persisted Twin.
                if not result['events_received']:
                    self.runtime.run_twin_inference(pid, frame['simulation_time'])
                frame['last_inference'] = result.get('inference')
            frame['next_due_wall'] = time.time() + 900 / frame['speed_multiplier']
            self._save(frame)
            return self.status(pid)

    def status(self, pid):
        with self.lock, self.runtime._lock:
            self.repo.get_patient(pid)
            session = self.sessions.get(pid)
            if not session:
                return {'patient_id': pid, 'status': 'NOT_CONFIGURED'}
            seconds = max(0., session['next_due_wall'] - time.time()) if session['next_due_wall'] else None
            result = {k: session.get(k) for k in ('simulation_id', 'patient_id', 'status', 'simulation_time', 'speed',
                      'speed_multiplier', 'profile_id', 'history_loaded', 'created_at', 'updated_at', 'error', 'restart_notice')}
            result.update(last_cgm=session['physiology']['glucose'],
                          next_event_time=(clock(session['simulation_time']) + timedelta(minutes=15)).isoformat(),
                          next_reading_wall_seconds=seconds,
                          next_reading_simulated_minutes=seconds * session['speed_multiplier'] / 60 if seconds is not None else None,
                          active_meal_influences_count=len(session['physiology']['meal_effects']),
                          wearable_state={k: session['physiology'][k] for k in ('heart_rate', 'steps_today', 'active_minutes_today', 'activity')},
                          provenance='Synthetic physiology for product simulation; not clinically validated')
            with self.clock.at(self.reference(pid)):
                latest=self.runtime.latest(pid)
                state=latest.get('twin_state')
                result['latest_twin_state']={k:state[k] for k in ('twin_state_id','prediction_id','reference_time','forecast','spike','status')} if state else None
                result['active_alert']=self.runtime.alerts.active(pid)['alert']
            check_owner(result, pid); return result

    def start_worker(self):
        if self.worker and self.worker.is_alive():
            return
        self.worker = threading.Thread(target=self._schedule, name='twincare-live-scheduler', daemon=True)
        self.worker.start()

    def _schedule(self):
        while not self.closed.is_set():
            with self.lock:
                due = [pid for pid, s in self.sessions.items() if s['status'] == 'RUNNING'
                       and s['next_due_wall'] is not None and s['next_due_wall'] <= time.time()]
            for pid in due:
                try:
                    with self.lock:
                        if self.sessions[pid]['status'] == 'RUNNING':
                            self.step(pid)
                except Exception as error:
                    with self.lock:
                        session = self.sessions[pid]
                        session.update(status='ERROR', next_due_wall=None,
                                       error='Live source update failed (' + type(error).__name__ + '). Reconnect or resume to retry.')
                        self._save(session)
            self.wakeup.wait(.25); self.wakeup.clear()

    def close(self):
        self.closed.set(); self.wakeup.set()
        if self.worker:
            self.worker.join(timeout=10)
