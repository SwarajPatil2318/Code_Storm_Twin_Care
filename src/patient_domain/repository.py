"""SQLite repository: each child query requires a patient ID; no current-patient state."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
from .models import (Patient, ClinicalProfile, ENTITY_TYPES, AuditEvent, identifier,
                     json_safe, now, owned_view, validate_id)

DEFAULT_DB = Path('data/patient_domain/twincare.sqlite3')
TABLES = {name: name.replace('-', '_') for name in ENTITY_TYPES}


class NotFound(LookupError):
    pass


class Repository:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        if str(path) == ':memory:':
            raise ValueError('Use an isolated file database for reproducible tests')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.resolve() == DEFAULT_DB.parent.resolve():
            os.chmod(self.path.parent, 0o700)
        with self.connection() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('Unsupported database version')
            if version == 0:
                if db.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]:
                    raise ValueError('Refusing to migrate an unrecognized database')
                self._migrate(db)
        os.chmod(self.path, 0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _migrate(self, db):
        db.execute('CREATE TABLE migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)')
        db.execute('''CREATE TABLE patients (patient_id TEXT PRIMARY KEY, data TEXT NOT NULL
                   CHECK(json_valid(data) AND json_extract(data,'$.patient_id')=patient_id))''')
        db.execute('''CREATE TABLE clinical_profiles (patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id)
                   ON DELETE CASCADE, data TEXT NOT NULL CHECK(json_valid(data) AND
                   json_extract(data,'$.patient_id')=patient_id))''')
        for name in TABLES.values():
            # A global entity ID is immutable in ownership. Composite keys support same-owner references.
            extras = ''
            ref_column = ''
            if name == 'predictions':
                ref_column = ', twin_state_id TEXT'
                extras = ', FOREIGN KEY(patient_id,twin_state_id) REFERENCES twin_states(patient_id,entity_id)'
            elif name == 'alerts':
                ref_column = ', prediction_id TEXT'
                extras = ', FOREIGN KEY(patient_id,prediction_id) REFERENCES predictions(patient_id,entity_id)'
            db.execute(f'''CREATE TABLE {name} (entity_id TEXT PRIMARY KEY,
                         patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE{ref_column},
                         data TEXT NOT NULL CHECK(json_valid(data) AND json_extract(data,'$.patient_id')=patient_id
                         AND json_extract(data,'$.entity_id')=entity_id), UNIQUE(patient_id,entity_id){extras})''')
            db.execute(f'CREATE INDEX {name}_owner ON {name}(patient_id)')
        db.execute('INSERT INTO migrations VALUES(1,?)', (now(),))
        db.execute('PRAGMA user_version=1')

    @staticmethod
    def _require(db, patient_id):
        validate_id(patient_id)
        row = db.execute('SELECT data FROM patients WHERE patient_id=?', (patient_id,)).fetchone()
        if not row:
            raise NotFound('Patient not found')
        return json.loads(row[0])

    def _audit(self, db, patient_id, action, entity_type, target_id, fields=()):
        event = AuditEvent(patient_id=patient_id, entity_id=identifier('entity'), action=action,
                           entity_type=entity_type, target_id=target_id,
                           synthetic=self._require(db, patient_id)['synthetic'],
                           details={'field_names': sorted(fields)})
        db.execute('INSERT INTO audit_events(entity_id,patient_id,data) VALUES(?,?,?)',
                   (event.entity_id, patient_id, json_safe(owned_view(event))))

    def create_patient(self, **values):
        if set(values) - {'display_name', 'date_of_birth', 'sex', 'status', 'synthetic', 'patient_id'}:
            raise ValueError('Unsupported patient fields')
        patient = Patient(**{'patient_id': identifier('patient'), **values})
        patient.validate()
        with self.connection() as db:
            db.execute('INSERT INTO patients VALUES(?,?)', (patient.patient_id, json_safe(patient.view())))
            profile = ClinicalProfile(patient_id=patient.patient_id).view()
            db.execute('INSERT INTO clinical_profiles VALUES(?,?)', (patient.patient_id, json_safe(profile)))
            self._audit(db, patient.patient_id, 'create', 'patient', patient.patient_id, values)
        return patient.view()

    def list_patients(self):
        with self.connection() as db:
            rows = db.execute('SELECT data FROM patients ORDER BY patient_id').fetchall()
        return [self._patient(json.loads(row[0])).view() for row in rows]

    @staticmethod
    def _patient(value):
        return Patient(**{k: v for k, v in value.items() if k not in ('age_years', 'age_as_of')})

    def get_patient(self, patient_id):
        with self.connection() as db:
            value = self._require(db, patient_id)
        return self._patient(value).view()

    def update_patient(self, patient_id, **changes):
        if not changes or set(changes) - {'display_name', 'date_of_birth', 'sex', 'status'}:
            raise ValueError('Only mutable basic patient fields may be updated')
        with self.connection() as db:
            patient = replace(self._patient(self._require(db, patient_id)), **changes, updated_at=now())
            patient.validate()
            db.execute('UPDATE patients SET data=? WHERE patient_id=?', (json_safe(patient.view()), patient_id))
            self._audit(db, patient_id, 'update', 'patient', patient_id, changes)
        return patient.view()

    def get_profile(self, patient_id):
        with self.connection() as db:
            self._require(db, patient_id)
            row = db.execute('SELECT data FROM clinical_profiles WHERE patient_id=?', (patient_id,)).fetchone()
        return json.loads(row[0])

    def put_profile(self, profile):
        value = profile.view()
        json_safe(value)
        with self.connection() as db:
            self._require(db, profile.patient_id)
            for item in profile.fields.values():
                if item.source_document_id:
                    if not db.execute('SELECT 1 FROM documents WHERE patient_id=? AND entity_id=?',
                                      (profile.patient_id, item.source_document_id)).fetchone():
                        raise ValueError('Source document is not owned by this patient')
            db.execute('UPDATE clinical_profiles SET data=? WHERE patient_id=?', (json_safe(value), profile.patient_id))
            self._audit(db, profile.patient_id, 'update', 'clinical_profile', profile.patient_id, profile.fields)
        return value

    def save(self, kind, entity):
        if kind not in ENTITY_TYPES or type(entity) is not ENTITY_TYPES[kind]:
            raise ValueError('Entity type mismatch')
        value = owned_view(entity)
        table = TABLES[kind]
        with self.connection() as db:
            self._require(db, entity.patient_id)
            row = db.execute(f'SELECT patient_id FROM {table} WHERE entity_id=?', (entity.entity_id,)).fetchone()
            if row and row[0] != entity.patient_id:
                raise ValueError('Cannot reassign record ownership')
            columns = ['entity_id', 'patient_id', 'data']
            vals = [entity.entity_id, entity.patient_id, json_safe(value)]
            for ref, reference_table in (('twin_state_id', 'twin_states'), ('prediction_id', 'predictions')):
                if ref in value:
                    if value[ref] and not db.execute(f'SELECT 1 FROM {reference_table} WHERE patient_id=? AND entity_id=?',
                                                   (entity.patient_id, value[ref])).fetchone():
                        raise ValueError('Referenced entity is not owned by this patient')
                    columns.append(ref)
                    vals.append(value[ref])
            update = ','.join(f'{c}=excluded.{c}' for c in columns[2:])
            db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in vals)}) "
                       f'ON CONFLICT(entity_id) DO UPDATE SET {update}', vals)
            if kind != 'audit-events':
                self._audit(db, entity.patient_id, 'store', kind, entity.entity_id)
        return value

    def list_owned(self, patient_id, kind):
        if kind not in TABLES:
            raise ValueError('Unknown collection')
        with self.connection() as db:
            self._require(db, patient_id)
            rows = db.execute(f'SELECT data FROM {TABLES[kind]} WHERE patient_id=? ORDER BY entity_id', (patient_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_owned(self, patient_id, kind, entity_id):
        if kind not in TABLES:
            raise ValueError('Unknown collection')
        with self.connection() as db:
            self._require(db, patient_id)
            row = db.execute(f'SELECT data FROM {TABLES[kind]} WHERE patient_id=? AND entity_id=?',
                             (patient_id, entity_id)).fetchone()
            if not row:
                raise NotFound('Record not found')
        return json.loads(row[0])

    def reset_synthetic(self):
        """CLI/test only. Refuse to remove even one non-synthetic patient's records."""
        with self.connection() as db:
            if db.execute("SELECT 1 FROM patients WHERE json_extract(data,'$.synthetic') != 1").fetchone():
                raise ValueError('Reset refused: database contains non-synthetic patients')
            db.execute('DELETE FROM patients')
