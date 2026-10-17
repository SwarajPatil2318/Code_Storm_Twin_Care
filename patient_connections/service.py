"""Transactional P1 connection/observation extension with patient-owned queries."""
from datetime import datetime, timezone, timedelta
import json
from src.patient_domain.models import (DeviceConnection, Observation, AuditEvent, identifier,
                                      now, owned_view, json_safe)
from src.patient_domain.repository import Repository, NotFound
from .contracts import CanonicalEvent, SOURCES, clock, TYPES
from .adapters import ReplayAdapter, WearableAdapter, SCENARIOS


class ConnectionError(ValueError):
    def __init__(self, message, status=409): self.status=status;super().__init__(message)


class Connections:
    def __init__(self, repository=None, adapters=None, time=None):
        self.repo=repository or Repository()
        self.adapters=adapters if adapters is not None else {'CGM':ReplayAdapter(),'WEARABLE':WearableAdapter()}
        self.time=time or (lambda:datetime.now(timezone.utc))
        with self.repo.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS connections_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)')
            versions=[r[0] for r in db.execute('SELECT version FROM connections_migrations')]
            if any(v!=1 for v in versions): raise ValueError('Unknown connections migration')
            if not versions:
                db.execute("CREATE UNIQUE INDEX p3_connection_source ON device_connections(patient_id,json_extract(data,'$.source_type')) WHERE json_extract(data,'$.connection_schema_version')=3")
                # P1 stores timestamps inside JSON; no second observations table is needed.
                db.execute("CREATE INDEX p3_observed_at ON observations(patient_id,json_extract(data,'$.observed_time'))")
                for op in ('INSERT','UPDATE'):
                    db.execute(f"""CREATE TRIGGER p3_event_owner_{op.lower()} BEFORE {op} ON observations
                        WHEN json_extract(NEW.data,'$.source_provenance.canonical_event.schema_version') IS NOT NULL
                        AND (coalesce(json_extract(NEW.data,'$.source_provenance.canonical_event.patient_id')=NEW.patient_id,0)=0
                        OR coalesce(json_extract(NEW.data,'$.source_provenance.canonical_event.event_id')=NEW.entity_id,0)=0)
                        BEGIN SELECT RAISE(ABORT,'Canonical owner mismatch'); END""")
                for table in ('observations','device_connections'):
                    for op in ('INSERT','UPDATE'):
                        db.execute(f"""CREATE TRIGGER p3_{table}_owner_{op.lower()} BEFORE {op} ON {table}
                            WHEN coalesce(json_extract(NEW.data,'$.patient_id')=NEW.patient_id,0)=0
                            OR coalesce(json_extract(NEW.data,'$.entity_id')=NEW.entity_id,0)=0
                            BEGIN SELECT RAISE(ABORT,'Owner mismatch'); END""")
                db.execute('INSERT INTO connections_migrations VALUES(1,?)',(now(),))

    def _stamp(self):return self.time().isoformat()
    def _audit(self,db,pid,source,action,details=None):
        value=owned_view(AuditEvent(patient_id=pid,entity_id=identifier('entity'),action=action,
            entity_type='data_connection',target_id=source,details=details or {},synthetic=self.repo._require(db,pid)['synthetic']))
        db.execute('INSERT INTO audit_events VALUES(?,?,?)',(value['entity_id'],pid,json_safe(value)))
    def _get(self,db,pid,source):
        self.repo._require(db,pid)
        if source not in SOURCES:raise NotFound('Source not found')
        row=db.execute("SELECT data FROM device_connections WHERE patient_id=? AND json_extract(data,'$.source_type')=? AND json_extract(data,'$.connection_schema_version')=3",(pid,source)).fetchone()
        return json.loads(row[0]) if row else None
    def _save(self,db,value):
        value['updated_at']=self._stamp()
        db.execute('INSERT INTO device_connections VALUES(?,?,?) ON CONFLICT(entity_id) DO UPDATE SET data=excluded.data WHERE patient_id=excluded.patient_id',
                   (value['entity_id'],value['patient_id'],json_safe(value)))
    def _empty(self,pid,source):
        return {'connection_id':None,'patient_id':pid,'source_type':source,'provider_type':None,'status':'NOT_CONNECTED',
                'connected_at':None,'last_sync_at':None,'last_event_at':None,'event_count':0,'data_start_at':None,
                'data_end_at':None,'quality_status':'INSUFFICIENT','metadata':{},'provenance':{},'created_at':None,'updated_at':None}
    def _events(self,db,pid,source=None,kind=None,start=None,end=None):
        self.repo._require(db,pid)
        if source is not None and source not in SOURCES:raise ValueError('Unknown source')
        if kind is not None and kind not in TYPES:raise ValueError('Unknown event type')
        params=[pid];sql="SELECT data FROM observations WHERE patient_id=? AND json_extract(data,'$.source_provenance.canonical_event.schema_version')='twincare-event/1'"
        for value,path in [(source,'source_type'),(kind,'event_type')]:
            if value is not None:sql+=f" AND json_extract(data,'$.source_provenance.canonical_event.{path}')=?";params.append(value)
        if start:sql+=" AND json_extract(data,'$.observed_time')>=?";params.append(clock(start).isoformat())
        if end:sql+=" AND json_extract(data,'$.observed_time')<=?";params.append(clock(end).isoformat())
        if start and end and clock(start)>clock(end):raise ValueError('Invalid time range')
        sql+=" ORDER BY json_extract(data,'$.observed_time'),entity_id"
        return [json.loads(r[0])['source_provenance']['canonical_event'] for r in db.execute(sql,params)]
    def observations(self,pid,source=None,event_type=None,start=None,end=None,limit=100,recent=False):
        if type(limit) is not int or not 1<=limit<=500:raise ValueError('Limit must be 1–500')
        with self.repo.connection() as db:events=self._events(db,pid,source,event_type,start,end)
        return {'patient_id':pid,'count':len(events),'data_start_at':events[0]['observed_at'] if events else None,
                'data_end_at':events[-1]['observed_at'] if events else None,'latest':events[-1] if events else None,
                'items':events[-limit:] if recent else events[:limit],'active_model_input':False}
    def _quality(self,source,events):
        if not events:return 'INSUFFICIENT', ['No observations received']
        if self.time()-clock(events[-1]['observed_at'])>timedelta(hours=1 if source=='CGM' else 24):return 'STALE',['Last source observation is historical or outside the recency window']
        if any(e['quality']['status']!='GOOD' for e in events):return 'PARTIAL',['Missing or suspect observations present']
        if source=='CGM':
            if len(events)<9 or clock(events[-1]['observed_at'])-clock(events[0]['observed_at'])<timedelta(hours=2):return 'INSUFFICIENT',['Less than two hours of native CGM history']
            gaps=[(clock(b['observed_at'])-clock(a['observed_at'])).total_seconds()/60 for a,b in zip(events,events[1:])]
            if any(abs(g-15)>1 for g in gaps):return 'PARTIAL',['Cadence differs from the supplied native 15-minute sequence']
        elif source=='WEARABLE' and not {'HEART_RATE','ACTIVE_MINUTES'}<={e['event_type'] for e in events}:
            return 'PARTIAL',['Heart rate or active-minutes context is unavailable']
        return 'GOOD',[]
    def _refresh(self,db,pid,source):
        value=self._get(db,pid,source)
        if not value:return self._empty(pid,source)
        lease=value['metadata'].get('sync_started_at')
        if value['status']=='SYNCING' and lease and self.time()-clock(lease)>timedelta(minutes=2):
            value.update({'status':'ERROR','error':'The previous sync did not finish. Try syncing again.'})
            self._audit(db,pid,source,'SOURCE_SYNC_FAILED',{'reason':'sync_lease_expired'})
        events=self._events(db,pid,source)
        quality,reasons=self._quality(source,events)
        value.update({'event_count':len(events),'data_start_at':events[0]['observed_at'] if events else None,
            'data_end_at':events[-1]['observed_at'] if events else None,'last_event_at':events[-1]['observed_at'] if events else None,
            'quality_status':quality,'quality_reasons':reasons})
        if value['status'] in ('CONNECTED','STALE'):value['status']='STALE' if quality=='STALE' else 'CONNECTED'
        self._save(db,value);return value
    def _ehr(self,db,pid):
        self.repo._require(db,pid)
        value=self._empty(pid,'EHR');value['provider_type']='medical_records'
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE name='ingestion_documents'").fetchone()
        docs=[json.loads(r[0]) for r in db.execute('SELECT data FROM ingestion_documents WHERE patient_id=?',(pid,))] if exists else []
        profile=json.loads(db.execute('SELECT data FROM clinical_profiles WHERE patient_id=?',(pid,)).fetchone()[0])
        ready=bool(profile.get('confirmation_history'))
        pending=any(d['processing_status'] not in ('CONFIRMED','FAILED') for d in docs)
        failed=sum(d['processing_status']=='FAILED' for d in docs)
        value.update({'status':'CONNECTED' if ready else 'ERROR' if failed and not pending else 'NOT_CONNECTED','display_status':'Ready · review pending' if ready and pending else 'Ready' if ready else 'Needs review' if pending else 'Needs attention' if failed else 'Not set up',
            'event_count':len(docs),'quality_status':'GOOD' if ready else 'INSUFFICIENT','last_sync_at':profile['updated_at'] if ready else None,
            'metadata':{'record_count':len(docs),'confirmed_count':sum(d['processing_status']=='CONFIRMED' for d in docs),'review_pending':pending,'failed_count':failed},
            'provenance':{'origin':'P2 confirmed ClinicalProfile and actual ingestion documents; read only'}})
        return value
    def list(self,pid):
        with self.repo.connection() as db:
            return {'patient_id':pid,'items':[self._ehr(db,pid) if source=='EHR' else self._refresh(db,pid,source) for source in SOURCES]}
    def status(self,pid,source):
        with self.repo.connection() as db:return self._ehr(db,pid) if source=='EHR' else self._refresh(db,pid,source)
    def connect(self,pid,source,config):
        if source not in ('CGM','WEARABLE'):raise ConnectionError('This source is managed through its own product workflow.')
        if set(config)!={'scenario','acknowledge_synthetic'} or config['scenario'] not in SCENARIOS or config['acknowledge_synthetic'] is not True:
            raise ConnectionError('Choose an available source and acknowledge its synthetic provenance.',400)
        with self.repo.connection() as db:
            patient=self.repo._require(db,pid);previous=self._get(db,pid,source)
            if previous and previous['status'] in ('CONNECTING','SYNCING'):raise ConnectionError('Source operation already in progress.')
            if previous and previous['metadata']['scenario']!=config['scenario']:raise ConnectionError('This patient already has history from a different scenario. Reconnect the same source.')
            if previous and previous['status'] in ('CONNECTED','STALE'):return previous
            base=owned_view(DeviceConnection(patient_id=pid,entity_id=previous['entity_id'] if previous else identifier('entity'),
                device_type=source.lower(),source='P3 patient source adapter',synthetic=True))
            value={**self._empty(pid,source),**base,**(previous or {}),'connection_id':base['entity_id'],'source_type':source,
                'provider_type':'replay_adapter' if source=='CGM' else 'simulated_wearable','connection_schema_version':3,
                'status':'CONNECTING','connected_at':previous.get('connected_at') if previous else None,
                'last_sync_at':previous.get('last_sync_at') if previous else None,
                'metadata':{**(previous or {}).get('metadata',{}),'scenario':config['scenario'],'acknowledge_synthetic':True},
                'provenance':{'origin':'existing synthetic project sources','synthetic':True,'commercial_provider_connected':False},
                'event_count':(previous or {}).get('event_count',0),'quality_status':(previous or {}).get('quality_status','INSUFFICIENT')}
            self._save(db,value);return value
    def activate(self,pid,source):
        with self.repo.connection() as db:
            value=self._get(db,pid,source)
            if not value or value['status']!='CONNECTING':raise ConnectionError('Start connecting this source first.')
            try:
                adapter=self.adapters.get(source)
                if adapter is None:raise ValueError('Unavailable')
                adapter.connect(pid,value['metadata']);value.update({'status':'CONNECTED','connected_at':value.get('connected_at') or self._stamp(),'error':None})
                self._audit(db,pid,source,'SOURCE_CONNECTED')
            except Exception:
                value.update({'status':'ERROR','error':'The data source could not be connected. Try again.'})
                self._audit(db,pid,source,'SOURCE_CONNECTION_FAILED')
            self._save(db,value);return value
    def disconnect(self,pid,source):
        with self.repo.connection() as db:
            value=self._get(db,pid,source)
            if not value:raise ConnectionError('This source is not connected.')
            if value['status']=='SYNCING':raise ConnectionError('Wait for the current sync before disconnecting.')
            adapter=self.adapters.get(source)
            if adapter:adapter.disconnect(pid)
            value['status']='DISCONNECTED';self._save(db,value);self._audit(db,pid,source,'SOURCE_DISCONNECTED')
            return value
    def _insert(self,db,pid,source,event):
        data=dict(event);data.pop('active_model_input',None)
        value=CanonicalEvent(**data).view(as_of=self.time())
        if value['patient_id']!=pid or value['source_type']!=source:raise ValueError('Owner/source mismatch')
        con=self._get(db,pid,source)
        if value['provider_type']!=con['provider_type']:raise ValueError('Provider mismatch')
        if value['provenance']['synthetic']!=con['provenance']['synthetic']:raise ValueError('Adapter provenance mismatch')
        event_id=value['event_id'];exists=db.execute('SELECT patient_id,data FROM observations WHERE entity_id=?',(event_id,)).fetchone()
        if exists:
            if exists[0]!=pid:raise ValueError('Owner mismatch')
            previous=json.loads(exists[1])['source_provenance']['canonical_event']
            if previous['payload']!=value['payload']:raise ValueError('Existing source event has different content')
            return False
        key,unit=TYPES[value['event_type']][1:]
        observation=Observation(patient_id=pid,entity_id=event_id,concept=value['event_type'],raw_value=value['payload'],
            original_unit=value['provenance'].get('original_unit',unit),observed_time=value['observed_at'],available_time=value['received_at'],
            source_provenance={'canonical_event':value},usable_value=value['payload'][key],canonical_unit=unit,
            quality_status=value['quality']['status'],quality_reasons=value['quality'].get('reasons',[]),synthetic=value['provenance']['synthetic'])
        db.execute('INSERT INTO observations VALUES(?,?,?)',(event_id,pid,json_safe(owned_view(observation))))
        return True
    def ingest(self,pid,source,events):
        if not isinstance(events,list) or not 1<=len(events)<=100:raise ValueError('Supply 1–100 canonical events')
        with self.repo.connection() as db:
            value=self._get(db,pid,source)
            if not value or value['status'] not in ('CONNECTED','STALE'):raise ConnectionError('Connect the source before adding observations.')
            count=sum(self._insert(db,pid,source,e) for e in events)
            self._audit(db,pid,source,'EVENT_INGESTED',{'events_received':count,'duplicates':len(events)-count})
            return {'events_received':count,'duplicates':len(events)-count,'connection':self._refresh(db,pid,source)}
    def sync(self,pid,source):
        with self.repo.connection() as db:
            value=self._get(db,pid,source)
            if not value or value['status'] not in ('CONNECTED','STALE','ERROR'):raise ConnectionError('Connect this source before syncing.')
            if not value.get('connected_at'):raise ConnectionError('Finish connecting before syncing.')
            value['status']='SYNCING';value['metadata'].update({'sync_token':identifier('entity'),'sync_started_at':self._stamp()})
            self._save(db,value);self._audit(db,pid,source,'SOURCE_SYNC_STARTED')
        try:
            adapter=self.adapters.get(source)
            if not adapter:raise ValueError('Adapter unavailable')
            events,cursor,exhausted=adapter.sync(pid,value,self._stamp())
            with self.repo.connection() as db:
                current=self._get(db,pid,source)
                if current['status']!='SYNCING' or current['metadata'].get('sync_token')!=value['metadata']['sync_token']:
                    raise ConnectionError('Source state changed during sync.')
                count=sum(self._insert(db,pid,source,e) for e in events)
                current.update({'status':'CONNECTED','last_sync_at':self._stamp(),'error':None})
                current['metadata'].update({'cursor':cursor,'exhausted':exhausted,'last_sync_result':{'events_received':count,'duplicates':len(events)-count,'status':'completed'}})
                self._save(db,current);self._audit(db,pid,source,'EVENT_INGESTED',{'events_received':count,'duplicates':len(events)-count})
                self._audit(db,pid,source,'SOURCE_SYNC_COMPLETED',{'events_received':count})
                return {'events_received':count,'duplicates':len(events)-count,'connection':self._refresh(db,pid,source)}
        except Exception:
            with self.repo.connection() as db:
                current=self._get(db,pid,source)
                if current and current['status']=='SYNCING' and current['metadata'].get('sync_token')==value['metadata']['sync_token']:
                    current.update({'status':'ERROR','error':f'{source.title()} data could not be synced. Try again.'})
                    current['metadata']['last_sync_result']={'status':'failed','events_received':0}
                    self._save(db,current);self._audit(db,pid,source,'SOURCE_SYNC_FAILED')
            raise ConnectionError(f'{source.title()} data could not be synced. Try again.',503) from None
