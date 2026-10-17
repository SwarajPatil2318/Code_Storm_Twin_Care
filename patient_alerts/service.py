"""Persisted P6 state -> exact frozen policy -> patient-owned alert/outbox."""
from datetime import timezone
import json
import math
from pathlib import Path
import time
import numpy as np
from patient_connections.contracts import clock,check_owner
from src.patient_domain.models import Alert,AuditEvent,identifier,owned_view,json_safe
from src.patient_domain.repository import NotFound
from .providers import TwinCareWearableAdapter

ROOT=Path(__file__).resolve().parents[1]


class Alerts:
    def __init__(self,inference,provider=None):
        self.inference=inference;self.repo=inference.repo;self.time=inference.readiness.time
        self.provider=provider or TwinCareWearableAdapter()
        self.policy=json.loads((ROOT/'artifacts/productization_p7/alert_policy.json').read_text())
        risk=inference.contract['risk']
        if risk['threshold']!=self.policy['threshold'] or risk['sha256']!=self.policy['risk_model_sha256']:
            raise ValueError('Frozen alert policy mismatch')
        self.metrics=[]
        with self.repo.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS alert_policy_states(
                patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id) ON DELETE CASCADE,
                data TEXT NOT NULL CHECK(json_valid(data)))''')
            db.execute('''CREATE TABLE IF NOT EXISTS alert_evaluations(
                patient_id TEXT NOT NULL,twin_state_id TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),
                PRIMARY KEY(patient_id,twin_state_id),
                FOREIGN KEY(patient_id,twin_state_id) REFERENCES twin_states(patient_id,entity_id) ON DELETE CASCADE)''')
            db.execute('''CREATE TABLE IF NOT EXISTS notification_deliveries(
                notification_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,alert_id TEXT NOT NULL,
                provider TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),UNIQUE(patient_id,alert_id,provider),
                FOREIGN KEY(patient_id,alert_id) REFERENCES alerts(patient_id,entity_id) ON DELETE CASCADE)''')
            for table in ('alert_policy_states','alert_evaluations','notification_deliveries'):
                for operation in ('INSERT','UPDATE'):
                    db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_owner_{operation.lower()} BEFORE {operation} ON {table}
                        WHEN coalesce(json_extract(NEW.data,'$.patient_id')=NEW.patient_id,0)=0
                        BEGIN SELECT RAISE(ABORT,'Alert owner mismatch'); END""")

    def stamp(self):return self.time().astimezone(timezone.utc).isoformat()

    def _audit(self,db,pid,action,target,details=None):
        patient=self.repo._require(db,pid)
        value=owned_view(AuditEvent(patient_id=pid,entity_id=identifier('entity'),created_at=self.stamp(),
            synthetic=patient['synthetic'],action=action,entity_type='alert',target_id=target,details=details or {}))
        db.execute('INSERT INTO audit_events VALUES(?,?,?)',(value['entity_id'],pid,json_safe(value)))

    def _get(self,db,pid,aid):
        self.repo._require(db,pid)
        row=db.execute('SELECT data FROM alerts WHERE patient_id=? AND entity_id=?',(pid,aid)).fetchone()
        if not row:raise NotFound('Alert not found')
        value=json.loads(row[0])
        if value.get('alert_schema_version')!=7:raise NotFound('P7 alert not found')
        return value

    def _update(self,db,value):
        check_owner(value,value['patient_id'])
        db.execute('UPDATE alerts SET data=? WHERE patient_id=? AND entity_id=?',
            (json_safe(value),value['patient_id'],value['alert_id']))

    def _source(self,db,pid,sid):
        self.repo._require(db,pid)
        row=db.execute('SELECT rowid,data FROM twin_states WHERE patient_id=? AND entity_id=?',(pid,sid)).fetchone()
        if not row:raise NotFound('Twin state not found')
        stored=json.loads(row['data']);state=stored.get('payload',{})
        if stored.get('module')!='metabolic' or state.get('schema_version')!='twincare-metabolic-twin/6':
            raise ValueError('Persisted generalized P6 state required')
        prediction=db.execute('SELECT data FROM predictions WHERE patient_id=? AND entity_id=? AND twin_state_id=?',
            (pid,state.get('prediction_id'),sid)).fetchone()
        trace=db.execute('SELECT data FROM inference_traces WHERE patient_id=? AND prediction_id=?',
            (pid,state.get('prediction_id'))).fetchone()
        if not prediction or not trace:raise ValueError('P6 prediction/trace linkage required')
        prediction=json.loads(prediction[0]);trace=json.loads(trace[0])
        if trace['twin_state_id']!=sid or state['twin_state_id']!=sid or state['trace_id']!=trace['trace_id']:
            raise ValueError('P6 linkage mismatch')
        if prediction['output']['spike']!=state['spike'] or trace['output']['spike']!=state['spike'] or prediction['output']['forecast']!=state['forecast'] or trace['output']['forecast']!=state['forecast'] or prediction['output']['predicted_peak']!=state['predicted_peak']:
            raise ValueError('P6 output mismatch')
        current=state.get('current_glucose')
        if type(current) not in (int,float) or not math.isfinite(current) or np.float32(current)!=np.float32(trace['features']['values'][8]):
            raise ValueError('P6 observed-current/feature linkage mismatch')
        for value in (state,prediction,trace):check_owner(value,pid)
        return row['rowid'],stored,state,trace

    def _eligible(self,state,trace):
        risk=state['spike']
        if risk.get('available') is not True:return 'RISK_UNAVAILABLE'
        probability=risk.get('probability')
        if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<=probability<=1:return 'INVALID_RISK'
        if risk['threshold']!=self.policy['threshold'] or risk['model_version']!=self.policy['risk_model_version']:return 'POLICY_MISMATCH'
        if trace.get('risk_model_sha256')!=self.policy['risk_model_sha256'] or not trace.get('artifact_execution',{}).get('risk',{}).get('executed'):return 'MODEL_PROVENANCE_UNAVAILABLE'
        if state['current_glucose']>180:return 'INCIDENT_RULE_NOT_APPLICABLE'
        freshness=state['freshness']['CGM'];reference=clock(state['reference_time'])
        age=freshness.get('latest_reading_age_minutes');now=self.time()
        if reference>now:return 'FUTURE_STATE'
        if freshness.get('stale') or type(age) not in (int,float) or not math.isfinite(age):return 'STALE_STATE'
        if age+(now-reference).total_seconds()/60>self.inference.contract['readiness']['stale_after_minutes']:return 'STALE_STATE'
        if not state['readiness']['capabilities']['short_horizon_forecast']:return 'STATE_NOT_READY'
        live=self.inference.readiness.evaluate(state['patient_id'],now.isoformat())
        if not live['capabilities']['short_horizon_forecast']:return 'LIVE_SOURCE_NOT_READY'
        return None

    def evaluate_alerts(self,pid,twin_state_id):
        started=time.perf_counter()
        # Source checks are read-only. Transition and outbox insertion share one
        # transaction; provider execution occurs only after a valid alert commits.
        with self.repo.connection() as db:
            ordinal,stored,state,trace=self._source(db,pid,twin_state_id)
            previous_evaluation=db.execute('SELECT data FROM alert_evaluations WHERE patient_id=? AND twin_state_id=?',(pid,twin_state_id)).fetchone()
        if previous_evaluation:
            result=json.loads(previous_evaluation[0]);result['cached']=True
            if result.get('alert_id'):self.dispatch(pid,result['alert_id'])
            return result
        ineligible=self._eligible(state,trace);prepared=time.perf_counter()
        with self.repo.connection() as db:
            # Recheck idempotency after the read-only eligibility work, so two
            # service processes cannot both create the same transition.
            existing=db.execute('SELECT data FROM alert_evaluations WHERE patient_id=? AND twin_state_id=?',(pid,twin_state_id)).fetchone()
            if existing:return {**json.loads(existing[0]),'cached':True}
            row=db.execute('SELECT data FROM alert_policy_states WHERE patient_id=?',(pid,)).fetchone()
            previous=json.loads(row[0]) if row else None
            reference=clock(state['reference_time'])
            if previous and (reference<clock(previous['reference_time']) or reference==clock(previous['reference_time']) and ordinal<=previous['source_ordinal']):ineligible='OUT_OF_ORDER'
            result={'patient_id':pid,'twin_state_id':twin_state_id,'prediction_id':state['prediction_id'],'trace_id':state['trace_id'],
                'reference_time':state['reference_time'],'evaluated_at':self.stamp(),'policy_version':self.policy['version'],
                'eligible':ineligible is None,'decision':ineligible,'alert_id':None,'cached':False}
            if ineligible is None:
                condition='ABOVE' if state['spike']['probability']>=self.policy['threshold'] else 'BELOW'
                active=previous.get('active_alert_id') if previous else None
                result['previous_condition']=previous['condition'] if previous else 'UNKNOWN';result['condition']=condition
                if previous and previous['condition']=='BELOW' and condition=='ABOVE':
                    active=self._create(db,pid,state,stored);result['decision']='ALERT_CREATED';result['alert_id']=active
                elif active and condition=='BELOW':
                    alert=self._get(db,pid,active);alert.update({'state':'RESOLVED','status':'RESOLVED','resolved_at':self.stamp(),
                        'resolved_reference_time':state['reference_time'],'resolved_by_twin_state_id':twin_state_id})
                    self._update(db,alert);self._audit(db,pid,'ALERT_RESOLVED',active,{'twin_state_id':twin_state_id})
                    result['decision']='ALERT_RESOLVED';result['alert_id']=active;active=None
                else:result['decision']='BASELINE_ESTABLISHED' if previous is None else 'CONDITION_CHANGED_NO_ALERT' if previous['condition']!=condition else 'CONDITION_UNCHANGED'
                policy_state={'patient_id':pid,'reference_time':state['reference_time'],'source_ordinal':ordinal,
                    'twin_state_id':twin_state_id,'condition':condition,'active_alert_id':active,'policy_version':self.policy['version']}
                db.execute('INSERT INTO alert_policy_states VALUES(?,?) ON CONFLICT(patient_id) DO UPDATE SET data=excluded.data',(pid,json_safe(policy_state)))
            db.execute('INSERT INTO alert_evaluations VALUES(?,?,?)',(pid,twin_state_id,json_safe(result)))
        persisted=time.perf_counter()
        if result['decision']=='ALERT_CREATED':self.dispatch(pid,result['alert_id'])
        delivered=time.perf_counter()
        self.metrics.append({'evaluation_ms':(prepared-started)*1000,'transition_persistence_ms':(persisted-prepared)*1000,
            'delivery_ms':(delivered-persisted)*1000,'state_to_notification_ms':(delivered-started)*1000,'decision':result['decision']})
        self.metrics=self.metrics[-1000:]
        return result

    def _create(self,db,pid,state,stored):
        aid=identifier('entity');stamp=self.stamp()
        alert=owned_view(Alert(patient_id=pid,entity_id=aid,created_at=stamp,observed_time=state['reference_time'],
            severity='informational',message='Elevated glucose risk. Review your Twin and continue monitoring.',
            status='ACTIVE',prediction_id=state['prediction_id'],synthetic=stored['synthetic']))
        alert.update({'alert_schema_version':7,'alert_id':aid,'alert_type':self.policy['alert_type'],'reference_time':state['reference_time'],
            'twin_state_id':state['twin_state_id'],'trace_id':state['trace_id'],'risk_probability':state['spike']['probability'],
            'threshold':self.policy['threshold'],'current_glucose':state['current_glucose'],'predicted_peak':state['predicted_peak']['glucose'],
            'predicted_peak_horizon':state['predicted_peak']['approximate_horizon'],'state':'ACTIVE','acknowledged_at':None,'resolved_at':None,
            'created_by_policy_version':self.policy['version'],'delivery_status':'PENDING',
            'why_now_evidence_refs':[{'twin_state_id':state['twin_state_id'],'path':f'why_now[{i}]'} for i in range(len(state['why_now']))],
            'provenance':state['provenance'],'model_scope':state['evidence']['clinical_status']})
        db.execute('INSERT INTO alerts(entity_id,patient_id,prediction_id,data) VALUES(?,?,?,?)',(aid,pid,state['prediction_id'],json_safe(alert)))
        nid=identifier('entity')
        delivery={'patient_id':pid,'notification_id':nid,'alert_id':aid,'provider':self.provider.name,'created_at':stamp,
            'delivered_at':None,'acknowledged_at':None,'status':'PENDING','attempts':0,'transport':'backend_inbox','native_push':False,
            'content':{'title':'TwinCare','heading':'Elevated glucose risk','message':'Review your Twin and continue monitoring.',
                'current_glucose':alert['current_glucose'],'predicted_peak':alert['predicted_peak'],
                'review_twin_url':'/twin-inference?patient='+pid,'source_alert_id':aid}}
        db.execute('INSERT INTO notification_deliveries VALUES(?,?,?,?,?)',(nid,pid,aid,self.provider.name,json_safe(delivery)))
        self._audit(db,pid,'ALERT_CREATED',aid,{'prediction_id':state['prediction_id'],'twin_state_id':state['twin_state_id']})
        return aid

    def dispatch(self,pid,aid,retry=False):
        with self.repo.connection() as db:
            alert=self._get(db,pid,aid)
            row=db.execute('SELECT data FROM notification_deliveries WHERE patient_id=? AND alert_id=? AND provider=?',(pid,aid,self.provider.name)).fetchone()
            if not row:raise NotFound('Delivery not found')
            delivery=json.loads(row[0])
            if alert['state']=='RESOLVED' or delivery['status'] in ('DELIVERED','ACKNOWLEDGED'):return delivery
            if delivery['status']!='PENDING' and not retry:return delivery
            connection=db.execute("SELECT data FROM device_connections WHERE patient_id=? AND json_extract(data,'$.source_type')='WEARABLE' AND json_extract(data,'$.connection_schema_version')=3",(pid,)).fetchone()
            connection=json.loads(connection[0]) if connection else None
            connected=connection and connection.get('connected_at') and connection['status'] in ('CONNECTED','STALE','SYNCING')
            delivery['attempts']+=1
            if not connected:delivery['status']='NOT_DELIVERED_NO_DEVICE'
            else:
                try:
                    receipt=self.provider.send_alert(alert,self.repo._require(db,pid),delivery['notification_id'])
                    if receipt.get('patient_id')!=pid or receipt.get('alert_id')!=aid or receipt.get('notification_id')!=delivery['notification_id'] or receipt.get('status')!='DELIVERED':raise ValueError('Invalid provider receipt')
                    delivery.update({'status':'DELIVERED','delivered_at':self.stamp()})
                except Exception:delivery['status']='FAILED';delivery['failure_reason']='Provider could not deliver the notification'
            # This prototype provider performs only a synchronous local inbox
            # handoff. A network provider should use an outbox worker and the
            # notification ID as its transport idempotency key.
            db.execute('UPDATE notification_deliveries SET data=? WHERE patient_id=? AND notification_id=?',
                (json_safe(delivery),pid,delivery['notification_id']))
            alert['delivery_status']=delivery['status'];self._update(db,alert)
            if delivery['status']=='DELIVERED':self._audit(db,pid,'ALERT_DELIVERED',aid,{'notification_id':delivery['notification_id'],'provider':self.provider.name})
            if delivery['status']=='FAILED':self._audit(db,pid,'ALERT_DELIVERY_FAILED',aid,{'notification_id':delivery['notification_id'],'provider':self.provider.name})
            return delivery

    def list(self,pid):
        return {'patient_id':pid,'items':[a for a in self.repo.list_owned(pid,'alerts') if a.get('alert_schema_version')==7]}

    def get(self,pid,aid):
        with self.repo.connection() as db:return self._get(db,pid,aid)

    def active(self,pid):
        items=[a for a in self.list(pid)['items'] if a['state'] in ('ACTIVE','ACKNOWLEDGED')]
        alert=max(items,key=lambda a:a['created_at']) if items else None
        return {'patient_id':pid,'alert':alert,'snapshot_live':self._snapshot_live(pid,alert) if alert else False}

    def _snapshot_live(self,pid,alert):
        age=(self.time()-clock(alert['reference_time'])).total_seconds()/60
        return 0<=age<=self.inference.contract['readiness']['stale_after_minutes'] and self.inference.readiness.evaluate(pid,self.stamp())['capabilities']['short_horizon_forecast']

    def deliveries(self,pid,aid):
        with self.repo.connection() as db:
            self._get(db,pid,aid)
            rows=db.execute('SELECT data FROM notification_deliveries WHERE patient_id=? AND alert_id=?',(pid,aid))
            return {'patient_id':pid,'items':[json.loads(row[0]) for row in rows]}

    def inbox(self,pid):
        self.repo.get_patient(pid)
        with self.repo.connection() as db:
            rows=db.execute('SELECT data FROM notification_deliveries WHERE patient_id=?',(pid,))
            deliveries=[json.loads(r[0]) for r in rows]
            items=[]
            for delivery in deliveries:
                alert=self._get(db,pid,delivery['alert_id'])
                if delivery['status'] in ('DELIVERED','ACKNOWLEDGED') and alert['state']!='RESOLVED':items.append({'delivery':delivery,'alert':alert})
        for item in items:item['snapshot_live']=self._snapshot_live(pid,item['alert'])
        return {'patient_id':pid,'items':sorted(items,key=lambda n:n['alert']['created_at'],reverse=True)}

    def acknowledge(self,pid,aid,client='dashboard'):
        if client not in ('dashboard','wearable'):raise ValueError('Unknown acknowledgement client')
        with self.repo.connection() as db:
            alert=self._get(db,pid,aid)
            if alert['acknowledged_at'] is not None or alert['state']=='RESOLVED':return alert
            stamp=self.stamp();alert.update({'state':'ACKNOWLEDGED','status':'ACKNOWLEDGED','acknowledged_at':stamp,'acknowledged_from':client})
            self._update(db,alert)
            for row in db.execute('SELECT notification_id,data FROM notification_deliveries WHERE patient_id=? AND alert_id=?',(pid,aid)).fetchall():
                delivery=json.loads(row['data'])
                if delivery['status']=='DELIVERED':
                    delivery.update({'status':'ACKNOWLEDGED','acknowledged_at':stamp})
                    db.execute('UPDATE notification_deliveries SET data=? WHERE patient_id=? AND notification_id=?',(json_safe(delivery),pid,row['notification_id']))
                    alert['delivery_status']='ACKNOWLEDGED';self._update(db,alert)
            self._audit(db,pid,'ALERT_ACKNOWLEDGED',aid,{'client':client})
            return alert

    def evidence(self,pid,aid):
        alert=self.get(pid,aid);state=self.repo.get_owned(pid,'twin-states',alert['twin_state_id'])['payload']
        return {'patient_id':pid,'alert':alert,'why_now':state['why_now'],'shadow':state['shadow'],
            'provenance':state['provenance'],'trace':self.inference.trace(pid,alert['prediction_id']),
            'deliveries':self.deliveries(pid,aid)['items']}
