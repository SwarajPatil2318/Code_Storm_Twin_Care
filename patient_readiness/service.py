"""Authoritative, causal, owner-scoped readiness and meaningful transition audit."""
from datetime import datetime,timezone,timedelta
import json
from src.patient_domain.models import AuditEvent,identifier,owned_view,json_safe
from patient_connections.contracts import clock,check_owner
from .contracts import Buffer,Shadow,MINUTE,np,nearest_indices,discovered_contract,build_buffer,feature_check


class Readiness:
    def __init__(self,meals,time=None):
        self.meals=meals;self.repo=meals.repo;self.time=time or (lambda:datetime.now(timezone.utc));self.contract=discovered_contract()
        with self.repo.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS readiness_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)')
            if any(r[0]!=1 for r in db.execute('SELECT version FROM readiness_migrations')):raise ValueError('Unknown readiness schema')
            db.execute('INSERT OR IGNORE INTO readiness_migrations VALUES(1,?)',(self._stamp(),))
            db.execute('''CREATE TABLE IF NOT EXISTS readiness_source_revisions(revision_id INTEGER PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,source_type TEXT NOT NULL,
                available_at TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)))''')
            db.execute('CREATE INDEX IF NOT EXISTS readiness_source_cutoff ON readiness_source_revisions(patient_id,source_type,available_at)')
            db.execute('''CREATE TABLE IF NOT EXISTS readiness_latest(patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id) ON DELETE CASCADE,
                evaluated_at TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)))''')
            for table in ['readiness_source_revisions','readiness_latest']:
                for operation in ['INSERT','UPDATE']:
                    db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_owner_{operation.lower()} BEFORE {operation} ON {table}
                        WHEN coalesce(json_extract(NEW.data,'$.patient_id')=NEW.patient_id,0)=0
                        BEGIN SELECT RAISE(ABORT,'Readiness owner mismatch'); END""")
    def _stamp(self):return self.time().astimezone(timezone.utc).isoformat()
    def _source(self,db,pid,source,reference):
        row=db.execute('SELECT data FROM readiness_source_revisions WHERE patient_id=? AND source_type=? AND available_at<=? ORDER BY available_at DESC,revision_id DESC LIMIT 1',
            (pid,source,reference.isoformat())).fetchone()
        if row:return json.loads(row[0])
        row=db.execute("SELECT data FROM device_connections WHERE patient_id=? AND json_extract(data,'$.source_type')=? AND json_extract(data,'$.connection_schema_version')=3",(pid,source)).fetchone()
        if row:
            value=json.loads(row[0])
            if clock(value['updated_at'])<=reference:return {k:value.get(k) for k in ['patient_id','source_type','status','connected_at','error']}
            return {'patient_id':pid,'source_type':source,'status':'NOT_CONNECTED','history_unknown':True}
        return {'patient_id':pid,'source_type':source,'status':'NOT_CONNECTED'}
    def _events(self,db,pid,source,reference):
        rows=db.execute("SELECT data FROM observations WHERE patient_id=? AND json_extract(data,'$.source_provenance.canonical_event.source_type')=?",(pid,source))
        result=[]
        for row in rows:
            event=json.loads(row[0])['source_provenance']['canonical_event']
            check_owner(event,pid)
            if clock(event['observed_at'])<=reference and clock(event['received_at'])<=reference:result.append(event)
        return sorted(result,key=lambda e:(clock(e['observed_at']),e['event_id']))
    def _clinical(self,db,pid,reference):
        profile=json.loads(db.execute('SELECT data FROM clinical_profiles WHERE patient_id=?',(pid,)).fetchone()[0])
        confirmed=[v for v in profile.get('confirmation_history',[]) if v.get('accepted_fields') and clock(v['confirmed_at'])<=reference]
        if confirmed:return {'status':'CONFIRMED','ready':True}
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE name='ingestion_documents'").fetchone()
        documents=[json.loads(r[0]) for r in db.execute('SELECT data FROM ingestion_documents WHERE patient_id=?',(pid,))] if exists else []
        pending=any(clock(d['uploaded_at'])<=reference for d in documents)
        return {'status':'REVIEW_REQUIRED' if pending else 'NO_RECORD','ready':False}
    def _derive(self,pid,reference):
        with self.repo.connection() as db:
            patient=self.repo._require(db,pid);clinical=self._clinical(db,pid,reference)
            source=self._source(db,pid,'CGM',reference);wearable=self._source(db,pid,'WEARABLE',reference)
            events=[e for e in self._events(db,pid,'CGM',reference) if e['event_type']=='CGM_READING']
            wearable_events=self._events(db,pid,'WEARABLE',reference)
        cutoff=int(reference.timestamp()*1e9);warnings=[];multiple=False
        try:buffer,rejected,unknown=build_buffer(pid,events)
        except ValueError:
            # Do not silently combine different device histories or clocks.
            buffer=Buffer(pid);rejected=len(events);unknown=False;multiple=True;warnings.append('Multiple incompatible CGM sources need review.')
        data,quality=feature_check(buffer,cutoff,unknown)
        t,g,usable,conflicts=buffer.glucose(cutoff)
        wanted=cutoff-np.arange(120,-1,-15)*MINUTE
        indices,found=nearest_indices(t,wanted,7.5*MINUTE) if len(t) else (np.array([],dtype=int),np.zeros(9,dtype=bool))
        slots=len(set(indices[found].tolist())) if len(t) else 0
        duration=(cutoff-int(t[0]))/MINUTE if len(t) else 0
        recent=t[t>=cutoff-120*MINUTE]
        recent_duration=float((recent[-1]-recent[0])/MINUTE) if len(recent)>1 else 0
        age=float((cutoff-int(t[-1]))/MINUTE) if len(t) else None
        stale=age is not None and age>self.contract['stale_after_minutes']
        gaps=np.diff(t)/MINUTE if len(t)>1 else np.array([])
        cadence_ok=not unknown and not multiple and (not len(gaps) or not (gaps<7.5).any())
        shadow=Shadow().predict(buffer,cutoff)
        personal_ready=bool(shadow['available'])
        personal_status='PERSONALIZATION_READY' if personal_ready else 'PERSONALIZATION_BUILDING' if len(t) else 'PERSONALIZATION_UNAVAILABLE'
        connected=source['status'] in ['CONNECTED','STALE','SYNCING'] and bool(source.get('connected_at'))
        source_error=source['status']=='ERROR'
        historical_features=False
        if len(t):
            # A backfilled reading may have arrived after its observation time.
            # Its historical capability can only be checked once receipt was possible.
            historical_reference=max(int(t[-1]),max(e.available_ns for e in buffer.visible(cutoff,'glucose')))
            historical_features=feature_check(buffer,historical_reference,unknown)[0] is not None
        forecast=clinical['ready'] and connected and not source_error and data is not None
        personalized=forecast and personal_ready
        if not clinical['ready']:state='PROFILE_INCOMPLETE'
        elif source_error:state='SOURCE_ERROR'
        elif not connected:state='CGM_NOT_CONNECTED'
        elif stale:state='DATA_STALE'
        elif data is None:state='COLLECTING_HISTORY'
        elif personalized:state='PERSONALIZED_TWIN_READY'
        else:state='PERSONALIZATION_BUILDING'
        if source.get('history_unknown'):warnings.append('Historical connection state was not recorded for this reference time.')
        if rejected or conflicts:warnings.append('Missing, suspect or conflicting glucose observations need review.')
        if unknown:warnings.append('Native 15-minute CGM cadence must be declared by the source.')
        if events and all(e['provenance']['synthetic'] for e in events):warnings.append('This history contains explicitly supplied synthetic engineering observations.')
        meal_context=self.meals.recent_context(pid,reference.isoformat())
        wearable_available=wearable['status'] in ['CONNECTED','STALE','SYNCING'] and any(e['quality']['status']=='GOOD' and reference-clock(e['observed_at'])<=timedelta(hours=24) for e in wearable_events)
        capabilities={'clinical_context':clinical['ready'],'short_horizon_forecast':forecast,'personal_shadow_twin':personalized,
            'multimodal_risk_context':False,'wearable_context':wearable_available,'meal_context':bool(meal_context['meals'])}
        requirements=[{'key':'clinical_context','ready':clinical['ready']},{'key':'CGM_connection','ready':connected and not source_error},
            {'key':'forecast_history','ready':data is not None},{'key':'personal_baseline','ready':personal_ready}]
        remaining=120-min(120,recent_duration) if len(t) and cadence_ok and len(gaps) and np.all(np.abs(gaps-15)<.01) and not stale and slots<9 else None
        if remaining is not None and remaining<=0:remaining=None
        result={'patient_id':pid,'overall_state':state,'requirements':requirements,'capabilities':capabilities,
            'blocking_reasons':[r['key'] for r in requirements if not r['ready']],'warnings':warnings,
            'progress':{'core_steps_ready':sum(r['ready'] for r in requirements),'core_steps_total':4},
            'clinical_context':clinical,'source_states':{'CGM':source['status'],'WEARABLE':wearable['status']},
            'source_freshness':{'CGM':{'latest_reading_age_minutes':age,'stale':stale}},
            'history':{'history_duration_minutes':round(float((t[-1]-t[0])/MINUTE),2) if len(t)>1 else 0,'recent_history_minutes':round(recent_duration,2),
                'required_history_minutes':120,'valid_reading_count':len(t),'expected_reading_count':9,'covered_slot_count':slots,
                'coverage_ratio':slots/9,'cadence_quality':'SUPPORTED' if cadence_ok else 'UNSUPPORTED',
                'missing_or_suspect_count':rejected,'conflict_count':len(conflicts),'quality_status':quality['status'],
                'features_generatable':data is not None,'historical_features_available':historical_features,'remaining_minutes':remaining},
            'personalization':{'status':personal_status,'ready':personal_ready,'scope':'Causal personal descriptive expectation; not clinically validated',
                'fallback':'Personal prefix median when time-of-day support is insufficient; no population Shadow fallback'},
            'evaluated_at':reference.isoformat(),'inference_enabled':False,'prediction':None}
        inspection={'forecast_contract':self.contract,'feature_validation':quality,'personalization_support':shadow.get('support',{}),
            'personal_history_elapsed_hours':duration/60,'warm_forecast_prefix_eligible':bool(data is not None and data['warm'][0]),
            'no_forecast_or_risk_model_loaded':True}
        check_owner(result,pid);return result,inspection
    def can_generate_forecast_features(self,pid,reference_time):return self._derive(pid,clock(reference_time))[0]['history']['features_generatable']
    def evaluate(self,pid,reference_time=None,inspect=False):
        reference=clock(reference_time) if reference_time is not None else self.time().astimezone(timezone.utc)
        if reference>self.time()+timedelta(minutes=5):raise ValueError('Future readiness is unavailable')
        result,inspection=self._derive(pid,reference)
        # Replay/inspection never rewinds or creates live lifecycle audit events.
        if reference_time is None:self._transition(result)
        if inspect:result={**result,'inspection':inspection}
        return result
    def _transition(self,result):
        pid=result['patient_id'];stamp=result['evaluated_at']
        summary={'patient_id':pid,'overall_state':result['overall_state'],'capabilities':result['capabilities'],
            'features_generatable':result['history']['features_generatable'],'stale':result['source_freshness']['CGM']['stale']}
        with self.repo.connection() as db:
            patient=self.repo._require(db,pid);row=db.execute('SELECT evaluated_at,data FROM readiness_latest WHERE patient_id=?',(pid,)).fetchone()
            if row and clock(row[0])>clock(stamp):return
            previous=json.loads(row[1]) if row else None
            if previous==summary:return
            actions=[]
            if previous is None:actions.append('TWIN_SETUP_STARTED')
            for key,action in [('clinical_context','CLINICAL_CONTEXT_READY'),('short_horizon_forecast','FORECAST_CAPABILITY_READY'),('personal_shadow_twin','PERSONALIZATION_READY')]:
                if summary['capabilities'][key] and (previous is None or not previous['capabilities'][key]):actions.append(action)
            if summary['features_generatable'] and (previous is None or not previous['features_generatable']):actions.append('CGM_HISTORY_SUFFICIENT')
            if summary['stale'] and (previous is None or not previous['stale']):actions.append('TWIN_DATA_STALE')
            if previous and previous['stale'] and not summary['stale'] and summary['features_generatable']:actions.append('TWIN_DATA_RECOVERED')
            if summary['overall_state']=='SOURCE_ERROR' and (previous is None or previous['overall_state']!='SOURCE_ERROR'):actions.append('TWIN_SOURCE_ERROR')
            if previous and previous['overall_state']!=summary['overall_state']:actions.append('TWIN_READINESS_CHANGED')
            for action in actions:
                event=owned_view(AuditEvent(patient_id=pid,entity_id=identifier('entity'),created_at=stamp,synthetic=patient['synthetic'],
                    action=action,entity_type='twin_readiness',target_id=None,details={'from':previous['overall_state'] if previous else None,'to':summary['overall_state']}))
                db.execute('INSERT INTO audit_events VALUES(?,?,?)',(event['entity_id'],pid,json_safe(event)))
            db.execute('INSERT INTO readiness_latest VALUES(?,?,?) ON CONFLICT(patient_id) DO UPDATE SET evaluated_at=excluded.evaluated_at,data=excluded.data',(pid,stamp,json_safe(summary)))
