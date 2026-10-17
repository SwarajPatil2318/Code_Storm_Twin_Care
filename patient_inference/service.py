"""Owner-scoped causal runtime, frozen artifacts, atomic P1 persistence."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np
from patient_connections.contracts import clock, check_owner
from patient_readiness.contracts import build_buffer, feature_check
from src.patient_domain.models import identifier, now, json_safe, TwinState, Prediction, owned_view
from src.patient_domain.repository import NotFound
from src.metabolic_d2.models import ForecastModel
from src.metabolic_d3.reference import register
from src.metabolic_d3.state import HORIZON_KEYS
from src.metabolic_d41.fusion import FusionAdapter
from src.metabolic_risk_v1.model import dynamic_vector
from src.metabolic_risk_v1 import NAME as RISK_NAME
from .clinical import confirmed_context
from .evidence import explain
from .readiness import cutoff_ns

ROOT=Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def vector(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values).ravel()]


class Inference:
    def __init__(self, readiness):
        self.readiness=readiness;self.repo=readiness.repo;self.meals=readiness.meals
        self.contract=json.loads((ROOT/'artifacts/productization_p6/model_contract.json').read_text())
        self._lock=threading.RLock();self._models=None;self._risk=None
        self.metrics=[]
        with self.repo.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS inference_traces(
                prediction_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,identity TEXT NOT NULL,
                reference_time TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),
                UNIQUE(patient_id,identity),
                FOREIGN KEY(patient_id,prediction_id) REFERENCES predictions(patient_id,entity_id) ON DELETE CASCADE)''')
            db.execute('CREATE INDEX IF NOT EXISTS inference_trace_owner ON inference_traces(patient_id,reference_time)')
            db.execute("""CREATE TABLE IF NOT EXISTS inference_trace_parts(
                patient_id TEXT NOT NULL,prediction_id TEXT NOT NULL,part_index INTEGER NOT NULL,
                data TEXT NOT NULL CHECK(json_valid(data)),PRIMARY KEY(patient_id,prediction_id,part_index),
                FOREIGN KEY(prediction_id) REFERENCES inference_traces(prediction_id) ON DELETE CASCADE)""")
            for operation in ('INSERT','UPDATE'):
                db.execute(f"""CREATE TRIGGER IF NOT EXISTS inference_trace_part_owner_{operation.lower()}
                    BEFORE {operation} ON inference_trace_parts
                    WHEN NOT EXISTS(SELECT 1 FROM inference_traces WHERE prediction_id=NEW.prediction_id AND patient_id=NEW.patient_id)
                    BEGIN SELECT RAISE(ABORT,'Inference trace part ownership mismatch'); END""")

            for operation in ('INSERT','UPDATE'):
                db.execute(f"""CREATE TRIGGER IF NOT EXISTS inference_trace_owner_{operation.lower()}
                    BEFORE {operation} ON inference_traces
                    WHEN coalesce(json_extract(NEW.data,'$.patient_id')=NEW.patient_id,0)=0
                    BEGIN SELECT RAISE(ABORT,'Inference trace ownership mismatch'); END""")

    def _load_forecast(self):
        if self._models is None:
            registered=register()
            if registered['model_hashes']!=self.contract['forecast']['sha256']:
                raise RuntimeError('Forecast artifact compatibility mismatch')
            self._models={name:ForecastModel.load(registered['paths'][name]) for name in ('cold','warm')}
        return self._models

    def _load_risk(self):
        if self._risk is None:
            candidate=FusionAdapter().model
            if list(candidate.feature_order)!=self.contract['risk']['encoded_order']:
                raise RuntimeError('Risk artifact compatibility mismatch')
            self._risk=candidate
        return self._risk

    def run_twin_inference(self, patient_id, reference_time):
        reference=clock(reference_time)
        with self._lock:
            return self._run(patient_id,reference)

    def _run(self,pid,reference):
        started=time.perf_counter();stamp=reference.isoformat()
        readiness=self.readiness.evaluate(pid,stamp,inspect=True)
        if not readiness['capabilities']['short_horizon_forecast']:
            return {'patient_id':pid,'reference_time':stamp,'status':'NOT_READY','readiness':readiness,'twin_state':None,'prediction_id':None}
        with self.repo.connection() as db:
            patient=self.repo._require(db,pid)
            clinical,profile=confirmed_context(db,pid,reference)
            source=self.readiness._source(db,pid,'CGM',reference)
            events=[e for e in self.readiness._events(db,pid,'CGM',reference)
                if e['event_type']=='CGM_READING' and clock(e['observed_at'])>=reference-timedelta(days=8)]
            wearable=[e for e in self.readiness._events(db,pid,'WEARABLE',reference)
                if e['quality']['status']=='GOOD' and clock(e['observed_at'])>=reference-timedelta(hours=24)]
        if not clinical['fields'] or source['status'] not in ('CONNECTED','STALE','SYNCING') or not source.get('connected_at'):
            return {'patient_id':pid,'reference_time':stamp,'status':'NOT_READY','readiness':readiness,'twin_state':None,'prediction_id':None}
        meal=self.meals.recent_context(pid,stamp)
        buffer,rejected,unknown=build_buffer(pid,events);cutoff=cutoff_ns(reference)
        data,quality=feature_check(buffer,cutoff,unknown)
        if data is None:
            return {'patient_id':pid,'reference_time':stamp,'status':'FEATURE_UNAVAILABLE','quality':quality,'twin_state':None,'prediction_id':None}
        data['pid']=np.array([pid])
        glucose_times,usable_glucose,_,_=buffer.glucose(cutoff)
        current=float(usable_glucose[-1])
        warm=bool(data['warm'][0] and readiness['personalization']['ready'])
        if bool(data['warm'][0])!=warm:
            return {'patient_id':pid,'reference_time':stamp,'status':'FEATURE_UNAVAILABLE','reason':'Warm readiness disagreement','twin_state':None,'prediction_id':None}
        mode='warm' if warm else 'cold'
        identity=digest({'patient_id':pid,'reference_time':stamp,'clinical':clinical,'events':events,
            'wearable':wearable,'meal':meal,'source':source,'contract':self.contract})
        with self.repo.connection() as db:
            row=db.execute('SELECT prediction_id FROM inference_traces WHERE patient_id=? AND identity=?',(pid,identity)).fetchone()
        if row:return self._result(pid,row[0],cached=True)
        prepared=time.perf_counter()
        try:models=self._load_forecast()
        except Exception:
            return {'patient_id':pid,'reference_time':stamp,'status':'MODEL_UNAVAILABLE','reason':'Frozen forecast artifact could not be loaded or verified','twin_state':None,'prediction_id':None}
        model_ready=time.perf_counter()
        try:predicted=np.asarray(models[mode].predict(data)[0],dtype=np.float32)
        except Exception:
            return {'patient_id':pid,'reference_time':stamp,'status':'MODEL_UNAVAILABLE','reason':'Forecast execution failed','twin_state':None,'prediction_id':None}
        if predicted.shape!=(4,) or not np.isfinite(predicted).all():
            return {'patient_id':pid,'reference_time':stamp,'status':'MODEL_UNAVAILABLE','reason':'Invalid artifact output','twin_state':None,'prediction_id':None}
        forecasted=time.perf_counter()
        risk={'available':False,'probability':None,'status':'Clinical context incomplete','missing_fields':clinical['missing_fields'],
            'model_version':RISK_NAME,'threshold':self.contract['risk']['threshold'],
            'definition':self.contract['risk']['definition'],'expected_time_window_minutes':[0,120],
            'timing_scope':'Probability over the full 2-hour window; no predicted exact event time'}
        dynamic=dynamic_vector(data,predicted[None,:]);encoded=None
        risk_loading_ms=0.;risk_execution_ms=0.
        if clinical['risk_context_available']:
            try:
                load_started=time.perf_counter();head=self._load_risk()
                risk_loading_ms=(time.perf_counter()-load_started)*1000
                encoded=head.features(data,predicted[None,:],profile)
                risk_started=time.perf_counter()
                probability=float(head.probabilities(data,predicted[None,:],profile)[0])
                risk_execution_ms=(time.perf_counter()-risk_started)*1000
                if not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('Invalid probability')
                risk.update({'available':True,'probability':probability,'context_complete':not clinical['missing_fields'],
                    'status':'Already above 180; incident rule not applicable' if current>180 else
                        'Above frozen alert threshold' if probability>=head.threshold else 'Below frozen alert threshold'})
            except Exception:
                risk['status']='Frozen risk artifact unavailable or input incompatible'
        risked=time.perf_counter()
        shadow,comparison,why,status=explain(buffer,cutoff,data,predicted,meal,wearable)
        current=next(item['value'] for item in why if item['signal']=='current_glucose')
        if risk['available'] and current<=180 and risk['probability']>=risk['threshold']:status='SPIKE_WATCH'
        explained=time.perf_counter()
        evidence_synthetic=any(e['provenance']['synthetic'] for e in events)
        state_id=identifier('entity');prediction_id=identifier('entity')
        trace_id=identifier('entity')
        current_synthetic=any(e.synthetic for e in buffer.visible(cutoff,'glucose') if e.observed_ns==glucose_times[-1])
        category='SIMULATED_LIVE_STREAM' if current_synthetic else 'OBSERVED'
        meal_category='SIMULATED_LIVE_STREAM' if any(m.get('synthetic') for m in meal.get('meals',[])) else 'OBSERVED'
        wearable_category='SIMULATED_LIVE_STREAM' if any(e['provenance']['synthetic'] for e in wearable) else 'OBSERVED'
        why_categories={'current_glucose':category,'recent_slope':'SIMULATED_LIVE_STREAM' if evidence_synthetic else 'OBSERVED',
            'recent_acceleration':'SIMULATED_LIVE_STREAM' if evidence_synthetic else 'OBSERVED',
            'departure_from_expected_now':'PERSONAL_EXPECTED','future_personal_departure':'PERSONAL_EXPECTED',
            'slope_departure_from_personal_norm':'PERSONAL_EXPECTED','recent_meal':meal_category,'received_wearable_context':wearable_category}
        forecast=dict(zip(HORIZON_KEYS,vector(predicted)))
        state_readiness={**readiness,'inference_enabled':True,'prediction':{'prediction_id':prediction_id},
            'capabilities':{**readiness['capabilities'],'risk_context':risk['available']},
            'inspection':{key:value for key,value in readiness['inspection'].items() if key!='no_forecast_or_risk_model_loaded'}}
        state_readiness['inspection']['readiness_derivation_loaded_models']=False
        payload={'schema_version':'twincare-metabolic-twin/6','patient_id':pid,'reference_time':stamp,
            'readiness':state_readiness,'current_glucose':current,'forecast':forecast,
            'predicted_peak':{'glucose':float(predicted.max()),'approximate_horizon':[30,60,90,120][int(predicted.argmax())],
                'scope':'Maximum of four forecast horizons; approximate timing only'},
            'forecast_mode':mode,'personalization':readiness['personalization'],'shadow':shadow,'divergence':comparison,
            'spike':risk,'why_now':why,'recent_context':{'meals':meal,'wearable':wearable,'clinical':clinical['fields']},
            'freshness':readiness['source_freshness'],'data_quality':quality,'status':status,
            'model_versions':{'forecast':'METABOLIC_D2_REFERENCE','risk':RISK_NAME},
            'provenance':{'current_glucose':category,'forecast':'MODEL_PREDICTED','predicted_peak':'MODEL_PREDICTED',
                'spike':'MODEL_PREDICTED','shadow':'PERSONAL_EXPECTED','divergence':'PERSONAL_EXPECTED',
                'clinical':'STATIC_CLINICAL','meals':meal_category,'wearable':wearable_category,
                'why_now':[{'signal':item['signal'],'source':item['source'],
                    'category':why_categories[item['signal']],
                    'method':'Structured descriptive evidence; not feature attribution'} for item in why],
                'freshness':'OBSERVED','readiness':'OBSERVED','data_quality':'OBSERVED','status':'MODEL_PREDICTED'},
            'evidence':{'synthetic_inputs':evidence_synthetic,'patient_synthetic':patient['synthetic'],
                'clinical_status':'NOT CLINICALLY VALIDATED','meal_wearable_predictive_use':False,
                'why_now_scope':'Structured descriptive evidence; not causal attribution'},
            'trace_id':trace_id,'twin_state_id':state_id,'prediction_id':prediction_id}
        names=self.contract['forecast'][mode];values=vector(np.column_stack([data['x'],data['personal']]) if warm else data['x'])
        trace={'patient_id':pid,'trace_id':trace_id,'prediction_id':prediction_id,'twin_state_id':state_id,
            'reference_time':stamp,'identity':identity,'forecast_mode':mode,'features':{'names':names,'values':values,'sha256':digest(values)},
            'risk_features':{'dynamic_names':self.contract['risk']['logical_dynamic'],'dynamic_values':vector(dynamic),
                'static_values':clinical['static_values'],'encoded_order':self.contract['risk']['encoded_order'],
                'encoded_values':vector(encoded) if encoded is not None else None,
                'encoded_sha256':digest(vector(encoded)) if encoded is not None else None},
            'inputs':{'CGM':[{'patient_id':pid,'event_id':e['event_id'],'observed_at':e['observed_at'],'received_at':e['received_at'],
                'quality':e['quality']['status']} for e in events], 'clinical':clinical['sources'],
                'meal_ids':[e['meal_id'] for e in meal.get('meals',[])], 'wearable_ids':[e['event_id'] for e in wearable],
                'meals':[{'patient_id':pid,'meal_id':e['meal_id'],'observed_at':e['meal_occurred_at'],'received_at':e['confirmed_at']} for e in meal.get('meals',[])],
                'wearable':[{'patient_id':pid,'event_id':e['event_id'],'observed_at':e['observed_at'],'received_at':e['received_at']} for e in wearable]},
            'source_snapshot':source,'static_mapping':clinical['mappings'],'models':self.contract['forecast']['sha256'],
            'artifact_execution':{'forecast':{'path':self.contract['forecast']['artifacts'][mode],
                'sha256':self.contract['forecast']['sha256'][mode],'executed':True},
                'risk':{'path':self.contract['risk']['path'],'sha256':self.contract['risk']['sha256'],'executed':risk['available']}},
            'risk_model_sha256':self.contract['risk']['sha256'],'preprocessors':{'forecast':self.contract['forecast']['preprocessor_sha256'],
                'risk':self.contract['risk']['preprocessor_sha256']},'output':{'forecast':forecast,'spike':risk},
            'timing_rules':'All observation, receipt, confirmation and revision times <= reference_time; UTC features',
            'rejected_CGM_count':rejected}
        assembled=time.perf_counter()
        timings={'feature_preparation_ms':(prepared-started)*1000,'forecast_model_loading_ms':(model_ready-prepared)*1000,
            'glucose_model_inference_ms':(forecasted-model_ready)*1000,'risk_model_inference_ms':risk_execution_ms,
            'risk_model_loading_ms':risk_loading_ms,'risk_stage_ms':(risked-forecasted)*1000,
            'shadow_why_now_ms':(explained-risked)*1000,'twin_state_assembly_ms':(assembled-explained)*1000}
        state=owned_view(TwinState(patient_id=pid,entity_id=state_id,cutoff_time=stamp,module='metabolic',schema_version='6',
            payload=payload,source_provenance={'trace_id':trace_id,'identity':identity},synthetic=evidence_synthetic))
        prediction=owned_view(Prediction(patient_id=pid,entity_id=prediction_id,prediction_time=stamp,horizon='30/60/90/120 minutes',
            prediction_kind='metabolic_glucose_and_spike_risk',model_version='METABOLIC_D2_REFERENCE+'+RISK_NAME,
            output={'forecast':forecast,'predicted_peak':payload['predicted_peak'],'spike':risk,'forecast_mode':mode},
            source_provenance={'trace_id':trace_id,'identity':identity,'feature_schema':names},twin_state_id=state_id,synthetic=evidence_synthetic))
        check_owner(trace,pid)
        with self.repo.connection() as db:
            # Across multiple service processes, the UNIQUE owner/identity key
            # arbitrates retries before writing any new state.
            existing=db.execute('SELECT prediction_id FROM inference_traces WHERE patient_id=? AND identity=?',(pid,identity)).fetchone()
            if existing:other=existing[0]
            else:
                other=None
                db.execute('INSERT INTO twin_states VALUES(?,?,?)',(state_id,pid,json_safe(state)))
                db.execute('INSERT INTO predictions(entity_id,patient_id,twin_state_id,data) VALUES(?,?,?,?)',
                    (prediction_id,pid,state_id,json_safe(prediction)))
                persisted=time.perf_counter()
                timings['persistence_ms']=(persisted-assembled)*1000
                timings['total_backend_ms']=(persisted-started)*1000
                trace['latency_ms']=timings
                db.execute('INSERT INTO inference_traces VALUES(?,?,?,?,?)',(prediction_id,pid,identity,stamp,self._trace_record(trace)))
                self._write_trace_parts(db,trace)
        if other:return self._result(pid,other,cached=True)
        timings['persistence_ms']=(time.perf_counter()-assembled)*1000
        timings['total_backend_ms']=(time.perf_counter()-started)*1000
        # Store commit-inclusive latency separately from the atomic prediction
        # write. This instrumentation update is not part of measured inference.
        trace['latency_ms']=timings
        with self.repo.connection() as db:
            db.execute('UPDATE inference_traces SET data=? WHERE patient_id=? AND prediction_id=?',(self._trace_record(trace),pid,prediction_id))
        self.metrics.append(timings);self.metrics=self.metrics[-1000:]
        return {'patient_id':pid,'reference_time':stamp,'status':'INFERRED','cached':False,'prediction_id':prediction_id,'twin_state':payload,'latency_ms':timings}

    def _result(self,pid,prediction_id,cached=False):
        prediction=self.repo.get_owned(pid,'predictions',prediction_id)
        state=self.repo.get_owned(pid,'twin-states',prediction['twin_state_id'])['payload']
        return {'patient_id':pid,'reference_time':state['reference_time'],'status':'INFERRED','cached':cached,
            'prediction_id':prediction_id,'twin_state':state}

    def latest(self,pid,reference_time=None):
        self.repo.get_patient(pid)
        with self.repo.connection() as db:
            params=[pid];condition=''
            if reference_time is not None:condition=' AND reference_time<=?';params.append(clock(reference_time).isoformat())
            row=db.execute('SELECT prediction_id FROM inference_traces WHERE patient_id=?'+condition+' ORDER BY reference_time DESC,rowid DESC LIMIT 1',params).fetchone()
        if not row:return {'patient_id':pid,'status':'NO_PREDICTION','twin_state':None,'prediction_id':None}
        return self._result(pid,row[0])

    def trace(self,pid,prediction_id):
        with self.repo.connection() as db:
            self.repo._require(db,pid)
            row=db.execute('SELECT data FROM inference_traces WHERE patient_id=? AND prediction_id=?',(pid,prediction_id)).fetchone()
        if not row:raise NotFound('Prediction trace not found')
        return self._read_trace(pid,prediction_id,json.loads(row[0]))

    def _trace_record(self,trace):
        # Keep the existing per-record limit. Large provenance is split losslessly;
        # prediction linkage and scientific outputs remain in the parent record.
        try:return json_safe(trace)
        except ValueError as error:
            if str(error)!='Record too large':raise
        inputs=json.dumps(trace['inputs'],allow_nan=False)
        parent={k:v for k,v in trace.items() if k!='inputs'}
        parent['_input_parts']={'count':(len(inputs)+32767)//32768,
                                'sha256':hashlib.sha256(inputs.encode()).hexdigest()}
        return json_safe(parent)

    def _write_trace_parts(self,db,trace):
        parent=json.loads(self._trace_record(trace))
        if '_input_parts' not in parent:return
        inputs=json.dumps(trace['inputs'],allow_nan=False)
        for index,start in enumerate(range(0,len(inputs),32768)):
            db.execute('INSERT INTO inference_trace_parts VALUES(?,?,?,?)',
                       (trace['patient_id'],trace['prediction_id'],index,json_safe(inputs[start:start+32768])))

    def _read_trace(self,pid,prediction_id,parent):
        if '_input_parts' not in parent:return parent
        metadata=parent.pop('_input_parts')
        with self.repo.connection() as db:
            rows=db.execute('SELECT part_index,data FROM inference_trace_parts WHERE patient_id=? AND prediction_id=? ORDER BY part_index',(pid,prediction_id)).fetchall()
        if [r[0] for r in rows]!=list(range(metadata['count'])):raise ValueError('Incomplete inference provenance')
        inputs=''.join(json.loads(r[1]) for r in rows)
        if hashlib.sha256(inputs.encode()).hexdigest()!=metadata['sha256']:raise ValueError('Inference provenance integrity mismatch')
        parent['inputs']=json.loads(inputs);check_owner(parent,pid)
        return parent
