"""Optional new multimodal runtime adapter; original D2/D3 remains immutable."""
import copy,datetime as dt,json,secrets,threading,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic_d2.protocol import digest
from src.metabolic_d3.events import timestamp,Event
from src.metabolic_d35.data import SAFE,medication_category
from src.metabolic_d41.runtime import Session,RuntimeStore,RuntimeUnavailable
from src.metabolic_risk_v1.model import RiskHead,dynamic_vector,DYNAMIC_NAMES
from src.metabolic_risk_v1 import NAME


def safe_demo_profile(state):
    patient=state['patient'];ehr=state['recent_context'].get('ehr')
    if not patient.get('synthetic') or not patient['id'].startswith('metabolic-demo:'):
        raise PermissionError('Runtime accepts explicit same-demo context only')
    if ehr is None or ehr.get('label')!='SYNTHETIC DEMO EHR':
        raise ValueError('Explicit synthetic EHR required for multimodal risk')
    demographics=ehr.get('demographics',{});birth=demographics.get('birth_date')
    age=None
    if birth:
        cutoff=dt.datetime.fromisoformat(state['timestamp'].replace('Z','+00:00')).date()
        born=dt.date.fromisoformat(birth);age=cutoff.year-born.year-((cutoff.month,cutoff.day)<(born.month,born.day))
        if not 18<=age<=120:age=None
    gender=demographics.get('gender');gender=gender if gender in ['female','male'] else None
    medications=ehr.get('medications',[])
    medication=medication_category(' '.join(v.get('text') or '' for v in medications))
    history=ehr.get('safe_history',{})
    values={'age_years':age,'sex':gender,'diabetes_duration_years':history.get('diabetes_duration_years'),
            'smoking_pack_years':history.get('smoking_pack_years'),'alcohol_history':history.get('alcohol_history'),
            'medication_context':medication,'other_medication_context':history.get('other_medication_context')}
    for name in ['diabetes_duration_years','smoking_pack_years']:
        value=values[name]
        if value is not None and (type(value) not in [int,float] or not np.isfinite(value) or value<0):
            raise ValueError('Invalid declared synthetic history')
    if values['alcohol_history'] is not None:values['alcohol_history']=str(values['alcohol_history']).strip().lower()
    if values['other_medication_context'] is not None:
        values['other_medication_context']=medication_category(values['other_medication_context'])
    return pd.DataFrame([{'participant_id':patient['id'],**values}]),values


class FusionAdapter:
    def __init__(self):
        self.manifest=json.loads(Path('artifacts/metabolic_risk_v1/manifest.json').read_text())
        if self.manifest['decision']!='GO':raise RuntimeUnavailable('Fusion head not promoted')
        final=self.manifest['final'];path=Path(final['path'])
        if digest(path)!=final['sha256']:raise RuntimeUnavailable('Fusion artifact hash mismatch')
        self.model=RiskHead.load(path)
        if self.model.version!=NAME or self.model.feature_order!=final['feature_order']:
            raise RuntimeUnavailable('Fusion schema/version mismatch')
        self.sha256=final['sha256'];self.lock=threading.Lock()

    def enrich(self,state,buffer,trace=None):
        start=time.perf_counter();profiles,static=safe_demo_profile(state)
        data,quality=buffer.features()
        if data is None:raise RuntimeUnavailable('Causal CGM risk input unavailable')
        data={**data,'pid':np.array([state['patient']['id']])}
        forecast=np.array([[state['forecast'][f'plus_{h}m'] for h in [30,60,90,120]]],dtype='float32')
        with self.lock:prob=float(self.model.probabilities(data,forecast,profiles)[0])
        elapsed=(time.perf_counter()-start)*1000;point=self.model.threshold
        state=copy.deepcopy(state)
        state['reference_spike']=copy.deepcopy(state['spike'])
        state['spike'].update({'probability':prob,'threshold':point,'model_version':NAME,
                              'evidence_label':'DEVELOPMENT-VALIDATED MULTIMODAL RISK HEAD',
                              'status':'Already above180; incident rule not applicable' if state['current_glucose']>180 else
                                       'Above multimodal alert threshold' if prob>=point else 'Below multimodal alert threshold'})
        state['multimodal_risk']={'model_version':NAME,'sha256':self.sha256,'trace_id':trace or 'MR-'+secrets.token_hex(8).upper(),
                                 'inference_ms':elapsed,'static_input':static,
                                 'dynamic_input':dict(zip(DYNAMIC_NAMES,dynamic_vector(data,forecast)[0].astype(float).tolist())),
                                 'personal_inputs_used':False,'shadow_predictive_use':False,
                                 'input_evidence':'SYNTHETIC DEMO EHR + SYNTHETIC CGM; actual safe schema',
                                 'evidence_label':'DEVELOPMENT-VALIDATED MULTIMODAL RISK HEAD',
                                 'probability':prob,'threshold':point,'history_retained':True,
                                 'missing_static_fields':[k for k,v in static.items() if v is None]}
        # Keep unchanged forecast/Shadow evidence distinct from the new risk evidence.
        state['evidence']['risk_model_version']=NAME
        state['evidence']['risk_static_predictive_use']=True
        state['evidence']['risk_wearable_predictive_use']=False
        state['evidence']['reference_predictive_scope']='wearable_EHR_predictive_use describes METABOLIC_D2_REFERENCE only; the separate risk head ingests safe EHR fields'
        state['evidence']['risk_evidence_label']='DEVELOPMENT-VALIDATED MULTIMODAL RISK HEAD'
        for entry in state['data_sources']:
            if entry['field']=='spike':entry.update({'source':NAME,'method':'Joint safe same-patient static + CGM/D2 trajectory inference; development validation'})
        state['data_sources'].extend([{'field':'multimodal_risk','category':'MODEL PREDICTED','source':NAME,'synthetic_input':True,
                                      'method':'Joint risk inference; static missingness explicit'},
                                     {'field':'reference_spike','category':'MODEL PREDICTED','source':'METABOLIC_D2_REFERENCE',
                                      'synthetic_input':True,'method':'Preserved historical D2 risk for provenance'}])
        if state['current_glucose']>180:state['status']='CURRENT_GLUCOSE_ELEVATED'
        elif prob>=point:state['status']='SPIKE_WATCH'
        elif state['divergence']['magnitude'] is not None and state['divergence']['magnitude']>=1 and np.mean(forecast[0]-np.array([state['shadow'][f'plus_{h}m'] for h in [30,60,90,120]]))>0:
            state['status']='PERSONALLY_ELEVATED_RESPONSE'
        elif data['x'][0,10]>=.5:state['status']='RISING'
        else:state['status']='STABLE'
        return state,elapsed


class FusionSession(Session):
    def __init__(self,*args,adapter,**kwargs):
        self.adapter=adapter;super().__init__(*args,**kwargs)
    def _emit(self,*args,**kwargs):
        saved={k:copy.deepcopy(getattr(self,k)) for k in ['latest','snapshots','logs','revision','touched'] if hasattr(self,k)}
        old_state=self.engine.last_state
        try:
            result=super()._emit(*args,**kwargs)
            state,elapsed=self.adapter.enrich(result['state'],self.engine.buffer,result['update']['trace_id'])
            result['state']=state;self.engine.last_state=state
            update=result['update'];update['risk_model_version']=NAME
            update['pipeline'].insert(-1,NAME+' (same-patient history + physiology)')
            update['timings_ms']['risk_inference_ms']=elapsed;update['timings_ms']['twin_update_ms']+=elapsed
            result['forecast_history'][-1]['spike_probability']=state['spike']['probability']
            result['forecast_history'][-1]['risk_model_version']=NAME
            result['forecast_history'][-1]['risk_sha256']=self.adapter.sha256
            update['risk_sha256']=self.adapter.sha256
            return result
        except Exception:
            for k,v in saved.items():setattr(self,k,v)
            self.engine.last_state=old_state
            raise



class FusionRuntimeStore(RuntimeStore):
    def __init__(self,demo,reference=None,adapter=None):
        super().__init__(demo,reference);self.adapter=adapter or FusionAdapter()
    def create(self,payload):
        if self.load_error:raise RuntimeUnavailable('Frozen trajectory unavailable')
        case=payload.get('case_id','predicted_spike');index=payload.get('start_index',0);mode=payload.get('mode','LIVE TWIN')
        if case not in self.cases or type(index) is not int or not 0<=index<len(self.cases[case]['timeline']):raise ValueError('Invalid demo case/index')
        if mode not in ['LIVE TWIN','HISTORICAL / DEMO REPLAY']:raise ValueError('Invalid mode')
        with self.lock:
            now=time.monotonic();self.sessions={k:v for k,v in self.sessions.items() if now-v.touched<self.TTL_SECONDS}
            if len(self.sessions)>=self.MAX_SESSIONS:raise RuntimeUnavailable('Session capacity reached')
            session=FusionSession(self.cases[case],self.reference,self.model_lock,index,mode,adapter=self.adapter)
            key=secrets.token_urlsafe(24);self.sessions[key]=session
        return {'session_id':key,**session.latest}
