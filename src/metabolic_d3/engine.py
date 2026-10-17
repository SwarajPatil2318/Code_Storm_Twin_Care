"""Canonical event -> causal features -> frozen D2 -> Shadow/WhyNow -> TwinState."""
import datetime as dt
import json,time
import numpy as np
from .buffer import Buffer
from .events import Event
from .reference import Reference,NAME
from .shadow import Shadow,divergence,CONFIG
from .state import HORIZON_KEYS,validate

def clock_string(ns,basis):
    value=str(np.datetime_as_string(np.datetime64(int(ns),'ns'),unit='ns'))
    return value+'Z' if basis=='UTC' else value

class TwinEngine:
    def __init__(self,patient,reference=None,clock_offset_minutes=0):
        self.patient=dict(patient);self.buffer=Buffer(patient['id'],clock_offset_minutes)
        self.reference=reference or Reference();self.shadow=Shadow();self.last_state=None;self.timings=[]
    def ingest(self,event):
        if bool(event.synthetic)!=bool(self.patient['synthetic']):raise ValueError('Patient/input evidence mismatch')
        changed=self.buffer.ingest(event)
        if not changed:return self.last_state
        self.last_state=self.snapshot();return self.last_state
    def snapshot(self,cutoff=None):
        cutoff=self.buffer.as_of if cutoff is None else cutoff
        if cutoff is None:raise ValueError('No received stream event')
        start=time.perf_counter();data,quality=self.buffer.features(cutoff)
        expected=self.shadow.predict(self.buffer,cutoff)
        prepared=time.perf_counter();pred=None;probability=None;mode=None
        if data is not None:pred,probability,mode=self.reference.predict(data)
        predicted=time.perf_counter()
        t,g,events,conflicts=self.buffer.glucose(cutoff)
        current=float(g[-1]) if len(g) else None
        received=self.buffer.visible(cutoff);contexts={}
        for concept in ['meal','heart_rate','steps','activity','ehr']:
            items=[v for v in received if v.concept==concept]
            contexts[concept]=items[-1] if items else None
        meal=contexts['meal'];ehr=contexts['ehr']
        if meal is not None and cutoff-meal.observed_ns>CONFIG['shadow_maximum_meal_age_minutes']*60_000_000_000:meal=None
        meal_context=None if meal is None else {'minutes_ago':float((cutoff-meal.observed_ns)/60_000_000_000),
            'meal_type':(meal.context or {}).get('meal_type'),'synthetic':meal.synthetic,'source':meal.source,'predictive_use':'Shadow context only; not a D2 input'}
        activity={name:{'value':v.value,'unit':v.unit,'observed_time':v.observed_time,'source':v.source,'synthetic':v.synthetic,'predictive_use':False}
            for name in ['heart_rate','steps','activity'] if (v:=contexts[name]) is not None}
        clinical={} if ehr is None else ehr.context
        why=[];d={'magnitude':None,'label':'Forecast or personal history unavailable','components':None}
        if current is not None:why.append({'signal':'current_glucose','value':current,'unit':'mg/dL','source':'CGM',
            'meaning':'Most recent usable received value; its observation time and age are reported separately.'})
        if data is not None:
            slope=float(data['x'][0,10]);acceleration=float(data['x'][0,18])
            why.extend([{'signal':'recent_slope','value':slope,'unit':'mg/dL/min','source':'received CGM',
                'meaning':'Observed last-30-minute change; not a cause of the forecast.'},
                {'signal':'recent_acceleration','value':acceleration,'unit':'mg/dL/min²','source':'received CGM',
                'meaning':'Change in adjacent observed 15-minute slopes.'}])
            d=divergence(current,pred,expected)
            if expected['available']:
                why.extend([{'signal':'departure_from_expected_now','value':current-expected['trajectory'][0],'unit':'mg/dL','source':'Shadow Twin + CGM',
                    'meaning':'Current glucose compared with a received-history personal expectation.'},
                    {'signal':'future_personal_departure','value':float(np.mean(pred-np.asarray(expected['trajectory'][1:]))),'unit':'mg/dL','source':'D2 + Shadow Twin',
                    'meaning':'Mean predicted future departure; descriptive comparison, not a causal attribution.'}])
                typical=float(data['personal'][0,4])
                if np.isfinite(typical):why.append({'signal':'slope_departure_from_personal_norm','value':slope-typical,
                    'observed':slope,'expected':typical,'unit':'mg/dL/min','source':'CGM + causal personal prefix',
                    'meaning':'Recent observed slope compared with the median adjacent-observation slope in the causal historical prefix.'})
        if meal_context is not None:why.append({'signal':'recent_meal','value':meal_context['minutes_ago'],'unit':'minutes','source':meal.source,
            'meaning':'Received meal timing is context; the frozen D2 forecaster does not use it. No meal cause is inferred.'})
        if data is None:status='INSUFFICIENT_DATA'
        elif current>180:status='CURRENT_GLUCOSE_ELEVATED'
        elif probability>=self.reference.risk.threshold:status='SPIKE_WATCH'
        elif d['magnitude'] is not None and d['magnitude']>=1 and np.mean(pred-np.asarray(expected['trajectory'][1:]))>0:status='PERSONALLY_ELEVATED_RESPONSE'
        elif slope>=CONFIG['rising_display_slope_mg_dl_per_minute']:status='RISING'
        else:status='STABLE'
        risk_status='Unavailable' if probability is None else 'Already above 180; incident rule not applicable' if current>180 else 'Above frozen alert threshold' if probability>=self.reference.risk.threshold else 'Below frozen alert threshold'
        shadow_values=expected['trajectory'] if expected['available'] else [None]*5
        state={'patient':self.patient,'timestamp':clock_string(cutoff,self.buffer.time_basis),'current_glucose':current,
            'forecast':dict(zip(HORIZON_KEYS,[None]*4 if pred is None else pred.astype(float).tolist())),
            'shadow':{'current':shadow_values[0],**dict(zip(HORIZON_KEYS,shadow_values[1:])),
                'available':expected['available'],'method':expected['method'],'support':expected['support'],
                'scope':'Personal historical expectation; not a validated intervention simulation'},
            'divergence':d,'predicted_peak':{'glucose':None if pred is None else float(pred.max()),
                'approximate_horizon':None if pred is None else int([30,60,90,120][pred.argmax()]),'scope':'Maximum of four forecast horizons; approximate timing only'},
            'spike':{'probability':probability,'status':risk_status,'threshold':self.reference.risk.threshold,
                'expected_time_window_minutes':[0,120],'definition':'Current ≤180, then two adjacent native CGM readings >180 within 120 minutes',
                'timing_scope':'Probability over the full 2-hour window; no predicted exact event time'},
            'why_now':why,'recent_context':{'meal':meal_context,'activity':activity or None,'medication':clinical.get('medications') or None,
                'ehr':clinical or None},'data_sources':[],
            'evidence':{'model_version':NAME,'evidence_label':'REAL OPEN-DATA MODEL VALIDATION',
                'input_evidence':'SYNTHETIC DEMO CONTEXT' if self.patient['synthetic'] else 'RECEIVED SOURCE OBSERVATIONS',
                'shadow_evidence':'EXPLORATORY PERSONAL EXPECTED','model_hashes':self.reference.registration['model_hashes'],
                'forecast_mode':mode,'clinical_status':'NOT CLINICALLY VALIDATED','wearable_EHR_predictive_use':False},
            'status':status,'data_quality':quality,
            'render_hints':{'symbolic_blood_flow':True,'flow_scale_scope':'Illustrative glucose display; not measured hemodynamics',
                'current_flow':None if current is None else float(np.clip(current/120,.25,3)),
                'expected_flow':None if shadow_values[0] is None else float(np.clip(shadow_values[0]/120,.25,3)),
                'predicted_flow':dict(zip(HORIZON_KEYS,[None]*4 if pred is None else np.clip(pred/120,.25,3).astype(float).tolist())),
                'expected_future_flow':dict(zip(HORIZON_KEYS,[None]*4 if not expected['available'] else np.clip(np.array(shadow_values[1:])/120,.25,3).tolist())),
                'forecast_horizons_minutes':[30,60,90,120],'stream_label':'SIMULATED LIVE STREAM' if self.patient['synthetic'] else 'RECEIVED CGM STREAM'}}
        for field in set(state)-{'data_sources'}:
            category='MODEL PREDICTED' if field in ['forecast','predicted_peak','spike','status','render_hints','why_now','evidence'] else 'PERSONAL EXPECTED' if field in ['shadow','divergence'] else 'STATIC CLINICAL' if field in ['patient','recent_context'] else 'SIMULATED LIVE STREAM' if self.patient['synthetic'] else 'OBSERVED'
            state['data_sources'].append({'field':field,'category':category,'source':NAME if category=='MODEL PREDICTED' else 'Shadow Twin' if category=='PERSONAL EXPECTED' else 'Received context / identity' if category=='STATIC CLINICAL' else self.buffer.cgm_source or 'Canonical stream',
                'synthetic_input':self.patient['synthetic'],'method':'Deterministic derived display' if field in ['divergence','status','render_hints','why_now'] else 'Received observation / context' if category in ['OBSERVED','SIMULATED LIVE STREAM','STATIC CLINICAL'] else 'Frozen inference / causal historical expectation'})
        observation_category='SIMULATED LIVE STREAM' if self.patient['synthetic'] else 'OBSERVED'
        for index,item in enumerate(why):
            category='PERSONAL EXPECTED' if item['signal'] in ['departure_from_expected_now','future_personal_departure','slope_departure_from_personal_norm'] else observation_category
            state['data_sources'].append({'field':f'why_now[{index}]','category':category,'source':item['source'],
                'synthetic_input':self.patient['synthetic'],'method':'Deterministic comparison of the stated evidence; not feature attribution'})
        for field,category,source in [('recent_context.meal',observation_category,'Received meal logger'),
            ('recent_context.activity',observation_category,'Optional wearable context; excluded from D2'),
            ('recent_context.medication','STATIC CLINICAL','FHIR MedicationRequest; not administration'),
            ('recent_context.ehr','STATIC CLINICAL','SYNTHETIC DEMO EHR' if self.patient['synthetic'] else 'Received same-identity FHIR context')]:
            state['data_sources'].append({'field':field,'category':category,'source':source,'synthetic_input':self.patient['synthetic'],
                'method':'Received context; original timing/coding preserved, not a D2 predictive input'})
        state['data_sources'].sort(key=lambda item:item['field'])
        validate(state);serialized=time.perf_counter()
        self.timings.append({'feature_preparation_ms':(prepared-start)*1000,'prediction_ms':(predicted-prepared)*1000,
            'TwinState_serialization_ms':(serialized-predicted)*1000,'total_ms':(serialized-start)*1000})
        return state
