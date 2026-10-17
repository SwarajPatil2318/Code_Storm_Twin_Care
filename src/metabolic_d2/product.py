"""Typed, private metabolic product state; descriptive Shadow, never meal simulation."""
from typing import TypedDict
import datetime as dt
import numpy as np

class MetabolicTwinState(TypedDict):
    patient_id: str
    timestamp: str
    current_glucose: float
    forecast: dict
    predicted_peak: dict
    spike: dict
    personal_response: dict
    context: dict
    data_sources: dict
    why_now: list


def shadow_reference(data):
    """Flat expected recent-personal baseline, not a counterfactual glucose simulator."""
    baseline=data['personal'][:,0]
    return np.where(data['warm'][:,None],np.repeat(baseline[:,None],4,axis=1),np.nan).astype('float32')


def make_state(data,index,prediction,probability,threshold,model_hash):
    p=np.asarray(prediction,float);current=float(data['sequence'][index,-1])
    warm=bool(data['warm'][index]);baseline=float(data['personal'][index,0]) if warm else None
    expected=[baseline]*4 if warm else None
    age=float(data['meal'][index,0])
    meal={'minutes_since_annotation':age,'availability':'event-time assumption; receipt unknown'} if np.isfinite(age) else None
    static=data['static'][index]
    clinical={'age_years':float(static[0]),'sex_female':bool(static[1]),'availability':'stable demographics at onboarding assumption'} if np.isfinite(static).all() else None
    state:MetabolicTwinState={
        'patient_id':str(data['pid'][index]),
        'timestamp':dt.datetime.fromtimestamp(int(data['time'][index])/1e9,dt.timezone.utc).isoformat(),
        'current_glucose':current,
        'forecast':dict(zip(['plus_30m','plus_60m','plus_90m','plus_120m'],p.tolist())),
        'predicted_peak':{'glucose':float(p.max()),'approximate_horizon':int([30,60,90,120][p.argmax()]),'scope':'maximum of four forecast horizons; may miss between-horizon peaks'},
        'spike':{'probability':float(probability),'status':'Elevated' if probability>=threshold else 'Below alert threshold','threshold':float(threshold),'definition':'current ≤180 mg/dL, then two adjacent observed readings >180 within 120 minutes'},
        'personal_response':{'expected_trajectory':expected,'predicted_today':p.tolist(),
            'deviation':(p-baseline).tolist() if warm else None,
            'method':'flat median of causal historical prefix; descriptive expected reference, not causal meal simulation',
            'available':warm},
        'context':{'recent_meal':meal,'activity_if_available':None,'clinical_context':clinical},
        'data_sources':{'glucose':'raw ShanghaiT2DM, native observed CGM','units':'mg/dL',
            'receipt_policy':'offline observation-time availability assumption; scan/receipt not recorded',
            'meal':'source meal annotation, no fabricated nutrient values',
            'model_sha256':model_hash,'illustrative_anatomy_only':True},
        'why_now':[{'signal':'recent_glucose_slope','value':float((data['sequence'][index,-1]-data['sequence'][index,-3])/30),
            'unit':'mg/dL/min','meaning':'Observed change over the last 30 minutes; descriptive input, not a causal attribution.'}]
    }
    if warm:state['why_now'].append({'signal':'departure_from_personal_baseline','value':current-baseline,'unit':'mg/dL',
        'meaning':'Current glucose minus the median of prior observations ending before the input window.'})
    validate_state(state);return state


def validate_state(state):
    required=set(MetabolicTwinState.__annotations__)
    if set(state)!=required:raise ValueError('Incomplete metabolic state contract')
    if not state['patient_id'].startswith('ShanghaiT2D:'):raise ValueError('Participant namespace missing')
    dt.datetime.fromisoformat(state['timestamp'])
    if set(state['forecast'])!={'plus_30m','plus_60m','plus_90m','plus_120m'}:raise ValueError('Forecast horizons missing')
    values=[state['current_glucose'],*state['forecast'].values(),state['predicted_peak']['glucose'],state['spike']['probability']]
    if not np.isfinite(values).all():raise ValueError('Nonfinite product values')
    if not 0<=state['spike']['probability']<=1:raise ValueError('Invalid probability')
    if state['predicted_peak']['approximate_horizon'] not in [30,60,90,120]:raise ValueError('Unsupported peak precision')
    response=state['personal_response']
    if response['available']:
        if any(len(response[k])!=4 for k in ['expected_trajectory','predicted_today','deviation']):raise ValueError('Personal response horizons missing')
        if not np.isfinite([*response['expected_trajectory'],*response['deviation']]).all():raise ValueError('Invalid personal reference')
    elif response['expected_trajectory'] is not None or response['deviation'] is not None:raise ValueError('Unavailable personal reference invented')
    return state
