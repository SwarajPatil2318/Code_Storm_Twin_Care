"""Phase-7-ready contract with deterministic field-level provenance and evidence boundaries."""
import json,math
from typing import TypedDict
from .events import CATEGORIES

HORIZON_KEYS=['plus_30m','plus_60m','plus_90m','plus_120m']

class MetabolicTwinState(TypedDict):
    patient:dict
    timestamp:str
    current_glucose:float|None
    forecast:dict
    shadow:dict
    divergence:dict
    predicted_peak:dict
    spike:dict
    why_now:list
    recent_context:dict
    data_sources:list
    evidence:dict
    status:str
    data_quality:dict
    render_hints:dict

def validate(state):
    if set(state)!=set(MetabolicTwinState.__annotations__):raise ValueError('TwinState fields missing or unknown')
    if list(state['forecast'])!=HORIZON_KEYS:raise ValueError('Four ordered forecast horizons required')
    if not set(['current',*HORIZON_KEYS]).issubset(state['shadow']):raise ValueError('Five Shadow horizons required')
    forecast=list(state['forecast'].values())
    if any(v is None for v in forecast) and not all(v is None for v in forecast):raise ValueError('Partial forecast must not masquerade as complete output')
    if state['spike']['probability'] is not None and not 0<=state['spike']['probability']<=1:raise ValueError('Invalid probability')
    if state['predicted_peak']['approximate_horizon'] not in [None,30,60,90,120]:raise ValueError('Minute-precise peak unsupported')
    if all(v is not None for v in forecast):
        peak=max(forecast);horizon=[30,60,90,120][forecast.index(peak)]
        if state['predicted_peak']['glucose']!=peak or state['predicted_peak']['approximate_horizon']!=horizon:raise ValueError('Peak must match ordered forecast')
    shadow=[state['shadow'][key] for key in ['current',*HORIZON_KEYS]]
    if state['shadow']['available']!=all(v is not None for v in shadow):raise ValueError('Shadow availability/value mismatch')
    if not state['shadow']['available'] and any(v is not None for v in shadow):raise ValueError('Unavailable Shadow must remain null')
    fields=[item['field'] for item in state['data_sources']]
    required=set(state)-{'data_sources'}
    if not required.issubset(fields):raise ValueError('Every visible field needs provenance')
    if any(item['category'] not in CATEGORIES for item in state['data_sources']):raise ValueError('Unknown evidence category')
    if state['patient']['synthetic']:
        if not state['patient']['id'].startswith('metabolic-demo:'):raise ValueError('Synthetic identity namespace missing')
        if state['evidence']['input_evidence']!='SYNTHETIC DEMO CONTEXT':raise ValueError('Synthetic input mistaken for real validation')
    json.dumps(state,allow_nan=False)
    return state
