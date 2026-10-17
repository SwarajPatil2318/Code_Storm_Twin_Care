"""D3 structured signal semantics for the generalized patient state."""
import numpy as np
from src.metabolic_d3.shadow import Shadow, divergence, CONFIG
from src.metabolic_d3.state import HORIZON_KEYS


def explain(buffer, cutoff, data, predicted, meal_context, wearable):
    expected=Shadow().predict(buffer,cutoff)
    _,glucose,_,_=buffer.glucose(cutoff)
    current=float(glucose[-1]);slope=float(data['x'][0,10]);acceleration=float(data['x'][0,18])
    comparison=divergence(current,predicted,expected)
    why=[{'signal':'current_glucose','value':current,'unit':'mg/dL','source':'CGM',
            'meaning':'Most recent usable received value; its observation time and age are reported separately.'},
        {'signal':'recent_slope','value':slope,'unit':'mg/dL/min','source':'received CGM',
            'meaning':'Observed last-30-minute change; not a cause of the forecast.'},
        {'signal':'recent_acceleration','value':acceleration,'unit':'mg/dL/min²','source':'received CGM',
            'meaning':'Change in adjacent observed 15-minute slopes.'}]
    if expected['available']:
        why.extend([{'signal':'departure_from_expected_now','value':current-expected['trajectory'][0],'unit':'mg/dL','source':'Shadow Twin + CGM',
            'meaning':'Current glucose compared with a received-history personal expectation.'},
            {'signal':'future_personal_departure','value':float(np.mean(predicted-np.asarray(expected['trajectory'][1:]))),'unit':'mg/dL','source':'D2 + Shadow Twin',
            'meaning':'Mean predicted future departure; descriptive comparison, not a causal attribution.'}])
        typical=float(data['personal'][0,4])
        if np.isfinite(typical):why.append({'signal':'slope_departure_from_personal_norm','value':slope-typical,
            'observed':slope,'expected':typical,'unit':'mg/dL/min','source':'CGM + causal personal prefix',
            'meaning':'Recent observed slope compared with the median adjacent-observation slope in the causal historical prefix.'})
    for item in meal_context.get('evidence',[]):
        why.append({'signal':'recent_meal','value':item['minutes_since_meal'],'unit':'minutes','source':'Confirmed meal logger',
            'meaning':'Received meal timing is context; the frozen D2 forecaster does not use it. No meal cause is inferred.'})
    if wearable:
        why.append({'signal':'received_wearable_context','value':len(wearable),'unit':'observations','source':'Wearable',
            'meaning':'Received activity context; excluded from both trained feature vectors. No cause is inferred.'})
    shadow_values=expected['trajectory'] if expected['available'] else [None]*5
    shadow={'current':shadow_values[0],**dict(zip(HORIZON_KEYS,shadow_values[1:])),
        'available':expected['available'],'method':expected['method'],'support':expected['support'],
        'scope':'Personal historical expectation; not a validated intervention simulation'}
    status='CURRENT_GLUCOSE_ELEVATED' if current>180 else 'PERSONALLY_ELEVATED_RESPONSE' if comparison['magnitude'] is not None and comparison['magnitude']>=1 and np.mean(predicted-np.asarray(expected['trajectory'][1:]))>0 else 'RISING' if slope>=CONFIG['rising_display_slope_mg_dl_per_minute'] else 'STABLE'
    return shadow,comparison,why,status
