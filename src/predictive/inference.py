"""Private Phase5-compatible prediction state and association explanations."""
import copy
import math
import numpy as np
from src.predictive.protocol import SIGNALS
from src.predictive.metrics import basic


def groups(keys):
    result={'ehr_demographics':[], 'ehr_documented_history':[], 'ehr_labs':[]}
    for key in keys:
        if key.startswith('ehr__'):
            group='ehr_documented_history' if 'documented_' in key else 'ehr_labs' if 'lab_' in key else 'ehr_demographics'
        else:group='__'.join(key.split('__')[:2])
        result.setdefault(group,[]).append(key)
    return {k:v for k,v in result.items() if v}


def background(rows,assignments=None):
    if assignments is not None:
        from src.predictive.data import guard
        guard(rows,assignments,{'train'})
    elif not rows or any(r.get('role')!='train' for r in rows):raise ValueError('Background reference requires TRAIN rows')
    result={}
    for key in rows[0]['features']:
        values=[r['features'][key] for r in rows if r['features'][key] is not None]
        if any(isinstance(v,str) for v in values):
            from collections import Counter
            result[key]=Counter(values).most_common(1)[0][0]
        else:result[key]=float(np.median(values)) if values else None
    return result


def occlude(rows,group,reference):
    changed=[];keys=groups(reference).get(group,[])
    for row in rows:
        r={**row,'features':dict(row['features']),'sequence':row['sequence'].copy()}
        for key in keys:r['features'][key]=reference[key]
        if group.startswith('phys__'):
            signal=group.split('__')[1];j=SIGNALS.index(signal)
            r['sequence'][:,j]=reference.get(f'phys__{signal}__360__median')
            if reference.get(f'phys__{signal}__360__median') is None:
                r['sequence'][:,j]=np.nan;r['sequence'][:,7+j]=0;r['sequence'][:,14+j]=1
        changed.append(r)
    return changed


def explain_validation(model,rows,reference,repeats=5):
    """Patient-level donor permutation on validation; never ranks from test."""
    base=model.predict(rows);baseline=basic([r['label'] for r in rows],base)['auprc'];result={}
    rng=np.random.default_rng(2026);ids=sorted({r['patient_id'] for r in rows});byid={pid:[r for r in rows if r['patient_id']==pid] for pid in ids}
    for group,keys in groups(reference).items():
        drops=[]
        for _ in range(repeats):
            donors=dict(zip(ids,rng.permutation(ids)));changed=[]
            for row in rows:
                donor=byid[donors[row['patient_id']]][0]
                r={**row,'features':dict(row['features']),'sequence':row['sequence'].copy()}
                for key in keys:r['features'][key]=donor['features'][key]
                if group.startswith('phys__'):
                    j=SIGNALS.index(group.split('__')[1]);r['sequence'][:,[j,7+j,14+j]]=donor['sequence'][:,[j,7+j,14+j]]
                changed.append(r)
            drops.append(baseline-basic([r['label'] for r in rows],model.predict(changed))['auprc'])
        result[group]={'validation_auprc_drop_mean':float(np.mean(drops)),'repeat_sd':float(np.std(drops)), 'repeats':repeats}
    return result


def uncertainty(bundle,rows):
    ensemble=bundle['ensemble']
    if not ensemble:return np.full((len(rows),2),np.nan)
    probabilities=np.array([c.predict(m.predict(rows)) for m,c in ensemble])
    return np.quantile(probabilities,[.05,.95],axis=0).T


def prediction_states(bundle,rows):
    model=bundle['models'][bundle['selected']];cal=bundle['calibrators'][bundle['selected']]
    raw=model.predict(rows);risk=cal.predict(raw);interval=uncertainty(bundle,rows)
    reference=bundle['background'];ranked=sorted(bundle['explanations'],key=lambda g:bundle['explanations'][g]['validation_auprc_drop_mean'],reverse=True)[:8]
    changes={g:risk-cal.predict(model.predict(occlude(rows,g,reference))) for g in ranked}
    threshold=bundle['thresholds'][bundle['selected']]
    result=[]
    for i,row in enumerate(rows):
        valid=bool(row['audit']['history_supported'] and row['audit']['washout_pass'] and row['audit']['buffered_latest_map'] is not None and not row['audit'].get('diagnostic_outside_target_landmark',False))
        uncertainty_values=interval[i].tolist() if np.all(np.isfinite(interval[i])) else None
        contributions=sorted([{'group':g,'risk_change_on_reference_occlusion':float(changes[g][i]),'interpretation':'association, not causation'} for g in ranked],key=lambda x:abs(x['risk_change_on_reference_occlusion']),reverse=True)
        personal=row['features'];map_prefix='personal__systemicmean__'
        state={'schema_version':'PredictionState_v1','patient_id':row['patient_id'],'icu_stay_id':row['icu_stay_id'],'timestamp_offset_minutes':float(row['t0']),
               'raw_risk':float(raw[i]),'calibrated_risk':float(risk[i]),'uncertainty_interval90':uncertainty_values,
               'confidence_state':'conditional_bootstrap_interval' if uncertainty_values is not None else 'unavailable',
               'operating_state':'outside_training_eligibility' if not valid else 'high_risk' if risk[i]>=threshold['balanced'] else 'below_balanced_threshold',
               'contributing_features':contributions,'current_map':row['audit']['map_at_t0'],
               'personal_deviation_summary':{'available':bool(personal[map_prefix+'available']),'map_deviation':personal[map_prefix+'current_deviation'],'map_standardized_deviation':personal[map_prefix+'standardized_deviation']},
               'prediction_horizon_minutes':180,'data_quality_state':{'eligible_history':bool(row['audit']['history_supported']),'washout_pass':bool(row['audit']['washout_pass']),
                     'maximum_physiology_offset':row['audit']['max_physiology_offset'],'input_buffer_minutes':5},
               'threshold_alerts':{k:bool(valid and risk[i]>=t) for k,t in threshold.items()},'model_name':bundle['selected'],'research_only':True}
        validate_state(state);result.append(state)
    return result


def validate_state(state):
    required={'schema_version','patient_id','timestamp_offset_minutes','raw_risk','calibrated_risk','uncertainty_interval90','confidence_state','operating_state','contributing_features','current_map','personal_deviation_summary','prediction_horizon_minutes','data_quality_state'}
    if not required<=set(state) or state['schema_version']!='PredictionState_v1':raise ValueError('Invalid PredictionState schema')
    for key in ['raw_risk','calibrated_risk']:
        if not math.isfinite(state[key]) or not 0<=state[key]<=1:raise ValueError('Invalid probability')
    if state['prediction_horizon_minutes']!=180:raise ValueError('Target horizon changed')
    interval=state['uncertainty_interval90']
    if interval is not None and (len(interval)!=2 or not 0<=interval[0]<=interval[1]<=1):raise ValueError('Invalid uncertainty interval')
    if state['data_quality_state']['maximum_physiology_offset']>state['timestamp_offset_minutes']-5:raise ValueError('Prediction crosses buffer')
    return True


def feature_associations(model,rows,repeats=3):
    """Feature-level validation permutation, patient donor shared across all windows."""
    if not hasattr(model,'encoder'):return {}
    keys=model.encoder.keys;baseline=basic([r['label'] for r in rows],model.predict(rows))['auprc']
    rng=np.random.default_rng(2026);ids=sorted({r['patient_id'] for r in rows});donor_rows={pid:next(r for r in rows if r['patient_id']==pid) for pid in ids}
    results={}
    for key in keys:
        drops=[]
        for _ in range(repeats):
            donors=dict(zip(ids,rng.permutation(ids)));changed=[]
            for row in rows:
                r={**row,'features':dict(row['features'])};r['features'][key]=donor_rows[donors[row['patient_id']]]['features'][key];changed.append(r)
            drops.append(baseline-basic([r['label'] for r in rows],model.predict(changed))['auprc'])
        results[key]={'validation_auprc_drop':float(np.mean(drops)),'repeat_sd':float(np.std(drops)), 'interpretation':'association, not causation; correlated features limit marginal permutation'}
    coefficients={}
    if hasattr(model.estimator,'coef_'):coefficients=dict(zip(model.encoder.names,model.estimator.coef_[0].tolist()))
    return {'feature_permutation':results,'encoded_standardized_logistic_coefficients':coefficients}
