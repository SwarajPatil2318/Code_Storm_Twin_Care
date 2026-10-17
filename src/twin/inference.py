"""Frozen challenge definition, uncertainty and deterministic TwinState evidence."""
import numpy as np
from src.twin.protocol import PRIMARY_SIGNALS,PERSONAL_STATS,EVIDENCE
from src.synthetic.config import SIGNALS,UNITS
from src.twin.models import interval_probabilities,cumulative


def challenge_mask(data,config):
    definition=config['challenge'];normal_count=np.zeros(len(data['recent']),int);in_range=np.ones(len(data['recent']),bool)
    for signal,(lo,hi) in definition['population_ranges'].items():
        j=PRIMARY_SIGNALS.index(signal);value=data['recent'][:,j];observed=np.isfinite(value)&(data['recent_count'][:,j]>=definition['minimum_recent_count']);normal_count+=observed;in_range&=~observed|((value>=lo)&(value<=hi))
    z=(data['recent']-data['reference'])/np.maximum(data['scale'],1e-6);enough=(data['baseline_count']>=definition['minimum_baseline_count'])&(data['recent_count']>=definition['minimum_recent_count'])
    deviations=np.sum(enough&np.isfinite(z)&(abs(z)>=definition['minimum_personal_abs_z']),axis=1)
    return (normal_count>=definition['minimum_observed_range_signals'])&in_range&(deviations>=definition['minimum_personally_deviating_signals'])

def quality_uncertainty(data,indices=None):
    absolute=data['absolute'].reshape(len(data['absolute']),len(PRIMARY_SIGNALS),-1)
    selected=np.arange(len(PRIMARY_SIGNALS)) if indices is None else np.array(indices)
    coverage=absolute[:,selected,5];staleness=absolute[:,selected,6]/168;confidence=data['confidence'][:,selected]
    return np.clip(.5*(1-coverage.mean(axis=1))+.25*staleness.mean(axis=1)+.25*(1-confidence.mean(axis=1)),0,1)

def likely_window(hazards,threshold72):
    risk=cumulative(hazards);interval=interval_probabilities(hazards);names=['0-24h','24-48h','48-72h'];return ['no event expected within 72h' if p<threshold72 else names[int(np.argmax(q[:3]))] for p,q in zip(risk[:,2],interval)]

def finite(value):return float(value) if np.isfinite(value) else None

def twin_state(index,data,divergence,hazards,model_uncertainty,quality,thresholds,timestamp):
    p=cumulative(hazards[index:index+1])[0];observed={s:finite(data['recent'][index,j]) for j,s in enumerate(PRIMARY_SIGNALS)}
    expected={s:{'value':finite(divergence['expected'][index,j]),'lower':finite(divergence['lower'][index,j]),'upper':finite(divergence['upper'][index,j])} for j,s in enumerate(PRIMARY_SIGNALS)}
    deviations=[]
    for j,s in enumerate(PRIMARY_SIGNALS):
        if np.isfinite(divergence['z'][index,j]):
            actual=data['recent'][index,j];ref=divergence['expected'][index,j]
            deviations.append({'signal':s,'observed':finite(actual),'expected':finite(ref),'difference':finite(actual-ref),'relative_percent':finite((actual-ref)/abs(ref)*100) if abs(ref)>1e-6 else None,
              'standardized_residual':finite(divergence['z'][index,j]),'divergence_component':finite(divergence['components'][index,j]),'recent_48h_deviation_fraction':finite(data['persistence'][index,j]),'recent_count':int(data['recent_count'][index,j])})
    deviations.sort(key=lambda x:abs(x['standardized_residual']),reverse=True)
    missing=[{'signal':s,'received_recent_count':int(data['recent_count'][index,j]),'baseline_confidence':finite(data['confidence'][index,j])} for j,s in enumerate(PRIMARY_SIGNALS) if data['recent_count'][index,j]<2 or data['confidence'][index,j]<.5]
    changes=[{'signal':s,'recent_minus_prior_48h':finite(data['recent'][index,j]-data['prior_recent'][index,j])} for j,s in enumerate(PRIMARY_SIGNALS) if np.isfinite(data['recent'][index,j]-data['prior_recent'][index,j])]
    context=[{'feature':name,'value':finite(data['ehr'][index,k])} for k,name in enumerate(['age','female','bmi','ef','hf_severity','hypertension','diabetes','ckd','beta_blocker','ace_arb','diuretic','mra','sglt2','creatinine','sodium','hemoglobin','bnp']) if np.isfinite(data['ehr'][index,k])]
    mu=model_uncertainty[index];dq=float(quality[index]);confidence='limited' if dq>.5 else 'moderate' if max(mu)>.05 or dq>.25 else 'higher within synthetic POC'
    return {'patient_id':str(data['patient_ids'][index]),'timestamp':timestamp,'landmark_hour':int(data['landmark_hours'][index]),
      'risk':{'next_24h':float(p[0]),'next_48h':float(p[1]),'next_72h':float(p[2]),'likely_window':likely_window(hazards[index:index+1],thresholds['72']['balanced'])[0],'window_is_approximate':True},
      'divergence':{'tdi':finite(divergence['tdi'][index]),'trend':finite(divergence['tdi_features'][index,4]),'components':{s:finite(divergence['components'][index,j]) for j,s in enumerate(PRIMARY_SIGNALS)}},
      'observed_state':observed,'signal_units':{s:UNITS[SIGNALS.index(s)] for s in PRIMARY_SIGNALS},'state_aggregation':'mean of received usable observations in the preceding48h','expected_shadow_state':expected,'personalization':{'baseline_summary':{s:{'reference':finite(data['reference'][index,j]),'scale':finite(data['scale'][index,j]),'trusted_count':int(data['baseline_count'][index,j])} for j,s in enumerate(PRIMARY_SIGNALS)},'deviation_summary':deviations},
      'uncertainty':{'model_uncertainty':{'risk24_sd':float(mu[0]),'risk48_sd':float(mu[1]),'risk72_sd':float(mu[2]),'scope':'training-patient bootstrap model disagreement; not clinical confidence'},'data_quality_uncertainty':dq},
      'why_now':{'primary_driver':deviations[0]['signal'] if deviations else 'insufficient observed physiology','personal_deviations':deviations[:5],'temporal_changes':changes,'ehr_context':context,'missing_data':missing,'confidence':confidence,
        'interpretation':'Structured observed departures from expected physiology; descriptive evidence, not causal medical advice.'},**EVIDENCE}
