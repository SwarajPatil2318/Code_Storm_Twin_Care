"""Heterogeneous virtual patients; no real individual is used as a template."""
import datetime as dt
import hashlib
import numpy as np
from scipy.special import expit
from src.synthetic.config import rng


def patient(index,config,seed):
    random=rng(seed,index,1);names=config['phenotype_order'];name=random.choice(names,p=config['phenotype_proportions']);p=config['phenotypes'][name]
    context={'age':float(np.clip(random.normal(*p['age']),*config['ehr']['age_bounds'])),'female':int(random.random()<p['female']),
      'bmi':float(np.clip(random.normal(*p['bmi']),*config['ehr']['bmi_bounds'])),'ef':float(random.uniform(*p['ef'])),
      'hf_severity':float(random.uniform(*p['severity'])),
      'hypertension':int(random.random()<p['htn']),'diabetes':int(random.random()<p['diabetes']),'ckd':int(random.random()<p['ckd'])}
    ehr=config['ehr']
    for med in ['beta_blocker','ace_arb','diuretic','mra','sglt2']:
        context[med]=int(random.random()<ehr[med+'_probability'])
    values={'creatinine':ehr['creatinine_base']+ehr['creatinine_ckd_add']*context['ckd']+random.normal(0,ehr['creatinine_sd']),
      'sodium':ehr['sodium_base']-ehr['sodium_severity_effect']*context['hf_severity']+random.normal(0,ehr['sodium_sd']),
      'hemoglobin':ehr['hemoglobin_base']-ehr['hemoglobin_ckd_effect']*context['ckd']+random.normal(0,ehr['hemoglobin_sd']),
      'bnp':random.lognormal(ehr['bnp_logmean']+ehr['bnp_severity_effect']*context['hf_severity'],ehr['bnp_logsd'])}
    for key,value in values.items():context[key]=float(np.clip(value,*ehr['lab_bounds'][key])) if random.random()>ehr['lab_missing_probability'] else None
    # Offset-coded chronology and separate random identity eliminate an ordinal-ID proxy.
    identifier=hashlib.sha256(f'synthetic_v1:{seed}:{index}:identity'.encode()).hexdigest()[:24]
    start=dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=int(random.integers(0,180)))
    return {'patient_id':'synthetic:'+identifier,'index':index,'profile':name,'context':context,
      'start_time':start.isoformat(),'vulnerability_residual':float(random.normal(0,config['vulnerability']['unobserved_logit_sd']))}


def vulnerability(patient,baseline,config):
    c=patient['context'];v=config['vulnerability'];meds=sum(c[k] for k in ['beta_blocker','ace_arb','diuretic','mra','sglt2'])/5
    score=v['intercept']+v['severity']*c['hf_severity']+v['age_per_decade']*(c['age']-65)/10+v['ckd']*c['ckd']+v['diabetes']*c['diabetes']+v['hypertension']*c['hypertension']
    score+=v['low_bp']*(100-baseline[4])/20-v['medication_protection']*meds+patient['vulnerability_residual']
    return float(expit(score))
