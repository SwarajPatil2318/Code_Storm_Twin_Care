"""Population covariance, personal variance, healthy drift and circadian context."""
import numpy as np
from scipy.special import ndtr
from src.synthetic.config import rng,SIGNALS


def baseline(patient,config,grounding,seed):
    random=rng(seed,patient['index'],2);population=grounding['perheart']['population_baseline'];b=config['baseline'];profile=config['phenotypes'][patient['profile']]
    center=np.array(population['median_of_patient_medians'])
    z=random.multivariate_normal(np.zeros(6),np.array(population['shrunken_correlation']))
    home=np.array([np.interp(ndtr(z[j]),population['marginal_quantile_levels'],population['marginal_quantiles_patient_medians'][j]) for j in range(6)])
    home=center+(home-center)*np.array(b['home_population_sd_multiplier'])+np.array(b['home_population_location_offset'])+np.array(profile['offset']);bounds=np.array(b['mean_bounds']);home=np.clip(home,bounds[:,0],bounds[:,1])
    # MAP is derived from the same virtual cuff pressure pair, never sampled independently.
    hr,sbp,dbp,spo2,weight,temp=home;dbp=min(dbp,sbp-15)
    context=patient['context'];hrv=np.clip(random.normal(*b['hrv_mean_sd'])+b['hrv_age_effect_per20years']*(context['age']-65)/20,*b['hrv_mean_bounds'])
    rr=np.clip(random.normal(*b['rr_mean_sd']),*b['rr_mean_bounds']);daily_steps=np.clip(random.normal(*profile['steps']),*b['steps_per_day_bounds'])
    sleep=np.clip(random.normal(*b['sleep_mean_sd']),*b['sleep_mean_bounds'])
    mean=np.array([hr,hrv,sbp,dbp,(sbp+2*dbp)/3,spo2,rr,weight,daily_steps/24,sleep,temp])
    days=config['days'];drift=np.zeros((days,6));increments=random.normal(size=(days-1,6))*np.array(b['stable_baseline_daily_drift_sd'])
    drift[1:]=np.cumsum(increments,axis=0);daily=np.repeat(mean[None],days,axis=0)
    for source,col in enumerate([0,2,3,5,7,10]):daily[:,col]+=drift[:,source]
    daily[:,4]=(daily[:,2]+2*daily[:,3])/3
    return {'mean':mean,'stable_daily':daily,'variance_multiplier':float(random.lognormal(0,b['personal_variance_multiplier_logsd'])),
      'circadian_phase_hours':float(random.normal(0,1.2)),'circadian_multiplier':float(random.uniform(.7,1.3)),
      'daily_steps':float(daily_steps),'adherence':float(random.beta(*config['sensors']['adherence_beta']))}
