"""Coupled noisy physiological manifestations, independent non-event decoys."""
import numpy as np
from src.synthetic.config import rng,SIGNALS


def autoregressive(random,length,dimensions,rho):
    x=np.empty((length,dimensions));x[0]=random.normal(size=dimensions)
    innovation=random.normal(size=(length-1,dimensions))*np.sqrt(1-rho*rho)
    for t in range(1,length):x[t]=rho*x[t-1]+innovation[t-1]
    return x


def simulate(patient,baseline,state,config,seed,variant=None):
    random=rng(seed,patient['index'],4);p=config['physiology'];b=config['baseline'];length=config['days']*24;hours=np.arange(length);day=hours//24
    values=baseline['stable_daily'][day].copy();factor=autoregressive(random,length,6,b['hourly_ar']);daily_factor=autoregressive(random,config['days'],6,b['daily_ar'])
    load=np.array(b['factor_loadings_hr_bp_spo2_weight_temp']);home_noise=(b['hour_factor_weight']*factor@load.T+b['day_factor_weight']*daily_factor[day]@load.T)*np.array(b['daily_variability_sd'])*baseline['variance_multiplier']
    circle=np.sin(2*np.pi*(hours%24-baseline['circadian_phase_hours']-8)/24)*baseline['circadian_multiplier'];home_noise+=circle[:,None]*np.array(b['circadian_amplitudes'])
    for k,col in enumerate([0,2,3,5,7,10]):values[:,col]+=home_noise[:,k]
    # Hourly activity is a synthetic wrist summary; sleep is a daily morning proxy.
    wake,sleep=b['activity_waking_hours'];awake=np.maximum(0,np.sin(np.pi*((hours%24)-wake)/(sleep-wake)))
    activity_daily=np.exp(b['activity_day_logsd']*daily_factor[day,0]);activity=awake*activity_daily*np.exp(random.normal(0,p['steps_logsd'],length))
    # Normalize by the configured process expectation, never future samples.
    clock_reference=np.maximum(0,np.sin(np.pi*(np.arange(24)-wake)/(sleep-wake)))
    expected_activity=float(clock_reference.mean())*np.exp(.5*(b['activity_day_logsd']**2+p['steps_logsd']**2))
    activity/=max(expected_activity,1e-9);values[:,8]=baseline['daily_steps']/24*activity
    values[:,0]+=p['hr_activity_gain']*(activity-1)/3
    sleep_day=np.clip(baseline['mean'][9]+random.normal(0,p['sleep_daily_sd'],config['days'])-.15*daily_factor[:,0],3,10)
    values[:,9]=sleep_day[day];values[:,1]+=p['hrv_sleep_gain']*(values[:,9]-baseline['mean'][9])+random.normal(0,p['hrv_hourly_sd'],length)
    values[:,6]+=random.normal(0,p['rr_hourly_sd'],length)-.2*home_noise[:,3]
    latent=state['latent'];response=random.lognormal(0,p['path_response_logsd'],size=(1,4));a,c,pressure,ill=(latent*response).T
    values[:,0]+=p['hr_autonomic_gain']*a+p['hr_pressure_gain']*pressure
    values[:,1]+=p['hrv_autonomic_gain']*a;values[:,2]+=p['sbp_pressure_gain']*pressure;values[:,3]+=p['dbp_pressure_gain']*pressure
    values[:,5]+=p['spo2_congestion_gain']*c;values[:,6]+=p['rr_congestion_gain']*c
    values[:,7]+=p['weight_congestion_gain']*c;values[:,8]*=np.maximum(.25,1-p['activity_burden_fraction']*(a+c+pressure)/2)
    values[:,9]+=p['sleep_autonomic_gain']*a;values[:,10]+=p['temperature_illness_gain']*ill
    decoys=[];artifact_burst=np.zeros(length,bool);d=config['decoys'];decoy_random=rng(seed,patient['index'],5)
    for _ in range(decoy_random.poisson(d['episodes_per_60days']*config['days']/60)):
        start=int(decoy_random.integers(0,length-1));duration=int(decoy_random.uniform(*d['duration_hours']));end=min(length,start+duration);kind=str(decoy_random.choice(d['types']));strength=decoy_random.uniform(*d['strength_range'])
        shape=np.sin(np.linspace(0,np.pi,end-start));effect=np.array(d['effects'][kind])*strength
        values[start:end]+=shape[:,None]*effect[None,:]
        if kind in ['exercise','poor_sleep','mild_illness']:values[start:end,8]*=np.maximum(.2,1+effect[8]*shape)
        if kind=='sensor_artifact':artifact_burst[start:end]=True
        decoys.append({'start_hour':start,'end_hour':end,'kind':kind,'not_an_event_generator':True})
    # Positive activity and saturation have physical boundaries, not event thresholds.
    values[:,8]=np.maximum(values[:,8],0);values[:,5]=np.minimum(values[:,5],100)
    values[:,3]=np.minimum(values[:,3],values[:,2]-10);values[:,4]=(values[:,2]+2*values[:,3])/3
    before=values.copy();bounds=np.array([config['signal_bounds'][s] for s in SIGNALS]);values=np.clip(values,bounds[:,0],bounds[:,1]);values[:,4]=(values[:,2]+2*values[:,3])/3
    clipping=np.sum(before!=values,axis=0)
    return {'values':values.astype(np.float32),'decoys':decoys,'artifact_burst':artifact_burst,'clipping_counts':clipping}
