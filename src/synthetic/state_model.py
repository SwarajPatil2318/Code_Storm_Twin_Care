"""Interpretable stochastic semi-Markov episodes and latent-only event hazard."""
import numpy as np
from src.synthetic.config import rng

STATES=['STABLE','EARLY_STRESS','COMPENSATING','EARLY_DETERIORATION','WORSENING','ADVERSE_EVENT','RECOVERY']


def event_hazard(burden,vulnerability,hours,config):
    c=config['states'];return float(1-np.exp(-c['event_hazard_per_hour']*np.exp(c['event_vulnerability_gain']*(vulnerability-.5)+
      c['event_burden_gain']*burden+c['event_duration_gain']*min(hours/168,1))))


def simulate(patient,vulnerability,config,seed):
    random=rng(seed,patient['index'],3);settings=config['states'];length=config['days']*24;state=np.zeros(length,np.uint8);latent=np.zeros((length,4));hazard=np.zeros(length)
    paths=list(config['paths']);prob=[config['paths'][p]['probability'] for p in paths];episodes=[];event_time=None
    start=settings['stable_warmup_hours']+int(random.exponential(24/settings['episode_start_rate_per_day']))
    if random.random()<settings['never_episode_fraction']:start=length
    while start<length-12:
        rapid=random.random()<settings['rapid_fraction'];duration=int(random.uniform(*(settings['rapid_duration_hours'] if rapid else settings['gradual_duration_hours'])))
        name=random.choice(paths,p=prob);load=np.array(config['paths'][name]['loadings']);peak=random.uniform(*settings['burden_peak_range']);recover=random.random()<settings['recovery_probability']
        recovery_fraction=random.uniform(*settings['recovery_start_fraction']);stop=min(length,start+int(duration*(recovery_fraction if recover else 1)));transitions=[];episode_event=None
        for hour in range(start,stop):
            progress=(hour-start)/max(duration,1);level=peak*(.15+.85*progress)+random.normal(0,settings['burden_variation_sd']);level=max(0,level)
            latent[hour]=level*load;new=int(np.searchsorted(settings['state_progress_fractions'],progress,side='right'))+1
            state[hour]=new
            if not transitions or transitions[-1]['state']!=STATES[new]:transitions.append({'hour':hour,'state':STATES[new]})
            # Event randomness is independent of observation/sensor noise and decoy draws.
            if new>=3 and event_time is None:
                hazard[hour]=event_hazard(level,vulnerability,hour-start,config)
                if random.random()<hazard[hour]:
                    event_time=float(hour+random.uniform(.01,.99));episode_event=event_time;state[hour]=5
                    transitions.append({'hour':event_time,'state':'ADVERSE_EVENT'});stop=hour+1;break
        recovery_hours=int(random.uniform(*settings['recovery_hours']));recovery_end=min(length,stop+recovery_hours)
        initial=latent[stop-1].copy() if stop>start else np.zeros(4)
        for hour in range(stop,recovery_end):
            state[hour]=6;latent[hour]=initial*np.exp(-(hour-stop+1)/(recovery_hours/3))
        if stop<length:transitions.append({'hour':stop,'state':'RECOVERY'})
        if recovery_end<length:transitions.append({'hour':recovery_end,'state':'STABLE'})
        episodes.append({'onset_hour':float(start),'path':str(name),'rapid':bool(rapid),'planned_duration_hours':duration,
          'transition_times':transitions,'event_hour':episode_event,'recovery_hour':float(stop),'recovered_without_event':episode_event is None,'recovery_end_hour':float(recovery_end)})
        # After the first event, retain observed recovery and stable follow-up; future labels
        # are restricted to at-risk, pre-event landmarks, with no recurrent-event claim.
        if event_time is not None:break
        start=recovery_end+int(random.uniform(*settings['interepisode_quiet_hours']))+int(random.exponential(24/settings['episode_start_rate_per_day']))
    return {'state':state,'latent':latent,'event_hazard':hazard,'episodes':episodes,'event_time':event_time,'vulnerability':vulnerability}
