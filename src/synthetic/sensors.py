"""Different sampling opportunities, adherence, device errors and receipt delay."""
import numpy as np
from src.synthetic.config import rng,SIGNALS

QUALITY={'VALID':0,'MISSING':1,'SUSPECT':2,'INVALID':3}
REASONS={'NONE':0,'NOT_SCHEDULED':1,'DEVICE_REMOVED':2,'MISSING_NIGHT':3,'CHARGING':4,'ADHERENCE':5,'INVALID_READING':6,'REDUCED_SENSOR_SET':7}
SOURCES=['synthetic_wrist_hr','synthetic_night_hrv','synthetic_home_cuff','synthetic_home_cuff','synthetic_derived_cuff_map',
 'synthetic_home_oximeter','synthetic_home_resp_proxy','synthetic_home_scale','synthetic_wrist_activity','synthetic_sleep_proxy','synthetic_home_thermometer']


def observe(patient,baseline,state,physiology,config,seed,variant=None):
    variant=variant or {};random=rng(seed,patient['index'],6);c=config['sensors'];truth=physiology['values'];length=len(truth);hours=np.arange(length);clock=hours%24
    scheduled=np.zeros_like(truth,bool)
    for j,signal in enumerate(SIGNALS):
        kind=c['opportunities'][signal]
        scheduled[:,j]=True if kind=='hourly' else clock==6 if kind=='nightly' else (clock==8)|(clock==20) if kind in ['morning_evening','derived_bp'] else clock==8
    reasons=np.where(scheduled,0,1).astype(np.uint8);active=np.ones(length,bool)
    for _ in range(random.poisson(c['device_removed_start_per_day']*config['days'])):
        start=int(random.integers(0,length));duration=int(random.uniform(*c['device_removed_hours']));active[start:start+duration]=False
    wrist=[0,1,8,9];reasons[np.ix_(~active,wrist)]=REASONS['DEVICE_REMOVED']
    for day in range(config['days']):
        if random.random()<c['missing_night_probability']:
            idx=(hours//24==day)&((clock<8)|(clock>22));reasons[np.ix_(idx,wrist)]=REASONS['MISSING_NIGHT']
        charging=int(random.integers(*[int(x) for x in c['charging_hours']]));start=day*24+int(random.integers(9,17));reasons[start:start+charging,wrist]=REASONS['CHARGING']
    adherence=np.clip(baseline['adherence']*variant.get('adherence_multiplier',1)-c['state_adherence_penalty']*np.mean(state['latent'],axis=1),.1,.99)
    keep=random.random(truth.shape)<adherence[:,None]*(1-np.array(c['sensor_individual_missing_probability']))
    # Same cuff reading determines all three pressure availability masks.
    keep[:,3:5]=keep[:,2,None];reasons[:,3:5]=reasons[:,2,None]
    for signal in variant.get('remove',[]):reasons[:,SIGNALS.index(signal)]=REASONS['REDUCED_SENSOR_SET']
    reasons[(reasons==0)&~keep]=REASONS['ADHERENCE'];present=scheduled&(reasons==0)
    noise=random.normal(size=truth.shape)*np.array(c['measurement_noise_sd'])*variant.get('noise_multiplier',1)
    raw=truth.astype(float)+noise
    # Consumer count summaries have discrete, nonnegative ordinary counting noise.
    raw[:,8]=random.poisson(np.maximum(truth[:,8],0)).astype(float)
    noisy=random.random(truth.shape)<c['suspect_noise_probability'];raw+=np.where(noisy,noise*2,0)
    raw[:,5]=np.minimum(raw[:,5],100) # ordinary oximeter saturation/quantization ceiling
    raw[:,8]=np.maximum(raw[:,8],0) # ordinary count summaries cannot be negative
    artifact=(random.random(truth.shape)<c['artifact_probability'])|((physiology['artifact_burst'][:,None])&(random.random(truth.shape)<.12))
    bounds=np.array([config['quality_bounds'][signal] for signal in SIGNALS]);raw=np.where(artifact,bounds[:,1][None,:]+random.uniform(10,50,truth.shape),raw)
    quality=np.where(present,np.where(noisy,QUALITY['SUSPECT'],QUALITY['VALID']),QUALITY['MISSING']).astype(np.uint8)
    invalid=present&((raw<bounds[:,0])|(raw>bounds[:,1])|~np.isfinite(raw));quality[invalid]=QUALITY['INVALID'];reasons[invalid]=REASONS['INVALID_READING']
    inconsistent=present[:,2]&present[:,3]&(raw[:,2]<=raw[:,3]);quality[inconsistent,2:5]=QUALITY['INVALID'];reasons[inconsistent,2:5]=REASONS['INVALID_READING']
    raw[:,4]=(raw[:,2]+2*raw[:,3])/3
    valid_bp=np.isin(quality[:,2],[0,2])&np.isin(quality[:,3],[0,2]);quality[:,4]=np.maximum(quality[:,2],quality[:,3]);present[:,4]=present[:,2]&present[:,3]
    reasons[:,4]=np.maximum(reasons[:,2],reasons[:,3]);raw[~present]=np.nan
    usable=present&np.isin(quality,[QUALITY['VALID'],QUALITY['SUSPECT']]);values=np.where(usable,raw,np.nan)
    delay=np.where(random.random(truth.shape)<c['delay_probability'],random.uniform(*c['delay_hours'],truth.shape),random.uniform(0,.1,truth.shape))
    delay[:,3:5]=delay[:,2,None];available=np.where(present,hours[:,None]+delay,np.nan)
    return {'raw_value':raw.astype(np.float32),'value':values.astype(np.float32),'quality_status':quality,
      'missingness_mask':~usable,'scheduled':scheduled,'available_hour':available.astype(np.float32),'reason_code':reasons}
