"""Precompute ordinary bounded synthetic scenarios using the actual frozen D2 reference."""
import datetime as dt,json,time
import numpy as np
from src.metabolic_d2.protocol import write_json,digest
from .reference import ROOT,Reference,assert_protected
from .engine import TwinEngine
from .events import CGMAdapter,MealAdapter,SmartwatchAdapter
from .fhir import FHIRAdapter

BASE=dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)
def when(i):return (BASE+dt.timedelta(minutes=15*int(i))).isoformat()

def synthetic_ehr(patient):
    identifier=patient['id'].split(':',1)[1];subject={'reference':'Patient/'+identifier}
    older=(BASE-dt.timedelta(days=20)).isoformat()
    resources=[{'resourceType':'Patient','id':identifier,'name':[{'given':[patient['name']],'family':'Demo'}],
        'birthDate':'1969-01-01','gender':'female'},
        {'resourceType':'Condition','id':'t2d','subject':subject,'code':{'text':'Type 2 diabetes — synthetic demo diagnosis'},'onsetDateTime':older,'recordedDate':older},
        {'resourceType':'MedicationRequest','id':'medication','subject':subject,'status':'active','intent':'order','authoredOn':older,
            'medicationCodeableConcept':{'text':'Metformin — synthetic medication order'}},
        {'resourceType':'Observation','id':'hba1c','subject':subject,'status':'final','effectiveDateTime':older,'issued':older,
            'code':{'text':'HbA1c — synthetic demo value'},'valueQuantity':{'value':7.2,'unit':'%','system':'http://unitsofmeasure.org','code':'%'}}]
    return {'resourceType':'Bundle','type':'collection','entry':[{'resource':r} for r in resources]}

def profile(base,kind,n=321):
    minute=np.arange(n)*15%1440
    values=base+4*np.sin(2*np.pi*minute/1440)
    for hour in [8,13,19]:values+=12*np.exp(-((minute-(hour*60+45))/45)**2)
    if kind=='predicted_spike':values[-9:]=base+np.array([0,-1,0,0,-2,-1,4,20,50])
    if kind=='personally_atypical':values[-17:]=base+np.r_[np.linspace(0,40,9),np.repeat(40,8)]
    return values.astype(float)

def build():
    reference=Reference();cases=[];timings=[]
    for kind,name,base in [('predicted_spike','Maya',120),('normal_response','Nora',120),('personally_atypical','Isha',100)]:
        patient={'id':'metabolic-demo:'+kind,'name':name,'synthetic':True,'label':'SYNTHETIC DEMO PATIENT',
            'context':'Type 2 diabetes — synthetic display context','source_scope':'All identity/CGM/wearable/EHR inputs are synthetic; historical D2 model evidence is real Shanghai validation.'}
        # The ordinary comparator is an overnight two-hour window with no new
        # meal event. Its daily physiology is the same bounded baseline profile.
        n=305 if kind=='normal_response' else 321
        visible_start=296 if kind=='normal_response' else 288
        engine=TwinEngine(patient,reference);frames=[];events=[];ready_timings=[];values=profile(base,kind,n)
        ehr=FHIRAdapter().adapt(synthetic_ehr(patient),patient_id=patient['id'],source='synthetic-fhir',event_id='ehr',available_time=when(0),synthetic=True)
        engine.ingest(ehr);events.append(ehr.payload())
        for i,value in enumerate(values):
            minute=i*15%1440
            if minute in [8*60,13*60,19*60] and not (kind=='personally_atypical' and i>=288):
                meal=MealAdapter().adapt(event_id='meal-'+str(i),patient_id=patient['id'],source='synthetic-meal-logger',observed_time=when(i),available_time=when(i),
                    synthetic=True,context={'meal_type':{8*60:'breakfast',13*60:'lunch',19*60:'dinner'}[minute]})
                engine.ingest(meal);events.append(meal.payload())
            if i%4==0:
                hr=SmartwatchAdapter().adapt(event_id='watch-'+str(i),patient_id=patient['id'],source='synthetic-watch',concept='heart_rate',
                    observed_time=when(i),available_time=when(i),value=float(72+3*np.sin(i/12)),unit='bpm',synthetic=True)
                engine.ingest(hr);events.append(hr.payload())
            reading=CGMAdapter().adapt(event_id='cgm-'+str(i),patient_id=patient['id'],source='synthetic-cgm',observed_time=when(i),available_time=when(i),
                value=float(value),unit='mg/dL',synthetic=True,native_period_minutes=15)
            state=engine.ingest(reading);events.append(reading.payload())
            if i>=visible_start:
                frames.append(state);ready_timings.append(engine.timings[-1])
        timings.extend(ready_timings)
        final=frames[-1]
        cases.append({'case':kind,'patient':patient,'timeline':frames,'events':events,
            'visible_history_hours':(n-visible_start-1)/4,'prior_personal_history_hours':visible_start/4,
            'construction':'Fixed bounded synthetic daily pattern. Normal comparator: ordinary 02:00–04:00 overnight window; other scenarios: declared rising/plateau departures. Integration illustration, not clinical validation.',
            'case_check':{'current_glucose':final['current_glucose'],'peak':final['predicted_peak']['glucose'],
                'spike_probability':final['spike']['probability'],'divergence':final['divergence']['magnitude'],'status':final['status']}})
        print(json.dumps({'synthetic_case':kind,'frames':len(frames),'case_check':cases[-1]['case_check']}),flush=True)
    spike,normal,atypical=cases
    final=spike['timeline'][-1]
    assert final['current_glucose']<=180 and final['predicted_peak']['glucose']>180
    assert final['predicted_peak']['glucose']>max(final['shadow'][k] for k in ['plus_30m','plus_60m','plus_90m','plus_120m'])
    assert final['spike']['probability']>=final['spike']['threshold']
    assert all(f['spike']['probability']<f['spike']['threshold'] and f['divergence']['magnitude']<.5 for f in normal['timeline'])
    final=atypical['timeline'][-1]
    assert final['current_glucose']<=180 and final['divergence']['magnitude']>=1 and final['status']=='PERSONALLY_ELEVATED_RESPONSE'
    latency={key:{'median':float(np.median([d[key] for d in timings])),'p95':float(np.quantile([d[key] for d in timings],.95))} for key in timings[0]}
    output={'schema_version':'metabolic-d3-v1','reference':{'name':'METABOLIC_D2_REFERENCE','model_hashes':reference.registration['model_hashes'],
        'historical_validation':{'label':'REAL OPEN-DATA MODEL VALIDATION','dataset':'ShanghaiT2DM','holdout_participants':20,
            'MAE_mg_dl':[9.354,16.360,21.022,23.796],'spike_AUPRC':.498,'spike_AUROC':.863,'scope':'Historical frozen D2 aggregate evidence; not validation of synthetic demos or D3 Shadow'}},
        'cases':cases,'latency_ms':latency,'limitations':['Synthetic demo inputs','Shadow assessed on development only','No clinical deployment validation','No commercial device connectors','Wearable/EHR excluded from frozen D2 inputs','CGMacros external evaluation unspent']}
    path=ROOT/'demo.json';write_json(path,output)
    write_json(ROOT/'demo_audit.json',{'sha256':digest(path),'cases':3,'frames':sum(len(c['timeline']) for c in cases),'latency_ms':latency,
        'latency_samples':len(timings),'latency_scope':'Only visible CGM-triggered updates; sequential local smoke benchmark',
        'reference_hashes':reference.registration['model_hashes'],'synthetic_only':True})
    assert_protected();return output

if __name__=='__main__':build()
