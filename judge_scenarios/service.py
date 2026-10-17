"""Persisted input scheduler over unchanged P1–P8 services."""
from pathlib import Path
from datetime import timedelta
import hashlib,json,threading,time,uuid
from unittest.mock import patch
from src.patient_domain.repository import Repository
from patient_ingestion.service import IngestionService
from patient_meals.service import Meals
from patient_inference.readiness import InferenceReadiness
from patient_inference.connections import InferenceConnections
from patient_alerts.runtime import AlertInference
from patient_connections.contracts import CanonicalEvent,clock,check_owner
from patient_experience.views import ProductViews
from .clock import ReplayClock
from .catalog import START,PROFILES,ROOT

def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700);p.parent.chmod(0o700)
 p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');p.chmod(0o600)

class Scenarios:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True,mode=0o700);self.root.chmod(0o700)
  self.clock=ReplayClock();self.lock=threading.RLock();self.wakeup=threading.Event();self.stop_event=threading.Event()
  self.repo=Repository(self.root/'patients.sqlite3');self.records=IngestionService(self.repo,self.root/'records')
  self.meals=Meals(self.repo,self.root/'photos',time=self.clock.now)
  self.readiness=InferenceReadiness(self.meals,time=self.clock.now);self.runtime=AlertInference(self.readiness)
  self.connections=InferenceConnections(self.runtime);self.views=ProductViews(self.runtime,self.connections,self.records)
  self.sessions={};self.configs={}
  with self.repo.connection() as db:
   db.execute('CREATE TABLE IF NOT EXISTS judge_sessions(scenario TEXT PRIMARY KEY,data TEXT NOT NULL CHECK(json_valid(data)))')
   for row in db.execute('SELECT scenario,data FROM judge_sessions'):
    s=json.loads(row['data']);s['status']='PAUSED' if s['status']=='RUNNING' else s['status'];self.sessions[row['scenario']]=s
  self.worker=None
 def install(self,key,config):
  if key not in PROFILES:raise ValueError('Unknown scenario')
  forbidden={'forecast','risk','spike','predicted_peak','shadow','divergence','why_now','alert_trigger','alert_time'}
  def audit(v):
   if isinstance(v,dict):
    if forbidden.intersection(v):raise ValueError('Scenario config must contain inputs only')
    for x in v.values():audit(x)
   elif isinstance(v,list):
    for x in v:audit(x)
  audit(config)
  allowed={'candidate','source_kind','source_file','source_sha256','dataset','source_subject_namespace','recording_namespace','native_period_minutes','points','meal_minute','source_identity_link','input_prescreen_score'}
  if set(config)-allowed:raise ValueError('Only documented input configuration fields are allowed')
  if config.get('source_identity_link') is not False or config.get('native_period_minutes')!=15:raise ValueError('Unlinked native 15-minute trajectory required')
  points=config.get('points',[])
  if not points or any(set(p)-{'minute','value','source_observed_at','source_available_at','availability_basis','original_value','original_unit','quality','source_quality'} for p in points):raise ValueError('Canonical source input points required')
  minutes=[p['minute'] for p in points]
  if minutes!=sorted(set(minutes)):raise ValueError('Source times must be unique and increasing')
  if not any(m<=0 for m in minutes) or not any(m>0 for m in minutes):raise ValueError('Personal prefix and future inputs required')
  if not 0<=config.get('meal_minute',0)<min(m for m in minutes if m>0):raise ValueError('Meal must precede the first future sample')
  self.configs[key]=json.loads(json.dumps(config,allow_nan=False))
  if key in self.sessions and self.sessions[key]['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():
   self.sessions[key]['status']='ERROR';self.sessions[key]['error']='Input configuration changed; reset required';self._persist(key)
 def _persist(self,key):
  s=self.sessions[key]
  with self.repo.connection() as db:db.execute('INSERT INTO judge_sessions VALUES(?,?) ON CONFLICT(scenario) DO UPDATE SET data=excluded.data',(key,json.dumps(s,allow_nan=False)))
 def for_patient(self,pid):return next((k for k,s in self.sessions.items() if s['patient_id']==pid),None)
 def reference(self,key):return clock(self.sessions[key]['reference_time'])
 def _event(self,pid,point,config):
  at=START+timedelta(minutes=point['minute'])
  return CanonicalEvent(patient_id=pid,source_type='CGM',provider_type='replay_adapter',event_type='CGM_READING',observed_at=at.isoformat(),received_at=at.isoformat(),sequence_id=config['candidate']+':'+str(point['minute']),payload={'glucose_mg_dl':point['value'],'timestamp':at.isoformat()},quality={'status':point['quality']},raw_reference=config['source_file']+'#'+str(point['minute']),provenance={'origin':config['source_kind'],'synthetic':True,'dataset':config['dataset'],'source_file_sha256':config['source_sha256'],'source_observed_at':point['source_observed_at'],'source_available_at':point.get('source_available_at'),'source_availability_basis':point.get('availability_basis','fixture supplied'),'source_quality':point.get('source_quality'),'original_value':point['original_value'],'original_unit':point['original_unit'],'native_period_minutes':15,'recorded_source_is_synthetic':config['source_kind']=='synthetic_replay','replay_clock_rebased':True,'receipt_basis':'controlled native replay arrival at shifted observation time','patient_mapping':'synthetic judge identity; imported source trajectory has no clinical identity link'}).view(as_of=self.clock.now())
 def _wearable(self,pid):
  at=self.clock.stamp();events=[]
  for kind,payload in [('HEART_RATE',{'bpm':74}),('ACTIVE_MINUTES',{'minutes':12,'level':'light','window_minutes':90})]:
   events.append(CanonicalEvent(patient_id=pid,source_type='WEARABLE',provider_type='simulated_wearable',event_type=kind,observed_at=at,received_at=at,payload=payload,quality={'status':'GOOD'},provenance={'origin':'explicit judge wearable context','synthetic':True}).view(as_of=self.clock.now()))
  self.connections.ingest(pid,'WEARABLE',events)
 def _meal(self,pid,key):
  d=self.meals.manual_draft(pid);fields={'title':{'A':'Lunch: rice, beans and vegetables','B':'Balanced lunch','C':'Usual lunch'}[key],'foods':[{'name':'Rice and vegetables','estimated_portion_description':'One medium bowl','confidence':None,'carb_relevance':'HIGH' if key=='A' else 'MODERATE'}],'meal_occurred_at':self.clock.stamp(),'carbohydrate_category':'HIGH' if key=='A' else 'MODERATE','glycemic_impact_category':'HIGH' if key=='A' else 'MODERATE','estimated_carb_range_g':None,'user_notes':'Synthetic judge meal context; no causal claim.'}
  d=self.meals.review(pid,d['draft_id'],d['revision'],fields);return self.meals.confirm(pid,d['draft_id'],d['revision'],True)
 def reset(self,key):
  with self.lock:
   try:return self._reset(key)
   except Exception as error:
    config=self.configs[key]
    self.sessions[key]=self.sessions.get(key,{'scenario':key,'patient_id':'twincare-patient:'+str(uuid.uuid5(uuid.NAMESPACE_URL,'twincare-judge-p9:'+key)),'cursor':0,'reference_time':START.isoformat(),'speed':'Jury Demo','generation':0,'milestones':[],'events':[],'snapshots':[],'next_due':None,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
    self.sessions[key]['status']='ERROR';self.sessions[key]['error']='Reset failed: '+type(error).__name__;self._persist(key)
    raise
 def _reset(self,key):
  with self.lock,self.runtime._lock:
   config=self.configs[key];profile=PROFILES[key];pid='twincare-patient:'+str(uuid.uuid5(uuid.NAMESPACE_URL,'twincare-judge-p9:'+key))
   # Remove only this fixed judge owner, in this service's isolated database.
   # Document bytes are private patient-owned inputs and must not accumulate.
   old_files=[Path(d['storage_path']) for d in self.records.list_documents(pid)] if any(p['patient_id']==pid for p in self.repo.list_patients()) else []
   with self.repo.connection() as db:
    db.execute('PRAGMA defer_foreign_keys=ON');db.execute('DELETE FROM patients WHERE patient_id=?',(pid,))
   for path in old_files:
    if path.resolve().is_relative_to(self.records.storage):path.unlink(missing_ok=True)
   points=config['points'];prefix=[p for p in points if p['minute']<=0];future=[p for p in points if p['minute']>0]
   if not prefix or not future:raise ValueError('Prefix and future observations required')
   earliest=START+timedelta(minutes=min(p['minute'] for p in prefix))
   with self.clock.at(earliest),patch('patient_ingestion.service.now',self.clock.stamp):
    self.repo.create_patient(patient_id=pid,display_name=profile['display_name'],sex=profile['sex'],synthetic=True)
    content=json.dumps({'schema_version':'twincare-clinical/1','synthetic':True,'fields':{k:{'value':v} for k,v in profile.items() if k!='display_name'}}).encode()
    doc=self.records.upload(pid,'judge-clinical.json','application/json',content);candidate=self.records.extract(pid,doc['document_id']);candidate=self.records.edit(pid,doc['document_id'],candidate['revision'],[{'fact_id':f['fact_id'],'decision':'accept'} for f in candidate['facts']]);self.records.confirm(pid,doc['document_id'],candidate['revision'],True)
    for source in ('CGM','WEARABLE'):self.connections.connect(pid,source,{'scenario':'stable','acknowledge_synthetic':True});self.connections.activate(pid,source)
   with self.clock.at(START):
    self._wearable(pid)
    for offset in range(0,len(prefix),100):self.connections.ingest(pid,'CGM',[self._event(pid,p,config) for p in prefix[offset:offset+100]])
    actual=self.runtime.latest(pid)
   if actual['status']!='INFERRED':raise ValueError('Selected inputs did not make Twin ready')
   self.sessions[key]={'scenario':key,'patient_id':pid,'status':'READY','cursor':0,'reference_time':START.isoformat(),'speed':'Jury Demo','generation':self.sessions.get(key,{}).get('generation',0)+1,'milestones':[],'events':[],'snapshots':[],'error':None,'next_due':None,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}
   self._record(key,'initialized',actual);self._record(key,'twin_ready',actual);self._persist(key);return self.state(key)
 def _record(self,key,event_type,result=None):
  s=self.sessions[key];pid=s['patient_id'];result=result or self.runtime.latest(pid);state=result.get('twin_state')
  with self.clock.at(self.reference(key)):
   alert=self.runtime.alerts.active(pid);inbox=self.runtime.alerts.inbox(pid)['items']
  milestone={'event_type':event_type,'reference_time':s['reference_time'],'event_order':len(s['milestones']),'twin_state':state,'prediction':self.repo.get_owned(pid,'predictions',state['prediction_id']) if state else None,'alert_state':alert,'watch_state':inbox,'observations':self.connections.observations(pid,source='CGM',limit=1,recent=True)['items']}
  s['milestones'].append({'event_type':event_type,'reference_time':s['reference_time'],'trace_id':state['trace_id'] if state else None});s['snapshots'].append(milestone)
 def advance(self,key):
  with self.lock,self.runtime._lock:
   s=self.sessions[key];config=self.configs[key];future=[p for p in config['points'] if p['minute']>0]
   if s['cursor']>=len(future):s['status']='COMPLETED';self._persist(key);return self.state(key)
   if not any(m['event_type']=='meal_logged' for m in s['milestones']):
    meal_at=START+timedelta(minutes=config.get('meal_minute',0));s['reference_time']=meal_at.isoformat()
    with self.clock.at(meal_at):
     self._meal(s['patient_id'],key)
     # A confirmed contextual event is processed by the same P6/P7 service;
     # this captures the actual meal-time state, not a later sample's clock.
     actual=self.runtime.run_twin_inference(s['patient_id'],reference_time=meal_at.isoformat())
     self._record(key,'meal_logged',actual)
   point=future[s['cursor']];at=START+timedelta(minutes=point['minute']);s['reference_time']=at.isoformat()
   with self.clock.at(at):
    self._wearable(s['patient_id']);result=self.connections.ingest(s['patient_id'],'CGM',[self._event(s['patient_id'],point,config)])
    actual=self.runtime.latest(s['patient_id']);state=actual['twin_state'];check_owner(state,s['patient_id'])
    trace=self.runtime.trace(s['patient_id'],state['prediction_id']);evaluation=result.get('inference');alert=self.runtime.alerts.active(s['patient_id'])['alert'];inbox=self.runtime.alerts.inbox(s['patient_id'])['items']
    s['events'].append({'reference_time':at.isoformat(),'type':'CGM_RECEIVED_AND_TWIN_UPDATED','trace_id':state['trace_id'],'prediction_id':state['prediction_id'],'risk':state['spike']['probability'],'status':evaluation['status']})
    seen={m['event_type'] for m in s['milestones']}
    if 'upward_forecast' not in seen and state['predicted_peak']['glucose']-state['current_glucose']>=15:self._record(key,'upward_forecast',actual)
    if 'personal_departure' not in seen and state['divergence']['magnitude'] is not None and state['divergence']['magnitude']>=1:self._record(key,'personal_departure',actual)
    if 'risk_crossed' not in seen and alert:
     self._record(key,'risk_crossed',actual)
     if any(x['delivery']['status']=='DELIVERED' and x['alert']['alert_id']==alert['alert_id'] for x in inbox):self._record(key,'wearable_delivered',actual)
    crossing=next((x for x in s['snapshots'] if x['event_type']=='risk_crossed'),None)
    if crossing and 'observed_confirmation' not in seen and at>clock(crossing['reference_time']) and state['current_glucose']>crossing['twin_state']['current_glucose']:self._record(key,'observed_confirmation',actual)
    # Persist each actual computed frame for deterministic audit and offline video export, never runtime input.
    s['snapshots'].append({'event_type':'prediction_update','reference_time':at.isoformat(),'twin_state':state,'prediction':self.repo.get_owned(s['patient_id'],'predictions',state['prediction_id']),'trace':trace,'alert_state':self.runtime.alerts.active(s['patient_id']),'watch_state':inbox})
   s['cursor']+=1
   if s['cursor']==len(future):s['status']='COMPLETED'
   self._persist(key);return self.state(key)
 def teardown(self,key):
  with self.lock,self.runtime._lock:
   s=self.sessions.get(key)
   if s:
    pid=s['patient_id'];files=[Path(d['storage_path']) for d in self.records.list_documents(pid)] if any(p['patient_id']==pid for p in self.repo.list_patients()) else []
    with self.repo.connection() as db:
     db.execute('PRAGMA defer_foreign_keys=ON');db.execute('DELETE FROM patients WHERE patient_id=?',(pid,));db.execute('DELETE FROM judge_sessions WHERE scenario=?',(key,))
    for path in files:
     if path.resolve().is_relative_to(self.records.storage):path.unlink(missing_ok=True)
    del self.sessions[key]
   return {'scenario':key,'status':'UNINITIALIZED'}
 def state(self,key):
  s=self.sessions[key];total=sum(p['minute']>0 for p in self.configs[key]['points'])
  return {k:v for k,v in s.items() if k not in ('snapshots','next_due')}|{'total_events':total,'overview_url':f'/patients/{s["patient_id"]}/overview?judge={key}','wearable_url':f'/patients/{s["patient_id"]}/wearable?judge={key}'}
 def start(self,key,speed=None):
  with self.lock:
   s=self.sessions[key]
   if s['status']=='COMPLETED':raise ValueError('Reset the scenario before restarting')
   if speed is not None:
    if speed not in ('1×','4×','8×','Jury Demo'):raise ValueError('Invalid playback speed')
    s['speed']=speed
   if s['status']=='ERROR':raise ValueError('Reset failed scenario before restarting')
   s['status']='RUNNING';s['next_due']=time.monotonic();self._persist(key);self.wakeup.set();return self.state(key)
 def pause(self,key):
  with self.lock:
   if self.sessions[key]['status'] not in ('COMPLETED','ERROR'):self.sessions[key]['status']='PAUSED'
   self._persist(key);return self.state(key)
 def run_all(self,key):
  while self.sessions[key]['status']!='COMPLETED':self.advance(key)
  return self.state(key)
 def launch_worker(self):
  if self.worker and self.worker.is_alive():return
  def loop():
   while not self.stop_event.is_set():
    with self.lock:
     running=[k for k,s in self.sessions.items() if s['status']=='RUNNING']
     for key in running:
      s=self.sessions[key];now=time.monotonic()
      if s['next_due'] is None or now<s['next_due']:continue
      try:
       started=time.monotonic();before={m['event_type'] for m in s['milestones']};self.advance(key);count=self.state(key)['total_events'];speed=s['speed'];delay=60/count if speed=='Jury Demo' else 900/float(speed[:-1])
       if speed=='Jury Demo':
        discovered={m['event_type'] for m in s['milestones']}-before
        # Optional presentation dwell is driven by actual runtime milestones.
        # No predictions, event clocks or alert transitions are changed.
        if 'wearable_delivered' in discovered:delay=max(delay,8)
        elif 'observed_confirmation' in discovered:delay=max(delay,4)
        elif 'meal_logged' in discovered:delay=max(delay,3)
       s['next_due']=max(started+delay,time.monotonic())
      except Exception as error:s['status']='ERROR';s['error']=type(error).__name__;self._persist(key)
    self.wakeup.wait(.05);self.wakeup.clear()
  self.worker=threading.Thread(target=loop,daemon=True,name='judge-input-scheduler');self.worker.start()
 def close(self):self.stop_event.set();self.wakeup.set();self.worker.join(timeout=5) if self.worker else None
