"""Read source observations only. Preserve dataset namespace, units and clocks."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
START=datetime(2026,10,10,7,tzinfo=timezone.utc)
PROFILES={
 'A':{'display_name':'Nisha Rao','age_years':59,'sex':'female','diabetes_duration_years':11,'smoking_pack_years':0,'alcohol_history':'non-drinker','medication_context':'recorded_without_explicit_insulin_term','other_medication_context':'recorded_without_explicit_insulin_term','diagnoses':['Type 2 Diabetes'],'medications':[{'name':'Metformin','dose':'500 mg','frequency':'twice daily','status':'active'}]},
 'B':{'display_name':'Arjun Sen','age_years':47,'sex':'male','diabetes_duration_years':4,'smoking_pack_years':0,'alcohol_history':'non-drinker','medication_context':'recorded_without_explicit_insulin_term','other_medication_context':'recorded_without_explicit_insulin_term','diagnoses':['Type 2 Diabetes'],'medications':[{'name':'Sitagliptin','dose':'100 mg','frequency':'daily','status':'active'}]},
 'C':{'display_name':'Leela Shah','age_years':63,'sex':'female','diabetes_duration_years':13,'smoking_pack_years':3,'alcohol_history':'non-drinker','medication_context':'recorded_without_explicit_insulin_term','other_medication_context':'recorded_without_explicit_insulin_term','diagnoses':['Type 2 Diabetes'],'medications':[{'name':'Metformin','dose':'1000 mg','frequency':'daily','status':'active'}]},
}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def replay_candidates():
 path=ROOT/'data/metabolic_d3/demo.json';data=json.loads(path.read_text());result=[]
 for c in data['cases']:
  rows=sorted([e for e in c['events'] if e['concept']=='glucose'],key=lambda e:e['observed_time'])
  anchor=datetime.fromisoformat(c['timeline'][0]['timestamp'].replace('Z','+00:00'))
  if anchor.tzinfo is None:anchor=anchor.replace(tzinfo=timezone.utc)
  points=[]
  for e in rows:
   stamp=datetime.fromisoformat(e['observed_time'].replace('Z','+00:00'));stamp=stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
   points.append({'minute':int((stamp-anchor).total_seconds()/60),'value':e['value'],'source_observed_at':e['observed_time'],'source_available_at':e['available_time'],'original_value':e['original_value'],'original_unit':e['original_unit'],'quality':'MISSING' if e['value'] is None else 'GOOD'})
  result.append({'candidate': 'replay-'+c['case'],'source_kind':'synthetic_replay','source_file':str(path.relative_to(ROOT)),'source_sha256':digest(path),'dataset':'metabolic_d3_synthetic_replay','native_period_minutes':15,'points':points,'meal_minute':0,'source_identity_link':False})
 return result

def open_candidates(limit=12):
 # Input demonstration discovery only; not model selection, training, validation,
 # or a new holdout read. Use previously designated development sources only.
 manifest=json.loads((ROOT/'data/metabolic_d2/participant_manifest.json').read_text());ranked=[]
 for subject in sorted(manifest['development']):
  path=ROOT/'data/metabolic/canonical/ShanghaiT2D'/(subject.split(':')[1]+'.parquet')
  d=pd.read_parquet(path)
  raw_column=next(k for k in d if k.strip() in ('CGM','CGM (mg / dl)'))
  for recording,raw in d.groupby('_recording_id',sort=True):
   # Reject ambiguous native clocks instead of silently discarding source rows.
   if raw['_time_ns'].duplicated().any():continue
   raw=raw.sort_values('_time_ns');t=raw['_time_ns'].to_numpy();g=raw['glucose_mg_dl'].to_numpy()
   for i in range(221,len(g)-32,8):
    section=g[i-220:i+33];dt=np.diff(t[i-220:i+33])/60e9
    if not np.isfinite(section).all() or not np.all(np.abs(dt-15)<.01):continue
    if not (80<=g[i]<=150) or np.median(section[:212])>150:continue
    future=g[i:i+33];rise=float(future.max()-g[i]);peak=int(future.argmax())
    if rise<55 or future.max()<180 or peak<3:continue
    score=rise-abs(8-peak)*2
    ranked.append((score,path,subject,recording,raw,i))
 ranked.sort(key=lambda v:-v[0]);result=[];used=set()
 for score,path,subject,recording,raw,i in ranked:
  if recording in used:continue
  used.add(recording);anchor=int(raw.iloc[i]['_time_ns']);points=[]
  for _,row in raw.iloc[i-220:i+33].iterrows():
   ns=int(row['_time_ns']);stamp=pd.Timestamp(ns,unit='ns').isoformat()
   points.append({'minute':int((ns-anchor)/60e9),'value':float(row['glucose_mg_dl']),'source_observed_at':stamp,'source_available_at':None if pd.isna(row['availability_time']) else pd.Timestamp(row['availability_time']).isoformat(),'availability_basis':str(row['availability_basis']),'original_value':str(row[raw_column]),'original_unit':'mg/dL','quality':'GOOD','source_quality':str(row['data_quality'])})
  result.append({'candidate':f'open-native-{len(result)+1:02d}','source_kind':'open_data_replay','source_file':str(path.relative_to(ROOT)),'source_sha256':digest(path),'dataset':'ShanghaiT2D','source_subject_namespace':subject,'recording_namespace':'ShanghaiT2D:'+str(recording),'native_period_minutes':15,'points':points,'meal_minute':0,'source_identity_link':False,'input_prescreen_score':score})
  if len(result)>=limit:break
 return result

def all_candidates():return open_candidates()+replay_candidates()
