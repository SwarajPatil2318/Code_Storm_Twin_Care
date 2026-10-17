"""Immutable role-specific feature caches; no test construction before sealing."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
from src.twin.protocol import load,EVIDENCE,ABS_FEATURES,PERSONAL_FEATURES,EHR_COLUMNS,PRIMARY_SIGNALS,code_hash,feature_code_hash
from src.twin.access import Cohort
from src.twin.features import patient_history
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json

ARRAYS={'absolute':(len(ABS_FEATURES),),'personal':(len(PERSONAL_FEATURES),),'ehr':(len(EHR_COLUMNS),),'sequence':(28,len(PRIMARY_SIGNALS),4),
 'recent':(len(PRIMARY_SIGNALS),),'recent_count':(len(PRIMARY_SIGNALS),),'reference':(len(PRIMARY_SIGNALS),),'prior_recent':(len(PRIMARY_SIGNALS),),'scale':(len(PRIMARY_SIGNALS),),
 'persistence':(len(PRIMARY_SIGNALS),),'baseline_count':(len(PRIMARY_SIGNALS),),'confidence':(len(PRIMARY_SIGNALS),),'baseline_frozen':()}

def prepare(role,config=None,limit=None,consumer=False):
    c=config or load();out=Path(c['output']);cache_name=('consumer_' if consumer else '')+role;root=out/'features'/cache_name
    cohort=Cohort(c['cohort'],out,c);cohort.authorize(role);ids=sorted(cohort.patients(role));ids=ids[:limit] if limit else ids;wanted=set(ids)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Role cache is immutable; choose another development output for a fixture')
    root.mkdir(parents=True,mode=0o700);out.chmod(0o700);(out/'features').chmod(0o700);n=len(ids)*51;arrays={k:open_memmap(root/(k+'.npy'),mode='w+',dtype=np.float32,shape=(n,*shape)) for k,shape in ARRAYS.items()}
    context={r['patient_id']:r['context'] for r in cohort.rows('ehr',role) if r['patient_id'] in wanted}
    # The outcomes interface decodes only approved patient rows, and strips TTE.
    labels={(r['patient_id'],r['landmark_hour']):[r['event_0_24h'],r['event_24_48h'],r['event_48_72h']] for r in cohort.rows('outcomes',role) if r['patient_id'] in wanted}
    y=np.empty((n,3),np.uint8);patient_ids=[];hours=[];row=0;start=time.monotonic()
    for pid,stream in cohort.streams(role):
        if pid not in wanted:continue
        if consumer:
            stream={k:v.copy() for k,v in stream.items()}
            from src.synthetic.config import SIGNALS
            excluded=[j for j,s in enumerate(SIGNALS) if s not in c['consumer_signals']]
            stream['value'][:,excluded]=np.nan;stream['raw_value'][:,excluded]=np.nan;stream['missingness_mask'][:,excluded]=True;stream['quality_status'][:,excluded]=1;stream['available_hour'][:,excluded]=np.nan
        for hour,features in patient_history(context[pid],stream,c):
            for k,mem in arrays.items():mem[row]=features[k]
            y[row]=labels[(pid,hour)];patient_ids.append(pid);hours.append(hour);row+=1
        if len(patient_ids)//51%100==0:print({'phase6_features_patients':len(patient_ids)//51,'role':role,'target':len(ids)},flush=True)
    if row!=n:raise ValueError('Missing patient histories')
    for a in arrays.values():a.flush()
    np.save(root/'labels.npy',y);np.save(root/'patient_ids.npy',np.array(patient_ids));np.save(root/'landmark_hours.npy',np.array(hours,np.int16))
    hashes={p.name:file_hash(p) for p in root.glob('*.npy')}
    private_json(root/'manifest.json',{'role':role,'consumer_subset':consumer,'available_signals':c['consumer_signals'] if consumer else PRIMARY_SIGNALS,'patients':len(ids),'rows':n,'config_sha256':c['_sha256'],'feature_code_sha256':feature_code_hash(),'cohort_manifest_sha256':cohort.metadata_sha,'artifacts':hashes,'seconds':time.monotonic()-start,
      'source_features':['EHR whitelist','received observed wearable values/quality','causal historical personal reference'],'oracle_inputs':False,'outcome_fields_loaded':['event_0_24h','event_24_48h','event_48_72h'],**EVIDENCE})
    for p in root.iterdir():p.chmod(0o600)
    cohort.save_access_log('access_'+cache_name+'.json');print({'phase6_role_features_ready':role,'patients':len(ids),'landmarks':n},flush=True)
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['train','validation','test'],required=True);p.add_argument('--limit',type=int);p.add_argument('--consumer',action='store_true');a=p.parse_args();prepare(a.role,limit=a.limit,consumer=a.consumer)
