"""Streaming paired synthetic POC cohort. Large generation requires a pilot gate.

python -m src.synthetic.generate --patients 10000 --days 60 --seed 2026

Public/model features are separate from outcomes and private simulator truth.
Efficient hourly arrays use compressed NPZ, with explicit units/QC/receipt clocks.
Selected observations also export through the existing canonical schema/FHIR.
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from src.synthetic.config import load,rng,SIGNALS,UNITS,EVIDENCE,STATUS
from src.synthetic.patient_generator import patient,vulnerability
from src.synthetic.baseline_generator import baseline
from src.synthetic.state_model import simulate as states,STATES
from src.synthetic.physiology_model import simulate as physiology
from src.synthetic.sensors import observe,QUALITY,REASONS,SOURCES
from src.synthetic.outcomes import landmarks,MODEL_FEATURES
from src.synthetic.ehr_generator import bundle,instant
from src.data.io import dumps,file_hash
from src.data.schema import Provenance,SourceValue,Observation,record
from src.vitaldb_feasibility.source import private_json


def code_hash():
    h=hashlib.sha256()
    modules=['config.py','patient_generator.py','baseline_generator.py','state_model.py','physiology_model.py','sensors.py','outcomes.py','ehr_generator.py','generate.py']
    for name in sorted(modules):
        p=Path('src/synthetic')/name;h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


class StreamWriter:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);self.raw=self.path.open('wb');self.path.chmod(0o600)
        self.stream=gzip.GzipFile(filename='',mode='wb',fileobj=self.raw,mtime=0,compresslevel=5);self.count=0
    def write(self,row):self.stream.write((dumps(row)+'\n').encode());self.count+=1
    def close(self):self.stream.close();self.raw.close()


def npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);np.savez_compressed(path,**arrays);path.chmod(0o600)


def iter_rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def verify_pilot(config,seed,days,pilot=Path('data/synthetic/pilot/validation.json')):
    if not pilot.exists():raise ValueError('Large cohort denied:1000patient pilot validation is missing')
    gate=json.loads(pilot.read_text())
    if not gate['all_gates_passed'] or gate['patients']!=1000:raise ValueError('Large cohort denied:pilot gates did not pass')
    if gate['config_sha256']!=config['_config_sha256'] or gate['simulator_code_sha256']!=code_hash():raise ValueError('Large cohort denied:configuration/source changed after validation')
    if seed!=gate['seed'] or days!=gate['days']:raise ValueError('Large cohort denied:seed/duration differs from validated design')
    if gate['grounding_sha256']!=file_hash(Path(config['grounding_path'])):raise ValueError('Large cohort denied:grounding changed')
    # Evidence cannot be a stale success JSON pasted next to a changed dataset.
    base=pilot.parent
    for name,expected in gate['pilot_artifact_sha256'].items():
        path=base/name
        if not path.exists() or file_hash(path)!=expected:raise ValueError('Large cohort denied:pilot artifact changed')
    return file_hash(pilot)


def simulate_one(index,config,grounding,seed,variant=None):
    p=patient(index,config,seed);b=baseline(p,config,grounding,seed);v=vulnerability(p,b['mean'],config)
    state=states(p,v,config,seed)
    # Exogenous healthy baseline drift is independent of deterioration; paired
    # robustness scenarios hold latent events fixed, without a causal-treatment claim.
    if variant and 'offset' in variant:
        start=int(config['days']*variant.get('start_fraction',.5))
        b['stable_daily'][start:]+=np.array(variant['offset']);b['stable_daily'][:,4]=(b['stable_daily'][:,2]+2*b['stable_daily'][:,3])/3
    phys=physiology(p,b,state,config,seed,variant);stream=observe(p,b,state,phys,config,seed,variant)
    return p,b,state,phys,stream


def generate(patients=1000,days=60,seed=2026,output=None,config_path=None,variant='normal_noise',pilot_path=None):
    config=load(config_path) if config_path else load();config['days']=int(days)
    if not 30<=days<=90 or patients<1:raise ValueError('Positive cohort size and30–90day history required')
    gate_hash=None
    if patients>1000 and variant!='normal_noise':raise ValueError('Only the pilot-validated normal-noise design may scale above1000 patients')
    if patients>1000:gate_hash=verify_pilot(config,seed,days,Path(pilot_path) if pilot_path else Path('data/synthetic/pilot/validation.json'))
    root=Path(output or ('data/synthetic/pilot' if patients==1000 else 'data/synthetic/cohort10000' if patients==10000 else 'data/synthetic/sample'))
    if root.exists() and any(root.iterdir()):raise FileExistsError('Choose a new output directory; existing cohorts are immutable')
    root.mkdir(parents=True,exist_ok=True,mode=0o700);grounding_path=Path(config['grounding_path']);grounding=json.loads(grounding_path.read_text())
    perturbation=config['variants'][variant];writers={name:StreamWriter(root/name/'records.jsonl.gz') for name in ['patients','ehr','model_features','outcomes','ground_truth','quality']}
    started=time.monotonic();ids=[];batch=[];batch_names=[];events=0;landmark_count=0
    index_offset=1000 if patients>1000 else 0
    for ordinal in range(patients):
        index=ordinal+index_offset
        p,b,state,phys,stream=simulate_one(index,config,grounding,seed,perturbation);pid=p['patient_id'];ids.append(pid);events+=state['event_time'] is not None
        writers['patients'].write({'patient_id':pid,'dataset':'synthetic','observed_time':p['start_time'],'days':days,'age':p['context']['age'],'gender':'female' if p['context']['female'] else 'male','evidence':EVIDENCE,'evidence_label':STATUS})
        # Every EHR resource is available before the monitoring period; wearable demo
        # observations are written to a separate demo export, not baseline EHR context.
        ehr=bundle(p,grounding);writers['ehr'].write({'patient_id':pid,'available_hour':-24*20.9,'context':p['context'],'fhir':ehr,'evidence':EVIDENCE,'evidence_label':STATUS})
        writers['ground_truth'].write({'patient_id':pid,'profile':p['profile'],'event_time_hours':state['event_time'],'vulnerability':state['vulnerability'],
          'episodes':state['episodes'],'decoys':phys['decoys'],'stable_baseline_mean':b['mean'].tolist(),'adherence':b['adherence'],
          'stable_baseline_variance_multiplier':b['variance_multiplier'],'evidence':EVIDENCE,'evidence_label':STATUS})
        counts={name:int(np.sum(stream['quality_status']==code)) for name,code in QUALITY.items()}
        writers['quality'].write({'patient_id':pid,'quality_counts':counts,'scheduled_readings':int(stream['scheduled'].sum()),'usable_readings':int((~stream['missingness_mask']).sum()),
          'delayed_readings':int(np.sum(stream['available_hour']>np.arange(days*24)[:,None]+.1)),
          'true_value_clipping_counts':phys['clipping_counts'].tolist(),'evidence':EVIDENCE,'evidence_label':STATUS})
        for hour,features,label in landmarks(p['context'],stream,state['event_time'],config):
            window=hashlib.sha256((pid+':'+str(hour)).encode()).hexdigest()[:24];common={'window_id':window,'patient_id':pid,'landmark_hour':hour,'evidence':EVIDENCE,'evidence_label':STATUS}
            writers['model_features'].write({**common,'features':features});writers['outcomes'].write({**common,**label});landmark_count+=1
        batch.append((p,b,state,phys,stream))
        if ordinal<3:
            private_json(root/'fhir_demo'/f'patient_{ordinal}.json',bundle(p,grounding,stream))
            demo=StreamWriter(root/'canonical_demo'/f'patient_{ordinal}.jsonl.gz')
            for j,signal in enumerate(SIGNALS):
                for hour in np.flatnonzero(stream['scheduled'][:,j])[:3]:
                    raw=stream['raw_value'][hour,j];value=stream['value'][hour,j];status=next(k for k,v in QUALITY.items() if v==stream['quality_status'][hour,j]);prov=Provenance('synthetic','observations/npz',str(hour),signal,signal,'synthetic_state_space_sensor',config['_config_sha256'],UNITS[j],'synthetic_scenario')
                    sv=SourceValue(float(raw) if np.isfinite(raw) else None,float(value) if np.isfinite(value) else None,status,[],UNITS[j],prov)
                    obs=Observation('synthetic:'+window+':'+signal+':'+str(hour),pid,'synthetic',signal,instant(p,hour) if np.isfinite(raw) else None,'absolute_utc',instant(p,stream['available_hour'][hour,j]) if np.isfinite(stream['available_hour'][hour,j]) else None,'simulated_receipt',sv,metadata={'synthetic':True,'evidence_label':STATUS,'sensor_source':SOURCES[j],'scheduled_opportunity_time':instant(p,hour)})
                    demo.write(record(obs))
            demo.close()
        if len(batch)==config['batch_patients'] or ordinal==patients-1:
            number=len(batch_names);name=f'batch_{number:05d}.npz';batch_names.append(name)
            public={'patient_ids':np.array([x[0]['patient_id'] for x in batch]),'signals':np.array(SIGNALS),'units':np.array(UNITS),'observed_hour':np.arange(days*24,dtype=np.int32),
              'sensor_source':np.array(SOURCES),'evidence':np.array(EVIDENCE),'evidence_label':np.array(STATUS)}
            for key in stream:public[key]=np.stack([x[4][key] for x in batch])
            npz(root/'observations'/name,**public)
            npz(root/'ground_truth'/name,patient_ids=public['patient_ids'],state=np.stack([x[2]['state'] for x in batch]),
              latent=np.stack([x[2]['latent'] for x in batch]).astype(np.float32),event_hazard=np.stack([x[2]['event_hazard'] for x in batch]).astype(np.float32),
              true_physiology=np.stack([x[3]['values'] for x in batch]),stable_daily=np.stack([x[1]['stable_daily'] for x in batch]).astype(np.float32),evidence=np.array(EVIDENCE),evidence_label=np.array(STATUS))
            batch.clear()
        if (ordinal+1)%100==0:print({'synthetic_patients_generated':ordinal+1,'target':patients,'variant':variant},flush=True)
    for writer in writers.values():writer.close()
    # Split only after every full history is generated; identical simulator rules.
    ordering=sorted(ids,key=lambda p:hashlib.sha256(f'{seed}:split:{p}'.encode()).hexdigest());a=int(.7*patients);b=int(.85*patients)
    roles={p:'train' if i<a else 'validation' if i<b else 'test' for i,p in enumerate(ordering)}
    private_json(root/'splits.json',{'assignments':roles,'proportions':[.7,.15,.15],'generated_before_split':True,'evidence':EVIDENCE,'evidence_label':STATUS})
    private_json(root/'schemas.json',{'model_features':MODEL_FEATURES,'metadata_never_model_inputs':['patient_id','window_id','landmark_hour','evidence','evidence_label'],
      'observations':'NPZ patient,time,signal axes: value/raw_value/quality/missingness/scheduled/available_hour/source/units; no latent state or outcomes',
      'private_ground_truth':['profile','state','latent','event_hazard','event_time_hours','episodes','decoys','vulnerability','stable_daily','stable_baseline_mean'],
      'quality_codes':QUALITY,'reason_codes':REASONS,'latent_states':STATES,'evidence':EVIDENCE,'evidence_label':STATUS})
    artifacts={str(p.relative_to(root)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()}
    manifest={'version':config['version'],'seed':seed,'patients':patients,'days':days,'event_patients':int(events),'event_patient_prevalence':events/patients,'landmarks':landmark_count,
      'variant':variant,'patient_index_offset':index_offset,'config_sha256':config['_config_sha256'],'simulator_code_sha256':code_hash(),'grounding_sha256':file_hash(grounding_path),'grounding_sources_sha256':grounding['sources_sha256'],
      'batches':batch_names,'evidence':EVIDENCE,'evidence_label':STATUS,'not_clinical_validation':True,'final_predictor_trained':False,'pilot_gate_sha256':gate_hash,'artifacts':artifacts}
    private_json(root/'manifest.json',manifest)
    private_json(root/'runtime.json',{'seconds':time.monotonic()-started,'bytes':sum(x['bytes'] for x in artifacts.values()),'patients':patients,'evidence':EVIDENCE,'evidence_label':STATUS})
    for path in root.rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
    print({'synthetic_generation_complete':True,'patients':patients,'event_patients':events,'landmarks':landmark_count,'seconds':round(time.monotonic()-started,2)},flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--patients',type=int,default=1000);parser.add_argument('--days',type=int,default=60);parser.add_argument('--seed',type=int,default=2026);parser.add_argument('--output');parser.add_argument('--config');parser.add_argument('--variant',default='normal_noise');parser.add_argument('--pilot-validation')
    args=parser.parse_args();generate(args.patients,args.days,args.seed,args.output,args.config,args.variant,args.pilot_validation)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
