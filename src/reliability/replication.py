"""Fresh seed-2027 simulation using unchanged validated Phase 5 core functions.

The old generator's CLI scaling gate is intentionally seed-2026-only. This
separate driver verifies those validated rules, then applies the user-authorized
new seed after the Phase 6.5 freeze. No simulator parameter is adjusted.
"""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from src.synthetic.config import load as sim_config,SIGNALS,UNITS
from src.synthetic.generate import simulate_one,StreamWriter,npz,code_hash,verify_pilot,iter_rows
from src.synthetic.outcomes import labels
from src.synthetic.validation import physiological_summary
from src.data.io import file_hash
from src.reliability.access import load_config,root,save_json,timed,ORIGINAL
from src.reliability.sealing import verify


def generate(config=None):
    c=config or load_config();frozen=verify(c);directory=Path(c['replication']);start=time.monotonic()
    if (directory/'manifest.json').exists():
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['seed']!=2027 or manifest['frozen_manifest_sha256']!=file_hash(root(c)/'PHASE6_5_FROZEN_MANIFEST.json'):
            raise ValueError('Existing replication identity mismatch')
        return manifest
    if directory.exists() and any(directory.iterdir()):raise FileExistsError('Partial generation exists; do not create a second cohort')
    sim=sim_config();sim['days']=60
    pilot=verify_pilot(sim,2026,60)
    grounding=json.loads(Path(sim['grounding_path']).read_text())
    directory.mkdir(parents=True,mode=0o700)
    writers={name:StreamWriter(directory/name/'records.jsonl.gz') for name in ['patients','ehr','outcomes','ground_truth','quality']}
    names=[];batch=[];events=0;count=0;ids=[]
    for ordinal in range(2500):
        index=1000+ordinal
        p,b,state,phys,stream=simulate_one(index,sim,grounding,2027,sim['variants']['normal_noise'])
        pid=p['patient_id'];ids.append(pid);events+=state['event_time'] is not None
        common={'patient_id':pid,'evidence':'SYNTHETIC POC','evidence_label':'NOT CLINICALLY VALIDATED'}
        writers['patients'].write({**common,'dataset':'synthetic','observed_time':p['start_time'],'days':60,'age':p['context']['age']})
        writers['ehr'].write({**common,'available_hour':-24*20.9,'context':p['context']})
        writers['ground_truth'].write({**common,'profile':p['profile'],'event_time_hours':state['event_time'],
            'episodes':state['episodes'],'decoys':phys['decoys'],'stable_baseline_mean':b['mean'].tolist()})
        writers['quality'].write({**common,'true_value_clipping_counts':phys['clipping_counts'].tolist()})
        for hour in range(168,1369,24):
            writers['outcomes'].write({**common,'landmark_hour':hour,**labels(state['event_time'],hour,1440)});count+=1
        batch.append((p,b,state,phys,stream))
        if len(batch)==sim['batch_patients'] or ordinal==2499:
            name=f'batch_{len(names):05d}.npz';names.append(name)
            public={'patient_ids':np.array([x[0]['patient_id'] for x in batch]),'signals':np.array(SIGNALS),'units':np.array(UNITS),
                'observed_hour':np.arange(1440,dtype=np.int32),'evidence':np.array('SYNTHETIC POC'),'evidence_label':np.array('NOT CLINICALLY VALIDATED')}
            for key in stream:public[key]=np.stack([x[4][key] for x in batch])
            npz(directory/'observations'/name,**public)
            npz(directory/'ground_truth'/name,patient_ids=public['patient_ids'],state=np.stack([x[2]['state'] for x in batch]),
                latent=np.stack([x[2]['latent'] for x in batch]).astype(np.float32),true_physiology=np.stack([x[3]['values'] for x in batch]),
                stable_daily=np.stack([x[1]['stable_daily'] for x in batch]).astype(np.float32))
            batch.clear()
        if (ordinal+1)%100==0:print({'phase65_replication_generated':ordinal+1,'target':2500},flush=True)
    for writer in writers.values():writer.close()
    old=set(json.loads((ORIGINAL/'splits.json').read_text())['assignments'])
    if set(ids)&old:raise ValueError('Replication overlaps original patient identities')
    artifacts={str(p.relative_to(directory)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in directory.rglob('*') if p.is_file()}
    manifest={'version':sim['version'],'seed':2027,'patients':2500,'days':60,'event_patients':int(events),
        'event_patient_prevalence':events/2500,'landmarks':count,'batches':names,'variant':'normal_noise','patient_index_offset':1000,
        'config_sha256':sim['_config_sha256'],'simulator_code_sha256':code_hash(),'grounding_sha256':file_hash(Path(sim['grounding_path'])),
        'frozen_manifest_sha256':file_hash(root(c)/'PHASE6_5_FROZEN_MANIFEST.json'),'validated_seed2026_pilot_sha256':pilot,
        'new_seed_authorization':'User requires an independent seed-2027 replication; same validated simulator rules',
        'artifacts':artifacts,'evidence':'SYNTHETIC POC','evidence_label':'NOT CLINICALLY VALIDATED',
        'replication_role':'All 2500 patients held out for one frozen Phase 6.5 evaluation; no development split'}
    save_json(directory/'manifest.json',manifest)
    for p in directory.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    timed(c,'replication_generation',start,{'patients':2500,'seed':2027})
    print({'phase65_replication_generation_complete':True,'patients':2500,'event_patients':int(events),'landmarks':count},flush=True)
    return manifest


def audit(config=None):
    c=config or load_config();verify(c);directory=Path(c['replication']);start=time.monotonic()
    m=json.loads((directory/'manifest.json').read_text());sim=sim_config()
    gt={r['patient_id']:r for r in iter_rows(directory/'ground_truth/records.jsonl.gz')}
    grounding=json.loads(Path(sim['grounding_path']).read_text())
    phys=physiological_summary(directory,m,gt,grounding,sim)
    cfg=sim['validation'];realism={}
    for signal,r in phys['comparison'].items():
        realism[signal]=bool(r['wasserstein_real_sd']<=cfg['realism_max_wasserstein_real_sd'] and r['maximum_quantile_distance_real_sd']<=cfg['realism_max_quantile_distance_real_sd']
            and cfg['within_sd_ratio_limits'][0]<=r['within_sd_ratio']<=cfg['within_sd_ratio_limits'][1]
            and cfg['day_change_sd_ratio_limits'][0]<=r['day_change_sd_ratio']<=cfg['day_change_sd_ratio_limits'][1])
    reproduced=True
    with np.load(directory/'observations'/m['batches'][0],allow_pickle=False) as archive:
        for i in [0,1]:
            p,b,s,ph,st=simulate_one(1000+i,sim,grounding,2027,sim['variants']['normal_noise'])
            reproduced &= p['patient_id']==str(archive['patient_ids'][i]) and all(np.array_equal(st[k],archive[k][i],equal_nan=True) for k in st)
    label_ok=True;counts={};positive=np.zeros(3,int)
    for row in iter_rows(directory/'outcomes/records.jsonl.gz'):
        hour=row['landmark_hour'];pid=row['patient_id'];counts[pid]=counts.get(pid,0)+1
        expected=labels(gt[pid]['event_time_hours'],hour,1440)
        label_ok &= all(row[k]==v for k,v in expected.items())
        positive += [row['event_0_24h'],row['event_24_48h'],row['event_48_72h']]
    safe=True
    for name in m['batches']:
        with np.load(directory/'observations'/name,allow_pickle=False) as z:
            safe &= not bool(set(z.files)&{'state','latent','event_time','event_hazard','stable_daily'})
            safe &= z['signals'].tolist()==SIGNALS and z['units'].tolist()==UNITS
    gates={'patient_count_and_fixed_schedule':len(gt)==2500 and set(counts.values())=={51} and sum(counts.values())==127500,
        'seed_and_unchanged_rules':m['seed']==2027 and m['config_sha256']==sim['_config_sha256'] and m['simulator_code_sha256']==code_hash(),
        'artifact_integrity':all(file_hash(directory/p)==meta['sha256'] for p,meta in m['artifacts'].items()),
        'reproducibility_sample':bool(reproduced),'physiological_bounds':phys['bounds_ok'],'pressure_order':phys['pressure_order'],
        'causal_receipt':phys['causal_receipt'],'labels_correct':bool(label_ok),'schema_and_hidden_state_exclusion':bool(safe),
        'event_prevalence':sim['states']['event_prevalence_acceptance'][0]<=m['event_patient_prevalence']<=sim['states']['event_prevalence_acceptance'][1],
        'basic_home_realism':all(realism.values()),'temporal_correlation':all(phys['daily_stats'][s]['lag1_daily_correlation']>=cfg['daily_lag1_min'] for s in realism),
        'bounded_missingness':all(cfg['sensor_missing_fraction_range'][0]<=x<=cfg['sensor_missing_fraction_range'][1] for x in phys['missing_fraction_by_signal'].values())}
    result={'gates':{k:bool(v) for k,v in gates.items()},'all_passed':all(gates.values()),'patients':2500,'landmarks':127500,
        'event_patients':m['event_patients'],'event_patient_prevalence':m['event_patient_prevalence'],
        'horizon_prevalence':(np.cumsum(positive)/127500).tolist(),'realism_by_signal':realism,
        'scope':'Existing essential generator integrity/realism gates only; no diagnostic model search or clinical validation'}
    save_json(directory/'replication_audit.json',result);timed(c,'replication_essential_audit',start)
    print({'phase65_replication_audit_passed':result['all_passed'],'failed':[k for k,v in gates.items() if not v]},flush=True)
    if not result['all_passed']:raise ValueError('Replication generation failed essential gates; preserve without retuning')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):generate();audit()
