"""Small paired sensor/baseline stress cohorts; latent event histories are fixed."""
import json
from pathlib import Path
import numpy as np
from src.synthetic.generate import generate,iter_rows
from src.synthetic.config import load,SIGNALS
from src.vitaldb_feasibility.source import private_json


def run(patients=128,root='data/synthetic/variants'):
    root=Path(root);config=load();summaries={};events={};truth={}
    for name in config['variants']:
        directory=root/name
        if not (directory/'manifest.json').exists():generate(patients,60,2026,directory,variant=name)
        manifest=json.loads((directory/'manifest.json').read_text());events[name]={r['patient_id']:r['event_time_hours'] for r in iter_rows(directory/'ground_truth/records.jsonl.gz')}
        total=np.zeros(len(SIGNALS));available=np.zeros(len(SIGNALS));error=[[] for _ in SIGNALS];baseline_shift=[]
        for batch in manifest['batches']:
            with np.load(directory/'observations'/batch) as a,np.load(directory/'ground_truth'/batch) as b:
                total+=a['scheduled'].sum(axis=(0,1));available+=(~a['missingness_mask']).sum(axis=(0,1))
                for j in range(len(SIGNALS)):
                    valid=~a['missingness_mask'][:,:,j];delta=a['value'][:,:,j][valid]-b['true_physiology'][:,:,j][valid];error[j].extend(delta.tolist())
                baseline_shift.extend((b['stable_daily'][:,-1]-b['stable_daily'][:,0]).tolist())
        summaries[name]={'patients':patients,'missing_fraction_at_opportunities':dict(zip(SIGNALS,(1-available/np.maximum(total,1)).tolist())),
          'measurement_error_sd':dict(zip(SIGNALS,[float(np.std(x)) if x else None for x in error])),
          'mean_true_baseline_end_minus_start':dict(zip(SIGNALS,np.mean(baseline_shift,axis=0).tolist())),
          'evidence':'SYNTHETIC POC','evidence_label':'NOT VALIDATED'}
    fixed=all(e==events['normal_noise'] for e in events.values())
    if not fixed:raise ValueError('Paired variants changed latent event labels')
    result={'variants':summaries,'event_histories_identical':fixed,'scope':'128matched virtual patients per variant; no final predictor or clinical robustness claim',
      'evidence':'SYNTHETIC POC','evidence_label':'SYNTHETIC POC VALIDATED'}
    private_json(root/'variant_summary.json',result);print({'small_variants_complete':len(summaries),'paired_latent_events_unchanged':fixed},flush=True)
    return result

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run()
