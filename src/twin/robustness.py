"""Frozen stress inference and complete-history sensor deletion.

Removal happens before baseline/Shadow/TDI construction. No removed signal
survives indirectly through a baseline freeze or multivariate aggregate.
"""
import json
from pathlib import Path
import numpy as np
import joblib
from src.twin.protocol import EVIDENCE,load,PRIMARY_SIGNALS
from src.twin.access import Cohort,OBS_KEYS,selective_npz,selective_json
from src.twin.features import patient_history
from src.twin.prepare import ARRAYS
from src.twin.train import view,predict
from src.twin.models import cumulative
from src.twin.metrics import horizon_results,paired_ap,patient_bootstrap
from src.twin.sealing import verify
from src.synthetic.config import SIGNALS
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json

VARIANTS=['normal_noise','high_sensor_noise','high_missingness','reduced_sensor_set','baseline_shift']

def register(config):
    root=Path('data/synthetic/variants');paths=[root/'variant_summary.json']+[root/name/'manifest.json' for name in VARIANTS]
    certificate=json.loads(paths[0].read_text())
    if not certificate['event_histories_identical']:raise ValueError('Stress pairing not certified')
    record={'original_sha256':{str(p):file_hash(p) for p in paths},'variants':VARIANTS,'sensor_ablations':config['sensor_ablations'],'removal_scope':'Complete observed history before all feature construction','test_used':False,**EVIDENCE};private_json(Path(config['output'])/'robustness_registration.json',record);return record

def masked_stream(stream,removed):
    s={k:v.copy() for k,v in stream.items()};ix=[SIGNALS.index(name) for name in removed]
    s['value'][:,ix]=np.nan;s['raw_value'][:,ix]=np.nan;s['missingness_mask'][:,ix]=True;s['quality_status'][:,ix]=1;s['available_hour'][:,ix]=np.nan;return s

def encode(context,stream,pid,config,removed=()):
    history=list(patient_history(context,masked_stream(stream,removed) if removed else stream,config));data={k:np.stack([features[k] for _,features in history]).astype(np.float32) for k in ARRAYS}
    data['patient_ids']=np.full(len(history),pid);data['landmark_hours']=np.array([h for h,_ in history]);return data

def concatenate(patients):return {k:np.concatenate([d[k] for d in patients]) for k in patients[0]}

def model_prediction(name,data,engines,config):
    directory=Path(config['output'])/'models'/name;model=joblib.load(directory/'selected.joblib');cal=joblib.load(directory/'calibration.joblib');x,_,dr=view(name,data,engines);return cumulative(cal.transform(predict(model,x,data))),dr

def run_ablations(data,engines,reference,config):
    """Checkpoint one patient's finished inference, then never replay it on resume."""
    cohort=Cohort(config['cohort'],config['output'],config);cohort.authorize('test');out=Path(config['output']);checkpoint=out/'sensor_ablation_predictions';checkpoint.mkdir(exist_ok=True,mode=0o700)
    contexts={r['patient_id']:r['context'] for r in cohort.rows('ehr','test')};position={str(p):np.flatnonzero(data['patient_ids']==p) for p in np.unique(data['patient_ids'])};models={name:(joblib.load(out/'models'/name/'selected.joblib'),joblib.load(out/'models'/name/'calibration.joblib')) for name in ['full_twin']}
    names=config['sensor_ablations'];final={name:np.empty_like(reference) for name in names};quality={name:np.empty(len(reference)) for name in names};tdi={name:np.empty(len(reference)) for name in names}
    for ordinal,(pid,stream) in enumerate(cohort.streams('test')):
        path=checkpoint/(pid.split(':')[-1]+'.npz');ix=position[pid]
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                for name in names:final[name][ix]=z[name];quality[name][ix]=z[name+'_quality'];tdi[name][ix]=z[name+'_tdi']
        else:
            record={}
            for name in names:
                removed=['systolic_bp','diastolic_bp','map'] if name=='bp' else [name];d=encode(contexts[pid],stream,pid,config,removed);x,_,dr=view('full_twin',d,engines);model,cal=models['full_twin'];p=cumulative(cal.transform(predict(model,x,d)));final[name][ix]=p;tdi[name][ix]=dr['tdi']
                from src.twin.inference import quality_uncertainty
                quality[name][ix]=quality_uncertainty(d);record[name]=p;record[name+'_quality']=quality[name][ix];record[name+'_tdi']=tdi[name][ix]
            record.update({k:np.array(v) for k,v in EVIDENCE.items()});temporary=path.with_suffix('.partial.npz');np.savez_compressed(temporary,**record);temporary.chmod(0o600);temporary.replace(path)
        if (ordinal+1)%100==0:print({'phase6_sensor_ablation_patients':ordinal+1,'target':1500,'removed_signals':6},flush=True)
    thresholds=json.loads((out/'models/full_twin/thresholds.json').read_text())['thresholds'];result={}
    from src.twin.analysis import summary
    for name,p in final.items():
        result[name]={'metrics':horizon_results(data['labels'],p,thresholds),'paired_auprc72_vs_intact':paired_ap(data['labels'].sum(axis=1),p[:,2],reference[:,2],data['patient_ids'],config['metric_bootstraps']),'tdi':summary(tdi[name]),'data_quality_uncertainty':summary(quality[name]),'removed_before_baseline':True}
    cohort.save_access_log('access_sensor_ablations.json');return {'experiments':result,'model':'Frozen full_twin, unchanged calibrator and thresholds','no_retraining':True,**EVIDENCE}

def stress_data(name,config,consumer=False):
    verify(config);out=Path(config['output']);registration=json.loads((out/'robustness_registration.json').read_text())
    if not (out/'evaluation_started.json').exists():raise ValueError('Stress analysis must follow frozen evaluation start')
    for p,h in registration['original_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Original stress cohort identity changed')
    root=Path('data/synthetic/variants')/name;manifest=json.loads((root/'manifest.json').read_text());ids=set(json.loads((root/'splits.json').read_text())['assignments']);main=Cohort(config['cohort'],out,config)
    if ids&set(main.roles):raise ValueError('Stress participants overlap the main cohort')
    def check(path):
        if file_hash(path)!=manifest['artifacts'][str(path.relative_to(root))]['sha256']:raise ValueError('Stress artifact changed')
    for table in ['ehr','outcomes']:check(root/table/'records.jsonl.gz')
    contexts={r['patient_id']:r['context'] for r in selective_json(root/'ehr/records.jsonl.gz',ids)};labels={(r['patient_id'],r['landmark_hour']):[r['event_0_24h'],r['event_24_48h'],r['event_48_72h']] for r in selective_json(root/'outcomes/records.jsonl.gz',ids)};patients=[]
    for batch in manifest['batches']:
        path=root/'observations'/batch;check(path)
        with np.load(path,allow_pickle=False) as z:pids=z['patient_ids'].tolist()
        streams=selective_npz(path,range(len(pids)),OBS_KEYS)
        for i,pid in enumerate(pids):
            stream={k:a[i] for k,a in streams.items()};removed=[s for s in SIGNALS if s not in config['consumer_signals']] if consumer else []
            d=encode(contexts[pid],stream,pid,config,removed);d['labels']=np.array([labels[pid,int(h)] for h in d['landmark_hours']],np.uint8);patients.append(d)
    return concatenate(patients)

def run_stress(engines,config):
    out=Path(config['output']);result={};baseline=None
    for variant in VARIANTS:
        checkpoint=out/f'stress_{variant}.json'
        if checkpoint.exists():result[variant]=json.loads(checkpoint.read_text());continue
        data=stress_data(variant,config);consumer=stress_data(variant,config,consumer=True);variant_result={}
        for name,d in [('population_fusion',data),('full_twin',data),('consumer_twin',consumer)]:
            prob,dr=model_prediction(name,d,engines,config);thresholds=json.loads((out/'models'/name/'thresholds.json').read_text())['thresholds']
            from src.twin.inference import quality_uncertainty
            from src.twin.analysis import summary
            variant_result[name]={'metrics':horizon_results(d['labels'],prob,thresholds),'patient_bootstrap95':patient_bootstrap(d['labels'],prob,d['patient_ids'],thresholds,config['metric_bootstraps']),'tdi':summary(dr['tdi']),'quality_uncertainty':summary(quality_uncertainty(d,[PRIMARY_SIGNALS.index(s) for s in config['consumer_signals']] if name=='consumer_twin' else None))}
            np.savez_compressed(out/f'stress_{variant}_{name}_predictions.npz',probabilities=prob,labels=d['labels'],patient_ids=d['patient_ids'],**{k:np.array(v) for k,v in EVIDENCE.items()});(out/f'stress_{variant}_{name}_predictions.npz').chmod(0o600)
        value={'variant':variant,'patients':len(np.unique(data['patient_ids'])),'rows':len(data['labels']),'models':variant_result,'paired_event_histories':True,'scope':'Small matched synthetic cohort, overlapping Phase5 pilot identities; no external validation',**EVIDENCE};private_json(checkpoint,value);result[variant]=value
        print({'phase6_stress_variant_complete':variant},flush=True)
    paired={}
    for variant in VARIANTS[1:]:
        paired[variant]={}
        for name in ['population_fusion','full_twin','consumer_twin']:
            with np.load(out/f'stress_normal_noise_{name}_predictions.npz',allow_pickle=False) as z:baseline={k:z[k] for k in z.files}
            with np.load(out/f'stress_{variant}_{name}_predictions.npz',allow_pickle=False) as z:current={k:z[k] for k in z.files}
            if not np.array_equal(baseline['patient_ids'],current['patient_ids']) or not np.array_equal(baseline['labels'],current['labels']):raise ValueError('Matched stress labels/patient alignment changed')
            paired[variant][name]=paired_ap(current['labels'].sum(axis=1),current['probabilities'][:,2],baseline['probabilities'][:,2],current['patient_ids'],config['metric_bootstraps'])
    return {'variants':result,'paired_ap72_variant_minus_normal':paired,'training_or_threshold_changes':False,**EVIDENCE}
