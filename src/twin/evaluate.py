"""One journalled final evaluation of the frozen synthetic TEST cohort.

All comparisons, controls, oracle analyses and stress experiments belong to
the same predeclared evaluation transaction. Resume reads completed outputs;
it neither retunes models nor starts another test-selection opportunity.
"""
import json,datetime as dt,gzip
from pathlib import Path
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
from src.twin.protocol import load,EVIDENCE,PRIMARY_SIGNALS
from src.twin.sealing import begin,complete,verify
from src.twin.prepare import prepare
from src.twin.cache import read_cache,take
from src.twin.train import view,predict
from src.twin.models import cumulative
from src.twin.access import Cohort
from src.twin.metrics import horizon_results,patient_bootstrap,paired_ap,basic
from src.twin.analysis import shadow_quality,oracle_annotations,baseline_quality,tdi_analysis,event_aligned,warning_times,decoy_results,uncertainty_results,summary
from src.twin.inference import challenge_mask,quality_uncertainty,twin_state,finite
from src.twin.robustness import run_ablations,run_stress
from src.vitaldb_feasibility.source import private_json
from src.data.io import file_hash


def save_array(path,**arrays):
    arrays.update({k:np.array(v) for k,v in EVIDENCE.items()});path=Path(path);temporary=path.with_suffix('.partial.npz');np.savez_compressed(temporary,**arrays);temporary.chmod(0o600);temporary.replace(path)

def frozen_predict(name,data,engines,config):
    directory=Path(config['output'])/'models'/name;model=joblib.load(directory/'selected.joblib');cal=joblib.load(directory/'calibration.joblib');x,_,dr=view(name,data,engines);raw=predict(model,x,data);hazards=cal.transform(raw);return hazards,cumulative(hazards),dr

def metric_record(name,data,p,thresholds,pred_path,metric_path,config):
    record={'model':name,'patients':len(np.unique(data['patient_ids'])),'rows':len(data['labels']),'horizons':horizon_results(data['labels'],p,thresholds),'patient_bootstrap95':patient_bootstrap(data['labels'],p,data['patient_ids'],thresholds,config['metric_bootstraps']),'predictions_sha256':file_hash(pred_path),**EVIDENCE};private_json(metric_path,record);return record

def ehr_controls(data,engines,config,reference):
    ids=np.unique(data['patient_ids']);rng=np.random.default_rng(config['seed']);shuffled=rng.permutation(ids);first={str(p):np.flatnonzero(data['patient_ids']==p)[0] for p in ids};mapping=dict(zip(ids,shuffled));context=np.stack([data['ehr'][first[str(mapping[p])]] for p in data['patient_ids']]);permuted={**data,'ehr':context};_,p,_=frozen_predict('full_twin',permuted,engines,config)
    out={'permutation_unit':'Whole EHR context shuffled by patient and held fixed across that patient history','mean_absolute_risk72_change':float(np.mean(abs(p[:,2]-reference[:,2]))),'paired_auprc72_original_minus_permuted':paired_ap(data['labels'].sum(axis=1),reference[:,2],p[:,2],data['patient_ids'],config['metric_bootstraps'])}
    # Copy two actual observed EHR contexts to identical physiology rows. This
    # checks model input responsiveness, not a causal intervention or treatment.
    score=data['ehr'][:,0]+10*data['ehr'][:,4]+15*data['ehr'][:,7];good=np.isfinite(score);lo=np.flatnonzero(good)[np.argmin(score[good])];hi=np.flatnonzero(good)[np.argmax(score[good])];sample=take(data,np.arange(len(data['labels']))%127==0);low={**sample,'ehr':np.broadcast_to(data['ehr'][lo],sample['ehr'].shape)};high={**sample,'ehr':np.broadcast_to(data['ehr'][hi],sample['ehr'].shape)};_,lp,_=frozen_predict('full_twin',low,engines,config);_,hp,_=frozen_predict('full_twin',high,engines,config)
    out['fixed_physiology_context_response']={'rows':len(lp),'signed_higher_context_minus_lower_risk72':summary(hp[:,2]-lp[:,2]),'mean_absolute_difference72':float(np.mean(abs(hp[:,2]-lp[:,2]))),'scope':'Noncausal model-response diagnostic with two actual synthetic contexts; no treatment recommendation'};out.update(EVIDENCE);return out

def run(config=None):
    c=config or load();out=Path(c['output']);verify(c);transaction=begin(c)
    if (out/'evaluation_complete.json').exists():return json.loads((out/'test_results.json').read_text())
    for consumer in [False,True]:
        root=out/'features'/('consumer_test' if consumer else 'test')
        if not (root/'manifest.json').exists():
            if root.exists() and any(root.iterdir()):raise ValueError('Incomplete test preparation requires recovery of the same transaction, not silent overwrite')
            prepare('test',c,consumer=consumer)
    data=read_cache('test',c);consumer=read_cache('test',c,consumer=True)
    if not np.array_equal(data['patient_ids'],consumer['patient_ids']) or not np.array_equal(data['landmark_hours'],consumer['landmark_hours']) or not np.array_equal(data['labels'],consumer['labels']):raise ValueError('Consumer prediction schedule mismatch')
    validation=read_cache('validation',c);engines=joblib.load(out/'shadow_engines.joblib');development=json.loads((out/'development_results.json').read_text());models={};probabilities={};full_dr=None;full_h=None
    pool=ThreadPoolExecutor(max_workers=3);pending={}
    for name in c['model_variants']:
        d=consumer if name=='consumer_twin' else data;pred_path=out/f'test_predictions_{name}.npz';metric_path=out/f'test_metrics_{name}.json';thresholds=json.loads((out/'models'/name/'thresholds.json').read_text())['thresholds']
        if pred_path.exists():
            with np.load(pred_path,allow_pickle=False) as z:h=z['hazards'];p=z['probabilities']
            if name=='full_twin':_,_,full_dr=frozen_predict(name,d,engines,c);full_h=h
        else:
            h,p,dr=frozen_predict(name,d,engines,c);save_array(pred_path,hazards=h,probabilities=p,labels=d['labels'],patient_ids=d['patient_ids'],landmark_hours=d['landmark_hours'])
            if name=='full_twin':full_dr=dr;full_h=h
        if not np.all(np.isfinite(p)) or np.any(np.diff(p,axis=1)<-1e-12):raise ValueError('Invalid final cumulative probabilities')
        if metric_path.exists():models[name]=json.loads(metric_path.read_text())
        else:
            pending[name]=pool.submit(metric_record,name,d,p,thresholds,pred_path,metric_path,c)
        probabilities[name]=p;print({'phase6_final_frozen_predictions_saved':name,'transaction_event_id':transaction['event_id']},flush=True)
    for name,future in pending.items():
        models[name]=future.result();print({'phase6_final_frozen_metrics_complete':name,'transaction_event_id':transaction['event_id']},flush=True)
    pool.shutdown(wait=True)
    comparisons={}
    for left,right in [('personalized_fusion','population_fusion'),('full_twin','population_fusion'),('full_twin','personalized_fusion'),('full_twin','full_no_tdi'),('full_twin','full_no_ehr'),('population_fusion','wearable_only'),('personal_normal_only','wearable_only'),('consumer_twin','full_twin')]:
        comparisons[left+'_vs_'+right]=paired_ap(data['labels'].sum(axis=1),probabilities[left][:,2],probabilities[right][:,2],data['patient_ids'],c['metric_bootstraps'])
    challenge=challenge_mask(data,c);challenge_result={'definition':c['challenge'],'rows':int(challenge.sum()),'patients':len(set(data['patient_ids'][challenge])),'outcome_used_to_define_subset':False,'models':{},'comparisons':{},**EVIDENCE}
    if challenge.any():
        for name in ['population_fusion','personalized_fusion','full_twin']:
            thresholds=json.loads((out/'models'/name/'thresholds.json').read_text())['thresholds'];challenge_result['models'][name]={'horizons':horizon_results(data['labels'][challenge],probabilities[name][challenge],thresholds),'patient_bootstrap95':patient_bootstrap(data['labels'][challenge],probabilities[name][challenge],data['patient_ids'][challenge],thresholds,c['metric_bootstraps'])}
        for name in ['personalized_fusion','full_twin']:challenge_result['comparisons'][name+'_vs_population_fusion']=paired_ap(data['labels'][challenge].sum(axis=1),probabilities[name][challenge,2],probabilities['population_fusion'][challenge,2],data['patient_ids'][challenge],c['metric_bootstraps'])
    private_json(out/'personalization_challenge.json',challenge_result)
    full_p=probabilities['full_twin'];full_thresholds=json.loads((out/'models/full_twin/thresholds.json').read_text())['thresholds'];cohort=Cohort(c['cohort'],out,c)
    # Oracle evaluation begins only now, after model selection and predictions.
    oracle=oracle_annotations(cohort,'test',data);val_oracle=oracle_annotations(cohort,'validation',validation);shadow={'validation_visible_nonevent_proxy':shadow_quality(engines['shadow'],validation,validation['labels'].sum(axis=1)==0,c['metric_bootstraps']),'validation_oracle_stable48':shadow_quality(engines['shadow'],validation,val_oracle['stable48'],c['metric_bootstraps']),'test_oracle_stable48':shadow_quality(engines['shadow'],data,oracle['stable48'],c['metric_bootstraps']),**EVIDENCE}
    private_json(out/'shadow_results.json',shadow);baseline=baseline_quality(data,oracle);private_json(out/'baseline_results.json',baseline);tdi=tdi_analysis(data,full_dr,oracle);private_json(out/'tdi_results.json',tdi);aligned=event_aligned(data,full_p,full_dr,oracle,c);private_json(out/'event_aligned.json',aligned);warnings=warning_times(data,full_p,full_thresholds,oracle);decoys=decoy_results(data,full_p,full_thresholds,oracle);private_json(out/'decoy_results.json',decoys);ehr=ehr_controls(data,engines,c,full_p);private_json(out/'ehr_fusion_control.json',ehr)
    pre_event=(~np.isfinite(oracle['event_hour']))|(data['landmark_hours']<oracle['event_hour']);pre_event_result={'rows':int(pre_event.sum()),'scope':'Evaluation-only sensitivity analysis; primary fixed schedule retains recovery/post-event landmarks and cannot validate recurrence','horizons':horizon_results(data['labels'][pre_event],full_p[pre_event],full_thresholds),**EVIDENCE};private_json(out/'pre_event_only_results.json',pre_event_result)
    ensemble_results={};full_sd=None;full_quality=quality_uncertainty(data)
    for name in sorted({'full_twin',development['primary_model']}):
        d=consumer if name=='consumer_twin' else data;x,_,_=view(name,d,engines);cal=joblib.load(out/'models'/name/'calibration.joblib');samples=[]
        for member in range(c['uncertainty']['bootstrap_members']):
            model=joblib.load(out/'models'/name/f'bootstrap_{member}.joblib');samples.append(cumulative(cal.transform(predict(model,x,d))))
        samples=np.stack(samples);q=quality_uncertainty(d,[PRIMARY_SIGNALS.index(s) for s in c['consumer_signals']] if name=='consumer_twin' else None);uq,sd=uncertainty_results(d,probabilities[name],samples,q);ensemble_results[name]=uq;save_array(out/f'test_uncertainty_{name}.npz',samples=samples,sd=sd,quantile05=np.quantile(samples,.05,axis=0),quantile95=np.quantile(samples,.95,axis=0),data_quality=q)
        if name=='full_twin':full_sd=sd
    private_json(out/'uncertainty_results.json',{'models':ensemble_results,**EVIDENCE})
    # Select representative outputs deterministically from observed challenge
    # membership/risk extremes; oracle events never alter the prediction itself.
    order=list(np.flatnonzero(challenge)[:5])+list(np.argsort(full_p[:,2])[-5:])+list(np.argsort(full_p[:,2])[:5]);order=list(dict.fromkeys(order));states=[];generated_at=dt.datetime.now(dt.timezone.utc).isoformat();trajectory=engines['shadow'].trajectory(take(data,np.array(order,int)));starts={row['patient_id']:row['observed_time'] for row in cohort.rows('patients','test')}
    for k,i in enumerate(order):
        timestamp=(dt.datetime.fromisoformat(starts[str(data['patient_ids'][i])].replace('Z','+00:00'))+dt.timedelta(hours=int(data['landmark_hours'][i]))).isoformat()
        state=twin_state(i,data,full_dr,full_h,full_sd,full_quality,full_thresholds,timestamp);state['generated_at_utc']=generated_at;state['retrospective_7day_template']={'relative_hours':trajectory['relative_hours'].tolist(),'actual_six_hour_bin_values':[[finite(v) for v in row] for row in data['sequence'][i,:,:,0]],'observed_mask':data['sequence'][i,:,:,1].astype(bool).tolist(),'expected':[[finite(v) for v in row] for row in trajectory['expected'][k]],'lower':[[finite(v) for v in row] for row in trajectory['lower'][k]],'upper':[[finite(v) for v in row] for row in trajectory['upper'][k]],'scope':trajectory['range_scope']};states.append(state)
    private_json(out/'twin_states.json',{'states':states,'model':'full_twin',**EVIDENCE});save_array(out/'test_twin_arrays.npz',tdi=full_dr['tdi'],z=full_dr['z'],expected=full_dr['expected'],components=full_dr['components'],challenge=challenge,quality=full_quality)
    reasons_path=out/'why_now.jsonl.gz'
    if not reasons_path.exists():
        temporary=reasons_path.with_suffix('.partial.gz');record_count=0
        with temporary.open('wb') as raw:
            temporary.chmod(0o600)
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=5) as stream:
                for i in range(len(data['labels'])):
                    timestamp=(dt.datetime.fromisoformat(starts[str(data['patient_ids'][i])].replace('Z','+00:00'))+dt.timedelta(hours=int(data['landmark_hours'][i]))).isoformat();state=twin_state(i,data,full_dr,full_h,full_sd,full_quality,full_thresholds,timestamp)
                    record={key:state[key] for key in ['patient_id','timestamp','landmark_hour','risk','divergence','signal_units','uncertainty','why_now']};record.update(EVIDENCE);stream.write((json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode());record_count+=1
        temporary.replace(reasons_path);private_json(out/'why_now_manifest.json',{'rows':record_count,'model':'full_twin','sha256':file_hash(reasons_path),'deterministic_structured_evidence':True,'LLM_generated':False,**EVIDENCE})
    sensor_path=out/'sensor_ablations.json'
    sensor=json.loads(sensor_path.read_text()) if sensor_path.exists() else run_ablations(data,engines,full_p,c)
    if not sensor_path.exists():private_json(sensor_path,sensor)
    stress=run_stress(engines,c);private_json(out/'stress_results.json',stress);cohort.save_access_log('access_final_oracle_evaluation.json')
    event_count=sum(row['event_time_hours'] is not None for row in oracle['truth'].values())
    result={'primary_model':development['primary_model'],'full_twin_model':'full_twin','test_event_patients':event_count,'test_event_patient_prevalence':event_count/1500,'models':models,'paired_auprc72_differences':comparisons,'challenge':challenge_result,'shadow':shadow,'baseline':baseline,'tdi':tdi,'ehr_control':ehr,'warning_times_full_twin':warnings,'decoys':decoys,'uncertainty':ensemble_results,'sensor_ablations':sensor,'stress':stress,'pre_event_sensitivity_analysis':pre_event_result,'transaction_event_id':transaction['event_id'],'joint_test_evaluations':1,'bootstrap_scope':'Patient-cluster uncertainty conditional on frozen fit/calibration/thresholds; dependence across daily landmarks preserved',**EVIDENCE}
    private_json(out/'test_results.json',result)
    from src.twin.plots import generate
    from src.twin.report import generate as report
    generate(c);complete(result,c);report(c);print({'phase6_joint_final_evaluation_complete':True,'test_patients':1500,'test_landmarks':76500},flush=True);return result

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
