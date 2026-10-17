"""Development-only fitting, explicit ablations and validation model selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.twin.protocol import load,EVIDENCE,columns,PRIMARY_SIGNALS,PERSONAL_STATS,ALLOWLIST,require_role
from src.twin.access import Cohort
from src.twin.cache import read_cache,validation_groups,matrix
from src.twin.shadow import ShadowTwin
from src.twin.divergence import Divergence
from src.twin.models import HazardModel,cumulative
from src.twin.temporal import TemporalHazard
from src.twin.calibration import select_calibration,operating_points
from src.twin.metrics import horizon_results,paired_ap
from src.vitaldb_feasibility.source import private_json


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);joblib.dump(value,path);path.chmod(0o600)

def predict(model,x,data):return model.predict_hazards(x,data['sequence']) if isinstance(model,TemporalHazard) else model.predict_hazards(x)

def population_reference(data,shadow):
    d=dict(data);n=len(data['recent']);d['reference']=np.broadcast_to(shadow.population,(n,len(PRIMARY_SIGNALS))).copy()
    # Generic dispersion learned on TRAIN stable references, not patient variance.
    d['scale']=np.broadcast_to(shadow.population_scale,d['reference'].shape).copy();d['confidence']=np.zeros_like(d['reference']);d['baseline_count']=np.zeros_like(d['reference']);d['persistence']=(abs(d['recent']-d['reference'])/d['scale']>1.5).astype(float);d['persistence'][~np.isfinite(d['recent'])]=np.nan;return d

def without_ehr(data):return {**data,'ehr':np.zeros_like(data['ehr'])}

def view(name,data,engines):
    if name=='consumer_twin':working=data;shadow=engines['consumer_shadow'];div=engines['consumer_divergence']
    elif name in ['full_no_ehr','personal_normal_only']:working=without_ehr(data);shadow=engines['no_ehr_shadow'];div=engines['no_ehr_divergence']
    elif name=='full_no_personal':working=population_reference(data,engines['shadow']);shadow=engines['population_shadow'];div=engines['population_divergence']
    else:working=data;shadow=engines['shadow'];div=engines['divergence']
    forecast=shadow.predict(working);dr=div.transform(working,forecast);x,names=matrix(working,dr)
    return x,names,dr

def model_columns(name,names,config):
    if name=='full_no_personal':return [i for i,n in enumerate(names) if not n.startswith('personal__')]
    if name=='personal_normal_only':
        selected=[]
        for i,n in enumerate(names):
            group,signal,key=n.split('__') if n.count('__')==2 else (*n.split('__'),None)
            if group=='personal' and key in ['sd','count','confidence','frozen','recent_deviation','recent_z','persistence'] or group=='shadow' and key in ['scale','residual','z','persistence','acceleration','count','width'] or group=='tdi':selected.append(i)
        return selected
    return columns(name,names,config['consumer_signals'])

def learning_hash():
    h=hashlib.sha256()
    for name in ['models.py','temporal.py','shadow.py','divergence.py','train.py']:
        h.update(name.encode());h.update((Path('src/twin')/name).read_bytes())
    return h.hexdigest()

def train(config=None,only=None):
    c=config or load();out=Path(c['output']);cohort=Cohort(c['cohort'],out,c)
    if (out/'selection_seal.json').exists():raise ValueError('Model development is closed after selection seal')
    train=read_cache('train',c);validation=read_cache('validation',c);groups,val_roles=validation_groups(validation['patient_ids'],c);selection=groups['selection']
    engines_path=out/'shadow_engines.joblib'
    if engines_path.exists():engines=joblib.load(engines_path)
    else:
        shadow=ShadowTwin(c).fit(train,cohort.roles);shadow.population_scale=np.maximum(np.nanstd(train['recent'][shadow.stable_selection],axis=0),1)
        divergence=Divergence(c).fit(train,shadow.predict(train),cohort.roles,shadow.stable_selection)
        no_ehr_data=without_ehr(train);no_ehr=ShadowTwin(c).fit(no_ehr_data,cohort.roles);nd=Divergence(c).fit(no_ehr_data,no_ehr.predict(no_ehr_data),cohort.roles,no_ehr.stable_selection)
        population_data=population_reference(train,shadow);pop=ShadowTwin(c).fit(population_data,cohort.roles);pd=Divergence(c).fit(population_data,pop.predict(population_data),cohort.roles,pop.stable_selection)
        engines={'shadow':shadow,'divergence':divergence,'no_ehr_shadow':no_ehr,'no_ehr_divergence':nd,'population_shadow':pop,'population_divergence':pd};save(engines_path,engines)
        quality=shadow.quality(validation,validation['labels'].sum(axis=1)==0);private_json(out/'shadow_validation_proxy.json',{'stable_definition':'No near-term event labels; evaluation proxy, not hidden stable truth','signals':quality,**EVIDENCE})
    results={};models={};preds={};model_names=c['model_variants'] if only is None else only
    for name in model_names:
        path=out/'models'/name/'selected.joblib'
        if path.exists():
            record=json.loads((path.parent/'selection.json').read_text())
            if record['config_sha256']!=c['_sha256'] or record['learning_code_sha256']!=learning_hash():raise ValueError('Stale model checkpoint')
            selected=joblib.load(path);models[name]=selected;results[name]=record;continue
        if name=='consumer_twin':
            training=read_cache('train',c,consumer=True);held=read_cache('validation',c,consumer=True)
            if 'consumer_shadow' not in engines:
                cs=ShadowTwin(c,available_signals=c['consumer_signals']).fit(training,cohort.roles);cd=Divergence(c).fit(training,cs.predict(training),cohort.roles,cs.stable_selection);engines['consumer_shadow']=cs;engines['consumer_divergence']=cd;save(engines_path,engines)
        else:training=train;held=validation
        x,names,_=view(name,training,engines);vx,_,_=view(name,held,engines);cols=[] if name=='prevalence' else model_columns(name,names,c);trials=[];best=None;best_ap=-1
        specifications=[{'kind':'prevalence'}] if name=='prevalence' else c['tabular_candidates']+(c['temporal_candidates'] if name in ['population_fusion','full_twin'] else [])
        for spec in specifications:
            model=TemporalHazard(spec,cols,names).fit(x,training['sequence'],training['labels'],training['patient_ids'],cohort.roles) if spec['kind']=='causal_tcn' else HazardModel(spec,cols,names).fit(x,training['labels'],training['patient_ids'],cohort.roles)
            raw=predict(model,vx,held);prob=cumulative(raw);ap=float(average_precision_score(held['labels'][selection].sum(axis=1),prob[selection,2]));trials.append({'spec':spec,'selection_auprc72':ap})
            print({'phase6_candidate':name,'kind':spec['kind'],'validation_selection_auprc72':round(ap,5)},flush=True)
            if ap>best_ap:best_ap=ap;best=model;best_raw=raw
        save(path,best);np.save(path.parent/'validation_raw_hazards.npy',best_raw);(path.parent/'validation_raw_hazards.npy').chmod(0o600)
        result={'model':name,'config_sha256':c['_sha256'],'learning_code_sha256':learning_hash(),'spec':best.spec,'candidates':trials,'selected_auprc72':best_ap,'selection_patients':c['validation_roles']['selection'],'input_features':best.names,'test_used':False,**EVIDENCE}
        private_json(path.parent/'selection.json',result);results[name]=result;models[name]=best
        del x,vx
    private_json(out/'model_selection.json',{'models':results,'validation_subroles':val_roles,'selection_criterion':c['primary_selection'],'complete_model_variants':model_names,'test_metrics_used':False,**EVIDENCE})
    cohort.save_access_log('access_model_development.json');print({'phase6_models_fitted':len(results),'test_access':False},flush=True)
    return results

def finalize_development(config=None):
    c=config or load();out=Path(c['output']);cohort=Cohort(c['cohort'],out,c);validation=read_cache('validation',c);groups,val_roles=validation_groups(validation['patient_ids'],c);results={}
    for name in c['model_variants']:
        directory=out/'models'/name;raw=np.load(directory/'validation_raw_hazards.npy');cal_group=groups['calibration'];op_group=groups['operating'];cal,trials=select_calibration(raw[cal_group],validation['labels'][cal_group],validation['patient_ids'][cal_group],cohort.roles,c)
        calibrated=cal.transform(raw);prob=cumulative(calibrated);threshold=operating_points(prob[op_group],validation['labels'][op_group],validation['patient_ids'][op_group],cohort.roles,c)
        save(directory/'calibration.joblib',cal);private_json(directory/'thresholds.json',{'thresholds':threshold,'validation_patients':c['validation_roles']['operating'],'fit_patient_ids':sorted(set(validation['patient_ids'][op_group].astype(str))),'test_used':False,**EVIDENCE})
        np.save(directory/'validation_hazards.npy',calibrated);(directory/'validation_hazards.npy').chmod(0o600)
        result={'model':name,'calibration_method':cal.method,'calibration_trials':trials,'selection':json.loads((directory/'selection.json').read_text()),'validation_development_metrics':horizon_results(validation['labels'],prob,threshold),**EVIDENCE};private_json(directory/'development.json',result);results[name]=result
    selection=groups['selection'];scores={name:results[name]['selection']['selected_auprc72'] for name in c['model_variants']};primary=max(scores,key=scores.get)
    comparisons={}
    for left,right in [('personalized_fusion','population_fusion'),('full_twin','population_fusion'),('full_twin','personalized_fusion'),('full_twin','full_no_tdi'),('full_twin','full_no_ehr')]:
        a=cumulative(np.load(out/'models'/left/'validation_raw_hazards.npy'))[selection,2];b=cumulative(np.load(out/'models'/right/'validation_raw_hazards.npy'))[selection,2]
        comparisons[left+'_vs_'+right]=paired_ap(validation['labels'][selection].sum(axis=1),a,b,validation['patient_ids'][selection],n=c['metric_bootstraps'])
    private_json(out/'development_results.json',{'models':results,'primary_model':primary,'full_twin_model':'full_twin','paired_selection_ap_differences':comparisons,'test_used':False,'calibration_and_threshold_roles_disjoint':True,**EVIDENCE})
    print({'phase6_development_finalized':True,'primary_model':primary,'test_used':False},flush=True)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):finalize_development() if a.finalize else train(only=a.models)
