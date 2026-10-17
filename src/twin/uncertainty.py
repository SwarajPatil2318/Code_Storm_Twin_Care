"""Training-patient bootstrap, frozen preprocessing/Shadow and fixed calibration."""
import json,hashlib
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from src.twin.protocol import load,EVIDENCE
from src.twin.access import Cohort
from src.twin.cache import read_cache
from src.twin.models import HazardModel,cumulative
from src.twin.temporal import TemporalHazard
from src.twin.train import view,predict,save,learning_hash
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def bootstrap_seed(seed,name,member):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{member}'.encode()).digest()[:8],'little')

def fit(config=None):
    c=config or load();out=Path(c['output']);cohort=Cohort(c['cohort'],out,c)
    if (out/'selection_seal.json').exists():raise ValueError('Uncertainty refits must precede test seal')
    development=json.loads((out/'development_results.json').read_text());engines=joblib.load(out/'shadow_engines.joblib');train=read_cache('train',c);val=read_cache('validation',c);ids=np.unique(train['patient_ids']);models={}
    for name in sorted({'full_twin',development['primary_model']}):
        train=read_cache('train',c,consumer=name=='consumer_twin');val=read_cache('validation',c,consumer=name=='consumer_twin');ids=np.unique(train['patient_ids'])
        directory=out/'models'/name;x,names,_=view(name,train,engines);vx,_,_=view(name,val,engines);base=joblib.load(directory/'selected.joblib');cal=joblib.load(directory/'calibration.joblib');predictions=[]
        for member in range(c['uncertainty']['bootstrap_members']):
            path=directory/f'bootstrap_{member}.joblib'
            provenance={'config_sha256':c['_sha256'],'learning_code_sha256':learning_hash(),'base_sha256':file_hash(directory/'selected.joblib'),'seed':bootstrap_seed(c['seed'],name,member)}
            if path.exists():
                if json.loads(path.with_suffix('.json').read_text())!=provenance:raise ValueError('Stale uncertainty checkpoint')
                model=joblib.load(path)
            else:
                rng=np.random.default_rng(provenance['seed'])
                drawn=rng.choice(ids,size=len(ids),replace=True);unique,counts=np.unique(drawn,return_counts=True);multiplicity=dict(zip(unique.astype(str),counts.tolist()))
                model=TemporalHazard(base.spec,base.columns,names).fit(x,train['sequence'],train['labels'],train['patient_ids'],cohort.roles,multiplicity=multiplicity) if isinstance(base,TemporalHazard) else HazardModel(base.spec,base.columns,names).fit(x,train['labels'],train['patient_ids'],cohort.roles,multiplicity=multiplicity)
                save(path,model)
                private_json(path.with_suffix('.json'),provenance)
            predictions.append(cumulative(cal.transform(predict(model,vx,val))));print({'uncertainty_bootstrap_member':member+1,'model':name,'test_used':False},flush=True)
        samples=np.stack(predictions);np.save(directory/'validation_ensemble.npy',samples);(directory/'validation_ensemble.npy').chmod(0o600);models[name]={'members':len(samples),'calibration':'fixed validation-only calibrator','scope':'Conditional model disagreement with fixed train-only feature transformation/Shadow; neither coverage guarantee nor clinical uncertainty'}
    private_json(out/'uncertainty_fit.json',{'models':models,'test_used':False,**EVIDENCE});return models

if __name__=='__main__':
    with threadpool_limits(limits=1):fit()
