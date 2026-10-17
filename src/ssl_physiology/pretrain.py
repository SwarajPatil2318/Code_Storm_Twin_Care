"""Two small SSL objectives; selection uses VitalDB validation only."""
from collections import Counter
import copy,json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from src.models.common import save_model
from src.vitaldb_feasibility.source import private_json
from src.ssl_physiology.protocol import ROOT,SIGNALS,PROTOCOL
from src.ssl_physiology.encoder import TCN,Normalizer,Adam,mask_input,allowed_channels
from src.ssl_physiology.cohort import save_npz

def load(minutes=60):
    rows=json.loads((ROOT/f'vitaldb_{minutes}_rows.json').read_text())
    with np.load(ROOT/f'vitaldb_{minutes}.npz',allow_pickle=False) as z:seq=z['sequence'];future=z['future']
    train=np.array([r['role']=='train' for r in rows]);return rows,seq,future,train

def future_targets(seq,future,norm):
    extended=np.concatenate([seq[:,:,:6],future],axis=1);target=norm.targets(extended[:,3:3+seq.shape[1]])
    return target,np.isfinite(target)

def prepare_targets(task,x,target,rng,mode):
    if task=='causal_masked_reconstruction':inp,mask=mask_input(x,rng);y=x[:,:,:6]
    else:inp=x;y=target;mask=np.isfinite(y)
    excluded=[j for j in range(6) if j not in allowed_channels(mode)];mask[:,:,excluded]=False
    return inp,y,mask

def train_one(task,mode,seq,future,rows,train,epochs=60):
    roles={r['patient_id']:r['role'] for r in rows};norm=Normalizer().fit(seq[train],[r['patient_id'] for r,t in zip(rows,train) if t],roles)
    x=norm.transform(seq,mode);target,_=future_targets(seq,future,norm);model=TCN(mode)
    counts=Counter(r['patient_id'] for r,t in zip(rows,train) if t);weight=np.array([1/counts[r['patient_id']] if t else 0 for r,t in zip(rows,train)])
    opt=Adam(model.params,.003);rng=np.random.default_rng(2026);indices=np.flatnonzero(train);curve=[];pilot=None
    for epoch in range(epochs):
        losses=[]
        for start in range(0,len(indices),128):
            if start==0:order=rng.permutation(indices)
            ix=order[start:start+128];inp,y,mask=prepare_targets(task,x[ix],target[ix],rng,mode)
            loss,gradient=model.loss_gradient(inp,y,mask,weight[ix],.001)
            if not np.isfinite(loss):raise ValueError('Nonfinite SSL loss')
            opt.update(model.params,gradient);losses.append(loss)
        if epoch==9:pilot=assess(model,norm,seq,future,train,task,mode)
        if (epoch+1)%10==0:
            held=np.flatnonzero(~train);inp,y,mask=prepare_targets(task,x[held],target[held],np.random.default_rng(777),mode)
            validation_loss=model.loss_gradient(inp,y,mask,l2=0)[0];curve.append({'epoch':epoch+1,'train_loss_with_l2':float(np.mean(losses)),'heldout_objective_mse':validation_loss})
            print({'ssl_task':task,'mode':mode,'epoch':epoch+1,'heldout_objective_mse':round(validation_loss,4)},flush=True)
    final=assess(model,norm,seq,future,train,task,mode)
    bundle={'encoder':model,'normalizer':norm,'task':task,'mode':mode,'fit_patients':sorted(counts),'validation_patients':sorted({r['patient_id'] for r,t in zip(rows,train) if not t}),
            'epochs':epochs,'curve':curve,'pilot10':pilot,'heldout':final,'config':PROTOCOL}
    name=task+'__'+mode;save_model(ROOT/(name+'.joblib'),bundle)
    save_npz(ROOT/(name+'_latent.npz'),latent=model.embedding(norm.transform(seq,mode)))
    return bundle

def metric_channels(truth,prediction,norm):
    result={}
    for j,signal in enumerate(SIGNALS):
        mask=np.isfinite(truth[:,j])&np.isfinite(prediction[:,j])
        if sum(mask)<2:result[signal]=None;continue
        error=prediction[mask,j]-truth[mask,j]
        result[signal]={'n':int(sum(mask)),'normalized_mae':float(np.mean(abs(error))),
                        'mae_original_units':float(np.mean(abs(error))*norm.scale[j]),
                        'r2':float(r2_score(truth[mask,j],prediction[mask,j]))}
    return result

def linear_probe(z,truth,train):
    prediction=np.full_like(truth,np.nan)
    for j in range(truth.shape[1]):
        good=train&np.isfinite(truth[:,j]);prediction[:,j]=Ridge(alpha=10).fit(z[good],truth[good,j]).predict(z) if sum(good)>3 else np.nan
    return prediction

def assess(model,norm,seq,future,train,task,mode):
    x=norm.transform(seq,mode);z=model.embedding(x);truth=norm.targets(future[:,-1]);current=norm.targets(seq[:,-1,:6]);held=~train
    prediction=linear_probe(z,truth,train);properties=linear_probe(z,norm.targets(np.nanmedian(seq[:,-3:,:6],axis=1)),train)
    handcrafted=np.concatenate([np.nan_to_num(norm.targets(seq[:,-1,:6]),nan=0),np.nan_to_num(norm.targets(np.nanmean(seq[:,-6:,:6],axis=1)),nan=0),np.mean(seq[:,:,6:12],axis=1)],axis=1)
    excluded=[j for j in range(6) if j not in allowed_channels(mode)]
    handcrafted[:,excluded]=0;handcrafted[:,np.array(excluded,dtype=int)+6]=0;handcrafted[:,np.array(excluded,dtype=int)+12]=0
    base=linear_probe(handcrafted,truth,train)
    result={'future15_linear_probe':metric_channels(truth[held],prediction[held],norm),
            'future15_persistence':metric_channels(truth[held],current[held],norm),
            'future15_handcrafted_linear':metric_channels(truth[held],base[held],norm),
            'current_property_linear_probe':metric_channels(norm.targets(np.nanmedian(seq[:,-3:,:6],axis=1))[held],properties[held],norm)}
    allowed=allowed_channels(mode);result['selection_mean_future_normalized_mae']=float(np.mean([result['future15_linear_probe'][SIGNALS[j]]['normalized_mae'] for j in allowed if result['future15_linear_probe'][SIGNALS[j]]]))
    if task=='causal_masked_reconstruction':
        inp,mask=mask_input(x[held],np.random.default_rng(777));mask[:,:,excluded]=False;pred=model.forward(inp)@model.params['out']+model.params['out_bias']
        error=np.where(mask,pred-x[held,:,:6],0)
        result['masked_reconstruction_normalized_mae']=float(sum(abs(error).ravel())/sum(mask.ravel()))
    return result

def run():
    rows,seq,future,train=load();models={};results={}
    with threadpool_limits(limits=1):
        for task in PROTOCOL['ssl_tasks']:
            bundle=train_one(task,'all',seq,future,rows,train);models[task]=bundle
            results[task+'__all']={k:bundle[k] for k in ['curve','pilot10','heldout','epochs']}
        selected=min(models,key=lambda t:models[t]['heldout']['selection_mean_future_normalized_mae'])
        for mode in ['no_map','no_bp']:
            bundle=train_one(selected,mode,seq,future,rows,train);results[selected+'__'+mode]={k:bundle[k] for k in ['curve','pilot10','heldout','epochs']}
    private_json(ROOT/'ssl_results.json',{'selected_task':selected,'selection_basis':'heldoutVitalDB future15minute linear probe normalizedMAE; no eICU scores used','models':results,'architectures_evaluated':['compact_causal_tcn'],'gru_transformer':'not justified; small cohort and compact audited existing NumPy approach','independent_validation_still_required':True})
    print({'selected_ssl_task':selected,'pretrained_encoders':len(results),'eicu_test_accessed':False})

if __name__=='__main__':run()
