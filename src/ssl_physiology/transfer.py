"""Nested development CV; no original-test reader or prediction path exists."""
import copy,json
from collections import Counter
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import joblib
from threadpoolctl import threadpool_limits
from src.predictive.models import Tabular,Encoder
from src.predictive.metrics import basic,Calibrator
from src.predictive.data import weights
from src.ssl_physiology.protocol import ROOT,PROTOCOL
from src.ssl_physiology.encoder import TCN,Normalizer,Adam
from src.ssl_physiology.cohort import save_npz
from src.vitaldb_feasibility.source import private_json,sha

def folds(rows,seed=2026,n=5):
    patients={r['patient_id'] for r in rows};label={p:int(any(r['label'] for r in rows if r['patient_id']==p)) for p in patients};result={}
    for y in [0,1]:
        order=sorted([p for p in patients if label[p]==y],key=lambda p:sha((str(seed)+':'+p).encode()))
        for i,p in enumerate(order):result[p]=i%n
    return result

class LatentHead:
    def __init__(self,c=.1):self.c=c
    def fit(self,x,rows):
        self.scaler=StandardScaler().fit(x);self.head=LogisticRegression(C=self.c,max_iter=1000,random_state=2026).fit(self.scaler.transform(x),[r['label'] for r in rows],sample_weight=weights(rows));return self
    def predict(self,x):return self.head.predict_proba(self.scaler.transform(x))[:,1]

class FineTune:
    def __init__(self,bundle,epochs=20,scratch=False):self.bundle=bundle;self.epochs=epochs;self.scratch=scratch
    def fit(self,seq,rows,ehr=None):
        self.encoder=copy.deepcopy(self.bundle['encoder']) if not self.scratch else TCN()
        if self.scratch:self.norm=Normalizer().fit(seq,[r['patient_id'] for r in rows],{r['patient_id']:'train' for r in rows})
        else:self.norm=self.bundle['normalizer']
        x=self.norm.transform(seq);z=self.encoder.embedding(x);features=np.concatenate([z,ehr],axis=1) if ehr is not None else z
        head=LatentHead(.1).fit(features,rows);self.center=head.scaler.mean_;self.scale=head.scaler.scale_;self.params={**self.encoder.params,'head':head.head.coef_[0].copy(),'bias':np.array(head.head.intercept_[0])}
        anchor={k:v.copy() for k,v in self.encoder.params.items()};opt=Adam(self.params,.0005);w=weights(rows);y=np.array([r['label'] for r in rows]);rng=np.random.default_rng(2026)
        keys=['w0','b0','w1','b1','w2','b2','head','bias'] if self.scratch else ['w2','b2','head','bias']
        self.fit_patients=sorted({r['patient_id'] for r in rows})
        for epoch in range(self.epochs):
            order=rng.permutation(len(rows))
            for start in range(0,len(rows),128):
                idx=order[start:start+128];h,states=self.encoder.forward(x[idx],True);z=np.concatenate([np.mean(h,axis=1),h[:,-1]],axis=1)
                f=np.concatenate([z,ehr[idx]],axis=1) if ehr is not None else z;f=(f-self.center)/self.scale
                p=expit(f@self.params['head']+self.params['bias']);d=(p-y[idx])*w[idx]/sum(w[idx])
                dz=(d[:,None]*self.params['head']/self.scale)[:,:16];dh=np.broadcast_to(dz[:,:8,None],(len(idx),8,h.shape[1])).transpose(0,2,1).copy()/h.shape[1];dh[:,-1]+=dz[:,8:16]
                gradient=self.encoder.backward(states,dh);gradient['head']=f.T@d+.1*self.params['head'];gradient['bias']=np.array(sum(d))
                for k in keys:
                    if k.startswith('w'):gradient[k]+=.1*self.params[k]+(0 if self.scratch else .1*(self.params[k]-anchor[k]))
                opt.update(self.params,gradient,keys)
        # Both dictionaries share convolution arrays. Early-layer invariance is checked.
        if not self.scratch:
            for k in ['w0','b0','w1','b1']:
                if not np.array_equal(self.encoder.params[k],anchor[k]):raise ValueError('Early pretrained layer changed')
        return self
    def predict(self,seq,ehr=None):
        z=self.encoder.embedding(self.norm.transform(seq));f=np.concatenate([z,ehr],axis=1) if ehr is not None else z
        return expit(((f-self.center)/self.scale)@self.params['head']+self.params['bias'])

def tabular_fit(train,held,modality,config):
    model=Tabular(modality,config).fit(train,{r['patient_id']:'train' for r in train});return model,model.predict(held)

def tabular_nested(rows,train_idx,held_idx,modality,only_logistic=False):
    training=[rows[i] for i in train_idx];held=[rows[i] for i in held_idx];split=folds(training,n=3)
    configs=[{'kind':'logistic','penalty':'l2','C':c,'balanced':False} for c in [.1,1]]
    if not only_logistic:configs+=[{'kind':'hist_gradient_boosting','leaves':3,'iterations':100,'balanced':False}]
    predictions=[];scores=[]
    for config in configs:
        oof=np.zeros(len(training))
        for f in range(3):
            fit=[r for r in training if split[r['patient_id']]!=f];indices=[i for i,r in enumerate(training) if split[r['patient_id']]==f]
            _,p=tabular_fit(fit,[training[i] for i in indices],modality,config);oof[indices]=p
        predictions.append(oof);scores.append(average_precision_score([r['label'] for r in training],oof))
    selected=int(np.argmax(scores));model,p=tabular_fit(training,held,modality,configs[selected]);return model,p,predictions[selected],{'config':configs[selected],'inner_ap':scores}

def numeric_matrix(rows,modality='ehr'):
    # Encoder fit separately by outer/inner callers; there is no global eICU preprocessing.
    return Encoder().fit(rows,{r['patient_id']:'train' for r in rows},modality)

def latent_nested(matrix,rows,train_idx,held_idx):
    training=[rows[i] for i in train_idx];split=folds(training,n=3);results=[]
    for c in [.1,1]:
        oof=np.zeros(len(training))
        for f in range(3):
            a=np.array([split[r['patient_id']]!=f for r in training]);model=LatentHead(c).fit(matrix[train_idx][a],[r for r,yes in zip(training,a) if yes]);oof[~a]=model.predict(matrix[train_idx][~a])
        results.append(oof)
    best=int(np.argmax([average_precision_score([r['label'] for r in training],p) for p in results]));model=LatentHead([.1,1][best]).fit(matrix[train_idx],training)
    return model,model.predict(matrix[held_idx]),results[best],{'C':[.1,1][best]}

def ehr_latent_nested(latent,rows,train_idx,held_idx):
    training=[rows[i] for i in train_idx];held=[rows[i] for i in held_idx];split=folds(training,n=3);results=[]
    for c in [.1,1]:
        oof=np.zeros(len(training))
        for f in range(3):
            a=np.array([split[r['patient_id']]!=f for r in training]);fit=[r for r,yes in zip(training,a) if yes];val=[r for r,yes in zip(training,a) if not yes]
            encoder=numeric_matrix(fit);x=np.concatenate([latent[train_idx][a],encoder.transform(fit)],axis=1);v=np.concatenate([latent[train_idx][~a],encoder.transform(val)],axis=1)
            oof[~a]=LatentHead(c).fit(x,fit).predict(v)
        results.append(oof)
    best=int(np.argmax([average_precision_score([r['label'] for r in training],p) for p in results]));encoder=numeric_matrix(training)
    x=np.concatenate([latent[train_idx],encoder.transform(training)],axis=1);v=np.concatenate([latent[held_idx],encoder.transform(held)],axis=1)
    model=LatentHead([.1,1][best]).fit(x,training);return (model,encoder),model.predict(v),results[best],{'C':[.1,1][best]}

def finetune_nested(bundle,seq,rows,train_idx,held_idx,ehr=False,scratch=False):
    training=[rows[i] for i in train_idx];held=[rows[i] for i in held_idx];split=folds(training,n=3);oof=np.zeros(len(training))
    for f in range(3):
        a=np.array([split[r['patient_id']]!=f for r in training]);fit=[r for r,yes in zip(training,a) if yes];val=[r for r,yes in zip(training,a) if not yes]
        encoder=numeric_matrix(fit) if ehr else None
        model=FineTune(bundle,scratch=scratch).fit(seq[train_idx][a],fit,encoder.transform(fit) if ehr else None)
        oof[~a]=model.predict(seq[train_idx][~a],encoder.transform(val) if ehr else None)
    encoder=numeric_matrix(training) if ehr else None;model=FineTune(bundle,scratch=scratch).fit(seq[train_idx],training,encoder.transform(training) if ehr else None)
    p=model.predict(seq[held_idx],encoder.transform(held) if ehr else None);return (model,encoder),p,oof,{'epochs':20,'early_layers_frozen':not scratch}

def calibrate(oof,raw,training):
    # Calibration labels/parameters originate exclusively inside the outer training population.
    cal=Calibrator('platt').fit(oof,training,{r['patient_id']:'validation' for r in training});return cal,cal.predict(raw)

def references(bundle,seq):
    norm=bundle['normalizer'];encoder=bundle['encoder'];recent=encoder.embedding(norm.transform(seq[:,-12:]));refseq=seq[:,:24]
    observed=refseq[:,:,6:11]>0;corecoverage=np.mean(observed,axis=1)>=.8
    map_ok=np.nanmin(refseq[:,:,0],axis=1)>=75;available=np.all(corecoverage,axis=1)&map_ok
    reference=encoder.embedding(norm.transform(refseq));delta=np.where(available[:,None],recent-reference,0)
    return recent,np.concatenate([recent,delta,available[:,None]],axis=1),available

def run():
    rows=json.loads((ROOT/'eicu_development_rows.json').read_text());access=json.loads((ROOT/'development_access.json').read_text());allowed=set(access['allowed_patients'])
    if any(r['patient_id'] not in allowed for r in rows):raise ValueError('eICU outside development')
    with np.load(ROOT/'eicu_development.npz',allow_pickle=False) as z:seq=z['sequence']
    selected=json.loads((ROOT/'ssl_results.json').read_text())['selected_task'];bundles={mode:joblib.load(ROOT/(selected+'__'+mode+'.joblib')) for mode in ['all','no_map','no_bp']}
    latent={mode:b['encoder'].embedding(b['normalizer'].transform(seq,mode)) for mode,b in bundles.items()};random=TCN().embedding(bundles['all']['normalizer'].transform(seq))
    recent,delta,reference_available=references(bundles['all'],seq);save_npz(ROOT/'development_latents.npz',**latent,random=random,recent=recent,delta=delta,reference_available=reference_available)
    fold_reports=[];oof_rows=[];fold_manifest={}
    with threadpool_limits(limits=1):
        for seed in [2026,2031]:
            assignments=folds(rows,seed);fold_manifest[str(seed)]=assignments
            for fold in range(5):
                train_idx=np.array([i for i,r in enumerate(rows) if assignments[r['patient_id']]!=fold]);held_idx=np.array([i for i,r in enumerate(rows) if assignments[r['patient_id']]==fold]);training=[rows[i] for i in train_idx];held=[rows[i] for i in held_idx]
                root=ROOT/'fold_models'/f'seed{seed}_fold{fold}';root.mkdir(parents=True,exist_ok=True,mode=0o700);results={};predictions={}
                for name,modality in [('map_only','map_only'),('bp_only','bp_only'),('handcrafted','phys'),('handcrafted_ehr','fused'),('map_logistic','map_only')]:
                    model,p,oof,config=tabular_nested(rows,train_idx,held_idx,modality,name=='map_logistic');cal,cp=calibrate(oof,p,training)
                    results[name]={'raw':basic([r['label'] for r in held],p),'calibrated':basic([r['label'] for r in held],cp),'config':config};predictions[name]=(p,cp);save_private(root/(name+'.joblib'),{'model':model,'calibrator':cal,'train_patients':sorted({r['patient_id'] for r in training}),'held_patients':sorted({r['patient_id'] for r in held})})
                map_encoder=numeric_matrix(training,'map_only');map_features=map_encoder.transform(rows)
                variants={'latent_frozen':latent['all'],'latent_no_map':latent['no_map'],'latent_no_bp':latent['no_bp'],'latent_plus_map':np.concatenate([latent['all'],map_features],axis=1),
                          'latent_delta':delta,'latent_recent':np.concatenate([recent,reference_available[:,None]],axis=1),'random_encoder':random}
                for name,matrix in variants.items():
                    # Combined MAP features need inner-train encoders; helper below handles this separately.
                    if name=='latent_plus_map':model,p,oof,config=combined_map_nested(latent['all'],rows,train_idx,held_idx)
                    else:model,p,oof,config=latent_nested(matrix,rows,train_idx,held_idx)
                    cal,cp=calibrate(oof,p,training);results[name]={'raw':basic([r['label'] for r in held],p),'calibrated':basic([r['label'] for r in held],cp),'config':config};predictions[name]=(p,cp)
                    save_private(root/(name+'.joblib'),{'model':model,'calibrator':cal,'train_patients':sorted({r['patient_id'] for r in training}),'held_patients':sorted({r['patient_id'] for r in held})})
                for name,matrix in [('latent_frozen_ehr',latent['all']),('latent_delta_ehr',delta),('latent_recent_ehr',np.concatenate([recent,reference_available[:,None]],axis=1))]:
                    model,p,oof,config=ehr_latent_nested(matrix,rows,train_idx,held_idx);cal,cp=calibrate(oof,p,training);results[name]={'raw':basic([r['label'] for r in held],p),'calibrated':basic([r['label'] for r in held],cp),'config':config};predictions[name]=(p,cp)
                    save_private(root/(name+'.joblib'),{'model':model,'calibrator':cal,'train_patients':sorted({r['patient_id'] for r in training}),'held_patients':sorted({r['patient_id'] for r in held})})
                for name,ehr,scratch in [('latent_finetuned',False,False),('latent_finetuned_ehr',True,False),('supervised_scratch',False,True)]:
                    model,p,oof,config=finetune_nested(bundles['all'],seq,rows,train_idx,held_idx,ehr,scratch);cal,cp=calibrate(oof,p,training);results[name]={'raw':basic([r['label'] for r in held],p),'calibrated':basic([r['label'] for r in held],cp),'config':config};predictions[name]=(p,cp)
                    save_private(root/(name+'.joblib'),{'model':model,'calibrator':cal,'train_patients':sorted({r['patient_id'] for r in training}),'held_patients':sorted({r['patient_id'] for r in held})})
                fold_reports.append({'seed':seed,'fold':fold,'train_patients':len({r['patient_id'] for r in training}),'held_patients':len({r['patient_id'] for r in held}),'held_positive_patients':len({r['patient_id'] for r in held if r['label']}),'windows':len(held),'positive_windows':sum(r['label'] for r in held),'models':results})
                for j,i in enumerate(held_idx):oof_rows.append({'seed':seed,'fold':fold,'row_index':int(i),'patient_id':rows[i]['patient_id'],'label':rows[i]['label'],'hard':rows[i]['audit']['hard'],'normal80':rows[i]['audit']['map_stratum']=='clearly_normal','raw':{n:float(p[0][j]) for n,p in predictions.items()},'calibrated':{n:float(p[1][j]) for n,p in predictions.items()}})
                private_json(ROOT/'fold_metrics.json',fold_reports);private_json(ROOT/'development_oof.json',oof_rows);private_json(ROOT/'development_folds.json',fold_manifest)
                print({'development_seed':seed,'fold':fold,'models':len(results),'map_ap':round(results['map_only']['raw']['auprc'],4),'latent_ap':round(results['latent_frozen']['raw']['auprc'],4),'test_accessed':False},flush=True)
    return fold_reports

def save_private(path,bundle):joblib.dump(bundle,path);path.chmod(0o600)

def combined_map_nested(latent,rows,train_idx,held_idx):
    # Same fold-local availability/missing/category rules as the EHR helper, using MAP features.
    return combined_nested(latent,rows,train_idx,held_idx,'map_only')

def combined_nested(latent,rows,train_idx,held_idx,modality):
    training=[rows[i] for i in train_idx];held=[rows[i] for i in held_idx];split=folds(training,n=3);results=[]
    for c in [.1,1]:
        oof=np.zeros(len(training))
        for f in range(3):
            a=np.array([split[r['patient_id']]!=f for r in training]);fit=[r for r,yes in zip(training,a) if yes];val=[r for r,yes in zip(training,a) if not yes];encoder=numeric_matrix(fit,modality)
            x=np.concatenate([latent[train_idx][a],encoder.transform(fit)],axis=1);v=np.concatenate([latent[train_idx][~a],encoder.transform(val)],axis=1);oof[~a]=LatentHead(c).fit(x,fit).predict(v)
        results.append(oof)
    best=int(np.argmax([average_precision_score([r['label'] for r in training],p) for p in results]));encoder=numeric_matrix(training,modality)
    x=np.concatenate([latent[train_idx],encoder.transform(training)],axis=1);v=np.concatenate([latent[held_idx],encoder.transform(held)],axis=1);model=LatentHead([.1,1][best]).fit(x,training)
    return (model,encoder),model.predict(v),results[best],{'C':[.1,1][best],'modality':modality}

if __name__=='__main__':
    # CLI artifacts must refer to importable classes rather than __main__.
    from src.ssl_physiology.transfer import run as canonical_run
    canonical_run()
