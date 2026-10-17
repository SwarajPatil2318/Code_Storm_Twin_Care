"""Bounded multi-output candidate families with train-participant-only preprocessing."""
import pickle,random
from pathlib import Path
import numpy as np
import xgboost as xgb
import torch
from torch import nn
from sklearn.linear_model import Ridge
from .protocol import config


def subject_weights(pid):
    _,inverse,counts=np.unique(pid,return_inverse=True,return_counts=True)
    w=1/counts[inverse];return (w/w.mean()).astype('float32')


class TrainPreprocessor:
    def fit(self,x,pid,allowed):
        if not set(pid).issubset(set(allowed)):raise PermissionError('Non-training participant in preprocessing fit')
        self.training_participants=sorted(set(pid))
        self.median=np.nanmedian(x,axis=0);self.median=np.where(np.isfinite(self.median),self.median,0)
        xx=np.where(np.isfinite(x),x,self.median)
        w=subject_weights(pid);self.mean=np.average(xx,axis=0,weights=w)
        self.scale=np.sqrt(np.average((xx-self.mean)**2,axis=0,weights=w));self.scale=np.maximum(self.scale,1e-6)
        return self
    def transform(self,x):return ((np.where(np.isfinite(x),x,self.median)-self.mean)/self.scale).astype('float32')


def features(data,variant):
    x=data['x']
    if variant=='personal':x=np.column_stack([x,data['personal']])
    elif variant=='meal':x=np.column_stack([x,data['meal']])
    elif variant=='static':x=np.column_stack([x,data['static']])
    return x.astype('float32')


def train_mask(data,variant):
    if variant in ['personal','warm_population']:return data['warm']
    if variant=='meal':return data['meal_eligible']
    if variant=='static':return np.isfinite(data['static']).all(1)
    return np.ones(len(data['y']),bool)


def huber_objective(pred,dmatrix):
    c=config()['gradient_boosting'];y=dmatrix.get_label().reshape(pred.shape)
    error=pred-y;delta=c['huber_delta_mg_dl'];root=np.sqrt(1+(error/delta)**2)
    weights=dmatrix.get_weight()
    grad=error/root;hess=1/(root**3)
    if len(weights):grad*=weights[:,None];hess*=weights[:,None]
    return grad.astype('float32'),hess.astype('float32')


class CompactGRU(nn.Module):
    def __init__(self,extra=0):
        super().__init__();size=config()['gru']['hidden_size']
        self.gru=nn.GRU(1,size,batch_first=True)
        self.head=nn.Sequential(nn.Linear(size+extra,32),nn.ReLU(),nn.Linear(32,4))
    def forward(self,sequence,extra):
        _,state=self.gru(sequence)
        return self.head(torch.cat([state[-1],extra],dim=1))


class ForecastModel:
    def __init__(self,family,variant='population'):
        self.family=family;self.variant=variant
    def fit(self,data,allowed,seed=2026):
        c=config();mask=train_mask(data,self.variant)
        selected={k:v[mask] for k,v in data.items()};pid=selected['pid']
        if not len(pid):raise ValueError('No eligible training landmarks')
        if not set(pid).issubset(allowed):raise PermissionError('Training participant isolation violation')
        self.training_participants=sorted(set(pid));self.train_windows=len(pid)
        current=selected['sequence'][:,-1]
        residual=selected['y']-current[:,None];w=subject_weights(pid)
        self.preprocessor=TrainPreprocessor().fit(features(selected,self.variant),pid,allowed)
        x=self.preprocessor.transform(features(selected,self.variant))
        if self.family=='mean':
            self.means=np.zeros((24,4),'float32');hour=((selected['time']//60_000_000_000//60)%24).astype(int)
            globalmean=np.average(selected['y'],axis=0,weights=w)
            for h in range(24):
                use=hour==h;self.means[h]=np.average(selected['y'][use],axis=0,weights=w[use]) if use.any() else globalmean
        elif self.family=='ridge':
            self.estimator=Ridge(alpha=c['ridge_alpha']).fit(x,residual,sample_weight=w)
        elif self.family=='boosting':
            p=c['gradient_boosting'];params={'tree_method':'hist','multi_strategy':'multi_output_tree','num_target':4,
                'objective':'reg:squarederror','base_score':0,'max_depth':p['max_depth'],'eta':p['learning_rate'],
                'max_bin':p['max_bin'],'nthread':p['nthread'],'reg_lambda':p['reg_lambda'],
                'min_child_weight':p['min_child_weight'],'seed':seed,'verbosity':0}
            self.estimator=xgb.train(params,xgb.DMatrix(x,label=residual,weight=w),num_boost_round=p['rounds'],obj=huber_objective)
        elif self.family=='gru':
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            torch.set_num_threads(c['gru']['threads']);torch.use_deterministic_algorithms(True)
            # Sequence scaling from training inputs only, subject-balanced, shared by all time steps.
            seq=selected['sequence'].astype('float32')
            self.seq_mean=float(np.average(seq.mean(1),weights=w))
            self.seq_scale=max(float(np.sqrt(np.average(((seq-self.seq_mean)**2).mean(1),weights=w))),1)
            self.target_scale=self.seq_scale
            seq=((seq-self.seq_mean)/self.seq_scale)[:,:,None]
            # Time-of-day always present; additional personal/meal/static branch only when requested.
            extra=x[:,19:] if self.variant=='population' else np.column_stack([x[:,19:21],x[:,21:]])
            self.estimator=CompactGRU(extra.shape[1]);opt=torch.optim.AdamW(self.estimator.parameters(),lr=c['gru']['learning_rate'],weight_decay=c['gru']['weight_decay'])
            tensors=[torch.from_numpy(a.astype('float32')) for a in [seq,extra,residual/self.target_scale,w]]
            rng=np.random.default_rng(seed);self.loss_history=[]
            delta=c['gru']['huber_delta_mg_dl']/self.target_scale
            for epoch in range(c['gru']['epochs']):
                order=rng.permutation(len(pid));total=0
                for start in range(0,len(pid),c['gru']['batch_size']):
                    ids=order[start:start+c['gru']['batch_size']]
                    out=self.estimator(tensors[0][ids],tensors[1][ids]);error=out-tensors[2][ids]
                    robust=torch.nn.functional.huber_loss(out,tensors[2][ids],reduction='none',delta=delta).mean(1)
                    loss=(robust*tensors[3][ids]).mean();opt.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.estimator.parameters(),1);opt.step();total+=float(loss.detach())*len(ids)
                self.loss_history.append(total/len(pid))
        else:raise ValueError('Unknown candidate family')
        return self
    def predict(self,data):
        current=data['sequence'][:,-1]
        if self.family=='mean':
            hour=((data['time']//60_000_000_000//60)%24).astype(int);return self.means[hour]
        x=self.preprocessor.transform(features(data,self.variant))
        if self.family=='ridge':residual=self.estimator.predict(x)
        elif self.family=='boosting':residual=self.estimator.predict(xgb.DMatrix(x))
        else:
            seq=((data['sequence']-self.seq_mean)/self.seq_scale).astype('float32')[:,:,None]
            extra=x[:,19:] if self.variant=='population' else np.column_stack([x[:,19:21],x[:,21:]])
            self.estimator.eval();parts=[]
            with torch.no_grad():
                for start in range(0,len(seq),1024):parts.append(self.estimator(torch.from_numpy(seq[start:start+1024]),torch.from_numpy(extra[start:start+1024])).numpy())
            residual=np.concatenate(parts)*self.target_scale
        return (current[:,None]+residual).astype('float32')
    def save(self,path):
        path=Path(path);path.parent.mkdir(exist_ok=True,parents=True,mode=0o700)
        with path.open('wb') as f:pickle.dump(self,f,protocol=5)
        path.chmod(0o600)
    @staticmethod
    def load(path):
        # Only locally generated, hash-verified artifacts are eligible; never untrusted pickle.
        with Path(path).open('rb') as f:return pickle.load(f)


def baseline(data,name):
    current=data['sequence'][:,-1]
    if name=='persistence':return np.repeat(current[:,None],4,axis=1)
    if name=='trend':
        slope=(data['sequence'][:,-1]-data['sequence'][:,-3])/30
        return current[:,None]+slope[:,None]*np.array([30,60,90,120])
    raise ValueError('Unknown baseline')
