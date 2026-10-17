"""Compact NumPy causal TCN, multi-horizon survival loss and fusion side branch.

Learned kernels and reverse-mode gradients; minibatches permit all TRAIN windows
without materializing several full-size activation tensors.
"""
import numpy as np
from scipy.special import expit
from src.twin.models import Normalizer,risk_sets
from src.twin.protocol import require_role,validate_features

class TemporalHazard:
    def __init__(self,spec,columns,names):self.spec=spec;self.columns=columns;self.names=[names[i] for i in columns];validate_features(self.names);self.params={}
    def initialize(self,input_dim,side_dim):
        rng=np.random.default_rng(2026);width=self.spec['channels']
        for l,dim in enumerate([input_dim,width,width]):self.params['w'+str(l)]=rng.normal(0,np.sqrt(1/(3*dim)),(3,dim,width));self.params['b'+str(l)]=np.zeros(width)
        self.params['side_w']=rng.normal(0,.03,(side_dim,width));self.params['side_b']=np.zeros(width);self.params['head']=rng.normal(0,.03,(width*3,3));self.params['bias']=np.zeros(3)
    def forward(self,x,side,cache=False):
        h=x;states=[]
        for layer,dilation in enumerate(self.spec['dilations']):
            parts=[];z=np.broadcast_to(self.params['b'+str(layer)],(*h.shape[:2],self.spec['channels'])).copy()
            for k in range(3):
                part=np.zeros_like(h);shift=k*dilation
                if shift<h.shape[1]:part[:,shift:]=h[:,:h.shape[1]-shift] if shift else h
                parts.append(part);z+=part@self.params['w'+str(layer)][k]
            h=np.tanh(z);states.append((parts,h,dilation))
        embedding=np.tanh(side@self.params['side_w']+self.params['side_b']);pooled=np.c_[h.mean(axis=1),h[:,-1],embedding];logits=pooled@self.params['head']+self.params['bias']
        return (logits,(states,pooled,embedding)) if cache else logits
    def loss_gradient(self,x,side,y,weights):
        logits,(states,pooled,embedding)=self.forward(x,side,True);mass=max(float(weights.sum()),1);error=(expit(logits)-y)*weights/mass
        loss=float(np.sum(weights*(np.logaddexp(0,logits)-y*logits))/mass);g={'head':pooled.T@error,'bias':error.sum(axis=0)};dp=error@self.params['head'].T;width=self.spec['channels']
        dz=dp[:,2*width:]*(1-embedding**2);g['side_w']=side.T@dz;g['side_b']=dz.sum(axis=0)
        dh=np.broadcast_to(dp[:,:width,None],(len(x),width,x.shape[1])).transpose(0,2,1).copy()/x.shape[1];dh[:,-1]+=dp[:,width:2*width]
        for layer in reversed(range(len(states))):
            parts,h,dilation=states[layer];dz=dh*(1-h*h);g['b'+str(layer)]=dz.sum(axis=(0,1));previous=np.zeros_like(parts[0]);kernels=[]
            for k,part in enumerate(parts):
                kernels.append(np.einsum('ntc,ntf->cf',part,dz,optimize=True));shift=k*dilation
                if shift<x.shape[1]:previous[:,:x.shape[1]-shift]+=dz[:,shift:]@self.params['w'+str(layer)][k].T
            g['w'+str(layer)]=np.stack(kernels);dh=previous
        for key,value in self.params.items():
            if key.endswith('_w') or key.startswith('w') or key=='head':loss+=self.spec['l2']*float(np.sum(value*value))/2;g[key]+=self.spec['l2']*value
        return loss,g
    def normalize_sequence(self,seq):
        z=np.asarray(seq);seen=(z[:,:,:,1]>0)&np.isfinite(z[:,:,:,0]);values=np.where(seen,np.clip((z[:,:,:,0]-self.center)/self.scale,-8,8),0)
        return np.concatenate([values,seen.astype(float),z[:,:,:,2],z[:,:,:,3]],axis=2)
    def fit(self,x,sequence,y,ids,roles,multiplicity=None):
        require_role(ids,roles,{'train'});self.normalizer=Normalizer().fit(x[:,self.columns],ids,roles);side=self.normalizer.transform(x[:,self.columns])
        self.center=[];self.scale=[]
        for j in range(sequence.shape[2]):
            values=np.asarray(sequence[:,: ,j,0]);v=values[np.isfinite(values)];self.center.append(float(np.median(v)) if len(v) else 0);self.scale.append(max(float(np.std(v)),1) if len(v) else 1)
        self.center=np.array(self.center);self.scale=np.array(self.scale);self.initialize(sequence.shape[2]*4,side.shape[1]);mask=risk_sets(y);sample_weight=np.ones(len(y)) if multiplicity is None else np.array([multiplicity.get(str(p),0) for p in ids],float);weights=mask.astype(float)*sample_weight[:,None]
        for j in range(3):
            pos=float(np.sum(y[:,j]*weights[:,j]));neg=float(np.sum((1-y[:,j])*weights[:,j]));weights[:,j]*=np.where(y[:,j],(pos+neg)/(2*max(pos,1)),(pos+neg)/(2*max(neg,1)))
        first={k:np.zeros_like(v) for k,v in self.params.items()};second={k:np.zeros_like(v) for k,v in self.params.items()};rng=np.random.default_rng(2026);step=0;self.losses=[]
        for epoch in range(self.spec['epochs']):
            ordering=rng.permutation(len(y));total=0;size=self.spec['batch_size']
            for start in range(0,len(y),size):
                chosen=ordering[start:start+size];seq=self.normalize_sequence(sequence[chosen]);loss,gradient=self.loss_gradient(seq,side[chosen],y[chosen],weights[chosen]);step+=1;total+=loss*len(chosen)
                if not np.isfinite(loss):raise ValueError('Nonfinite temporal loss')
                norm=np.sqrt(sum(np.sum(g*g) for g in gradient.values()))
                for k,g in gradient.items():
                    g*=min(1,5/max(norm,1e-9));first[k]=.9*first[k]+.1*g;second[k]=.999*second[k]+.001*g*g;self.params[k]-=self.spec['learning_rate']*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
            self.losses.append(total/len(y));print({'phase6_tcn_epoch':epoch+1,'epochs':self.spec['epochs'],'loss':round(self.losses[-1],5)},flush=True)
        self.fit_patients=sorted(set(ids[sample_weight>0].tolist()));return self
    def predict_hazards(self,x,sequence):
        side=self.normalizer.transform(x[:,self.columns]);out=[]
        for start in range(0,len(x),2048):out.append(expit(self.forward(self.normalize_sequence(sequence[start:start+2048]),side[start:start+2048])))
        return np.concatenate(out)
