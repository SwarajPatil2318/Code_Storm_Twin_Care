"""Compact causal TCN with analytically differentiable SSL and transfer heads."""
import copy
import numpy as np

def allowed_channels(mode):
    return [0,1,2,3,4,5] if mode=='all' else [1,2,3,4,5] if mode=='no_map' else [3,4,5] if mode=='no_bp' else (_ for _ in ()).throw(ValueError('Unknown signal ablation'))

class Normalizer:
    def fit(self,seq,patients,roles):
        if any(roles.get(p)!='train' for p in patients):raise ValueError('Normalizer received held-out subject')
        v=np.asarray(seq)[:,:,:6];m=np.asarray(seq)[:,:,6:12]>0
        self.center=[];self.scale=[]
        for j in range(6):
            x=v[:,:,j][m[:,:,j]&np.isfinite(v[:,:,j])]
            center=float(np.median(x)) if len(x) else 0;scale=float(np.subtract(*np.quantile(x,[.75,.25]))) if len(x) else 1
            self.center.append(center);self.scale.append(scale if scale>1e-6 else max(float(np.std(x)),1) if len(x) else 1)
        self.center=np.array(self.center);self.scale=np.array(self.scale);self.fit_patients=sorted(set(patients));return self
    def transform(self,seq,mode='all'):
        s=np.asarray(seq,float);m=(s[:,:,6:12]>0)&np.isfinite(s[:,:,:6]);v=np.where(m,np.clip((s[:,:,:6]-self.center)/self.scale,-8,8),0)
        age=s[:,:,12:18].copy();allowed=allowed_channels(mode)
        excluded=[j for j in range(6) if j not in allowed];v[:,:,excluded]=0;m[:,:,excluded]=False;age[:,:,excluded]=1
        return np.concatenate([v,m.astype(float),age],axis=2)
    def targets(self,values):return (values-self.center)/self.scale

def mask_input(x,rng,probability=.25):
    """Mask observed values and recompute age from surviving past samples only."""
    z=x.copy();observed=z[:,:,6:12]>0
    chosen=(rng.random(observed.shape)<probability)&observed
    # Short contiguous masks make reconstruction harder than isolated channel interpolation.
    chosen[:,1:]|=chosen[:,:-1]&observed[:,1:]
    z[:,:,:6][chosen]=0;z[:,:,6:12][chosen]=0
    for i in range(len(z)):
        for j in range(6):
            last=None
            for t in range(z.shape[1]):
                if z[i,t,6+j]>0:last=t
                z[i,t,12+j]=1 if last is None else min(1,(t-last)*5/360)
    return z,chosen

class TCN:
    def __init__(self,mode='all',channels=8,seed=2026):
        self.mode=mode;self.channels=channels;self.dilations=[1,2,4];rng=np.random.default_rng(seed);self.params={}
        for layer,n in enumerate([18,channels,channels]):
            self.params[f'w{layer}']=rng.normal(0,np.sqrt(1/(3*n)),(3,n,channels));self.params[f'b{layer}']=np.zeros(channels)
        self.params['out']=rng.normal(0,.05,(channels,6));self.params['out_bias']=np.zeros(6)
    def forward(self,x,cache=False):
        h=x;states=[]
        for layer,dilation in enumerate(self.dilations):
            parts=[];z=np.zeros((*h.shape[:2],self.channels))+self.params[f'b{layer}']
            for k in range(3):
                shift=k*dilation;part=np.zeros_like(h)
                if shift<h.shape[1]:part[:,shift:]=h[:,:h.shape[1]-shift] if shift else h
                parts.append(part);z+=part@self.params[f'w{layer}'][k]
            h=np.tanh(z);states.append((parts,h,dilation))
        return (h,states) if cache else h
    def embedding(self,x):
        h=self.forward(x);return np.concatenate([np.mean(h,axis=1),h[:,-1]],axis=1)
    def backward(self,states,dh):
        gradients={}
        for layer in reversed(range(3)):
            parts,h,dilation=states[layer];dz=dh*(1-h*h);gradients[f'b{layer}']=np.sum(dz,axis=(0,1));gw=[];previous=np.zeros_like(parts[0]);t=h.shape[1]
            for k,part in enumerate(parts):
                gw.append(np.einsum('ntc,ntf->cf',part,dz,optimize=True));shift=k*dilation
                if shift<t:previous[:,:t-shift]+=dz[:,shift:]@self.params[f'w{layer}'][k].T
            gradients[f'w{layer}']=np.stack(gw);dh=previous
        return gradients
    def loss_gradient(self,x,target,mask,weights=None,l2=0):
        h,states=self.forward(x,True);prediction=h@self.params['out']+self.params['out_bias']
        m=mask.astype(float)
        if weights is not None:m*=weights[:,None,None]
        den=max(float(sum(m.ravel())),1);error=np.where(mask,prediction-np.nan_to_num(target,nan=0),0)
        loss=float(sum((error*error*m).ravel())/den);dp=2*error*m/den
        gradients=self.backward(states,dp@self.params['out'].T)
        gradients['out']=np.einsum('ntc,ntf->cf',h,dp,optimize=True);gradients['out_bias']=np.sum(dp,axis=(0,1))
        for k,v in self.params.items():
            if k.startswith('w') or k=='out':loss+=l2*np.sum(v*v)/2;gradients[k]+=l2*v
        return loss,gradients

class Adam:
    def __init__(self,params,lr):self.first={k:np.zeros_like(v) for k,v in params.items()};self.second=copy.deepcopy(self.first);self.step=0;self.lr=lr
    def update(self,params,gradient,keys=None):
        self.step+=1;norm=np.sqrt(sum(np.sum(g*g) for g in gradient.values()));factor=min(1,5/max(norm,1e-9))
        for k in keys if keys is not None else params:
            g=gradient[k]*factor;self.first[k]=.9*self.first[k]+.1*g;self.second[k]=.999*self.second[k]+.001*g*g
            params[k]-=self.lr*(self.first[k]/(1-.9**self.step))/(np.sqrt(self.second[k]/(1-.999**self.step))+1e-8)
