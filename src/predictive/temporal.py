"""Small NumPy causal TCN: learned kernels and exact reverse-mode gradients."""
import numpy as np
from scipy.special import expit
from src.predictive.models import Encoder
from src.predictive.data import guard,weights


class CompactTCN:
    def __init__(self,fusion=False,l2=.01,epochs=150):
        self.fusion=fusion;self.l2=l2;self.epochs=epochs;self.modality='fused' if fusion else 'phys'
        self.config={'kind':'compact_tcn','fusion':fusion,'l2':l2,'epochs':epochs,'channels':4,'dilations':[1,4,16]};self.converged=True

    def initialize(self,channels=21,ehr_dim=0,prevalence=.05):
        rng=np.random.default_rng(2026);self.params={}
        for layer,c in enumerate([channels,4,4]):
            self.params['w'+str(layer)]=rng.normal(0,.08,(3,c,4));self.params['b'+str(layer)]=np.zeros(4)
        if ehr_dim:
            self.params['ehr_w']=rng.normal(0,.05,(ehr_dim,4));self.params['ehr_b']=np.zeros(4)
        self.params['head']=rng.normal(0,.05,8+(4 if ehr_dim else 0));self.params['bias']=np.array(np.log(prevalence/(1-prevalence)))

    def forward(self,x,ehr=None,cache=False):
        h=x;states=[]
        for layer,dilation in enumerate([1,4,16]):
            shifted=[];z=np.zeros((*h.shape[:2],4))+self.params['b'+str(layer)]
            for k in range(3):
                shift=k*dilation;part=np.zeros_like(h)
                if shift<h.shape[1]:part[:,shift:]=h[:,:h.shape[1]-shift] if shift else h
                shifted.append(part);z+=part@self.params['w'+str(layer)][k]
            h=np.tanh(z);states.append((shifted,h,dilation))
        pooled=np.concatenate([np.mean(h,axis=1),h[:,-1]],axis=1);ehr_h=None
        if ehr is not None and 'ehr_w' in self.params:
            ehr_h=np.tanh(ehr@self.params['ehr_w']+self.params['ehr_b']);pooled=np.concatenate([pooled,ehr_h],axis=1)
        logit=pooled@self.params['head']+self.params['bias']
        return (logit,(states,pooled,ehr_h)) if cache else logit

    def loss_gradient(self,x,y,w,ehr=None):
        logit,(states,pooled,ehr_h)=self.forward(x,ehr,True);normalized=w/sum(w)
        loss=float(sum(normalized*(np.logaddexp(0,logit)-y*logit)))
        gradients={};dlogit=(expit(logit)-y)*normalized
        gradients['head']=pooled.T@dlogit;gradients['bias']=np.array(sum(dlogit));dpool=dlogit[:,None]*self.params['head']
        if ehr_h is not None:
            dz=dpool[:,8:]*(1-ehr_h**2);gradients['ehr_w']=ehr.T@dz;gradients['ehr_b']=sum(dz)
        t=x.shape[1];dh=np.broadcast_to(dpool[:,:4,None],(len(x),4,t)).transpose(0,2,1).copy()/t
        dh[:,-1]+=dpool[:,4:8]
        for layer in reversed(range(3)):
            shifted,h,dilation=states[layer];dz=dh*(1-h**2)
            gradients['b'+str(layer)]=np.sum(dz,axis=(0,1));gw=[];previous=np.zeros_like(shifted[0])
            for k,part in enumerate(shifted):
                gw.append(np.einsum('ntc,ntf->cf',part,dz,optimize=True));shift=k*dilation
                if shift<t:
                    update=dz[:,shift:]@self.params['w'+str(layer)][k].T
                    previous[:,:t-shift]+=update
            gradients['w'+str(layer)]=np.stack(gw);dh=previous
        for key,value in self.params.items():
            if key.startswith('w') or key in ['ehr_w','head']:
                loss+=self.l2*float(sum((value**2).ravel()))/2;gradients[key]+=self.l2*value
        return loss,gradients

    def normalize(self,rows):
        seq=np.stack([r['sequence'] for r in rows]);values=seq[:,:,:7]
        standardized=np.where(np.isfinite(values),(values-self.means)/self.scales,0.)
        return np.concatenate([np.clip(standardized,-8,8),seq[:,:,7:]],axis=2)

    def fit(self,rows,assignments,multiplicity=None):
        guard(rows,assignments,{'train'});active=[r for r in rows if multiplicity is None or multiplicity.get(r['patient_id'],0)>0]
        reference=active if multiplicity is None else [r for r in active for _ in range(multiplicity[r['patient_id']])]
        raw=np.stack([r['sequence'][:,:7] for r in reference])
        self.means=np.array([np.mean(raw[:,:,j][np.isfinite(raw[:,:,j])]) if np.any(np.isfinite(raw[:,:,j])) else 0 for j in range(7)])
        self.scales=np.array([np.std(raw[:,:,j][np.isfinite(raw[:,:,j])]) if np.any(np.isfinite(raw[:,:,j])) else 1 for j in range(7)])
        self.scales[self.scales<1e-8]=1
        self.encoder=Encoder().fit(reference,assignments,'ehr') if self.fusion else None
        x=self.normalize(rows);ehr=self.encoder.transform(rows) if self.encoder else None
        y=np.array([r['label'] for r in rows]);w=weights(rows)
        if multiplicity is not None:w*=np.array([multiplicity.get(r['patient_id'],0) for r in rows])
        # Defensible train-only class weighting; calibrate later on separate validation patients.
        a=sum(w[y==0]);b=sum(w[y==1]);w*=np.where(y==1,(a+b)/(2*b),(a+b)/(2*a))
        self.initialize(ehr_dim=ehr.shape[1] if ehr is not None else 0,prevalence=.5)
        first={k:np.zeros_like(v) for k,v in self.params.items()};second={k:np.zeros_like(v) for k,v in self.params.items()};self.loss_history=[]
        for step in range(1,self.epochs+1):
            loss,gradient=self.loss_gradient(x,y,w,ehr);norm=np.sqrt(sum(np.sum(g**2) for g in gradient.values()))
            self.loss_history.append(loss)
            if not np.isfinite(loss):raise ValueError('Nonfinite TCN training loss')
            for key,g in gradient.items():
                g*=min(1.,5/max(norm,1e-10));first[key]=.9*first[key]+.1*g;second[key]=.999*second[key]+.001*g*g
                self.params[key]-=.005*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        self.fit_patients=sorted({r['patient_id'] for r in active});return self

    def predict(self,rows):
        return expit(self.forward(self.normalize(rows),self.encoder.transform(rows) if self.encoder else None))
    @property
    def complexity(self):return 4
