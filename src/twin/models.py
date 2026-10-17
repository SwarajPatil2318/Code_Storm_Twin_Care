"""Discrete conditional hazards with train-only transformations."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from src.twin.protocol import require_role,validate_features


def risk_sets(labels):
    y=np.asarray(labels);return np.c_[np.ones(len(y),bool),y[:,0]==0,(y[:,0]+y[:,1])==0]

def cumulative(hazard):
    h=np.clip(np.asarray(hazard,float),0,1);return 1-np.cumprod(1-h,axis=1)

def interval_probabilities(hazard):
    h=np.clip(np.asarray(hazard,float),0,1);survival=np.c_[np.ones(len(h)),np.cumprod(1-h,axis=1)[:,:-1]]
    return np.c_[h*survival,np.prod(1-h,axis=1)]

class Normalizer:
    def fit(self,x,ids,roles):
        require_role(ids,roles,{'train'})
        import warnings
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(x,axis=0)
        self.median=np.where(np.isfinite(self.median),self.median,0);filled=np.where(np.isfinite(x),x,self.median)
        self.scale=np.std(filled,axis=0);self.scale=np.where(self.scale>1e-6,self.scale,1);self.fit_patients=sorted(set(ids.tolist()));return self
    def transform(self,x):
        missing=~np.isfinite(x);z=np.where(missing,self.median,x);return np.c_[np.clip((z-self.median)/self.scale,-10,10),missing.astype(float)].astype(np.float32)

class ConstantHead:
    def __init__(self,probability):self.p=float(probability)
    def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))

class HazardModel:
    def __init__(self,spec,columns,names):self.spec=spec;self.columns=columns;self.names=[names[i] for i in columns];validate_features(self.names)
    def fit(self,x,y,ids,roles,multiplicity=None):
        require_role(ids,roles,{'train'});values=x[:,self.columns];self.normalizer=Normalizer().fit(values,ids,roles)
        data=self.normalizer.transform(values);at_risk=risk_sets(y);self.heads=[];weights=np.ones(len(y)) if multiplicity is None else np.array([multiplicity.get(str(p),0) for p in ids],float)
        for horizon in range(3):
            mask=at_risk[:,horizon]&(weights>0);target=y[mask,horizon];w=weights[mask];rate=float(np.average(target,weights=w))
            if self.spec['kind']=='prevalence' or len(set(target))<2:model=ConstantHead(rate)
            elif self.spec['kind']=='logistic':
                model=LogisticRegression(C=self.spec['C'],max_iter=300,class_weight='balanced',random_state=2026).fit(data[mask],target,sample_weight=w)
            elif self.spec['kind']=='hist_gradient_boosting':
                model=HistGradientBoostingClassifier(max_leaf_nodes=self.spec['leaves'],max_iter=self.spec['iterations'],learning_rate=self.spec['learning_rate'],l2_regularization=1,early_stopping=False,random_state=2026)
                pos=float(np.sum(w*target));neg=float(np.sum(w*(1-target)));balanced=w*np.where(target==1,(pos+neg)/(2*max(pos,1)),(pos+neg)/(2*max(neg,1)))
                model.fit(data[mask],target,sample_weight=balanced)
            else:raise ValueError('Unknown head family')
            self.heads.append(model)
        self.fit_patients=sorted(set(ids[weights>0].tolist()));return self
    def predict_hazards(self,x):
        data=self.normalizer.transform(x[:,self.columns]);return np.stack([h.predict_proba(data)[:,1] for h in self.heads],axis=1)
