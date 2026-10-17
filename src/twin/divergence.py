"""Covariance-aware physiology divergence, never renamed event probability."""
import numpy as np
from src.twin.protocol import PRIMARY_SIGNALS,SHADOW_FEATURES,TDI_FEATURES,require_role

class Divergence:
    def __init__(self,config):self.c=config
    def fit(self,data,forecast,roles,stable):
        require_role(data['patient_ids'],roles,{'train'});z=(data['recent']-forecast['expected'])/np.maximum(data['scale'],1e-6)
        good=stable.copy();center=[]
        for j in range(z.shape[1]):
            x=z[good,j];x=x[np.isfinite(x)];center.append(float(np.median(x)) if len(x) else 0)
        self.center=np.array(center);filled=np.where(np.isfinite(z[good]),np.clip(z[good]-self.center,-6,6),0)
        covariance=np.cov(filled,rowvar=False);diag=np.diag(np.maximum(np.diag(covariance),.25));shrink=self.c['tdi']['covariance_shrinkage'];self.covariance=(1-shrink)*covariance+shrink*diag+np.eye(z.shape[1])*.05
        distance=self.distance(z[good]);finite=distance[np.isfinite(distance)];self.distance_scale=max(float(np.quantile(finite,self.c['tdi']['stable_distance_quantile'])),.5)
        self.fit_patients=sorted(set(data['patient_ids'][good].tolist()));return self
    def distance(self,z):
        seen=np.isfinite(z);out=np.full(len(z),np.nan);cache={}
        # Only 2^10 submatrices; normalized by available dimension, not raw size.
        patterns=np.sum(seen*(2**np.arange(z.shape[1])),axis=1)
        for pattern in np.unique(patterns):
            indices=np.flatnonzero(patterns==pattern);available=seen[indices[0]]
            if not available.any():continue
            covariance=self.covariance[np.ix_(available,available)];inverse=np.linalg.pinv(covariance)
            values=np.clip(z[indices][:,available]-self.center[available],-8,8);out[indices]=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',values,inverse,values)/available.sum(),0))
        return out
    def transform(self,data,forecast):
        residual=data['recent']-forecast['expected'];z=residual/np.maximum(data['scale'],1e-6);distance=self.distance(z)
        tdi=1-np.exp(-np.log(2)*(distance/self.distance_scale)**2);available=np.isfinite(z);coverage=available.mean(axis=1);confidence=np.nanmean(data['confidence'],axis=1)*coverage
        systems=(abs(z)>self.c['tdi']['deviation_z']).sum(axis=1);accel=(data['recent']-data['prior_recent'])/np.maximum(data['scale'],1e-6)
        components=np.where(available,abs(z)/np.maximum(np.nansum(abs(z),axis=1,keepdims=True),1e-6),np.nan)
        trend=np.zeros(len(z));ids=data['patient_ids'];hours=data['landmark_hours']
        for i in range(1,len(z)):
            if ids[i]==ids[i-1] and hours[i]>hours[i-1] and np.isfinite(tdi[i]) and np.isfinite(tdi[i-1]):trend[i]=(tdi[i]-tdi[i-1])*24/(hours[i]-hours[i-1])
        shadow=[]
        for j in range(z.shape[1]):shadow.extend([forecast['expected'][:,j],data['scale'][:,j],residual[:,j],z[:,j],data['persistence'][:,j],accel[:,j],data['recent_count'][:,j],forecast['width'][:,j]])
        sf=np.stack(shadow,axis=1);tf=np.c_[tdi,systems,coverage,confidence,trend,components]
        return {'shadow_features':sf.astype(np.float32),'tdi_features':tf.astype(np.float32),'tdi':tdi,'components':components,'z':z,'residual':residual,'expected':forecast['expected'],'lower':forecast['lower'],'upper':forecast['upper'],'confidence':confidence,'coverage':coverage}
