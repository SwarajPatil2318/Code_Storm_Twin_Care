"""Train-only stable-reference regression, independent of event classifier."""
import hashlib
import numpy as np
from sklearn.linear_model import Ridge
from src.twin.protocol import PRIMARY_SIGNALS,PERSONAL_STATS,require_role


def safe_median(a,axis=0):
    import warnings
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);x=np.nanmedian(a,axis=axis)
    return np.where(np.isfinite(x),x,0)

def prepare_input(data,j):
    # Only own trusted historical reference and EHR; observed current trajectory
    # never enters the expected-stable predictor. This also prevents other missing
    # sensors from silently entering a consumer-subset Shadow prediction.
    n=len(data['ehr']);counts=data['baseline_count'][:,j]
    own=np.c_[data['reference'][:,j],data['scale'][:,j],np.log1p(counts),data['confidence'][:,j]]
    return np.c_[data['ehr'],own]

class ShadowTwin:
    def __init__(self,config,available_signals=None):self.c=config;self.available_signals=available_signals or PRIMARY_SIGNALS;self.models=[];self.normalizers=[];self.population=[];self.width=[];self.fit_patients=[];self.interval_patients=[]
    def fit(self,data,roles):
        ids=data['patient_ids'].astype(str);require_role(ids,roles,{'train'});unique=sorted(set(ids));boundary=int(len(unique)*self.c['shadow']['fit_patient_fraction'])
        ordered=sorted(unique,key=lambda p:hashlib.sha256((p+':shadow').encode()).hexdigest());fit_ids=set(ordered[:boundary]);interval_ids=set(ordered[boundary:])
        fit=np.array([p in fit_ids for p in ids]);held=~fit;near_event=data['labels'].sum(axis=1)>0
        # Future outcomes define TRAIN sample eligibility only, as preregistered.
        visible_z=(data['recent']-data['reference'])/np.maximum(data['scale'],1e-6)
        self.stable_selection=(~near_event)&np.all((~np.isfinite(visible_z))|(abs(visible_z)<self.c['shadow']['stable_proxy_max_z']),axis=1)
        for j,s in enumerate(PRIMARY_SIGNALS):
            x=prepare_input(data,j);y=data['recent'][:,j];valid=np.isfinite(y)&np.isfinite(data['reference'][:,j])&self.stable_selection
            good=valid&fit
            if s not in self.available_signals:
                self.models.append(None);self.normalizers.append(None);self.population.append(np.nan);self.width.append(np.nan);continue
            if good.sum()<20:raise ValueError('Insufficient stable train references')
            med=safe_median(x[good]);missing=~np.isfinite(x);filled=np.where(missing,med,x);scale=np.std(filled[good],axis=0);scale=np.where(scale>1e-6,scale,1)
            inputs=np.c_[(filled-med)/scale,missing.astype(float)]
            # Predict a bounded correction, not raw cross-patient physiology.
            residual=y-data['reference'][:,j];model=Ridge(alpha=self.c['shadow']['ridge_alpha']).fit(inputs[good],residual[good])
            self.models.append(model);self.normalizers.append((med,scale));self.population.append(float(np.median(y[good])))
            predicted=data['reference'][:,j]+np.clip(model.predict(inputs),-data['scale'][:,j],data['scale'][:,j])
            cal=valid&held;error=abs(y[cal]-predicted[cal]);self.width.append(float(np.quantile(error,self.c['shadow']['interval_coverage'],method='higher')) if len(error) else float(np.median(data['scale'][good,j])*2))
        self.population=np.array(self.population);self.width=np.maximum(self.width,.001);self.fit_patients=sorted(fit_ids);self.interval_patients=sorted(interval_ids)
        return self
    def predict(self,data):
        n=len(data['ehr']);expected=np.empty((n,len(PRIMARY_SIGNALS)))
        for j,model in enumerate(self.models):
            if model is None:expected[:,j]=np.nan;continue
            x=prepare_input(data,j);med,scale=self.normalizers[j];missing=~np.isfinite(x);filled=np.where(missing,med,x);inputs=np.c_[(filled-med)/scale,missing.astype(float)]
            correction=np.clip(model.predict(inputs),-data['scale'][:,j],data['scale'][:,j]);expected[:,j]=data['reference'][:,j]+correction
            # Without adequate personal history use explicit population fallback.
            bad=~np.isfinite(expected[:,j]);expected[bad,j]=self.population[j]
        widths=np.broadcast_to(self.width,expected.shape).copy();widths*=1+(1-data['confidence'])
        return {'expected':expected,'lower':expected-widths,'upper':expected+widths,'width':widths,'population':np.broadcast_to(self.population,expected.shape)}
    def trajectory(self,data):
        """Retrospective stable template at this cutoff; not earlier issued forecasts.

        Mean-response ranges are calibrated separately. Bin-level ranges below
        additionally use trusted observed variability and are descriptive.
        """
        forecast=self.predict(data);personal=data['personal'].reshape(len(data['personal']),len(PRIMARY_SIGNALS),len(PERSONAL_STATS))
        day=personal[:,:,PERSONAL_STATS.index('day_reference')];night=personal[:,:,PERSONAL_STATS.index('night_reference')];values=[]
        for relative in range(-162,1,6):
            hour=(data['landmark_hours']+relative)%24;reference=np.where(((hour>6)&(hour<=18))[:,None],day,night);reference=np.where(np.isfinite(reference),reference,forecast['expected'])
            correction=np.where(np.isfinite(data['reference']),forecast['expected']-data['reference'],0);values.append(reference+correction)
        expected=np.stack(values,axis=1);width=np.maximum(forecast['width'],1.645*data['scale'])[:,None,:]
        return {'expected':expected,'lower':expected-width,'upper':expected+width,'relative_hours':np.arange(-162,1,6),'range_scope':'Descriptive bin ranges; calibrated mean-response intervals evaluated separately.'}
    def quality(self,data,mask):
        forecast=self.predict(data);y=data['recent'];result={}
        for j,s in enumerate(PRIMARY_SIGNALS):
            good=mask&np.isfinite(y[:,j]);pred=forecast['expected'][:,j];pop=forecast['population'][:,j]
            result[s]={'rows':int(good.sum()),'shadow_mae':float(np.mean(abs(pred[good]-y[good,j]))) if good.any() else None,'population_mae':float(np.mean(abs(pop[good]-y[good,j]))) if good.any() else None,
              'interval_coverage':float(np.mean((y[good,j]>=forecast['lower'][good,j])&(y[good,j]<=forecast['upper'][good,j]))) if good.any() else None,'mean_width':float(np.mean(2*forecast['width'][good,j])) if good.any() else None}
        return result
