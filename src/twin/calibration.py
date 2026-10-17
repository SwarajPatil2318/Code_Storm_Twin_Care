"""Validation-only conditional-hazard calibration and operating thresholds."""
import hashlib
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from src.twin.protocol import require_role
from src.twin.models import cumulative,risk_sets

class HazardCalibrator:
    def __init__(self,method):self.method=method;self.models=[]
    def fit(self,hazards,y,ids,roles):
        require_role(ids,roles,{'validation'});at_risk=risk_sets(y);self.fit_patients=sorted(set(ids.tolist()))
        for j in range(3):
            mask=at_risk[:,j];p=np.clip(hazards[mask,j],1e-6,1-1e-6);target=y[mask,j]
            if self.method=='none':model=None
            elif self.method=='isotonic':model=IsotonicRegression(out_of_bounds='clip').fit(p,target)
            elif self.method=='platt':
                z=logit(p)
                def objective(theta):
                    a=theta[0]*z+theta[1];residual=expit(a)-target
                    return float(np.mean(np.logaddexp(0,a)-target*a)+.00001*(theta[0]-1)**2),np.array([np.mean(residual*z)+.00002*(theta[0]-1),np.mean(residual)])
                fit=minimize(objective,[1,0],jac=True,bounds=[(.001,10),(-20,20)],method='L-BFGS-B')
                if not fit.success:raise ValueError('Validation hazard calibration failed')
                model=fit.x
            else:raise ValueError('Unknown calibration')
            self.models.append(model)
        return self
    def transform(self,hazards):
        p=np.clip(hazards,1e-6,1-1e-6);out=[]
        for j,model in enumerate(self.models):
            out.append(p[:,j] if self.method=='none' else model.predict(p[:,j]) if self.method=='isotonic' else expit(model[0]*logit(p[:,j])+model[1]))
        return np.stack(out,axis=1)

def select_calibration(hazards,y,ids,roles,config):
    require_role(ids,roles,{'validation'});fold=np.array([int(hashlib.sha256((str(p)+':calibration').encode()).hexdigest()[:8],16)%config['calibration_folds'] for p in ids]);trials={};truth=np.cumsum(y,axis=1)
    for method in config['calibration_candidates']:
        oof=np.zeros_like(hazards)
        for i in range(config['calibration_folds']):
            fit=fold!=i;held=~fit;cal=HazardCalibrator(method).fit(hazards[fit],y[fit],ids[fit],roles);oof[held]=cal.transform(hazards[held])
        trials[method]={'oof_brier':float(np.mean((cumulative(oof)-truth)**2)),'conditional_oof_hazards':oof}
    best=min(x['oof_brier'] for x in trials.values());chosen=next(m for m in config['calibration_candidates'] if trials[m]['oof_brier']<=best+1e-5)
    final=HazardCalibrator(chosen).fit(hazards,y,ids,roles)
    return final,{k:{'oof_brier':x['oof_brier']} for k,x in trials.items()}

def operating_points(probability,y,ids,roles,config):
    require_role(ids,roles,{'validation'});out={};truth=np.cumsum(y,axis=1)
    for j in range(3):
        p=probability[:,j];target=truth[:,j];order=np.argsort(-p,kind='stable');scores=p[order];a=target[order];tp=np.cumsum(a);fp=np.cumsum(1-a);totalpos=max(a.sum(),1);totalneg=max(len(a)-a.sum(),1)
        ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1];threshold=scores[ends];tp=tp[ends];fp=fp[ends];sensitivity=tp/totalpos;ppv=tp/np.maximum(tp+fp,1);f1=2*tp/np.maximum(totalpos+tp+fp,1)
        high=np.flatnonzero(sensitivity>=config['operating_points']['high_sensitivity_target']);h=high[0] if len(high) else len(ends)-1
        balanced=int(np.argmax(f1));valid=np.flatnonzero(fp/len(a)<=config['operating_points']['low_burden_false_alerts_per_patient_day']);low=valid[np.argmax(sensitivity[valid])] if len(valid) else None
        out[str((j+1)*24)]={'high_sensitivity':float(threshold[h]),'balanced':float(threshold[balanced]),'low_alert_burden':float(threshold[low]) if low is not None else float(np.nextafter(max(p),np.inf))}
    return out
