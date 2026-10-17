"""Horizon metrics and paired patient-cluster bootstrap with frozen predictions."""
import numpy as np
from scipy.special import logit,expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve


def basic(y,p,threshold=.5,w=None):
    y=np.asarray(y,int);p=np.asarray(p,float);w=np.ones(len(y)) if w is None else np.asarray(w,float);mass=w.sum();pos=np.sum(w*y);neg=mass-pos;alert=p>=threshold
    tp=np.sum(w*y*alert);fp=np.sum(w*(1-y)*alert);tn=np.sum(w*(1-y)*~alert);fn=np.sum(w*y*~alert)
    divide=lambda a,b:float(a/b) if b else None
    return {'rows':len(y),'prevalence':divide(pos,mass),'auprc':float(average_precision_score(y,p,sample_weight=w)) if pos and neg else None,'auroc':float(roc_auc_score(y,p,sample_weight=w)) if pos and neg else None,
      'brier':float(np.average((p-y)**2,weights=w)) if mass else None,'sensitivity':divide(tp,tp+fn),'specificity':divide(tn,tn+fp),'ppv':divide(tp,tp+fp),'npv':divide(tn,tn+fn),'f1':divide(2*tp,2*tp+fp+fn),
      'false_alerts':float(fp),'alerts':float(tp+fp),'false_alerts_per_patient_day':divide(fp,mass),'alerts_per_patient_day':divide(tp+fp,mass),'threshold':float(threshold)}

def calibration_diagnostics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);z=logit(np.clip(p,1e-6,1-1e-6));intercept=None;slope=None
    if len(set(y))==2 and np.std(z)>1e-8:
        def objective(theta):
            a=theta[0]+theta[1]*z;d=expit(a)-y;return float(np.mean(np.logaddexp(0,a)-y*a)),np.array([d.mean(),np.mean(d*z)])
        fit=minimize(objective,[0,1],jac=True,method='BFGS',options={'gtol':1e-6})
        if fit.success:intercept=float(fit.x[0]);slope=float(fit.x[1])
    # Equal-count bins are more informative for rare risks than fixed .1 bins.
    ordered=np.argsort(p,kind='stable');bins=[];ece=0
    for group in np.array_split(ordered,10):
        if not len(group):continue
        predicted=float(p[group].mean());observed=float(y[group].mean());ece+=len(group)/len(p)*abs(predicted-observed);bins.append({'n':len(group),'predicted':predicted,'observed':observed,'min':float(p[group].min()),'max':float(p[group].max())})
    return {'slope':slope,'intercept':intercept,'ece_equal_count':float(ece),'bins':bins,'diagnostic_only':True}

def horizon_results(labels,probabilities,thresholds):
    truth=np.cumsum(labels,axis=1);out={}
    for j in range(3):
        key=str((j+1)*24);point=thresholds[key];out[key]={'metrics':basic(truth[:,j],probabilities[:,j],point['balanced']),
          'calibration':calibration_diagnostics(truth[:,j],probabilities[:,j]),'operating':{name:basic(truth[:,j],probabilities[:,j],t) for name,t in point.items()}}
    return out

class RankedMetrics:
    """Pre-sort scores once; resampling retains exact tied-score AP/AUC."""
    def __init__(self,y,p,groups,threshold):
        self.y=np.asarray(y);self.p=np.asarray(p);self.groups=groups;self.threshold=threshold;self.order=np.argsort(-p,kind='stable');scores=self.p[self.order];self.ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1];self.logits=logit(np.clip(self.p,1e-6,1-1e-6))
    def weighted(self,counts,calibration=False):
        w=counts[self.groups];order=self.order;y=self.y[order];weights=w[order];tp=np.cumsum(weights*y)[self.ends];fp=np.cumsum(weights*(1-y))[self.ends];pos=tp[-1];neg=fp[-1]
        out=basic_fast(self.y,self.p,w,self.threshold)
        if calibration:
            intercept,slope=weighted_calibration(self.y,self.logits,w);out['calibration_intercept']=intercept;out['calibration_slope']=slope
        if pos and neg:
            dpos=np.diff(np.r_[0,tp]);out['auprc']=float(np.sum(dpos*tp/np.maximum(tp+fp,1))/pos);out['auroc']=float(np.trapezoid(np.r_[0,tp/pos],np.r_[0,fp/neg]))
        else:out['auprc']=None;out['auroc']=None
        return out

def basic_fast(y,p,w,t):
    mass=w.sum();alert=p>=t;tp=np.sum(w*y*alert);fp=np.sum(w*(1-y)*alert);tn=np.sum(w*(1-y)*~alert);fn=np.sum(w*y*~alert);div=lambda a,b:float(a/b) if b else None
    return {'brier':float(np.sum(w*(p-y)**2)/mass) if mass else None,'prevalence':div(np.sum(w*y),mass),'sensitivity':div(tp,tp+fn),'specificity':div(tn,tn+fp),'ppv':div(tp,tp+fp),'npv':div(tn,tn+fn),'f1':div(2*tp,2*tp+fp+fn),'alerts_per_patient_day':div(tp+fp,mass),'false_alerts_per_patient_day':div(fp,mass)}

def patient_bootstrap(labels,probs,ids,thresholds,n=1000,seed=2026):
    unique,groups=np.unique(ids,return_inverse=True);rng=np.random.default_rng(seed);truth=np.cumsum(labels,axis=1);scorers=[RankedMetrics(truth[:,j],probs[:,j],groups,thresholds[str((j+1)*24)]['balanced']) for j in range(3)]
    keys=['auprc','auroc','brier','prevalence','sensitivity','specificity','ppv','npv','f1','alerts_per_patient_day','false_alerts_per_patient_day','calibration_intercept','calibration_slope'];samples=[{key:[] for key in keys} for _ in range(3)]
    for _ in range(n):
        counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        for j,scorer in enumerate(scorers):
            for key,value in scorer.weighted(counts,calibration=True).items():
                if value is not None:samples[j].setdefault(key,[]).append(value)
    return {str((j+1)*24):{key:{'interval95':np.quantile(values,[.025,.975]).tolist() if len(values)>=.8*n else None,'valid_resamples':len(values),'requested':n} for key,values in sample.items()} for j,sample in enumerate(samples)}

def paired_ap(labels,left,right,ids,n=1000,seed=2026):
    unique,groups=np.unique(ids,return_inverse=True);rng=np.random.default_rng(seed);a=RankedMetrics(labels,left,groups,.5);b=RankedMetrics(labels,right,groups,.5);deltas=[]
    for _ in range(n):
        counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));x=a.weighted(counts)['auprc'];y=b.weighted(counts)['auprc']
        if x is not None and y is not None:deltas.append(x-y)
    return {'difference':float(average_precision_score(labels,left)-average_precision_score(labels,right)),'interval95':np.quantile(deltas,[.025,.975]).tolist() if len(deltas)>=.8*n else None,'valid_resamples':len(deltas),'requested':n}


def weighted_calibration(y,z,w):
    if np.std(z)<1e-8 or not np.sum(w*y) or not np.sum(w*(1-y)):return None,None
    intercept=0.;slope=1.;mass=max(w.sum(),1)
    for _ in range(20):
        p=expit(intercept+slope*z);d=(p-y)*w/mass;curvature=p*(1-p)*w/mass
        gradient=np.array([d.sum(),np.dot(d,z)]);hessian=np.array([[curvature.sum(),np.dot(curvature,z)],[np.dot(curvature,z),np.dot(curvature,z*z)]])+np.eye(2)*1e-9
        try:update=np.linalg.solve(hessian,gradient)
        except np.linalg.LinAlgError:return None,None
        update*=min(1,2/max(np.max(abs(update)),1e-9));intercept-=update[0];slope-=update[1]
        if not np.isfinite(intercept+slope):return None,None
        if max(abs(update))<1e-5:return float(intercept),float(slope)
    return None,None
