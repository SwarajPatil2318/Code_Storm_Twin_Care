"""Cluster-aware estimates and calibration/threshold role gates."""
from collections import defaultdict,Counter
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from src.predictive.data import guard


def basic(y,p,w=None):
    y=np.asarray(y);p=np.asarray(p);w=np.ones(len(y)) if w is None else np.asarray(w)
    positive=sum(w*y);negative=sum(w*(1-y));total=sum(w)
    return {'auprc':float(average_precision_score(y,p,sample_weight=w)) if positive and negative else None,
            'auroc':float(roc_auc_score(y,p,sample_weight=w)) if positive and negative else None,
            'brier':float(np.average((p-y)**2,weights=w)) if total else None,
            'prevalence':float(np.average(y,weights=w)) if total else None}


def operating(rows,p,threshold,w=None):
    y=np.array([r['label'] for r in rows]);p=np.asarray(p);w=np.ones(len(rows)) if w is None else np.asarray(w)
    alert=p>=threshold;tp=float(sum(w*(y==1)*alert));fp=float(sum(w*(y==0)*alert));tn=float(sum(w*(y==0)*~alert));fn=float(sum(w*(y==1)*~alert))
    ratio=lambda a,b: a/b if b else None
    leads=[(r['event_onset']-r['t0'])/60 for r,a in zip(rows,alert) if a and r['label']]
    return {'threshold':float(threshold),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),
            'ppv':ratio(tp,tp+fp),'npv':ratio(tn,tn+fn),'f1':ratio(2*tp,2*tp+fp+fn),
            'false_alerts':fp,'alerts':tp+fp,'false_alerts_per_eligible_hour':ratio(fp,sum(w)*3),
            'false_alerts_per_patient':ratio(fp,len({r['patient_id'] for r in rows})),
            'false_alerts_per_stay':ratio(fp,len({r['icu_stay_id'] for r in rows})),
            'lead_time_hours':None if not leads else {'min':float(min(leads)),'median':float(np.median(leads)),'max':float(max(leads))}}


def patient_folds(rows,folds=3):
    labels={pid:int(any(r['label'] for r in rows if r['patient_id']==pid)) for pid in {r['patient_id'] for r in rows}}
    import hashlib
    result={}
    for label in [0,1]:
        ids=sorted([p for p,v in labels.items() if v==label],key=lambda p:hashlib.sha256(('2026:'+p).encode()).hexdigest())
        for i,p in enumerate(ids):result[p]=i%folds
    return result


class Calibrator:
    def __init__(self,method):self.method=method
    def fit(self,p,rows,assignments,multiplicity=None):
        guard(rows,assignments,{'validation'});y=np.array([r['label'] for r in rows]);w=np.ones(len(y))
        if multiplicity is not None:w*=np.array([multiplicity.get(r['patient_id'],0) for r in rows])
        self.fit_patients=sorted({r['patient_id'] for r in rows if multiplicity is None or multiplicity.get(r['patient_id'],0)})
        p=np.clip(p,1e-6,1-1e-6)
        if self.method=='platt':
            x=logit(p)
            def objective(theta):
                z=theta[0]*x+theta[1]
                loss=float(np.average(np.logaddexp(0,z)-y*z,weights=w)+.001*(theta[0]-1)**2)
                dz=(expit(z)-y)*w/sum(w)
                return loss,np.array([sum(dz*x)+.002*(theta[0]-1),sum(dz)])
            fit=minimize(objective,[1.,0.],jac=True,bounds=[(.001,10),(-15,15)],method='L-BFGS-B')
            if not fit.success:raise ValueError('Calibration failed')
            self.parameters=fit.x
        elif self.method=='isotonic':self.estimator=IsotonicRegression(out_of_bounds='clip').fit(p,y,sample_weight=w)
        elif self.method!='none':raise ValueError('Unknown calibration method')
        return self
    def predict(self,p):
        p=np.clip(p,1e-6,1-1e-6)
        if self.method=='platt':return expit(self.parameters[0]*logit(p)+self.parameters[1])
        if self.method=='isotonic':return self.estimator.predict(p)
        return p


def select_calibration(p,rows,assignments):
    guard(rows,assignments,{'validation'});folds=patient_folds(rows);trials={}
    for method in ['none','platt','isotonic']:
        oof=np.zeros(len(rows))
        for fold in range(3):
            fit=np.array([folds[r['patient_id']]!=fold for r in rows]);held=~fit
            c=Calibrator(method).fit(p[fit],[r for r,yes in zip(rows,fit) if yes],assignments)
            oof[held]=c.predict(p[held])
        trials[method]={'oof':oof,'metrics':basic([r['label'] for r in rows],oof)}
    best=min(v['metrics']['brier'] for v in trials.values())
    chosen=next(m for m in ['none','platt','isotonic'] if trials[m]['metrics']['brier']<=best+1e-4)
    final=Calibrator(chosen).fit(p,rows,assignments)
    return final,trials[chosen]['oof'],{m:v['metrics'] for m,v in trials.items()}


def thresholds(rows,p,assignments):
    guard(rows,assignments,{'validation'})
    trials=[operating(rows,p,t) for t in np.r_[np.unique(p),1.000001]]
    high=max((a for a in trials if a['sensitivity'] is not None and a['sensitivity']>=.8),key=lambda a:a['threshold'])
    balanced=max(trials,key=lambda a:(a['f1'] or 0,a['specificity'] or 0,a['threshold']))
    burden=max((a for a in trials if a['false_alerts_per_eligible_hour']<=.02),key=lambda a:(a['sensitivity'] or 0,a['ppv'] or 0,a['threshold']))
    return {name:a['threshold'] for name,a in [('high_sensitivity',high),('balanced',balanced),('burden_aware',burden)]}


def calibration_diagnostics(y,p):
    y=np.array(y);p=np.array(p);bins=[];ece=0
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if any(mask):
            pred=float(np.mean(p[mask]));observed=float(np.mean(y[mask]));n=int(sum(mask));ece+=n/len(p)*abs(pred-observed)
            bins.append({'lower':float(lo),'upper':float(hi),'n':n,'predicted':pred,'observed':observed})
    intercept=None;slope=None
    if len(set(y))==2 and np.std(p)>1e-8:
        fit=LogisticRegression(penalty=None,max_iter=1000).fit(logit(np.clip(p,1e-6,1-1e-6))[:,None],y)
        intercept=float(fit.intercept_[0]);slope=float(fit.coef_[0,0])
    return {'intercept':intercept,'slope':slope,'ece_equal_width_10_bins':float(ece) if len(p) else None,'bins':bins,'diagnostic_only_not_applied':True}


def bootstrap(rows,p,threshold,n=2000,seed=2026):
    if not rows:return {}
    rng=np.random.default_rng(seed);p=np.asarray(p);y=np.array([r['label'] for r in rows]);ids=sorted({r['patient_id'] for r in rows})
    lookup={pid:i for i,pid in enumerate(ids)};group=np.array([lookup[r['patient_id']] for r in rows]);samples=defaultdict(list)
    for _ in range(n):
        counts=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids));w=counts[group]
        for k,v in basic(y,p,w).items():
            if k!='prevalence' and v is not None:samples[k].append(v)
        alert=p>=threshold;tp=sum(w*y*alert);fp=sum(w*(1-y)*alert);tn=sum(w*(1-y)*~alert);fn=sum(w*y*~alert)
        for k,a,b in [('sensitivity',tp,tp+fn),('specificity',tn,tn+fp),('ppv',tp,tp+fp),('npv',tn,tn+fn),('f1',2*tp,2*tp+fp+fn),('false_alerts_per_eligible_hour',fp,sum(w)*3)]:
            if b:samples[k].append(float(a/b))
        bin_id=np.minimum((p*10).astype(int),9);ece=0.
        for b in range(10):
            mask=bin_id==b;mass=sum(w[mask])
            if mass:ece+=mass/sum(w)*abs(np.average(p[mask]-y[mask],weights=w[mask]))
        samples['ece_equal_width_10_bins'].append(float(ece))
    return {k:{'interval95':np.quantile(v,[.025,.975]).tolist() if len(v)>=.8*n else None,'valid_resamples':len(v),'requested':n} for k,v in samples.items()}


def paired_bootstrap(rows,left,right,n=2000):
    if not rows:return {'auprc_difference':None,'interval95':None,'valid_resamples':0}
    ids=sorted({r['patient_id'] for r in rows});lookup={p:i for i,p in enumerate(ids)};group=np.array([lookup[r['patient_id']] for r in rows]);rng=np.random.default_rng(2026);y=[r['label'] for r in rows];deltas=[]
    for _ in range(n):
        counts=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids));w=counts[group]
        a=basic(y,left,w)['auprc'];b=basic(y,right,w)['auprc']
        if a is not None and b is not None:deltas.append(a-b)
    a=basic(y,left)['auprc'];b=basic(y,right)['auprc']
    return {'auprc_difference':None if a is None or b is None else a-b,'interval95':np.quantile(deltas,[.025,.975]).tolist() if len(deltas)>=.8*n else None,'valid_resamples':len(deltas),'estimand':'paired patient-cluster bootstrap with frozen models'}
