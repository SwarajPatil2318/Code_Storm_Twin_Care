"""Participant-macro continuous metrics, descriptive subgroups and cluster bootstrap."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score


def finite(v):return float(v) if np.isfinite(v) else None


def point_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);error=p-y
    mae=np.mean(np.abs(error),axis=0);rmse=np.sqrt(np.mean(error**2,axis=0));bias=np.mean(error,axis=0)
    corr=[];r2=[]
    for j in range(y.shape[1]):
        corr.append(finite(np.corrcoef(y[:,j],p[:,j])[0,1]) if len(y)>1 and np.std(y[:,j])>0 and np.std(p[:,j])>0 else None)
        total=np.sum((y[:,j]-np.mean(y[:,j]))**2)
        r2.append(finite(1-np.sum(error[:,j]**2)/total) if total>0 else None)
    return {'MAE':mae.tolist(),'RMSE':rmse.tolist(),'bias':bias.tolist(),'Pearson':corr,'R2':r2}


def forecast_metrics(y,p,pid):
    if not len(y):return {'windows':0,'participants':0}
    subjects=np.unique(pid);per=[point_metrics(y[pid==s],p[pid==s]) for s in subjects]
    macro={k:[finite(np.nanmean([np.nan if d[k][h] is None else d[k][h] for d in per])) for h in range(4)]
           for k in ['MAE','RMSE','bias','Pearson','R2']}
    return {'windows':len(y),'participants':len(subjects),'subject_macro':macro,'pooled':point_metrics(y,p)}


def confidence(y,p,pid,repeats=1000,seed=2026):
    subjects=np.unique(pid);per=[point_metrics(y[pid==s],p[pid==s]) for s in subjects]
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(subjects),size=(repeats,len(subjects)))
    return {key:np.nanquantile(np.nanmean(np.array([[np.nan if v is None else v for v in d[key]] for d in per])[draw],axis=1),[.025,.975],axis=0).T.tolist()
            for key in ['MAE','RMSE','bias','Pearson','R2']}


def paired_gain(y,base,personal,pid,repeats=1000,seed=2026):
    subjects=np.unique(pid)
    gains=np.array([(np.abs(y[pid==s]-base[pid==s])-np.abs(y[pid==s]-personal[pid==s])).mean(0) for s in subjects])
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(subjects),size=(repeats,len(subjects)))
    return {'participants':len(subjects),'windows':len(y),'MAE_reduction':gains.mean(0).tolist(),
            'ci95':np.quantile(gains[draw].mean(1),[.025,.975],axis=0).T.tolist(),
            'participants_improved':(gains.mean(1)>0).sum().item()}


def subgroups(data,p):
    g=data['sequence'][:,-1];slope=(g-data['sequence'][:,-3])/30
    masks={'lower_lt70':g<70,'intermediate_70_180':(g>=70)&(g<=180),'elevated_gt180':g>180,
           'stable_abs_slope_lt0.5':np.abs(slope)<.5,'rising_slope_ge0.5':slope>=.5,'falling_slope_le_minus0.5':slope<=-.5}
    return {name:forecast_metrics(data['y'][m],p[m],data['pid'][m]) for name,m in masks.items()}


def event_metrics(y,p,threshold,pid=None):
    y=np.asarray(y,bool);p=np.asarray(p,float);positive=p>=threshold
    tp=int((y&positive).sum());fp=int((~y&positive).sum());tn=int((~y&~positive).sum());fn=int((y&~positive).sum())
    div=lambda a,b:float(a/b) if b else None
    bins=[]
    for lo in np.linspace(0,1,6)[:-1]:
        hi=lo+.2;use=(p>=lo)&(p<=hi if hi>=1 else p<hi)
        bins.append({'lower':float(lo),'upper':float(hi),'windows':int(use.sum()),
            'mean_probability':float(p[use].mean()) if use.any() else None,'observed_rate':float(y[use].mean()) if use.any() else None})
    ece=sum(b['windows']*abs(b['mean_probability']-b['observed_rate']) for b in bins if b['windows'])/len(y) if len(y) else None
    result={'windows':len(y),'positive_landmarks':int(y.sum()),'prevalence':float(y.mean()) if len(y) else None,
        'AUPRC':float(average_precision_score(y,p)) if y.any() else None,
        'AUROC':float(roc_auc_score(y,p)) if y.any() and (~y).any() else None,
        'sensitivity':div(tp,tp+fn),'specificity':div(tn,tn+fp),'PPV':div(tp,tp+fp),'NPV':div(tn,tn+fn),
        'Brier':float(np.mean((p-y)**2)) if len(y) else None,'ECE_5bins':ece,'calibration_bins':bins,
        'confusion':{'tp':tp,'fp':fp,'tn':tn,'fn':fn},'threshold':float(threshold)}
    if pid is not None:
        result['participants']=len(np.unique(pid));result['positive_participants']=sum(bool(y[pid==s].any()) for s in np.unique(pid))
    return result


def event_confidence(y,p,pid,threshold,repeats=1000,seed=2026):
    subjects=np.unique(pid);indices=[np.flatnonzero(pid==s) for s in subjects];rng=np.random.default_rng(seed)
    scores={k:[] for k in ['AUPRC','AUROC','sensitivity','specificity','PPV','NPV','Brier']}
    for _ in range(repeats):
        ids=np.concatenate([indices[j] for j in rng.integers(0,len(subjects),len(subjects))])
        m=event_metrics(y[ids],p[ids],threshold)
        for key in scores:scores[key].append(np.nan if m[key] is None else m[key])
    return {k:np.nanquantile(v,[.025,.975]).tolist() for k,v in scores.items()}
