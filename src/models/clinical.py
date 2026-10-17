"""Six-field clinical benchmark; test evaluation is a separately sealed operation."""
import csv
from pathlib import Path
import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                            confusion_matrix, precision_recall_curve, roc_curve)
from src.data.adapters.zigong import APPROVED_FEATURES as APPROVED, TARGET
from .common import *

CATEGORICAL = ['gender','ageCat','admission.ward','admission.way']
NUMERIC = ['visit.times','congestive.heart.failure']


def read_predictors(root, assignments, roles):
    with (Path(root)/'zigong/predictors.csv').open() as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames) != {'patient_id',*APPROVED}:
            raise ValueError('Predictor allowlist violated')
        rows = [r for r in reader if assignments.get(r['patient_id']) in roles]
    ids = [r.pop('patient_id') for r in rows]
    require_roles(ids, assignments, roles)
    x = np.empty((len(rows),6), dtype=object)
    for i,r in enumerate(rows):
        for j,k in enumerate(CATEGORICAL+NUMERIC):
            v=r[k]
            x[i,j] = (v if v else np.nan) if k in CATEGORICAL else (float(v) if v else np.nan)
    return ids,x


def read_labels(root, ids, assignments, roles):
    require_roles(ids, assignments, roles)
    wanted = set(ids)
    with (Path(root)/'zigong/labels.csv').open() as f:
        labels = {r['patient_id']:int(r[TARGET]) for r in csv.DictReader(f) if r['patient_id'] in wanted}
    if set(labels) != wanted: raise ValueError('Missing labels')
    return np.array([labels[p] for p in ids])


def preprocessing():
    return ColumnTransformer([
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
                                 ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),list(range(4))),
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                             ('scale',StandardScaler())]),[4,5])])


def candidates():
    result=[]
    for penalty in ['l2','l1']:
        for c in [0.1,1.,10.]:
            result.append((f'logistic_{penalty}_C{c}',LogisticRegression(C=c,penalty=penalty,solver='liblinear',max_iter=2000,random_state=SEED)))
    result.append(('logistic_l2_balanced',LogisticRegression(C=1,solver='liblinear',class_weight='balanced',random_state=SEED)))
    for depth in [1,2]:
        result.append((f'gradient_boosting_depth{depth}',GradientBoostingClassifier(n_estimators=60,max_depth=depth,min_samples_leaf=40,learning_rate=.05,random_state=SEED)))
    return result


class ClinicalRiskModel:
    """Probability predictor; fits base/OOF calibrator on training patients only."""
    def __init__(self, estimator=None):
        self.estimator=estimator

    def fit(self,x,y,ids,assignments):
        require_roles(ids,assignments,{'train'})
        if x.shape[1]!=6:raise ValueError('Expected exactly six approved input fields')
        self.fit_patient_ids_=tuple(ids)
        self.prevalence_=float(np.mean(y))
        if self.estimator is None:
            self.pipeline_=None; self.calibrator_=None
            return self
        pipeline=Pipeline([('preprocessing',preprocessing()),('model',clone(self.estimator))])
        oof=np.zeros(len(y)); self.calibration_folds_=[]
        for train,hold in StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y):
            model=clone(pipeline).fit(x[train],y[train])
            oof[hold]=model.predict_proba(x[hold])[:,1]
            self.calibration_folds_.append({'fit_ids':[ids[i] for i in train],'prediction_ids':[ids[i] for i in hold]})
        self.calibrator_=LogisticRegression(C=1e6,solver='lbfgs',random_state=SEED).fit(logit(np.clip(oof,1e-6,1-1e-6)).reshape(-1,1),y)
        self.pipeline_=pipeline.fit(x,y)
        return self

    def calibrate(self,x,y,ids,assignments,method):
        require_roles(ids,assignments,{'validation'})
        self.calibration_patient_ids_=tuple(ids)
        self.calibration_method_=method
        logits=logit(np.clip(self.raw(x),1e-6,1-1e-6))
        if method=='sigmoid':
            self.calibrator_=LogisticRegression(C=1e6,solver='lbfgs',random_state=SEED).fit(logits.reshape(-1,1),y)
        else:
            self.calibrator_=None
            self.offset_=float(minimize_scalar(lambda b:np.mean(np.logaddexp(0,logits+b)-y*(logits+b)),bounds=(-10,10),method='bounded').x)
        return self

    def raw(self,x):
        if x.shape[1]!=6:raise ValueError('Expected exactly six approved input fields')
        return np.full(len(x),self.prevalence_) if self.pipeline_ is None else self.pipeline_.predict_proba(x)[:,1]

    def predict(self,x):
        p=self.raw(x)
        if hasattr(self,'offset_'):return expit(logit(np.clip(p,1e-6,1-1e-6))+self.offset_)
        return p if self.calibrator_ is None else self.calibrator_.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]


def calibration_diagnostic(y,p):
    if np.ptp(p)<1e-10 or len(set(y))<2:
        return {'intercept':None,'slope':None,'reason':'constant probabilities or one outcome class'}
    model=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000).fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),y)
    return {'intercept':float(model.intercept_[0]),'slope':float(model.coef_[0,0]),'reason':'descriptive evaluation fit; not applied to predictions'}


def metric(y,p,threshold=.5,curves=True):
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    d={'n':len(y),'positives':int(sum(y)),'prevalence':float(np.mean(y)),
       'auprc':float(average_precision_score(y,p)), 'auroc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
       'brier':float(brier_score_loss(y,p)), 'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
       'specificity':float(tn/(tn+fp)) if tn+fp else None,'ppv':float(tp/(tp+fp)) if tp+fp else 0.,
       'recall':float(tp/(tp+fn)) if tp+fn else None,'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
       'threshold':float(threshold),'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}
    if curves:
        d['calibration']=calibration_diagnostic(y,p)
        d['calibration_curve']=[{'n':int(sum((p>=lo)&(p<(hi if hi<1 else 1.0001)))),
            'mean_probability':float(np.mean(p[(p>=lo)&(p<(hi if hi<1 else 1.0001))])),
            'observed_fraction':float(np.mean(y[(p>=lo)&(p<(hi if hi<1 else 1.0001))]))}
            for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]) if np.any((p>=lo)&(p<(hi if hi<1 else 1.0001)))]
        pr,rec,_=precision_recall_curve(y,p); fpr,tpr,_=roc_curve(y,p)
        d['pr_curve']={'precision':pr.tolist(),'recall':rec.tolist()}; d['roc_curve']={'fpr':fpr.tolist(),'tpr':tpr.tolist()}
    return d


def thresholds(y,p):
    choices=[metric(y,p,t,False) for t in np.unique(np.r_[p,1.0000001])]
    best=max(choices,key=lambda m:(m['f1'],m['specificity'],m['threshold']))
    sensitive=max((m for m in choices if m['sensitivity']>=.8),key=lambda m:(m['specificity'],m['threshold']))
    return {'max_validation_f1':best['threshold'],'validation_sensitivity_at_least_0.80':sensitive['threshold']}


def bootstrap(y,p,t,reps=2000):
    rng=np.random.default_rng(SEED); values={k:[] for k in ['auprc','auroc','brier','sensitivity','specificity','ppv','f1']}
    skipped=0
    for _ in range(reps):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))<2: skipped+=1;continue
        m=metric(y[i],p[i],t,False)
        for k in values:values[k].append(m[k])
    return {'method':'nonparametric patient bootstrap percentile; fixed fitted model and threshold', 'replicates':reps,'skipped_one_class':skipped,
            'intervals_95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in values.items()}}


def train(root='data/processed',output='data/results/phase3/clinical'):
    info=metadata(root,'zigong'); directory=sealed_directory(output)
    assignments=load_json(Path(root)/'zigong/splits.json')['assignments']
    ids,x=read_predictors(root,assignments,{'train'}); y=read_labels(root,ids,assignments,{'train'})
    vi,vx=read_predictors(root,assignments,{'validation'}); vy=read_labels(root,vi,assignments,{'validation'})
    from src.data.splits import rank
    calibration_ids=set()
    for label in [0,1]:
        ranked=sorted((pid for pid,v in zip(vi,vy) if v==label),key=lambda pid:rank(pid,SEED+1))
        calibration_ids.update(ranked[:len(ranked)//2])
    ci=np.array([pid in calibration_ids for pid in vi]);si=~ci
    results=[]; models={}
    for name,est in [('prevalence',None)]+candidates():
        model=ClinicalRiskModel(est).fit(x,y,ids,assignments); models[name]=model
        raw=model.raw(vx[si]); variants={}
        from copy import deepcopy
        for method in ['intercept_only','sigmoid']:
            calibrated=deepcopy(model).calibrate(vx[ci],vy[ci],[pid for pid,flag in zip(vi,ci) if flag],assignments,method)
            variants[method]=(calibrated,metric(vy[si],calibrated.predict(vx[si])))
        method=min(variants,key=lambda k:variants[k][1]['brier'])
        models[name]=variants[method][0]
        results.append({'name':name,'hyperparameters':{} if est is None else est.get_params(),
                        'raw_validation':metric(vy[si],raw),'calibrated_validation':variants[method][1],
                        'calibration_selected':method,'calibration_comparison':{k:v[1] for k,v in variants.items()},
                        'training_coefficients':[] if est is None or not hasattr(model.pipeline_.named_steps['model'],'coef_') else [
                            {'encoded_feature':n,'log_odds_coefficient':float(v)} for n,v in zip(model.pipeline_.named_steps['preprocessing'].get_feature_names_out(),model.pipeline_.named_steps['model'].coef_[0])],
                        'training_oof_calibration_slope':None if model.calibrator_ is None else float(model.calibrator_.coef_[0,0])})
    # Rounded AP differences <0.005 are treated as negligible; simpler candidate wins.
    top=max(r['calibrated_validation']['auprc'] for r in results)
    selected=next(r['name'] for r in results if r['calibrated_validation']['auprc']>=top-.005)
    model=models[selected]; rules=thresholds(vy[si],model.predict(vx[si]))
    interpret={}
    if model.pipeline_ is not None:
        pp=model.pipeline_.named_steps['preprocessing']; est=model.pipeline_.named_steps['model']
        if hasattr(est,'coef_'):
            interpret['coefficients']=[{'encoded_feature':n,'log_odds_coefficient':float(v)} for n,v in zip(pp.get_feature_names_out(),est.coef_[0])]
        else:
            perm=permutation_importance(model.pipeline_,vx[si],vy[si],scoring='average_precision',n_repeats=20,random_state=SEED)
            interpret['validation_permutation_importance']=[{'feature':k,'mean_ap_drop':float(v),'std':float(s)} for k,v,s in zip(CATEGORICAL+NUMERIC,perm.importances_mean,perm.importances_std)]
    save_model(directory/'model.joblib',model);save_model(directory/'prevalence.joblib',ClinicalRiskModel().fit(x,y,ids,assignments))
    write_json(directory/'selection.json',{'metadata':info,'selected':selected,'candidates':results,'thresholds':rules,
             'selection_rule':'highest calibrated validation AP; within 0.005 choose first simpler candidate in declared order',
             'calibration_method':'fit base model on train; fit sigmoid and intercept-only calibrators on stratified half of validation; choose by Brier on other validation half',
             'isotonic':'not attempted: 84 training positives and 28 validation positives too sparse',
             'fit_patient_ids':ids,'validation_patient_ids':vi,'calibration_patient_ids':sorted(calibration_ids),
             'selection_patient_ids':[pid for pid,flag in zip(vi,si) if flag],
             'validation_partition_counts':{'calibration':{'n':int(sum(ci)),'positives':int(sum(vy[ci]))},'selection':{'n':int(sum(si)),'positives':int(sum(vy[si]))}},'interpretability':interpret})
    write_json(directory/'selection_seal.json',{'hashes':artifact_hashes(directory),'code_sha256':info['code_sha256']})
    return {'selected':selected,'validation_auprc':next(r['calibrated_validation']['auprc'] for r in results if r['name']==selected),'candidates':len(results)}


def evaluate(root='data/processed',output='data/results/phase3/clinical'):
    directory=Path(output); seal=load_json(directory/'selection_seal.json');verify_hashes(directory,seal['hashes'])
    selection=load_json(directory/'selection.json')
    if code_hash()!=seal['code_sha256']:raise ValueError('Source changed after selection')
    if (directory/'test_metrics.json').exists():
        verify_hashes(directory,load_json(directory/'evaluation_seal.json')['hashes'])
        return {'status':'stored final test evaluation; no new test evaluation','metrics_file':str(directory/'test_metrics.json')}
    verify_inputs(root,selection['metadata'])
    # Exclusive persistent marker: any interrupted attempt requires investigation, never automatic retry.
    fd=os.open(directory/'TEST_EVALUATION_STARTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    assignments=load_json(Path(root)/'zigong/splits.json')['assignments']
    ids,x=read_predictors(root,assignments,{'test'}); y=read_labels(root,ids,assignments,{'test'})
    model=joblib.load(directory/'model.joblib');base=joblib.load(directory/'prevalence.joblib')
    p=model.predict(x);raw=model.raw(x)
    metrics={'selected':selection['selected'],'baseline':metric(y,base.predict(x)), 'raw_final':metric(y,raw),
             'calibrated_final':metric(y,p),'operating_points':{},'test_evaluation_count':1,
             'selection_sha256':file_hash(directory/'selection.json')}
    for rule,t in selection['thresholds'].items():
        metrics['operating_points'][rule]={'metrics':metric(y,p,t),'bootstrap':bootstrap(y,p,t)}
    write_rows(directory/'test_predictions.jsonl',[{'patient_id':pid,'observed':int(v),'raw_probability':float(r),'clinical_vulnerability':float(q)} for pid,v,r,q in zip(ids,y,raw,p)])
    write_json(directory/'test_metrics.json',metrics)
    write_json(directory/'evaluation_seal.json',{'hashes':artifact_hashes(directory,('evaluation_seal.json',))})
    return {'status':'final test evaluation sealed','selected':selection['selected']}
