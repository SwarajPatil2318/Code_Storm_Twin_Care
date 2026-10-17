"""Train-role encoded tabular baselines; reusable, serializable inference."""
import warnings
from collections import Counter
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from src.predictive.data import columns,guard,weights


class Encoder:
    def fit(self,rows,assignments,modality):
        guard(rows,assignments,{'train'});self.modality=modality
        self.keys=columns(modality,rows[0]['features'])
        self.categories={};self.numeric=[]
        for key in self.keys:
            vals=[r['features'][key] for r in rows]
            if any(isinstance(v,str) for v in vals):self.categories[key]=sorted({str(v) if v is not None else '__MISSING__' for v in vals})
            elif key in ['ehr__gender','ehr__unit_type','ehr__unit_admission_source'] or key.endswith('_unit'):
                self.categories[key]=sorted({str(v) if v is not None else '__MISSING__' for v in vals})
            else:self.numeric.append(key)
        raw=self.raw_numeric(rows);self.medians=np.array([np.median(c[np.isfinite(c)]) if np.any(np.isfinite(c)) else 0 for c in raw.T])
        filled=np.where(np.isfinite(raw),raw,self.medians);self.means=np.mean(filled,axis=0);self.scales=np.std(filled,axis=0);self.scales[self.scales<1e-8]=1
        self.names=self.numeric+['missing__'+k for k in self.numeric]
        for k,v in self.categories.items():self.names += [k+'='+x for x in v]+[k+'=__UNSEEN__']
        self.fit_patients=sorted({r['patient_id'] for r in rows});return self

    def raw_numeric(self,rows):
        return np.array([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in self.numeric] for r in rows])

    def transform(self,rows):
        raw=self.raw_numeric(rows);missing=~np.isfinite(raw);filled=np.where(missing,self.medians,raw)
        blocks=[(filled-self.means)/self.scales,missing.astype(float)]
        for key,categories in self.categories.items():
            values=[str(r['features'][key]) if r['features'][key] is not None else '__MISSING__' for r in rows]
            blocks.append(np.array([[float(v==c) for c in categories]+[float(v not in categories)] for v in values]))
        return np.concatenate(blocks,axis=1)

    def reference(self):
        result=dict(zip(self.numeric,self.medians.tolist()))
        for key,categories in self.categories.items():result[key]=categories[0] if categories else None
        return result


class Tabular:
    def __init__(self,modality,config):self.modality=modality;self.config=config
    def fit(self,rows,assignments,multiplicity=None):
        guard(rows,assignments,{'train'})
        reference=rows if multiplicity is None else [r for r in rows for _ in range(multiplicity.get(r['patient_id'],0))]
        self.encoder=Encoder().fit(reference,assignments,self.modality)
        x=self.encoder.transform(rows);y=np.array([r['label'] for r in rows]);w=weights(rows)
        if multiplicity is not None:w*=np.array([multiplicity.get(r['patient_id'],0) for r in rows])
        active=w>0
        if self.config.get('balanced'):
            total=[sum(w[y==k]) for k in [0,1]]
            w*=np.where(y==1,sum(total)/(2*total[1]),sum(total)/(2*total[0]))
        c=self.config
        if c['kind']=='logistic':
            self.estimator=LogisticRegression(penalty=c['penalty'],C=c['C'],solver='lbfgs',max_iter=1500,random_state=2026)
        else:
            self.estimator=HistGradientBoostingClassifier(max_leaf_nodes=c['leaves'],max_iter=c['iterations'],learning_rate=.05,
                           min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=2026)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);self.estimator.fit(x[active],y[active],sample_weight=w[active])
        self.converged=not any(issubclass(a.category,ConvergenceWarning) for a in caught)
        self.fit_patients=self.encoder.fit_patients;return self
    def predict(self,rows):return self.estimator.predict_proba(self.encoder.transform(rows))[:,1]
    @property
    def complexity(self):return 1 if self.config['kind']=='logistic' else 2


class Rule:
    def __init__(self,kind):self.kind=kind;self.modality='map_only';self.config={'kind':kind};self.converged=True
    def fit(self,rows,assignments,multiplicity=None):
        guard(rows,assignments,{'train'});self.fit_patients=sorted({r['patient_id'] for r in rows if multiplicity is None or multiplicity.get(r['patient_id'],0)>0})
        y=np.array([r['label'] for r in rows]);w=np.ones(len(rows))
        if multiplicity is not None:w*=np.array([multiplicity.get(r['patient_id'],0) for r in rows])
        self.prevalence=float(np.average(y,weights=w));return self
    def predict(self,rows):
        if self.kind=='prevalence':return np.full(len(rows),self.prevalence)
        latest=np.array([r['features']['phys__systemicmean__30__last'] for r in rows],dtype=float)
        if self.kind=='map_proximity':return expit((75-latest)/5)
        slopes=np.array([r['features']['phys__systemicmean__60__slope'] or 0 for r in rows])
        forecast=latest+3*slopes
        return expit((65-forecast)/10)
    @property
    def complexity(self):return 0


class LateFusion:
    def __init__(self,ehr,phys):self.ehr=ehr;self.phys=phys;self.modality='fused';self.config={'kind':'late_prediction_fusion'};self.converged=True
    def fit(self,rows,assignments,folds,multiplicity=None):
        guard(rows,assignments,{'train'});y=np.array([r['label'] for r in rows]);oof=np.zeros((len(rows),2))
        for fold in range(3):
            fit=[r for r in rows if folds[r['patient_id']]!=fold];held=[r for r in rows if folds[r['patient_id']]==fold]
            for j,base in enumerate([self.ehr,self.phys]):
                model=Tabular(base.modality,base.config).fit(fit,assignments,multiplicity)
                pred=model.predict(held);indexes=[i for i,r in enumerate(rows) if folds[r['patient_id']]==fold];oof[indexes,j]=pred
        w=weights(rows)
        if multiplicity is not None:w*=np.array([multiplicity.get(r['patient_id'],0) for r in rows])
        self.head=LogisticRegression(C=.1,max_iter=1000,random_state=2026).fit(self.logits(oof),y,sample_weight=w)
        self.inner_folds=folds
        self.fit_patients=sorted({r['patient_id'] for r in rows if multiplicity is None or multiplicity.get(r['patient_id'],0)>0});return self
    @staticmethod
    def logits(p):return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    def predict(self,rows):
        inputs=np.column_stack([self.ehr.predict(rows),self.phys.predict(rows)])
        return self.head.predict_proba(self.logits(inputs))[:,1]
    @property
    def complexity(self):return 3
