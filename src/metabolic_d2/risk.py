"""Auxiliary event classifier and participant-cross-fitted calibration, development only."""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from .models import TrainPreprocessor,subject_weights
from .metrics import event_metrics


def trajectory_score(prediction):
    # Two of four 30-minute forecasts above threshold is only a proxy for native 15-minute adjacency.
    return (np.sort(prediction,axis=1)[:,-2]-180)/20


def risk_features(data,prediction):
    return np.column_stack([data['x'],prediction-data['sequence'][:,-1,None]])


class RiskBundle:
    def fit_auxiliary(self,data,oof_prediction,allowed):
        use=data['sequence'][:,-1]<=180
        x=risk_features(data,oof_prediction)[use];pid=data['pid'][use];y=data['spike'][use]
        self.preprocessor=TrainPreprocessor().fit(x,pid,allowed)
        self.auxiliary=LogisticRegression(C=.1,max_iter=1000,random_state=2026).fit(
            self.preprocessor.transform(x),y,sample_weight=subject_weights(pid))
        self.training_participants=sorted(set(pid));return self

    def raw_scores(self,data,prediction):
        return {'trajectory':trajectory_score(prediction),
            'auxiliary':self.auxiliary.decision_function(self.preprocessor.transform(risk_features(data,prediction)))}

    def fit_calibration(self,data,prediction,folds):
        raw=self.raw_scores(data,prediction);y=data['spike'];eligible=data['sequence'][:,-1]<=180
        crossfit={k:np.zeros(len(y)) for k in raw}
        for fold in np.unique(folds):
            train=(folds!=fold)&eligible;valid=(folds==fold)&eligible
            if len(np.unique(y[train]))!=2:raise ValueError('Insufficient calibration classes')
            for k,scores in raw.items():
                calibrator=LogisticRegression(C=1,max_iter=1000,random_state=2026).fit(
                    scores[train,None],y[train],sample_weight=subject_weights(data['pid'][train]))
                crossfit[k][valid]=calibrator.predict_proba(scores[valid,None])[:,1]
        crossfit['combined_equal_average']=(crossfit['trajectory']+crossfit['auxiliary'])/2
        results={}
        for k,p in crossfit.items():
            results[k]=event_metrics(y,p,.5,data['pid'])
            results[k]['participant_macro_Brier']=float(np.mean([np.mean((p[data['pid']==s]-y[data['pid']==s])**2) for s in np.unique(data['pid'])]))
        order=['trajectory','auxiliary','combined_equal_average']
        self.method=min(order,key=lambda k:(results[k]['participant_macro_Brier'],order.index(k)))
        probability=crossfit[self.method]
        # Freeze the alert threshold using only cross-fitted calibration probabilities.
        fpr,tpr,thresholds=roc_curve(y,probability,sample_weight=subject_weights(data['pid']))
        valid=np.isfinite(thresholds)&(thresholds>=0)&(thresholds<=1)
        indices=np.flatnonzero(valid)
        best=max(indices,key=lambda i:(round(float((tpr[i]+1-fpr[i])/2),12),1-fpr[i],thresholds[i]))
        self.threshold=float(thresholds[best]);self.calibrators={}
        for k,scores in raw.items():
            self.calibrators[k]=LogisticRegression(C=1,max_iter=1000,random_state=2026).fit(
                scores[eligible,None],y[eligible],sample_weight=subject_weights(data['pid'][eligible]))
        self.calibration_participants=sorted(set(data['pid']))
        self.crossfit_results=results
        return {'selected_method':self.method,'threshold':self.threshold,'crossfit_candidates':results,
            'selected_crossfit_metrics':event_metrics(y,probability,self.threshold,data['pid']),
            'threshold_selection':'participant-weighted balanced accuracy; ties higher specificity',
            'calibration_participants':len(self.calibration_participants),'calibration_folds':len(np.unique(folds)),
            'target':'incident_two_readings_gt180','holdout_inspected':False},crossfit

    def probabilities(self,data,prediction):
        raw=self.raw_scores(data,prediction)
        p={k:self.calibrators[k].predict_proba(s[:,None])[:,1] for k,s in raw.items()}
        p['combined_equal_average']=(p['trajectory']+p['auxiliary'])/2
        for values in p.values():values[data['sequence'][:,-1]>180]=0
        return p

    def save(self,path):
        with path.open('wb') as stream:pickle.dump(self,stream,protocol=5)
        path.chmod(0o600)

    @staticmethod
    def load(path):
        with path.open('rb') as stream:return pickle.load(stream)
