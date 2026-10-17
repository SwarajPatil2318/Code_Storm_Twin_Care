"""Explicit two-stream logistic risk; no D2/Shadow mutations or missing-profile fallback."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.metabolic_d2.data import CGM_NAMES, subset
from src.metabolic_d2.models import TrainPreprocessor, subject_weights
from src.metabolic_d35.data import SAFE
from src.metabolic_d35.models import StaticEncoder
from . import NAME

DYNAMIC_NAMES = CGM_NAMES + [f'forecast_residual_{h}m' for h in [30,60,90,120]] + ['predicted_peak']
VARIANTS = ('static_only','dynamic_only','fused','shuffled_static')


def dynamic_vector(data, forecast):
    if data is None or forecast is None:
        raise ValueError('CGM and four D2 forecasts are required')
    x=np.asarray(data['x']);p=np.asarray(forecast)
    if x.ndim!=2 or x.shape[1]!=21 or p.shape!=(len(x),4) or not np.isfinite(p).all():
        raise ValueError('Invalid dynamic risk schema')
    current=np.asarray(data['sequence'])[:,-1,None]
    return np.column_stack([x,p-current,p.max(axis=1)]).astype('float32')


def linked_profiles(data, profiles, donors=None):
    if profiles is None or not set(['participant_id']+SAFE)<=set(profiles.columns):
        raise ValueError('Explicit same-person safe static profile required')
    if profiles['participant_id'].duplicated().any():raise ValueError('Ambiguous static identity')
    owners=[str(v) for v in data['pid']]
    if donors and (set(donors)!=set(owners) or set(donors.values())!=set(owners)):
        raise PermissionError('Profile shuffle must stay within training participants')
    wanted=[donors[v] if donors else v for v in owners]
    lookup=profiles.set_index('participant_id')
    if not set(wanted)<=set(lookup.index):raise ValueError('Same-person profile missing')
    result=lookup.loc[wanted].reset_index()
    result['participant_id']=owners
    return result


class RiskHead:
    def __init__(self,variant='fused'):
        if variant not in VARIANTS:raise ValueError('Unknown risk variant')
        self.variant=variant;self.version=NAME;self.calibrator=None

    def fit(self,data,forecast,profiles,allowed,donors=None):
        eligible=data['sequence'][:,-1]<=180
        forecast=np.asarray(forecast)[eligible]
        data=subset(data,eligible)
        ids=set(data['pid'])
        if not ids or not ids<=set(allowed):raise PermissionError('Risk training participant isolation violation')
        self.training_participants=sorted(ids)
        self.dynamic_preprocessor=None;self.static_encoder=None;self.static_preprocessor=None
        if self.variant!='static_only':
            self.dynamic_preprocessor=TrainPreprocessor().fit(dynamic_vector(data,forecast),data['pid'],allowed)
        if self.variant!='dynamic_only':
            unique=profiles[profiles['participant_id'].isin(ids)].copy()
            self.static_encoder=StaticEncoder().fit(unique,allowed)
            encoded=self.static_encoder.transform(linked_profiles(data,profiles,donors))
            self.static_preprocessor=TrainPreprocessor().fit(encoded,data['pid'],allowed)
        x=self.features(data,forecast,profiles,donors)
        self.estimator=LogisticRegression(C=.1,solver='lbfgs',penalty='l2',max_iter=2000,
                                         random_state=5102026,class_weight=None).fit(
            x,data['spike'],sample_weight=subject_weights(data['pid']))
        if self.estimator.n_iter_.max()>=2000:raise RuntimeError('Risk logistic did not converge')
        self.feature_order=(DYNAMIC_NAMES if self.dynamic_preprocessor else [])+(self.static_encoder.names if self.static_encoder else [])
        return self

    def features(self,data,forecast,profiles,donors=None):
        parts=[]
        if self.dynamic_preprocessor:parts.append(self.dynamic_preprocessor.transform(dynamic_vector(data,forecast)))
        if self.static_encoder:
            x=self.static_encoder.transform(linked_profiles(data,profiles,donors))
            parts.append(self.static_preprocessor.transform(x))
        if not parts:raise RuntimeError('Unfitted risk model')
        return np.column_stack(parts).astype('float32')

    def raw(self,data,forecast,profiles):
        return self.estimator.decision_function(self.features(data,forecast,profiles))

    def calibrate(self,data,forecast,profiles,allowed):
        if set(data['pid'])&set(self.training_participants) or not set(data['pid'])<=set(allowed):
            raise PermissionError('Calibration participants must be separate training-fold participants')
        self.calibration_participants=sorted(set(data['pid']))
        eligible=data['sequence'][:,-1]<=180
        forecast=np.asarray(forecast)[eligible]
        data=subset(data,eligible)
        scores=self.raw(data,forecast,profiles)
        self.calibrator=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,random_state=5102026).fit(
            scores[:,None],data['spike'],sample_weight=subject_weights(data['pid']))
        return self

    def probabilities(self,data,forecast,profiles):
        if self.calibrator is None:raise RuntimeError('Risk calibration required')
        probability=self.calibrator.predict_proba(self.raw(data,forecast,profiles)[:,None])[:,1]
        probability[data['sequence'][:,-1]>180]=0
        return probability

    def save(self,path):
        path=Path(path)
        with path.open('wb') as stream:pickle.dump(self,stream,protocol=5)
        path.chmod(0o600)

    @staticmethod
    def load(path):
        with Path(path).open('rb') as stream:return pickle.load(stream)
