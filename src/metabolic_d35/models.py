"""Fixed D2-family experimental clones; no production model saves or replacement."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from src.metabolic_d2.models import TrainPreprocessor, subject_weights, huber_objective
from src.metabolic_d2.protocol import config
from .data import NUMERIC, CATEGORICAL, dynamic_features


class StaticEncoder:
    def __init__(self, demographics=False):
        self.numeric = ['age_years'] if demographics else NUMERIC
        self.categorical = ['sex'] if demographics else CATEGORICAL

    def fit(self, profiles, allowed):
        ids = set(profiles['participant_id'])
        if not ids <= set(allowed) or len(ids) != len(profiles):
            raise PermissionError('Static encoder must fit unique training participants only')
        self.training_participants = sorted(ids)
        x = profiles[self.numeric].to_numpy(float)
        median = []
        for column in x.T:
            good = column[np.isfinite(column)]
            median.append(float(np.median(good)) if len(good) else 0.)
        self.median = np.array(median)
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        self.encoder.fit(self.categories(profiles))
        self.names = self.numeric + [n+'_missing' for n in self.numeric] + self.encoder.get_feature_names_out(self.categorical).tolist() + [n+'_missing' for n in self.categorical]
        return self

    def categories(self, profiles):
        return profiles[self.categorical].fillna('__missing__').astype(str).to_numpy()

    def transform(self, profiles):
        x = profiles[self.numeric].to_numpy(float)
        missing = ~np.isfinite(x)
        return np.column_stack([np.where(missing, self.median, x), missing,
                                self.encoder.transform(self.categories(profiles)),
                                profiles[self.categorical].isna().to_numpy()]).astype('float32')


def profiles_for(data, profiles, donors=None):
    lookup = profiles.set_index('participant_id')
    own = [str(p) for p in data['pid']]
    if donors and set(donors) != set(own): raise PermissionError('Shuffle must cover exactly this fold')
    requested = [donors[p] if donors else p for p in own]
    if not set(requested) <= set(lookup.index): raise PermissionError('Unknown same-cohort static profile')
    frame = lookup.loc[requested].reset_index()
    frame['participant_id'] = own  # Training identities remain dynamic-stream owners.
    return frame


class FusionModel:
    def __init__(self, variant):
        self.variant = variant

    def fit(self, data, profiles, allowed, seed, donors=None):
        ids = set(data['pid'])
        if not ids <= set(allowed): raise PermissionError('Non-training dynamic participant')
        if donors and (set(donors) != ids or set(donors.values()) != ids):
            raise PermissionError('Cross-fold static shuffle')
        self.training_participants = sorted(ids)
        self.dynamic_preprocessor = None
        self.static_encoder = None
        if self.variant != 'static_only':
            self.dynamic_preprocessor = TrainPreprocessor().fit(dynamic_features(data), data['pid'], allowed)
        if self.variant != 'dynamic_only':
            unique = profiles[profiles['participant_id'].isin(ids)].copy()
            if donors:
                # Fit on the permuted training profiles only (same multiset, unique owners).
                temp = {'pid': np.array(sorted(ids))}
                unique = profiles_for(temp, profiles, donors)
            self.static_encoder = StaticEncoder(self.variant == 'age_sex').fit(unique, allowed)
        x = self.features(data, profiles, donors=donors)
        w = subject_weights(data['pid'])
        self.static_offset = np.average(data['y'], axis=0, weights=w).astype('float32')
        offset = np.broadcast_to(self.static_offset, data['y'].shape) if self.variant == 'static_only' else data['sequence'][:, -1, None]
        p = config()['gradient_boosting']
        params = {'tree_method':'hist', 'multi_strategy':'multi_output_tree', 'num_target':4,
                  'objective':'reg:squarederror', 'base_score':0, 'max_depth':p['max_depth'],
                  'eta':p['learning_rate'], 'max_bin':p['max_bin'], 'nthread':p['nthread'],
                  'reg_lambda':p['reg_lambda'], 'min_child_weight':p['min_child_weight'], 'seed':seed, 'verbosity':0}
        self.estimator = xgb.train(params, xgb.DMatrix(x, label=data['y']-offset, weight=w),
                                   num_boost_round=p['rounds'], obj=huber_objective)
        return self

    def features(self, data, profiles, donors=None, dynamic=None):
        parts = []
        if self.dynamic_preprocessor:
            parts.append(self.dynamic_preprocessor.transform(dynamic_features(data) if dynamic is None else dynamic))
        if self.static_encoder:
            parts.append(self.static_encoder.transform(profiles_for(data, profiles, donors)))
        return np.column_stack(parts).astype('float32')

    def predict(self, data, profiles, donors=None, dynamic=None, current=None):
        residual = self.estimator.predict(xgb.DMatrix(self.features(data, profiles, donors, dynamic)))
        if self.variant == 'static_only': offset = self.static_offset
        else: offset = (data['sequence'][:, -1] if current is None else current)[:, None]
        return (offset + residual).astype('float32')

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open('wb') as stream: pickle.dump(self, stream, protocol=5)
        path.chmod(0o600)
