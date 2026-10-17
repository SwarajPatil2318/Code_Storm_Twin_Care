"""One regularized offset-logistic candidate; fixed Phase 6 core untouched."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from src.twin.models import risk_sets, cumulative


def temporal_recovery_rule(hazards, features, attenuation=.65):
    returning = features[:,:7]
    recovery = np.nan_to_num(features[:,-2],nan=0)
    persistence = np.nan_to_num(features[:,-1],nan=0)
    worsening = features[:,-3]
    eligible = (np.sum(returning > .25,axis=1)>=2) & (worsening<=.25) & (persistence<.6)
    factor = 1-attenuation*recovery*(1-persistence)*eligible
    return np.asarray(hazards)*factor[:,None]


class OffsetReliability:
    def __init__(self, l2=.001):
        self.l2 = l2

    def fit(self, hazards, features, labels, roles):
        if set(roles) != {'train'}:
            raise PermissionError('Reliability fitting is TRAIN-only')
        self.median = np.array([np.median(a[np.isfinite(a)]) if np.isfinite(a).any() else 0 for a in features.T])
        filled = np.where(np.isfinite(features), features, self.median)
        self.scale = np.maximum(filled.std(0), 0.01)
        x = self.transform(features)
        self.theta = []
        risk = risk_sets(labels)
        for j in range(3):
            keep = risk[:, j]
            a = x[keep]
            y = labels[keep, j]
            offset = logit(np.clip(hazards[keep, j], 1e-6, 1-1e-6))
            def objective(theta):
                z = offset + a @ theta
                residual = expit(z) - y
                penalty = self.l2 * np.dot(theta[1:], theta[1:]) / 2
                grad = a.T @ residual / len(y)
                grad[1:] += self.l2 * theta[1:]
                return float(np.mean(np.logaddexp(0, z)-y*z) + penalty), grad
            result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-11})
            if not result.success:
                raise ValueError('Reliability optimization did not converge')
            self.theta.append(result.x)
        return self

    def transform(self, features):
        missing = ~np.isfinite(features)
        x = np.clip((np.where(missing, self.median, features) - self.median) / self.scale, -6, 6)
        return np.c_[np.ones(len(x)), x, missing.astype(float)]

    def predict(self, hazards, features):
        x = self.transform(features)
        return expit(logit(np.clip(hazards, 1e-6, 1-1e-6)) + np.stack([x @ t for t in self.theta], 1))


class HorizonCalibration:
    """Choose a method per conditional interval, retaining monotonic cumulative risks."""
    def fit(self, hazards, labels, roles, methods):
        if set(roles) != {'calibration'}:
            raise PermissionError('Calibration fitting needs disjoint validation-calibration patients')
        self.methods = list(methods)
        self.models = []
        risk = risk_sets(labels)
        for j, method in enumerate(methods):
            keep = risk[:, j]
            p = np.clip(hazards[keep, j], 1e-6, 1-1e-6)
            y = labels[keep, j]
            if method == 'existing':
                model = None
            elif method == 'isotonic':
                model = IsotonicRegression(out_of_bounds='clip').fit(p, y)
            elif method == 'platt':
                x = np.c_[logit(p), np.ones(len(p))]
                def objective(theta):
                    z = x @ theta
                    return float(np.mean(np.logaddexp(0,z)-y*z) + 1e-5*(theta[0]-1)**2), x.T @ (expit(z)-y)/len(y) + [2e-5*(theta[0]-1), 0]
                fit = minimize(objective, [1,0], jac=True, bounds=[(.001,10),(-20,20)], method='L-BFGS-B')
                if not fit.success:
                    raise ValueError('Recalibration failed')
                model = fit.x
            else:
                raise ValueError('Unsupported bounded calibration method')
            self.models.append(model)
        return self

    def transform(self, hazards):
        p = np.clip(hazards, 1e-6, 1-1e-6)
        out = []
        for j, (method, model) in enumerate(zip(self.methods, self.models)):
            out.append(p[:,j] if method == 'existing' else model.predict(p[:,j]) if method == 'isotonic' else expit(model[0]*logit(p[:,j])+model[1]))
        return np.stack(out, 1)


def select_calibration(hazards, labels, cal_mask, selection_mask):
    # Only three candidates; fixed interval-order selection on selection Brier.
    methods = ['existing'] * 3
    trials = {}
    truth = np.cumsum(labels[selection_mask], axis=1)
    for j in range(3):
        values = {}
        for method in ['existing', 'platt', 'isotonic']:
            candidate = methods.copy()
            candidate[j] = method
            model = HorizonCalibration().fit(hazards[cal_mask], labels[cal_mask], ['calibration'], candidate)
            p = cumulative(model.transform(hazards[selection_mask]))
            values[method] = float(np.mean((p[:,j]-truth[:,j])**2))
        best = min(values.values())
        # Prefer simpler choices when Brier improvement is <=1e-6.
        methods[j] = next(m for m in ['existing', 'platt', 'isotonic'] if values[m] <= best + 1e-6)
        trials[str((j+1)*24)] = values
    return methods, trials
