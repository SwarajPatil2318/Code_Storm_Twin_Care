"""Twenty-one small causal reliability features from cached received bins.

No episode labels, future rows, or simulator-private state enters this layer.
Missing bins remain missing; differences require adjacent observed bins.
"""
import numpy as np
from src.twin.protocol import PRIMARY_SIGNALS

SIGNALS = ['heart_rate', 'hrv_rmssd', 'steps', 'systolic_bp', 'weight', 'spo2', 'sleep_hours']
NAMES = [s + '__return_toward_baseline' for s in SIGNALS] + [s + '__recent_abs_z' for s in SIGNALS] + [
    'deviating_bin_fraction', 'consecutive_deviating_bins', 'deviating_systems',
    'multisystem_agreement', 'worsening_fraction', 'recovery_score', 'persistence_score']


def _mean(values, axis):
    good = np.isfinite(values)
    total = np.where(good, values, 0).sum(axis=axis)
    n = good.sum(axis=axis)
    return np.divide(total, n, out=np.full_like(total, np.nan, dtype=float), where=n > 0)


def recovery_features(data):
    ix = [PRIMARY_SIGNALS.index(s) for s in SIGNALS]
    sequence = np.asarray(data['sequence'])[:, -8:, ix, 0].astype(float)
    # Cached references are causal as-of-current-cutoff summaries. Using them
    # here describes change within the history, not a forecast issued earlier.
    ref = np.asarray(data['reference'])[:, ix]
    scale = np.asarray(data['scale'])[:, ix]
    z = (sequence - ref[:, None, :]) / np.maximum(scale[:, None, :], 1e-6)
    absolute = np.abs(z)
    old = _mean(absolute[:, :4], 1)
    current = _mean(absolute[:, 4:], 1)
    returning = old - current
    observed = np.isfinite(absolute)
    dev = observed & (absolute >= 1.5)
    fraction = dev.sum((1, 2)) / np.maximum(observed.sum((1, 2)), 1)
    bin_dev = dev.sum(2) >= 2
    streak = np.zeros(len(sequence))
    active = np.ones(len(sequence), bool)
    for b in range(7, -1, -1):
        active &= bin_dev[:, b]
        streak += active
    systems = (np.isfinite(current) & (current >= 1.5)).sum(1)
    valid = np.isfinite(current).sum(1)
    agree = systems / np.maximum(valid, 1)
    worsening = np.sum(returning < -0.25, 1) / np.maximum(np.isfinite(returning).sum(1), 1)
    recovery = _mean(np.clip(returning / 2, 0, 1), 1)
    persistence = .5 * fraction + .5 * streak / 8
    extra = np.c_[fraction, streak, systems, agree, worsening, recovery, persistence]
    return np.c_[returning, current, extra].astype(np.float32)


def scores(features):
    return {'recovery_score': np.nan_to_num(features[:, -2], nan=0), 'persistent_deterioration_score': features[:, -1]}
