"""Observed-only forecasting feasibility and explicit causal feature gates.

Times are integer nanoseconds. Counts assume instantaneous observation availability
unless an actual receipt is supplied; this assumption must be reported separately.
No resampling, interpolation, fitting, row splits, or outcome-based feature selection.
"""
from dataclasses import dataclass
import numpy as np

MINUTE = 60_000_000_000
HORIZONS = (30, 60, 90, 120)
HISTORIES = (30, 60, 120, 180)

@dataclass
class Series:
    participant_id: str
    dataset: str
    diabetes_status: str
    time: np.ndarray
    glucose: np.ndarray
    period: float
    quality: dict
    availability: np.ndarray | None = None


def prepare(time, values, period):
    """Remove unusable values explicitly; collapse identical duplicates, quarantine conflicts."""
    time = np.asarray(time, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    good = (time != np.iinfo(np.int64).min) & np.isfinite(values) & (values > 0)
    qc = {'rows': len(time), 'unusable_rows': int((~good).sum()),
          'out_of_40_400_rows': int((good & ((values < 40) | (values > 400))).sum()),
          'backward_intervals_original': int((np.diff(time[good]) < 0).sum())}
    t, v = time[good], values[good]
    order = np.argsort(t, kind='stable'); t, v = t[order], v[order]
    unique, first, counts = np.unique(t, return_index=True, return_counts=True)
    lo = np.minimum.reduceat(v, first) if len(v) else np.array([])
    hi = np.maximum.reduceat(v, first) if len(v) else np.array([])
    conflict = lo != hi
    qc.update(duplicate_extra_rows=int((counts-1).sum()), conflicting_timestamps=int(conflict.sum()))
    t, v = unique[~conflict], v[first[~conflict]]
    delta = np.diff(t) / MINUTE
    qc['interval_bins'] = {name: int(mask.sum()) for name, mask in {
        '<native/2': delta < period/2,
        'native/2..1.5native': (delta >= period/2) & (delta <= 1.5*period),
        '1.5native..60m': (delta > 1.5*period) & (delta <= 60),
        '60m..24h': (delta > 60) & (delta <= 1440),
        '>24h': delta > 1440}.items()}
    qc['interval_minutes_quantiles'] = np.quantile(delta, [0,.5,.9,.99,1]).tolist() if len(delta) else []
    qc['usable_observations'] = len(t)
    qc['span_days'] = float((t[-1]-t[0])/(MINUTE*1440)) if len(t)>1 else 0
    # Duration excludes unobserved gaps; an equivalent day is not a calendar day.
    qc['observed_equivalent_days'] = len(t)*period/1440
    if len(t):
        dates = t // (MINUTE*1440)
        _, per_day = np.unique(dates, return_counts=True)
        qc['calendar_days_any'] = len(per_day)
        qc['calendar_days_70pct'] = int((per_day >= .7*1440/period).sum())
    else: qc.update(calendar_days_any=0, calendar_days_70pct=0)
    return t, v, qc


def nearest_indices(t, wanted, tolerance):
    right = np.searchsorted(t, wanted)
    r = np.minimum(right, len(t)-1); l = np.maximum(right-1, 0)
    idx = np.where(np.abs(t[l]-wanted) <= np.abs(t[r]-wanted), l, r)
    return idx, np.abs(t[idx]-wanted) <= tolerance


def window_masks(s, history):
    """History and future targets use actual time, never sample-index arithmetic."""
    t = s.time; n = len(t)
    if not n: return np.zeros(0,bool), {h:np.zeros(0,bool) for h in HORIZONS}, {h:np.zeros(0,int) for h in HORIZONS}
    tolerance = s.period * MINUTE / 2
    start, close = nearest_indices(t, t-history*MINUTE, tolerance)
    # Require full nominal history length (half-period tolerance only at its start).
    breaks = np.r_[0, np.cumsum((np.diff(t) > 1.5*s.period*MINUTE) |
                                (np.diff(t) < .5*s.period*MINUTE))]
    valid = close & (breaks == breaks[start]) & (start < np.arange(n))
    if s.availability is not None:
        # No observation in a history may arrive after this landmark.
        for i in np.flatnonzero(valid):
            if np.max(s.availability[start[i]:i+1]) > t[i]: valid[i] = False
    masks, targets = {}, {}
    for h in HORIZONS:
        idx, found = nearest_indices(t, t+h*MINUTE, tolerance)
        masks[h] = valid & found & (idx > np.arange(n)) & (breaks[idx] == breaks)
        targets[h] = idx
    return valid, masks, targets


def eligible_features(observation_time, availability_time, cutoff):
    """Unknown receipt means unavailable, including meals, insulin and activity."""
    if observation_time is None or availability_time is None: return False
    return observation_time <= cutoff and availability_time <= cutoff


def assert_participant_isolation(assignments, provenance_groups=None):
    """Raw/harmonized identities share one group; datasets otherwise remain separate."""
    seen = {}
    for participant, split in assignments:
        group = (provenance_groups or {}).get(participant, participant)
        if group in seen and seen[group] != split:
            raise ValueError('Participant/provenance group crosses splits')
        seen[group] = split


def training_scaler(train_rows, train_participants):
    """Diagnostic contract: rejects accidental evaluation-subject fitting."""
    if any(pid not in train_participants for pid, value in train_rows):
        raise ValueError('Preprocessing fit contains non-training participants')
    x = np.array([v for _,v in train_rows],float)
    return float(np.mean(x)), float(np.std(x))
