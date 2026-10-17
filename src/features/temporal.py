"""Causal reference summaries; current-time batches never enter their own history."""
from collections import defaultdict
import datetime as dt
import math
import statistics as stats

TEMPORAL_FEATURES = ('rolling_mean_7d','rolling_median_7d','rolling_std_7d','slope_per_day_7d',
                     'prior_baseline_median_30d','deviation_from_prior_baseline',
                     'recent_vs_baseline_delta','prior_count_7d','prior_count_30d',
                     'current_missing','current_invalid','time_since_last_usable_days')


def time_number(value, basis):
    if value is None:
        return None
    if basis == 'relative_source_day':
        return float(value)
    if basis == 'absolute_utc':
        return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()/86400
    raise ValueError('Unsupported temporal basis')


def slope(points):
    if len(points) < 2:
        return None
    times, values = zip(*points)
    mean_t, mean_v = stats.mean(times), stats.mean(values)
    denom = sum((t-mean_t)**2 for t in times)
    return sum((t-mean_t)*(v-mean_v) for t,v in points)/denom if denom > 0 else None


def causal_features(observations):
    """Use no labels, no population fitting, and strictly prior eligible measurements.

    Unknown source availability may use only the declared offline capture assumption
    for CHIRON/PerHeart. Known late availability is always respected.
    """
    groups = defaultdict(list)
    for item in observations:
        if item.dataset not in {'chiron','perheart'}:
            continue
        if item.available_time is None and not item.metadata.get('retrospective_feature_policy'):
            raise ValueError('Unknown availability requires an explicit retrospective feature policy')
        if item.value.usable_value is not None and (not isinstance(item.value.usable_value,(int,float)) or not math.isfinite(item.value.usable_value)):
            raise ValueError('Temporal values must be finite numeric measurements')
        if item.observed_time is not None and item.value.quality_status != 'DUPLICATE':
            groups[(item.patient_id,item.concept,item.time_basis)].append(item)
    for (patient,concept,basis), items in sorted(groups.items()):
        if len({item.value.unit for item in items}) > 1:
            raise ValueError('Temporal group has incompatible units; convert with provenance before use')
        items.sort(key=lambda item:(time_number(item.observed_time,basis),item.observation_id))
        batches = defaultdict(list)
        for item in items:
            batches[time_number(item.observed_time,basis)].append(item)
        history = []
        for cutoff,current in sorted(batches.items()):
            usable_history = []
            for prior in history:
                value = prior.value.usable_value
                observed = time_number(prior.observed_time,basis)
                available = time_number(prior.available_time,basis) if prior.available_time is not None else observed
                if value is not None and observed < cutoff and available <= cutoff:
                    usable_history.append((observed,float(value),prior.observation_id))
            recent = [(t,v) for t,v,_ in usable_history if cutoff-7 <= t < cutoff]
            baseline = [(t,v) for t,v,_ in usable_history if cutoff-30 <= t < cutoff]
            values = [v for _,v in recent]
            baseline_median = stats.median(v for _,v in baseline) if baseline else None
            current_values = [float(item.value.usable_value) for item in current if item.value.usable_value is not None
                              and (item.available_time is None or time_number(item.available_time,basis) <= cutoff)]
            current_value = stats.median(current_values) if current_values else None
            mean = stats.mean(values) if values else None
            selected = [(t,obsid) for t,_,obsid in usable_history if cutoff-30<=t<cutoff]
            yield {'patient_id':patient,'dataset':current[0].dataset,'concept':concept,
                   'timestamp':current[0].observed_time,'time_basis':basis,
                   'unit':current[0].value.unit,'unit_support':current[0].value.provenance.unit_support,
                   'availability_policy':'known_available_time_else_explicit_retrospective_capture_assumption',
                   'current_observation_ids':[item.observation_id for item in current],
                   'current_value':current_value,
                   'features':{
                       'rolling_mean_7d':mean,
                       'rolling_median_7d':stats.median(values) if values else None,
                       'rolling_std_7d':stats.stdev(values) if len(values)>1 else None,
                       'slope_per_day_7d':slope(recent),
                       'prior_baseline_median_30d':baseline_median,
                       'deviation_from_prior_baseline':current_value-baseline_median if current_value is not None and baseline_median is not None else None,
                       'recent_vs_baseline_delta':mean-baseline_median if mean is not None and baseline_median is not None else None,
                       'prior_count_7d':len(recent),'prior_count_30d':len(baseline),
                       'current_missing':not bool(current_values),
                       'current_invalid':any(item.value.quality_status=='INVALID' for item in current),
                       'time_since_last_usable_days':cutoff-max(t for t,_,_ in usable_history) if usable_history else None},
                   'provenance':{'transformation':'strict_prior_calendar_window_v1',
                                 'prior_observation_ids':[obsid for _,obsid in selected],
                                 'max_prior_observed_time':max(t for t,_ in selected) if selected else None,
                                 'cutoff_in_days':cutoff,'windows_in_days':[7,30],
                                 'baseline_uses_current_observation':False}}
            # Append the whole timestamp batch only after computing its prior reference.
            history.extend(current)
