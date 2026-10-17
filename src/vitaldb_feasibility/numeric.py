"""Raw numeric QC and causal minute summaries; retained labels use dense samples."""
import csv,io
import numpy as np
from src.vitaldb_feasibility.source import describe
SCREENS={'map':(10,200),'sbp':(30,300),'dbp':(10,200),'hr':(20,250),'spo2':(50,100),'rr':(1,80),'temperature':(30,43),'co':(.1,30),'cvp':(-10,50)}


def parse_numeric(body,track):
    reader=csv.reader(io.StringIO(body.decode('utf-8-sig')));header=next(reader)
    if header!=['Time',track]:raise ValueError('Unexpected numeric track schema')
    rows=[]
    for row in reader:
        if len(row)!=2:raise ValueError('Malformed numeric row')
        rows.append([float(x) if x.strip().lower() not in ['','nan'] else np.nan for x in row])
    raw=np.asarray(rows,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=2:raise ValueError('Empty numeric stream')
    return raw[:,0],raw[:,1]


def qc_numeric(t,v,concept):
    t=np.asarray(t,dtype=float);v=np.asarray(v,dtype=float)
    lo,hi=SCREENS[concept];finite=np.isfinite(t);t=t[finite];v=v[finite]
    order=np.argsort(t,kind='stable');t=t[order];v=v[order]
    unique,first,count=np.unique(t,return_index=True,return_counts=True);values=v[first].copy();conflicts=np.zeros(len(unique),dtype=bool)
    for j in np.flatnonzero(count>1):
        group=v[first[j]:first[j]+count[j]]
        if not np.all(group==group[0]):conflicts[j]=True;values[j]=np.nan
    missing=~np.isfinite(values);impossible=np.isfinite(values)&((values<lo)|(values>hi));usable=np.where(missing|impossible|conflicts,np.nan,values)
    dt=np.diff(unique);valid=np.isfinite(usable)
    jump=(dt>0)&(dt<=10)&valid[:-1]&valid[1:]&(np.abs(np.diff(usable))>40) if concept in ['map','sbp','dbp'] else np.zeros(len(dt),dtype=bool)
    flat_durations=[];flat_start=None;flat_last=None;flat_value=None
    for ti,vi in zip(unique,usable):
        continues=np.isfinite(vi) and flat_start is not None and vi==flat_value and ti-flat_last<=10
        if not continues:
            if flat_start is not None and flat_last-flat_start>=300:flat_durations.append(float((flat_last-flat_start)/60))
            flat_start=ti if np.isfinite(vi) else None;flat_value=vi
        flat_last=ti
    if flat_start is not None and flat_last-flat_start>=300:flat_durations.append(float((flat_last-flat_start)/60))
    summary={'flatline_runs_at_least5min_flagged_not_removed':len(flat_durations),'flatline_duration_minutes':describe(flat_durations),'input_records':len(t),'unique_records':len(unique),'missing_values':int(sum(missing)),'impossible_values':int(sum(impossible)),
             'identical_duplicate_records':int(sum(count-1))-int(sum(count[conflicts]-1)),'conflicting_timestamp_groups':int(sum(conflicts)),
             'usable_records':int(sum(valid)),'observed_span_minutes':float((unique[-1]-unique[0])/60) if len(unique)>1 else 0,
             'observed_sampling_interval_seconds':describe(dt[dt>0]),'sampling_rate_hz_median':float(1/np.median(dt[dt>0])) if any(dt>0) else None,
             'rapid_pressure_jumps_flagged_not_removed':int(sum(jump)), 'source_value_distribution':describe(values),
             'invalid_timestamp_records':int(sum(~finite)),'source_negative_offsets':int(sum(unique<0))}
    return unique,usable,summary


def dense_events(t,v,max_gap=10,duration=1800):
    """No interpolation. An onset is confirmed only after30min observed low run."""
    result=[];start=None;last=None;n=0
    def close():
        if start is not None and last-start>=duration and n>=int(duration/max_gap)+1:
            result.append({'onset_seconds':float(start),'last_low_seconds':float(last),'confirmation_seconds':float(confirmation),
                           'duration_minutes':float((last-start)/60),'low_observations':n})
    confirmation=None
    for ti,vi in zip(t,v):
        low=np.isfinite(vi) and vi<65
        if start is not None and (not low or ti-last>max_gap):close();start=None;last=None;n=0;confirmation=None
        if low:
            if start is None:start=ti
            last=ti;n+=1
            if confirmation is None and ti-start>=duration:confirmation=ti
    close();return result


def minute_summary(t,v):
    """Bins close at ceil(raw_seconds/60); a bin never contains a future value."""
    if not len(t):raise ValueError('Empty numeric stream')
    grid=np.arange(0,int(np.ceil(max(t)/60))+1);binned=np.ceil(t/60).astype(int)
    count=np.zeros(len(grid),dtype=np.int32);last=np.full(len(grid),np.nan);last_t=last.copy();med=last.copy();minimum=last.copy();maximum=last.copy();max_gap=last.copy()
    valid=np.isfinite(v)&(t>=0)
    for minute in np.unique(binned[valid]):
        mask=(binned==minute)&valid;times=t[mask];values=v[mask]
        count[minute]=len(values);last[minute]=values[-1];last_t[minute]=times[-1];med[minute]=np.median(values);minimum[minute]=min(values);maximum[minute]=max(values)
        max_gap[minute]=max(np.diff(np.r_[max(0,(minute-1)*60),times,minute*60]))
    return {'time_minutes':grid,'count':count,'latest':last,'latest_raw_seconds':last_t,'median':med,'minimum':minimum,'maximum':maximum,'max_gap_seconds':max_gap}
