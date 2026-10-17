"""Causal personal response expectation, separate from calibrated D2 forecasts."""
import json
from pathlib import Path
import numpy as np
from src.metabolic.windows import MINUTE,nearest_indices

CONFIG=json.loads(Path('configs/metabolic_d3.json').read_text())
HORIZONS=np.array([0,30,60,90,120])

class Shadow:
    def predict(self,buffer,cutoff=None):
        cutoff=buffer.as_of if cutoff is None else cutoff
        t,g,_,_=buffer.glucose(cutoff)
        prefix=(t<cutoff-CONFIG['shadow_embargo_minutes']*MINUTE)&(t>=cutoff-CONFIG['shadow_lookback_days']*1440*MINUTE)
        tt,values=t[prefix],g[prefix]
        if len(values)<CONFIG['shadow_minimum_observations'] or not len(t) or cutoff-t[0]<50*60*MINUTE:
            return {'available':False,'trajectory':None,'method':'Insufficient causal personal history','support':{'observations':len(values)}}
        median=float(np.median(values));sd=float(np.std(values));shift=buffer.clock_offset_minutes*MINUTE
        expected=[];support=[]
        minute=(tt+shift)//MINUTE%1440;days=(tt+shift)//(1440*MINUTE)
        for horizon in HORIZONS:
            wanted=((cutoff+shift)//MINUTE+int(horizon))%1440
            distance=np.abs(minute-wanted);use=np.minimum(distance,1440-distance)<=CONFIG['shadow_hour_radius_minutes']
            count=int(use.sum());day_count=len(np.unique(days[use]));supported=count>=CONFIG['shadow_hour_minimum_samples'] and day_count>=CONFIG['shadow_hour_minimum_days']
            expected.append(float(np.median(values[use])) if supported else median)
            support.append({'samples':count,'days':day_count,'fallback_to_prefix_median':not supported})
        method='Causal personal time-of-day medians; prefix-median fallback'
        meals=list({(m.observed_ns,(m.context or {}).get('meal_type')):m for m in buffer.visible(cutoff,'meal')}.values())
        meals.sort(key=lambda m:m.observed_ns)
        eligible=[m for m in meals if 0<=cutoff-m.observed_ns<=CONFIG['shadow_maximum_meal_age_minutes']*MINUTE]
        analogs=[]
        if eligible:
            latest=eligible[-1];age=cutoff-latest.observed_ns
            now_type=(latest.context or {}).get('meal_type')
            for past in meals:
                if past.observed_ns>=latest.observed_ns:continue
                if now_type is not None and (past.context or {}).get('meal_type')!=now_type:continue
                wanted=past.observed_ns+age+HORIZONS*MINUTE
                indices,found=nearest_indices(tt,wanted,7.5*MINUTE)
                if not found.all() or (np.diff(indices)<=0).any():continue
                start,end=indices[0],indices[-1]
                if (np.diff(tt[start:end+1])>22.5*MINUTE).any():continue
                baseline_indices,baseline_found=nearest_indices(tt,np.array([past.observed_ns-15*MINUTE]),7.5*MINUTE)
                if not baseline_found[0]:continue
                analogs.append(values[indices]-values[baseline_indices[0]])
            premeal_idx,premeal_found=nearest_indices(t,np.array([latest.observed_ns-15*MINUTE]),7.5*MINUTE)
            if len(analogs)>=CONFIG['shadow_meal_minimum_episodes'] and premeal_found[0]:
                expected=(g[premeal_idx[0]]+np.median(analogs,axis=0)).tolist()
                method='Current observed pre-meal baseline plus median completed historical '+('same-type' if now_type is not None else 'untyped')+' meal response'
        return {'available':True,'trajectory':expected,'method':method,'median':median,'sd':sd,
            'support':{'observations':len(values),'hour_slots':support,'completed_meal_analogs':len(analogs)},
            'prefix_end_exclusive':int(cutoff-CONFIG['shadow_embargo_minutes']*MINUTE),
            'scope':'Personal expectation from received history; not a causal meal intervention or independently validated forecast'}


def divergence(actual,predicted,shadow):
    if not shadow['available']:return {'magnitude':None,'label':'Personal history unavailable','components':None}
    today=np.r_[actual,predicted];expected=np.asarray(shadow['trajectory']);delta=today-expected
    scale=max(shadow['sd'],CONFIG['divergence_sd_floor_mg_dl']);magnitude=float(np.mean(np.abs(delta))/scale)
    boundaries=CONFIG['divergence_display_boundaries']
    label='Near personal pattern' if magnitude<boundaries[0] else 'Different from personal pattern' if magnitude<boundaries[1] else 'Above personal pattern' if delta.mean()>0 else 'Below personal pattern'
    return {'magnitude':magnitude,'label':label,'components':{'signed_deviation_mg_dl':delta.tolist(),
        'mean_absolute_deviation_mg_dl':float(np.abs(delta).mean()),'standardized_mean_absolute_deviation':magnitude,
        'absolute_area_mg_dl_minutes':float(np.trapezoid(np.abs(delta),HORIZONS)),
        'excess_peak_mg_dl':float(max(predicted)-max(expected[1:])),
        'excess_maximum_rise_mg_dl_per_minute':float((np.diff(today)/30).max()-(np.diff(expected)/30).max()),
        'normalization_sd_mg_dl':scale},'scope':'Descriptive display heuristic; not calibrated clinical risk'}
