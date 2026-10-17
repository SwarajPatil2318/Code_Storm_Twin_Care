"""Prespecified observed-history risk sets; future data used only for labels/censoring."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from src.vitaldb_feasibility.source import ROOT,private_json,describe,cases
from src.vitaldb_feasibility.protocol import PROTOCOL


def coverage(summary,start,end):
    if start<0 or end>=len(summary['count']) or end<start:return False
    indexes=np.arange(int(np.ceil(start)),int(np.floor(end))+1)
    valid=(summary['count'][indexes]>=15)&(summary['max_gap_seconds'][indexes]<=10)
    if not len(valid):return False
    gaps=np.diff(np.r_[-1,np.flatnonzero(valid),len(valid)])-1
    return valid, float(np.mean(valid)),int(max(gaps))


def supported(summary,start,end,fraction):
    result=coverage(summary,start,end)
    return result is not False and result[1]>=fraction and result[2]<=2


def followup_supported(summary,start,end):
    observed=summary['latest_raw_seconds'];finite=observed[np.isfinite(observed)]
    return bool(len(finite) and max(finite)>=end*60 and supported(summary,start,end,.9))


def asof(summary,t0):
    # All cached bins close at their index; latest raw offset remains explicit.
    if t0<0:return None,None
    end=min(int(np.floor(t0)),len(summary['latest'])-1)
    valid=np.flatnonzero(np.isfinite(summary['latest'][:end+1]))
    if not len(valid):return None,None
    i=valid[-1];return float(summary['latest'][i]),float(summary['latest_raw_seconds'][i]/60)


def stable_state(summary,t0,lookback):
    cutoff=t0-5
    if not supported(summary,cutoff-lookback,cutoff,.8):return None,'insufficient_history'
    if not supported(summary,cutoff-120,cutoff,.8):return None,'insufficient_observed_washout'
    w=summary['minimum'][int(cutoff-120):int(cutoff)+1]
    if any(w[np.isfinite(w)]<65):return None,'recent_hypotension'
    latest,last_time=asof(summary,t0);buffered,_=asof(summary,cutoff)
    if latest is None or buffered is None:return None,'missing_current'
    minimum=None;trajectory=None
    if supported(summary,cutoff-60,cutoff,.8):
        indexes=np.arange(int(cutoff-60),int(cutoff)+1);values=summary['median'][indexes];valid=np.isfinite(values)
        minimum=float(np.nanmin(summary['minimum'][indexes]));trajectory=float(np.polyfit(indexes[valid]/60,values[valid],1)[0])
    normal80=latest>=80 and buffered>=80 and t0-last_time<=1
    state={'washed_out':True,'normal80':bool(normal80),
           'hard':bool(normal80 and minimum is not None and minimum>=75 and trajectory is not None and trajectory>=-1),
           'normal90':bool(latest>=90 and buffered>=90 and t0-last_time<=1 and minimum is not None and minimum>=80),
           'map_at_t0':latest,'current_raw_minute':last_time,'buffered_latest_map':buffered,'minimum_last60':minimum,
           'slope_last60_mmHg_per_hour':trajectory,'feature_cutoff':cutoff}
    return state,None


def episodes(events):
    result=[]
    for e in events:
        if result and e['onset_seconds']-result[-1]['last_low_seconds']<=7200:
            result[-1]['last_low_seconds']=e['last_low_seconds'];result[-1]['runs']+=1
        else:result.append({**e,'runs':1})
    return result


def causal_features(state):
    """T0 MAP is an audit stratum only; predictor fields stop at the 5-minute buffer."""
    return {k:state[k] for k in ['buffered_latest_map','minimum_last60','slope_last60_mmHg_per_hour','feature_cutoff']}


def coarse_events(summary):
    """Explicit cadence sensitivity; not evidence of continuously low native MAP."""
    grid=np.arange(0,len(summary['latest']),5);value=summary['latest'][grid].copy()
    stale=(grid-summary['latest_raw_seconds'][grid]/60)>1/6
    value[stale]=np.nan
    from src.vitaldb_feasibility.numeric import dense_events
    return dense_events(grid*60,value,max_gap=300,duration=1800)


def risk_sets(summary,events,case,history,horizon,band=False):
    eventlist=episodes(events);rows=[];excluded=Counter();seen={s:set() for s in PROTOCOL['stability']}
    first=max(history,120)+5
    for t0 in range(first,int(float(case['caseend'])/60)+1,30):
        excluded['candidate_landmarks']+=1;state,reason=stable_state(summary,t0,history)
        if reason:excluded[reason]+=1;continue
        if any(e['onset_seconds']<=t0*60<=e['last_low_seconds'] for e in eventlist):excluded['event_underway']+=1;continue
        if not followup_supported(summary,t0,t0+horizon+31):excluded['incomplete_future_censored']+=1;continue
        future=[(i,e) for i,e in enumerate(eventlist) if e['onset_seconds']>t0*60]
        future.sort(key=lambda ie:ie[1]['onset_seconds']);index,event=future[0] if future else (None,None)
        lead=(event['onset_seconds']/60-t0) if event is not None else None
        if lead is not None and lead<=15:excluded['event_within_minimum_lead']+=1;continue
        lower=max(15,horizon-30) if band else 15
        if band and lead is not None and lead<=lower:excluded['earlier_first_event_before_band']+=1;continue
        label=int(lead is not None and lead<=horizon)
        if label and event['confirmation_seconds']>(t0+horizon+31)*60:
            excluded['confirmation_outside_followup_censored']+=1;continue
        excluded['eligible_landmarks']+=1
        for stability in PROTOCOL['stability']:
            if not state[stability]:continue
            if label and index in seen[stability]:continue
            if label:seen[stability].add(index)
            rows.append({'case_id':case['caseid'],'patient_id':'vitaldb:subject:'+case['subjectid'],
                         'history_minutes':history,'horizon_minutes':horizon,'lead_band':band,'stability':stability,
                         't0_minutes':t0,'label':label,'event_index':index if label else None,
                         'lead_minutes':lead if label else None,'event_onset_minutes':event['onset_seconds']/60 if label else None,
                         'event_confirmation_minutes':event['confirmation_seconds']/60 if label else None,
                         'causal_features':causal_features(state),'audit_stability_state':state})
    return rows,dict(excluded)


def counts(rows):
    positive=[r for r in rows if r['label']];controls=[r for r in rows if not r['label']]
    return {'cases':len({r['case_id'] for r in rows}),'patients':len({r['patient_id'] for r in rows}),
            'positive_cases':len({r['case_id'] for r in positive}),'positive_patients':len({r['patient_id'] for r in positive}),
            'control_cases':len({r['case_id'] for r in controls}),'control_patients':len({r['patient_id'] for r in controls}),
            'windows':len(rows),'positive_windows':len(positive),'control_windows':len(controls),'window_prevalence':len(positive)/len(rows) if rows else None,
            'eligible_unique_events':len({(r['case_id'],r['event_index']) for r in positive}),'lead_minutes':describe([r['lead_minutes'] for r in positive]),
            'windows_not_independent':True}


def event_diagnostic():
    """Exact-horizon attrition check; not a relaxed or optimized target definition."""
    cohort=cases();events=[];stats={};identities={}
    for path in (ROOT/'numeric/map').glob('*.json'):
        meta=json.loads(path.read_text());case=cohort[meta['case_id']]
        if float(case['age'])<18:continue
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:summary={k:z[k] for k in z.files}
        for event in episodes(meta['events']):
            events.append(event)
            for history in [30,60,180,360]:
                for horizon in [30,60,120,180]:
                    key=f'history{history}_horizon{horizon}';x=stats.setdefault(key,Counter());x['episodes']+=1
                    groups=identities.setdefault(key,{'supported':set(),'normal80':set(),'patients':set()})
                    t0=event['onset_seconds']/60-horizon;cutoff=t0-5
                    if t0<0 or cutoff-history<0:x['insufficient_elapsed_history']+=1;continue
                    if not supported(summary,cutoff-history,cutoff,.8):x['history_qc_failure']+=1;continue
                    if not followup_supported(summary,t0,t0+horizon+31) or event['confirmation_seconds']>(t0+horizon+31)*60:
                        x['future_qc_or_end_censored']+=1;continue
                    x['history_and_future_supported']+=1;groups['supported'].add(case['caseid']);groups['patients'].add(case['subjectid'])
                    current,observed=asof(summary,t0)
                    if current is not None and current>=80 and t0-observed<=1:
                        x['current_normal80_no_washout_assurance']+=1;groups['normal80'].add(case['caseid'])
                    state,reason=stable_state(summary,t0,history)
                    if reason:x[reason]+=1
                    else:
                        x['prespecified_washedout_eventcentred']+=1
                        for stable in ['normal80','hard','normal90']:x[stable]+=int(state[stable])
    for k,groups in identities.items():
        stats[k].update(history_future_supported_cases=len(groups['supported']),history_future_supported_patients=len(groups['patients']),
                        current_normal80_cases_no_washout_assurance=len(groups['normal80']))
    report={'adult_episode_onset_minutes':describe([e['onset_seconds']/60 for e in events]),
            'event_centred_exact_horizon_diagnostic':{k:dict(v) for k,v in stats.items()},'diagnostic_not_new_primary_definition':True}
    private_json(ROOT/'event_attrition_diagnostic.json',report);return report


def analyze():
    cohort=cases();case_reports=[];results={};windows=[];censor={}
    for path in sorted((ROOT/'numeric/map').glob('*.json')):
        meta=json.loads(path.read_text());case=cohort[meta['case_id']]
        try:adult=float(case['age'])>=18
        except ValueError:adult=False
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:summary={k:archive[k] for k in archive.files}
        minute_valid=(summary['count']>=15)&(summary['max_gap_seconds']<=10)
        case_reports.append({'case_id':meta['case_id'],'patient_id':meta['patient_id'],'adult':adult,'native_event_count':len(meta['events']),
                            'episode_count':len(episodes(meta['events'])),'event_duration_minutes':describe([e['duration_minutes'] for e in meta['events']]),
                            'event_durations':[e['duration_minutes'] for e in meta['events']],
                            'usable_minutes':int(sum(minute_valid)),'usable_fraction_minutes':float(np.mean(minute_valid)),
                            'coarse5minute_event_count':len(coarse_events(summary)),'qc':meta['qc']})
        if not adult:continue
        for history in PROTOCOL['history_sensitivity_minutes']:
            for horizon in PROTOCOL['horizons_minutes']:
                for band in [False,True]:
                    rows,excluded=risk_sets(summary,meta['events'],case,history,horizon,band)
                    key=f'history{history}_horizon{horizon}_'+('band' if band else 'cumulative')
                    censor.setdefault(key,Counter()).update(excluded)
                    for stability in PROTOCOL['stability']:
                        selected=[r for r in rows if r['stability']==stability];results.setdefault(key+'_'+stability,[]).extend(selected)
                    windows.extend(rows)
    private_json(ROOT/'case_event_audit.json',case_reports)
    path=ROOT/'risk_windows.jsonl'
    with path.open('w') as f:
        for row in windows:f.write(json.dumps(row,allow_nan=False,sort_keys=True)+'\n')
    path.chmod(0o600)
    report={'protocol':PROTOCOL,'retrieved_map_cases':len(case_reports),'retrieved_map_patients':len({r['patient_id'] for r in case_reports}),
            'adult_cases':sum(r['adult'] for r in case_reports),'native_event_positive_cases':sum(bool(r['native_event_count']) for r in case_reports),
            'native_event_positive_patients':len({r['patient_id'] for r in case_reports if r['native_event_count']}),
            'native_event_count':sum(r['native_event_count'] for r in case_reports),'merged_episode_count':sum(r['episode_count'] for r in case_reports),
            'risk_sets':{k:counts(v) for k,v in results.items()},'censoring':{k:dict(v) for k,v in censor.items()},
            'all_window_count':len(windows),'overlapping_control_windows':'30minute landmarks; correlated within patients/cases, never independent sample counts',
            'models_trained':0,'eicu_test_accessed':False}
    for name,population in [('all_ages',case_reports),('adults',[r for r in case_reports if r['adult']])]:
        positive=[r for r in population if r['native_event_count']]
        usable=[r for r in population if r['usable_minutes']>=30 and r['usable_fraction_minutes']>=.8]
        report[name]={'cases':len(population),'patients':len({r['patient_id'] for r in population}),
                      'event_positive_cases':len(positive),'event_positive_patients':len({r['patient_id'] for r in positive}),
                      'no_observed_event_cases':len(population)-len(positive),'native_events':sum(r['native_event_count'] for r in population),
                      'merged_episodes':sum(r['episode_count'] for r in population),'recurrent_case_count':sum(r['episode_count']>1 for r in population),
                      'case_event_prevalence':len(positive)/len(population) if population else None,
                      'event_duration_minutes':describe([v for r in population for v in r['event_durations']]),
                      'coarse5minute_events':sum(r['coarse5minute_event_count'] for r in population),
                      'usable_map_cases_30minutes_80percent':len(usable),'usable_map_patients_30minutes_80percent':len({r['patient_id'] for r in usable}),
                      'usable_minutes':describe([r['usable_minutes'] for r in population]),
                      'no_observed_event_is_not_uncensored_negative':True}
    private_json(ROOT/'feasibility_report.json',report);event_diagnostic()
    print({k:report[k] for k in ['retrieved_map_cases','adult_cases','native_event_positive_cases','native_event_count','all_window_count','models_trained','eicu_test_accessed']});return report

if __name__=='__main__':analyze()
