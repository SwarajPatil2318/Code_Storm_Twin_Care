"""Measured numeric coverage, synchronized overlap, and clinical metadata aggregates."""
from collections import Counter,defaultdict
import json
import numpy as np
from src.vitaldb_feasibility.source import ROOT,cases,tracks,csv_rows,cached_metadata,describe,private_json
from src.vitaldb_feasibility.retrieve import TRACKS

COMBINATIONS={'A_MAP':['map'],'B_MAP_SBP_DBP':['map','sbp','dbp'],'C_BP_HR':['map','sbp','dbp','hr'],
              'D_BP_HR_SpO2':['map','sbp','dbp','hr','spo2'],'E_BP_HR_SpO2_RR':['map','sbp','dbp','hr','spo2','rr']}

def usable_minutes(summary,cadence):
    cadence=cadence or 2
    # Accommodate the documented 7s respiratory numeric source without imposing 2s cadence on it.
    minimum=max(1,int(np.ceil(60/(2*cadence))));gap=max(10,3*cadence)
    return (summary['count']>=minimum)&(summary['max_gap_seconds']<=gap)

def analyze():
    cohort=cases();track_index=defaultdict(set)
    for row in tracks():track_index[row['caseid']].add(row['tname'])
    selected=set(json.loads((ROOT/'multisignal_selection.json').read_text())['case_ids'])
    percase=defaultdict(dict);summaries={};report={}
    for concept in TRACKS:
        metadata=[];qualities=[]
        for path in sorted((ROOT/'numeric'/concept).glob('*.json')):
            meta=json.loads(path.read_text());metadata.append(meta)
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:summary={k:z[k] for k in z.files}
            interval=meta['qc']['observed_sampling_interval_seconds'];cadence=interval['median'] if interval else 2
            mask=usable_minutes(summary,cadence);fraction=float(np.mean(mask));n=int(sum(mask))
            item={'meta':meta,'mask':mask,'summary':summary,'usable':n>=30 and fraction>=.8,'usable_minutes':n,'usable_fraction':fraction}
            percase[meta['case_id']][concept]=item
            qualities.append(item)
        usable=[r for r in qualities if r['usable']]
        source_cases={i for i,names in track_index.items() if any(n in names for n in TRACKS[concept])}
        fields=['input_records','unique_records','missing_values','impossible_values','identical_duplicate_records','conflicting_timestamp_groups','usable_records',
                'rapid_pressure_jumps_flagged_not_removed','flatline_runs_at_least5min_flagged_not_removed','invalid_timestamp_records','source_negative_offsets']
        qc={k:sum(r['qc'][k] for r in metadata) for k in fields}
        report[concept]={'catalog_cases':len(source_cases),'catalog_patients':len({cohort[i]['subjectid'] for i in source_cases}),
                         'retrieved_cases':len(metadata),'retrieved_patients':len({r['patient_id'] for r in metadata}),
                         'retrieved_is_entire_catalog_union':len(metadata)==len(source_cases),
                         'usable_cases_30minutes_80percent':len(usable),'usable_patients_30minutes_80percent':len({r['meta']['patient_id'] for r in usable}),
                         'observed_span_minutes':describe([r['qc']['observed_span_minutes'] for r in metadata]),
                         'usable_minutes':describe([r['usable_minutes'] for r in qualities]),'usable_minute_fraction':describe([r['usable_fraction'] for r in qualities]),
                         'sample_rate_hz':describe([r['qc']['sampling_rate_hz_median'] for r in metadata if r['qc']['sampling_rate_hz_median'] is not None]),
                         'qc_totals':qc,'qc_case_flags':{k:sum(r['qc'][k]>0 for r in metadata) for k in ['impossible_values','rapid_pressure_jumps_flagged_not_removed','flatline_runs_at_least5min_flagged_not_removed']},
                         'missing_values_fraction_unique_records':qc['missing_values']/qc['unique_records'] if qc['unique_records'] else None,
                         'impossible_values_fraction_unique_records':qc['impossible_values']/qc['unique_records'] if qc['unique_records'] else None}
    overlaps={};quality_alignment=[]
    for name,concepts in COMBINATIONS.items():
        available=[i for i in sorted(selected) if all(c in percase[i] for c in concepts)]
        usable=[];common_durations=[];offsets=[]
        for caseid in available:
            size=len(percase[caseid]['map']['mask']);joint=np.ones(size,dtype=bool)
            latest_times=[]
            for concept in concepts:
                item=percase[caseid][concept];aligned=np.zeros(size,bool);n=min(size,len(item['mask']));aligned[:n]=item['mask'][:n];joint&=aligned
                t=np.full(size,np.nan);t[:n]=item['summary']['latest_raw_seconds'][:n];latest_times.append(t)
            common_durations.append(int(sum(joint)))
            if sum(joint)>=30 and np.mean(joint)>=.8:usable.append(caseid)
            if sum(joint):
                matrix=np.stack(latest_times);offsets.extend((np.max(matrix[:,joint],axis=0)-np.min(matrix[:,joint],axis=0)).tolist())
        catalog=[i for i,names in track_index.items() if all(any(n in names for n in TRACKS[c]) for c in concepts)]
        overlaps[name]={'selected_cases':len(selected),'retrieved_all_required_cases':len(available),'usable_cases':len(usable),
                        'usable_patients':len({cohort[i]['subjectid'] for i in usable}),'catalog_cases_upper_bound':len(catalog),
                        'catalog_patients_upper_bound':len({cohort[i]['subjectid'] for i in catalog}),
                        'common_usable_minutes':describe(common_durations),'latest_time_spread_seconds_common_minutes':describe(offsets),
                        'definition':'at least30 common usable minutes and >=80% of MAP cached minute grid; same recording-offset clock, no fill',
                        'whole_cohort_usable_count_is_lower_bound':True}
    for caseid in sorted(selected):
        if not all(c in percase[caseid] for c in ['map','sbp','dbp']):continue
        items=[percase[caseid][c] for c in ['map','sbp','dbp']];size=min(len(r['mask']) for r in items)
        joint=np.logical_and.reduce([r['mask'][:size] for r in items]);ts=np.stack([r['summary']['latest_raw_seconds'][:size] for r in items])
        joint&=(np.max(ts,axis=0)-np.min(ts,axis=0))<=5
        if not any(joint):continue
        mv,sv,dv=[r['summary']['latest'][:size][joint] for r in items]
        quality_alignment.append({'case_id':caseid,'checked_minutes':int(sum(joint)),'bp_ordering_inconsistent_minutes':int(sum((sv<mv)|(mv<dv)))})
    private_json(ROOT/'bp_alignment_audit.json',quality_alignment)
    result={'signals':report,'overlaps':overlaps,'selected_multisignal_cases':len(selected),
            'selected_multisignal_patients':len({cohort[i]['subjectid'] for i in selected}),
            'selection':'1200 outcome-independent hashed cases plus all >=575minute cases; catalog upper bounds never substituted for measured usable counts',
            'bp_ordering_checked_minutes':sum(r['checked_minutes'] for r in quality_alignment),
            'bp_ordering_inconsistent_minutes':sum(r['bp_ordering_inconsistent_minutes'] for r in quality_alignment),
            'synchronization_limit':'shared recording timestamps; no independent hardware clock or pulse transit verification; minute summaries cannot validate waveform phase alignment'}
    private_json(ROOT/'signal_audit.json',result)
    print({'numeric_concepts':len(report),'measured_multisignal_cases':len(selected),'overlaps':{k:(v['usable_cases'],v['usable_patients']) for k,v in overlaps.items()}})
    return result

def clinical_audit():
    cohort=cases();labs=csv_rows(cached_metadata('labs'),['caseid','dt','name','result']);by_name=Counter();timing=Counter();subjects=set();case_ids=set();invalid=0
    for row in labs:
        if row['caseid'] not in cohort:raise ValueError('Unresolved laboratory encounter')
        by_name[row['name']]+=1;case_ids.add(row['caseid']);subjects.add(cohort[row['caseid']]['subjectid'])
        try:
            offset=float(row['dt'])
            if not np.isfinite(offset):raise ValueError()
            label='before_recording' if offset<0 else 'during_recording' if offset<=float(cohort[row['caseid']]['caseend']) else 'after_recording'
            timing[label]+=1
        except ValueError:timing['unknown_offset']+=1
        try:
            if not np.isfinite(float(row['result'])):invalid+=1
        except ValueError:invalid+=1
    fhir_path=ROOT/'clinical_fhir_audit.json'
    enrichment=json.loads(fhir_path.read_text())['enriched_cases'] if fhir_path.exists() else None
    result={'lab_records':len(labs),'lab_names':dict(sorted(by_name.items())),'lab_cases':len(case_ids),'lab_patients':len(subjects),
            'relative_recording_timing':dict(timing),'nonnumeric_or_missing_results':invalid,
            'lab_units_in_csv':False,'lab_clinical_receipt_time_supplied':False,
            'preop_columns':[k for k in next(iter(cohort.values())) if k.startswith('preop_')],
            'future_blocked_columns':[k for k in next(iter(cohort.values())) if k.startswith('intraop_') or k in ['icu_days','death_inhosp','dis','aneend','opend','caseend']],
            'fhir_inspire_linked_cases':enrichment,'fhir_enrichment_count_kind':'measured Encounter tags when available; does not establish resource completeness or HF labels'}
    private_json(ROOT/'clinical_audit.json',result);print({'lab_records':len(labs),'lab_names':len(by_name),'lab_cases':len(case_ids),'relative_timing':dict(timing)})
    return result

if __name__=='__main__':clinical_audit();analyze()
