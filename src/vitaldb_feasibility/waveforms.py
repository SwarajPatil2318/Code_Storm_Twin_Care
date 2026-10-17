"""Twelve outcome-independent cases, one bounded native waveform chunk per track."""
import datetime as dt
import json,re,time,urllib.parse
import numpy as np
from src.vitaldb_feasibility.source import ROOT,get,private_json,describe,cases,tracks

def decode_sampled(resource):
    sampled=resource['valueSampledData']
    if sampled.get('dimensions')!=1 or float(sampled['period'])<=0:raise ValueError('Unsupported waveform shape')
    origin=sampled.get('origin',{});factor=float(sampled.get('factor',1));bias=float(origin.get('value',0))
    values=[]
    for token in sampled.get('data','').split():
        try:values.append(float(token)*factor+bias)
        except ValueError:values.append(np.nan)
    return np.asarray(values),float(sampled['period'])/1000,origin.get('unit'),factor,bias


def summarize(reports):
    cohort=cases();catalog={}
    for row in tracks():catalog.setdefault(row['caseid'],set()).add(row['tname'])
    required={'SNUADC/ART','SNUADC/ECG_II','SNUADC/PLETH'};ids=[i for i,names in catalog.items() if required<=names]
    usable=[r for r in reports if all(t['finite_samples']/t['samples']>=.9 and t['screened_values']/t['samples']<=.1 for t in r['tracks'])]
    report={'catalog_cases_upper_bound':len(ids),'catalog_patients_upper_bound':len({cohort[i]['subjectid'] for i in ids}),
        'catalog_with_map_union_upper_bound':sum(any(n in catalog[i] for n in ['Solar8000/ART_MBP','Solar8000/FEM_MBP','EV1000/ART_MBP']) for i in ids),
        'sampled_cases':len(reports),'snippet_screen_usable_cases':len(usable),'snippet_screen_usable_patients':len({r['patient_id'] for r in usable}),
        'snippet_screen_definition':'three60second separate-midpoint chunks >=90% finite and <=10% diagnostic ART/ECG screen flags; not fullcase waveform quality or temporal phase validation',
        'common_metadata_span_minutes':describe([r['common_track_span_minutes'] for r in reports]),
        'selected_chunk_start_spread_seconds':describe([r['selected_chunk_start_spread_seconds'] for r in reports])}
    private_json(ROOT/'waveform_overlap.json',report);return report

def run():
    selection=json.loads((ROOT/'waveform_selection.json').read_text());reports=[]
    for case in selection['cases']:
        caseid=case['caseid'];base=ROOT/'waveforms'/caseid
        query=urllib.parse.urlencode({'encounter':caseid,'category':'vital-signs','_summary':'true','_count':200})
        data,_=get('fhir/Observation?'+query,base/'summary.response',max_bytes=2**20)
        listing=json.loads(data)
        if len(listing.get('entry',[]))<listing.get('total',0):raise ValueError('Waveform metadata requires pagination')
        summaries={e['resource']['code']['coding'][0]['code']:e['resource'] for e in listing.get('entry',[])}
        rows=[]
        for code in selection['track_names']:
            record=summaries[code];period=record['effectivePeriod']
            start=dt.datetime.fromisoformat(period['start'].replace('Z','+00:00'));end=dt.datetime.fromisoformat(period['end'].replace('Z','+00:00'))
            midpoint=start+(end-start)/2
            query=urllib.parse.urlencode({'encounter':caseid,'code':code,'date':'ge'+midpoint.isoformat(),'_count':1})
            data,provenance=get('fhir/Observation?'+query,base/(code.split('/')[-1]+'.response'),max_bytes=2*2**20)
            payload=json.loads(data)
            if not payload.get('entry'):raise ValueError('Waveform chunk missing')
            resource=payload['entry'][0]['resource'];values,interval,unit,factor,bias=decode_sampled(resource)
            finite=np.isfinite(values);span=(end-start).total_seconds()/60
            screened=(values<0)|(values>300) if code=='SNUADC/ART' else (np.abs(values)>10) if code=='SNUADC/ECG_II' else np.zeros(len(values),bool)
            rows.append({'track':code,'track_span_minutes':span,'period_seconds':interval,'rate_hz':1/interval,
                         'chunk_start':resource['effectivePeriod']['start'],'chunk_end':resource['effectivePeriod']['end'],
                         'samples':len(values),'finite_samples':int(sum(finite)),'missing_samples':int(sum(~finite)),
                         'screened_values':int(sum(finite&screened)),'screen_note':'ART outside[0,300]mmHg / ECG abs>10mV diagnostic screen; PPG has no validated bound',
                         'constant_adjacent_fraction':float(np.mean(np.diff(values[finite])==0)) if sum(finite)>1 else None,
                         'native_unit':unit,'factor':factor,'origin':bias,'physical_distribution':describe(values),
                         'response_sha256':provenance['sha256'],'network_bytes':provenance['bytes']})
            time.sleep(.2)
        starts=[dt.datetime.fromisoformat(r['chunk_start'].replace('Z','+00:00')).timestamp() for r in rows]
        # Independently chosen track midpoints need not be the same recording time.
        periods=[summaries[c]['effectivePeriod'] for c in selection['track_names']]
        common_start=max(dt.datetime.fromisoformat(p['start'].replace('Z','+00:00')).timestamp() for p in periods)
        common_end=min(dt.datetime.fromisoformat(p['end'].replace('Z','+00:00')).timestamp() for p in periods)
        reports.append({'case_id':caseid,'patient_id':'vitaldb:subject:'+case['subjectid'],'tracks':rows,
                        'common_track_span_minutes':max(0,(common_end-common_start)/60),'selected_chunk_start_spread_seconds':max(starts)-min(starts)})
    private_json(ROOT/'waveform_audit.json',reports)
    summarize(reports)
    print({'waveform_cases':len(reports),'waveform_chunks':sum(len(r['tracks']) for r in reports),
           'sampled_waveforms_not_cohort_quality_estimate':True})
    return reports

if __name__=='__main__':run()
