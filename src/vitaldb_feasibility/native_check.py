"""Three positive engineering checks, never prevalence sampling or model selection."""
import json
import numpy as np
from src.vitaldb_feasibility.source import ROOT,get,sha,private_json
from src.vitaldb_feasibility.numeric import parse_numeric,qc_numeric,dense_events,minute_summary

def run():
    records=[json.loads(p.read_text()) for p in (ROOT/'numeric/map').glob('*.json')]
    chosen=sorted([r for r in records if r['events']],key=lambda r:sha(('2026:'+r['case_id']).encode()))[:3]
    checks=[]
    for record in chosen:
        endpoint=record['endpoint'].split('/')[-1]
        body,provenance=get(endpoint,ROOT/'native_checks'/(record['case_id']+'.response'),max_bytes=4*2**20)
        if provenance['sha256']!=record['response_sha256']:raise ValueError('Numeric source snapshot changed')
        t,v=parse_numeric(body,record['track']);t,v,qc=qc_numeric(t,v,'map')
        if qc!=record['qc'] or dense_events(t,v)!=record['events']:raise ValueError('Raw-native labels or QC did not reproduce')
        summary=minute_summary(t,v)
        with np.load(ROOT/'numeric/map'/(record['case_id']+'.npz')) as archive:
            for key in archive.files:np.testing.assert_equal(summary[key],archive[key])
        for event in record['events']:
            mask=t<=event['confirmation_seconds'];prefix=dense_events(t[mask],v[mask])
            matches=[e for e in prefix if e['onset_seconds']==event['onset_seconds']]
            if len(matches)!=1 or matches[0]['confirmation_seconds']!=event['confirmation_seconds']:
                raise ValueError('Future extension changed confirmed event onset/confirmation')
        checks.append({'case_id':record['case_id'],'events_checked':len(record['events']),
                       'source_body_sha256':provenance['sha256'],'qc_exact':True,'minute_arrays_exact':True,'native_events_exact':True,'prefix_confirmations_exact':True})
    report={'status':'PASS','engineering_cases':len(checks),'events_checked':sum(r['events_checked'] for r in checks),
            'selection':'hash-selected among positives solely to verify target implementation; never used for coverage or model selection',
            'checks':checks,'models_trained':0}
    private_json(ROOT/'native_check_report.json',report);print({k:report[k] for k in ['status','engineering_cases','events_checked','models_trained']});return report

if __name__=='__main__':run()
