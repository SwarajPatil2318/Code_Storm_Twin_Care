"""Local HTTP benchmark on existing synthetic demo streams only."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np


def benchmark(root='http://127.0.0.1:8097'):
    directory=Path('data/metabolic_d41');directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    timings=[];sample=None
    original=json.loads(Path('data/metabolic_d3/demo.json').read_text())
    expected={case['case']:case['timeline'] for case in original['cases']}
    def post(action,payload):
        start=time.perf_counter()
        request=Request(root+'/metabolic-api/metabolic/live/'+action,
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urlopen(request,timeout=20) as response:result=json.load(response)
        if 'update' in result:
            timings.append({**result['update']['timings_ms'],'http_round_trip_ms':(time.perf_counter()-start)*1000})
        return result
    for case in ['predicted_spike','normal_response','personally_atypical']:
        result=post('session',{'case_id':case,'start_index':0,'mode':'HISTORICAL / DEMO REPLAY'})
        assert result['state']==expected[case][0], 'Runtime state mismatch; private values omitted'
        try:
            while not result['complete']:
                result=post('advance',{'session_id':result['session_id'],'expected_revision':result['update']['revision']})
                assert result['state']==expected[case][result['index']], 'Runtime state mismatch; private values omitted'
            if case=='predicted_spike':sample=result
        finally:post('close',{'session_id':result['session_id']})
    summary={'samples':len(timings),'all_75_states_match_verified_cache':True,
        'stream_scope':'All 75 original synthetic D3 demo frames; no research cohort reads',
        'metrics_ms':{name:{'median':float(np.median([row[name] for row in timings])),
                             'p95':float(np.quantile([row[name] for row in timings],.95))} for name in timings[0]}}
    with urlopen(root+'/metabolic-api/metabolic/live/performance',timeout=10) as response:
        summary['startup_model_load_ms']=json.load(response)['startup_model_load_ms']
    incoming={'case_id':'predicted_spike','source':'cgm',
              'event_time':sample['update']['input']['event_time'],
              'receipt_time':sample['update']['input']['receipt_time'],
              'glucose_mg_dl':sample['state']['current_glucose']}
    for name,value in [('latency_aggregate.json',summary),('sample_runtime_response.json',sample),
                       ('sample_incoming_event.json',incoming)]:
        path=directory/name;path.write_text(json.dumps(value,indent=2)+'\n');path.chmod(0o600)
    print(json.dumps(summary))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='http://127.0.0.1:8097')
    benchmark(parser.parse_args().root)
