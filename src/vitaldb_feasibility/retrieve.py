"""Selective numeric-only cohort retrieval. No bulk case or cohort waveform GETs."""
import argparse,concurrent.futures,datetime as dt,gzip,json,threading,time,urllib.request
from pathlib import Path
import numpy as np
from src.vitaldb_feasibility.source import cases,tracks,ROOT,BASE,sha,decode,private_json,select_cases
from src.vitaldb_feasibility.numeric import parse_numeric,qc_numeric,dense_events,minute_summary
MAP_TRACKS=['Solar8000/ART_MBP','Solar8000/FEM_MBP','EV1000/ART_MBP']
TRACKS={'map':MAP_TRACKS,'sbp':['Solar8000/ART_SBP','Solar8000/FEM_SBP'],'dbp':['Solar8000/ART_DBP','Solar8000/FEM_DBP'],
        'hr':['Solar8000/HR'],'spo2':['Solar8000/PLETH_SPO2'],'rr':['Solar8000/RR_CO2','Primus/RR_CO2','Solar8000/RR'],
        'temperature':['Solar8000/BT'],'co':['EV1000/CO','Vigileo/CO','Vigilance/CO','CardioQ/CO'],'cvp':['Solar8000/CVP']}


class Budget:
    def __init__(self,max_bytes):self.maximum=max_bytes;self.used=0;self.reserved=0;self.lock=threading.Lock();self.next=0
    def reserve(self):
        with self.lock:
            if self.used+self.reserved+4*1024*1024+1>self.maximum:raise ValueError('Network budget exhausted before request')
            self.reserved+=4*1024*1024+1
            delay=max(0,self.next-time.monotonic());self.next=max(self.next,time.monotonic())+.2
        if delay:time.sleep(delay)
    def add(self,n):
        with self.lock:
            self.reserved-=4*1024*1024+1
            self.used+=n
            if self.used>self.maximum:raise ValueError('Frozen network-body budget exceeded')


def retrieve_one(case,record,concept,budget):
    root=ROOT/'numeric'/concept;root.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=root/(case['caseid']+'.npz');manifest=path.with_suffix('.json')
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if path.exists() and sha(path.read_bytes())==existing['cache_sha256']:return existing
        raise ValueError('Numeric cache checksum mismatch')
    budget.reserve();url=BASE+record['tid'];req=urllib.request.Request(url,headers={'User-Agent':'TWIN_PATIENT-feasibility-audit/1.0','Accept-Encoding':'identity'})
    try:
        with urllib.request.urlopen(req,timeout=60) as response:
            body=response.read(4*1024*1024+1);headers=dict(response.headers)
    except Exception:
        budget.add(0);raise
    budget.add(len(body))
    if len(body)>4*1024*1024:raise ValueError('Individual numeric response exceeds4MiB limit')
    decoded=decode(body);t,v=parse_numeric(decoded,record['tname']);t,v,qc=qc_numeric(t,v,concept)
    events=dense_events(t,v) if concept=='map' else []
    summary=minute_summary(t,v);np.savez_compressed(path,**summary);path.chmod(0o600)
    result={'case_id':case['caseid'],'patient_id':'vitaldb:subject:'+case['subjectid'],'concept':concept,'track':record['tname'],
            'endpoint':url,'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'body_bytes':len(body),'decoded_bytes':len(decoded),
            'response_sha256':sha(body),'decoded_sha256':sha(decoded),'headers':headers,'cache_sha256':sha(path.read_bytes()),
            'cache_bytes':path.stat().st_size,'qc':qc,'events':events,'raw_response_retained':False,'max_event_gap_seconds':10}
    private_json(manifest,result);return result


def run(concepts=('map',),workers=4,budget_bytes=512*1024*1024,limit=None,case_sample=None):
    cohort=cases();index={}
    for r in tracks():index[(r['caseid'],r['tname'])]=r
    requests=[]; selected=None
    if case_sample is not None:
        candidates=[c for c in cohort.values() if any((c['caseid'],n) in index for n in MAP_TRACKS)]
        # Outcome-independent whole-case sampling plus all cases long enough for primary history/horizon.
        chosen=select_cases(candidates,case_sample)
        selected={c['caseid'] for c in chosen}|{c['caseid'] for c in candidates if float(c['caseend'])>=575*60}
        private_json(ROOT/'multisignal_selection.json',{'seed':2026,'hash_sample_size':case_sample,
                     'long_case_duration_minutes':575,'case_ids':sorted(selected),'selection_uses_outcomes':False})
    for case in cohort.values():
        if selected is not None and case['caseid'] not in selected:continue
        for concept in concepts:
            record=next((index[(case['caseid'],name)] for name in TRACKS[concept] if (case['caseid'],name) in index),None)
            if record:requests.append((case,record,concept))
    # A pilot limit is label-free; a full run has no sample or duration exclusion.
    requests.sort(key=lambda request:sha(('2026:'+request[0]['subjectid']+':'+request[0]['caseid']+':'+request[2]).encode()))
    if limit is not None:requests=requests[:limit]
    budget=Budget(budget_bytes);failures=[];done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(retrieve_one,*request,budget):request for request in requests}
        for future in concurrent.futures.as_completed(futures):
            request=futures[future]
            try:future.result()
            except Exception as e:failures.append({'case_id':request[0]['caseid'],'concept':request[2],'error_type':type(e).__name__})
            done+=1
            if done%100==0:print({'numeric_requests_completed':done,'planned':len(requests),'failures':len(failures),'new_network_body_MiB':round(budget.used/2**20,2)},flush=True)
    report={'planned_requests':len(requests),'completed':done-len(failures),'failures':failures,'new_network_body_bytes':budget.used,'maximum_new_network_body_bytes':budget_bytes,'concepts':list(concepts),'workers':workers,'rate_limit_requests_per_second':5}
    report.update(retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),case_sample=case_sample)
    report_path=ROOT/('retrieval_'+('_'.join(concepts))+'.json')
    history_path=ROOT/('retrieval_history_'+('_'.join(concepts))+'.json')
    history=json.loads(history_path.read_text()) if history_path.exists() else ([json.loads(report_path.read_text())] if report_path.exists() else [])
    history.append(report);private_json(history_path,history)
    private_json(report_path,report);print({'retrieval_complete':True,'requests':len(requests),'failures':len(failures),'new_bytes':budget.used},flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--concepts',nargs='+',choices=sorted(TRACKS),default=['map']);p.add_argument('--workers',type=int,default=4);p.add_argument('--budget-mib',type=int,default=512);p.add_argument('--limit',type=int);p.add_argument('--case-sample',type=int);a=p.parse_args();run(a.concepts,a.workers,a.budget_mib*2**20,a.limit,a.case_sample)
