"""Bounded, resumable public GETs with private provenance and gzip handling."""
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import numpy as np
BASE='https://api.vitaldb.net/'
ROOT=Path('data/vitaldb')


def sha(body):return hashlib.sha256(body).hexdigest()

def private_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(path)


def decode(body,max_decoded_bytes=64*2**20):
    if body[:2]==b'\x1f\x8b':
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:decoded=stream.read(max_decoded_bytes+1)
    else:decoded=body
    if len(decoded)>max_decoded_bytes:raise ValueError('Decoded response exceeds64MiB safety limit')
    return decoded


def cached_metadata(name):
    path=ROOT/'metadata'/(name+'.response');return decode(path.read_bytes())


def csv_rows(body,required=()):
    reader=csv.DictReader(io.StringIO(body.decode('utf-8-sig')))
    if not reader.fieldnames or not set(required)<=set(reader.fieldnames):raise ValueError('API CSV schema changed')
    if len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError('Duplicate metadata column')
    rows=list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):raise ValueError('Malformed metadata row width')
    return rows


def get(endpoint,destination,max_bytes=16*1024*1024):
    """A response limit is enforced while reading; no unbounded bulk request."""
    path=Path(destination);manifest=path.with_suffix(path.suffix+'.provenance.json')
    if path.exists() and manifest.exists():
        meta=json.loads(manifest.read_text());body=path.read_bytes()
        if meta['sha256']!=sha(body):raise ValueError('Cached API response changed')
        if meta['endpoint']!=BASE+endpoint:raise ValueError('Cached endpoint differs from requested source')
        return decode(body),meta
    if endpoint.startswith(('http:','https:')) or '..' in endpoint:raise ValueError('Only relative public VitalDB API routes permitted')
    url=BASE+endpoint;req=urllib.request.Request(url,headers={'User-Agent':'TWIN_PATIENT-feasibility-audit/1.0','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=60) as response:
        body=response.read(max_bytes+1)
        if len(body)>max_bytes:raise ValueError('Response exceeds frozen byte limit')
        meta={'endpoint':url,'retrieved_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'bytes':len(body),
              'sha256':sha(body),'status':response.status,'headers':dict(response.headers)}
    decoded=decode(body);meta.update(decoded_bytes=len(decoded),decoded_sha256=sha(decoded))
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);path.write_bytes(body);path.chmod(0o600);private_json(manifest,meta)
    return decoded,meta


def cases():
    rows=csv_rows(cached_metadata('cases'),['caseid','subjectid','caseend'])
    if len({r['caseid'] for r in rows})!=len(rows):raise ValueError('Duplicate case identity')
    return {r['caseid']:r for r in rows}


def tracks():return csv_rows(cached_metadata('trks'),['caseid','tname','tid'])


def select_cases(rows,limit=None):
    """Deterministic label-free sample: hash subject then case; repeated patients identified."""
    result=sorted(rows,key=lambda r:(sha(('2026:'+r['subjectid']).encode()),sha(('2026:'+r['caseid']).encode())))
    return result if limit is None else result[:limit]


def describe(values):
    values=np.asarray([v for v in values if np.isfinite(v)],dtype=float)
    return None if not len(values) else {'n':len(values),'min':float(min(values)), 'q25':float(np.quantile(values,.25)),
              'median':float(np.median(values)),'q75':float(np.quantile(values,.75)),'q95':float(np.quantile(values,.95)),'max':float(max(values))}


def metadata_report():
    cohort=cases();trackrows=tracks();names=sorted({r['tname'] for r in trackrows});coverage={n:sorted({r['caseid'] for r in trackrows if r['tname']==n}) for n in names}
    duration=[(float(r['caseend'])-float(r['casestart']))/60 for r in cohort.values()]
    result={'cases':len(cohort),'patients':len({r['subjectid'] for r in cohort.values()}),'track_rows':len(trackrows),'distinct_track_names':len(names),
            'duration_minutes':describe(duration),'duration_at_least_hours':{str(h):sum(v>=h*60 for v in duration) for h in [1,2,3,4,6,9]},
            'clinical_columns':list(next(iter(cohort.values()))),'clinical_nonmissing':{k:sum(r[k].strip().lower() not in ['','nan','none'] for r in cohort.values()) for k in next(iter(cohort.values()))},
            'tracks':{n:{'cases':len(ids),'patients':len({cohort[i]['subjectid'] for i in ids}),'case_duration_minutes':describe([(float(cohort[i]['caseend'])-float(cohort[i]['casestart']))/60 for i in ids])} for n,ids in coverage.items()},
            'track_duration_note':'case duration is an upper bound; actual usable observation duration not yet measured',
            'case_ids':'private metadata only','models_trained':0,'eicu_test_accessed':False}
    private_json(ROOT/'metadata_audit.json',result);return result


if __name__=='__main__':
    result=metadata_report();print({k:result[k] for k in ['cases','patients','track_rows','distinct_track_names','duration_minutes','duration_at_least_hours','models_trained','eicu_test_accessed']})
