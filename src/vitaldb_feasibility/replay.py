"""Small read-only replay->canonical observation proof; no UI or clinical service."""
from dataclasses import asdict
import datetime as dt
import json,math,gzip
from src.data.schema import Observation,Provenance,SourceValue
from src.vitaldb_feasibility.source import ROOT,decode,sha,private_json
MAP={'Solar8000/HR':('heart_rate','bpm'),'Solar8000/ART_MBP':('invasive_map','mmHg'),
     'Solar8000/ART_SBP':('invasive_systolic_bp','mmHg'),'Solar8000/ART_DBP':('invasive_diastolic_bp','mmHg'),
     'Solar8000/PLETH_SPO2':('spo2','%'),'SNUADC/ECG_II':('ecg_lead_ii','mV'),
     'SNUADC/PLETH':('ppg_waveform','unitless'),'SNUADC/ART':('arterial_pressure_waveform','mmHg')}


def observations(payload,response_sha256,received_utc):
    resources=[e['resource'] for e in payload.get('entry',[])] if payload.get('resourceType')=='Bundle' else [payload]
    result=[]
    for resource in resources:
        code=next((c['code'] for c in resource.get('code',{}).get('coding',[]) if c.get('code') in MAP),None)
        if code is None:continue
        sampled=resource['valueSampledData']
        if sampled.get('dimensions')!=1:raise ValueError('Unsupported multidimensional SampledData')
        period=float(sampled['period'])/1000
        if period<=0:raise ValueError('Invalid sampled period')
        origin=sampled.get('origin',{});factor=float(sampled.get('factor',1));bias=float(origin.get('value',0))
        start=dt.datetime.fromisoformat(resource['effectivePeriod']['start'].replace('Z','+00:00'))
        end=dt.datetime.fromisoformat(resource['effectivePeriod']['end'].replace('Z','+00:00'))
        subject=resource['subject']['reference'].split('/')[-1];encounter=resource['encounter']['reference'].split('/')[-1]
        concept,unit=MAP[code];original_unit=origin.get('unit') or None
        for i,token in enumerate(sampled.get('data','').split()):
            time=start+dt.timedelta(seconds=i*period)
            if time>end:raise ValueError('Sample crosses declared replay window')
            value=None
            try:
                raw=float(token);value=raw*factor+bias
                if not math.isfinite(value):value=None
            except ValueError:raw=token
            reasons=['historical_surgical_replay','virtual_clock_not_clinical_receipt_time']
            if period>=1:reasons.append('replay_numeric_resampling_not_raw_measurement_cadence')
            if original_unit is None:reasons.append('unit_from_official_track_catalog_native_unit_missing')
            status='SUSPECT' if value is not None else 'MISSING'
            provenance=Provenance('vitaldb','pilot/replay_sample.response',resource['id']+':'+str(i),code,concept,
                       'physical=raw*factor+origin; replay clock preserved; documented catalog units',response_sha256,original_unit,'documented_catalog')
            source=SourceValue({'token':token,'factor':factor,'origin':origin},value,status,reasons,unit,provenance)
            record=Observation('vitaldb:case:'+encounter+':'+code+':'+time.isoformat(),'vitaldb:subject:'+subject,'vitaldb',concept,
                               time.isoformat(),'virtual_replay_utc',received_utc,'audit_ingestion_received_utc',source,
                               metadata={'case_id':encounter,'sampling_period_seconds':period,'replay_extensions':resource.get('extension',[]),'clinical_use_validated':False})
            result.append(asdict(record))
    return result


def proof():
    path=ROOT/'pilot/replay_sample.response';body=path.read_bytes();provenance=json.loads(path.with_suffix(path.suffix+'.provenance.json').read_text())
    rows=observations(json.loads(decode(body)),sha(body),provenance['retrieved_utc'])
    dest=ROOT/'pilot/canonical_replay_observations.jsonl.gz'
    canonical=''.join(json.dumps(row,allow_nan=False,sort_keys=True)+'\n' for row in rows).encode()
    dest.write_bytes(gzip.compress(canonical,mtime=0))
    dest.chmod(0o600)
    old=dest.with_suffix('')
    if old.exists():old.unlink()  # Own redundant proof output only, never previous-phase records.
    report={'canonical_records':len(rows),'concepts':sorted({r['concept'] for r in rows}),'source_response_sha256':sha(body),
            'canonical_sha256':sha(canonical),'cache_sha256':sha(dest.read_bytes()),'canonical_cache_bytes':dest.stat().st_size,
            'schema_version':'2.0.0 dataclass contract; separate VitalDB audit adapter, Phase2 registry unchanged',
            'future_UI_built':False,'clinical_live_feed':False,'public_replay_proven':True,
            'label_construction_uses_replay':False,'numeric_warning':'1Hz replay samples may be resampled; use original numeric CSV timestamps for labels and rate/missingness'}
    private_json(ROOT/'replay_proof.json',report);print(report);return report

if __name__=='__main__':proof()
