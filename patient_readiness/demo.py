"""Explicit isolated engineering history acceleration; no readiness override."""
from datetime import timedelta
from patient_connections.contracts import CanonicalEvent


def accelerate(readiness,connections,pid,minutes,acknowledge):
    if acknowledge is not True or type(minutes) is not int or minutes not in [0,30,45,105,120,3000]:raise ValueError('Choose and acknowledge an engineering history stage')
    if not readiness.repo.get_patient(pid)['synthetic']:raise ValueError('Engineering history requires an explicitly synthetic patient')
    current=connections.status(pid,'CGM')
    if current['status'] in ['NOT_CONNECTED','DISCONNECTED','ERROR']:
        connections.connect(pid,'CGM',{'scenario':'stable','acknowledge_synthetic':True});connections.activate(pid,'CGM')
    wall=readiness.time()
    # Latest reading must be within the runtime's 7.5-minute freshness gate.
    # End is rounded to a minute, with a patient-specific persistent anchor for native cadence.
    with readiness.repo.connection() as db:
        row=db.execute("SELECT data FROM observations WHERE patient_id=? AND json_extract(data,'$.source_provenance.canonical_event.provenance.origin')='P5_explicit_engineering_acceleration' ORDER BY json_extract(data,'$.observed_time') DESC LIMIT 1",(pid,)).fetchone()
    if row:
        import json
        from patient_connections.contracts import clock
        anchor=clock(json.loads(row[0])['observed_time']);elapsed=int((wall-anchor).total_seconds()//900)
        end=anchor+timedelta(minutes=15*max(0,elapsed))
    else:end=wall.replace(microsecond=0)
    if minutes:
        events=[]
        for offset in range(minutes,-1,-15):
            observed=end-timedelta(minutes=offset)
            events.append(CanonicalEvent(patient_id=pid,source_type='CGM',provider_type='replay_adapter',event_type='CGM_READING',
                observed_at=observed.isoformat(),received_at=wall.isoformat(),sequence_id='p5-history-'+observed.isoformat(),
                payload={'glucose_mg_dl':110,'timestamp':observed.isoformat()},quality={'status':'GOOD'},
                provenance={'origin':'P5_explicit_engineering_acceleration','synthetic':True,'native_period_minutes':15,
                    'original_unit':'mg/dL','dataset':'P5_generated_engineering_history','no_cohort_identity_link':True}).view(as_of=wall))
        for start in range(0,len(events),100):connections.ingest(pid,'CGM',events[start:start+100])
    return readiness.evaluate(pid)
