"""Canonical JSONL ingestion and deterministic cached simulated-live replay.

No commercial device transport is implemented. The JSONL adapter consumes the
explicit two-clock Event contract emitted by the four source adapters.
"""
import argparse,json
from pathlib import Path
from .events import Event
from .engine import TwinEngine
from .reference import ROOT,Reference

class CanonicalStreamAdapter:
    def observations(self,lines):
        for line in lines:
            if line.strip():yield Event(**json.loads(line))

def ingest_stream(patient,lines,reference=None):
    engine=TwinEngine(patient,reference)
    for event in CanonicalStreamAdapter().observations(lines):
        state=engine.ingest(event)
        if event.concept=='glucose':yield state

def cached_replay(case):
    """Emit an initial native 120m window followed by CGM-triggered cached states.

    Personal-prefix history is already incorporated during offline preparation;
    no model or feature calculation runs in this replay/browser path.
    """
    from .events import timestamp
    first=case['timeline'][0];cutoff=timestamp(first['timestamp'])[0]
    history=[e for e in case['events'] if e['concept']=='glucose'
        and cutoff-120*60_000_000_000<=timestamp(e['observed_time'])[0]<=cutoff
        and timestamp(e['available_time'])[0]<=cutoff]
    yield {'type':'initial_window','label':'SIMULATED LIVE STREAM','history':history,'state':first}
    for state in case['timeline'][1:]:
        yield {'type':'cgm_update','label':'SIMULATED LIVE STREAM','state':state}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['predicted_spike','normal_response','personally_atypical'],default='predicted_spike')
    parser.add_argument('--output',type=Path,required=True,help='Private local JSONL destination; no individual states are printed to logs')
    parser.add_argument('--recompute',action='store_true',help='Replay canonical synthetic events through frozen D2; otherwise use precomputed states')
    args=parser.parse_args()
    demo=json.loads((ROOT/'demo.json').read_text());case=next(c for c in demo['cases'] if c['case']==args.case)
    frames=ingest_stream(case['patient'],(json.dumps(e) for e in case['events']),Reference()) if args.recompute else cached_replay(case)
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with args.output.open('w') as stream:
        args.output.chmod(0o600)
        for item in frames:stream.write(json.dumps(item,allow_nan=False)+'\n')
    print(json.dumps({'synthetic_replay_exported':args.case,'recomputed':args.recompute,'path':str(args.output)}))

if __name__=='__main__':main()
