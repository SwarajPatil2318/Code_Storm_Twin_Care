"""Reproduce bounded metadata discovery, label-free sampling and replay input."""
import argparse,json,time,urllib.parse
from src.vitaldb_feasibility.source import ROOT,get,private_json,metadata_report,cases,tracks,select_cases,sha
from src.vitaldb_feasibility.replay import MAP

ENDPOINTS={'cases':'cases','trks':'trks','labs':'labs','openapi':'openapi.json','api_guide':'AGENTS.md',
           'capability':'fhir/metadata','locations':'fhir/Location?physicalType=ro'}

def metadata():
    manifest_path=ROOT/'metadata/retrieval_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name,endpoint in ENDPOINTS.items():
        path=ROOT/'metadata'/(name+'.response');provenance=path.with_suffix(path.suffix+'.provenance.json')
        if name in manifest and path.exists() and not provenance.exists():
            if sha(path.read_bytes())!=manifest[name]['sha256']:raise ValueError('Original metadata checksum mismatch')
            private_json(provenance,manifest[name])
        _,record=get(endpoint,path,max_bytes=32*2**20);manifest[name]=record
    private_json(manifest_path,manifest);metadata_report()
    cohort=cases();catalog={}
    for row in tracks():catalog.setdefault(row['caseid'],set()).add(row['tname'])
    names={'SNUADC/ART','SNUADC/ECG_II','SNUADC/PLETH'}
    selected=select_cases([c for c in cohort.values() if names<=catalog.get(c['caseid'],set())],12)
    private_json(ROOT/'waveform_selection.json',{'seed':2026,'cases':selected,'track_names':sorted(names),'selection_uses_outcomes':False})
    print({'metadata_tables':len(manifest),'waveform_cases_selected':len(selected)})

def replay():
    at='2000-01-01T00:30:00Z'
    get('fhir/Location/OR1/$status?'+urllib.parse.urlencode({'at':at}),ROOT/'pilot/replay_status.response')
    query=urllib.parse.urlencode({'at':at,'window':10,'code':','.join(MAP)})
    get('fhir/Location/OR1/$sample?'+query,ROOT/'pilot/replay_sample.response',max_bytes=2**20)
    print({'replay_window_seconds':10,'requested_concepts':len(MAP)})

def clinical():
    encounters=[];total=None
    for offset in range(0,6400,200):
        body,_=get('fhir/Encounter?'+urllib.parse.urlencode({'_count':200,'_offset':offset}),
                   ROOT/'clinical_fhir'/('encounters_'+str(offset)+'.response'),max_bytes=2*2**20)
        bundle=json.loads(body);total=bundle['total'];encounters.extend(e['resource'] for e in bundle.get('entry',[]))
        if len(encounters)>=total:break
        time.sleep(.2)
    if len(encounters)!=total:raise ValueError('Incomplete encounter pagination')
    if len({r['id'] for r in encounters})!=total:raise ValueError('Duplicate paginated encounter')
    enriched=[r for r in encounters if any(t.get('code')=='inspire-clinical' for t in r.get('meta',{}).get('tag',[]))]
    sample=sorted(enriched,key=lambda r:sha(('2026:'+r['id']).encode()))[:6];counts={};case_reports=[]
    for encounter in sample:
        one={'case_id':encounter['id'],'resources':{}}
        for kind in ['Procedure','Condition','MedicationAdministration']:
            data,_=get('fhir/'+kind+'?'+urllib.parse.urlencode({'encounter':encounter['id']}),
                       ROOT/'clinical_fhir'/encounter['id']/(kind+'.response'),max_bytes=2*2**20)
            resources=[e['resource'] for e in json.loads(data).get('entry',[])];fields=set().union(*(r.keys() for r in resources)) if resources else set()
            one['resources'][kind]={'records':len(resources),'fields':sorted(fields),
                 'timed_records':sum(any(k in r for k in ['effectiveDateTime','effectivePeriod','onsetDateTime','performedDateTime','performedPeriod']) for r in resources)}
            counts[kind]=counts.get(kind,0)+len(resources);time.sleep(.2)
        case_reports.append(one)
    private_json(ROOT/'clinical_fhir_audit.json',{'encounters':total,'enriched_cases':len(enriched),
        'enriched_patients':len({r['subject']['reference'] for r in enriched}),'sampled_enriched_cases':len(sample),
        'sample_resource_counts':counts,'sample':case_reports,'receipt_time_verified':False,'sample_does_not_establish_all_linked_resource_coverage':True})
    print({'encounters':total,'inspire_enriched_cases':len(enriched),'sampled_enriched_cases':len(sample),'resource_counts':counts})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['metadata','replay','clinical']);args=parser.parse_args()
    {'metadata':metadata,'replay':replay,'clinical':clinical}[args.mode]()
