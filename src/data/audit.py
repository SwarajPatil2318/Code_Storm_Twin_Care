"""Read-only, standard-library audit of the supplied archives and CSV files.

Run: python3 -m src.data.audit --input Datasets --output data/audit/report.json
No records, identifiers, names, or free-text questionnaire answers are emitted.
"""
import argparse
import collections as C
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as S
import zipfile
from urllib.parse import unquote

MISSING = {'', '?', 'NA', 'NaN', 'nan', 'null', 'None'}
ZIGONG = 'hospitalized-patients-with-heart-failure-integrating-electronic-healthcare-records-and-external-outcome-data-1.2.zip'


def number(v):
    try:
        n = float(v.replace(',', '.')) if isinstance(v, str) else float(v)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def ident(v):
    n = number(v)
    return str(int(n)) if n is not None and n.is_integer() else str(v)


def table(text, delimiter=','):
    raw = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    header = [v or f'_unnamed_{i}' for i, v in enumerate(raw[0])]
    if len(set(header)) != len(header):
        raise ValueError('Duplicate named headers')
    if any(len(r) != len(header) for r in raw[1:]):
        raise ValueError('Ragged CSV')
    return header, [dict(zip(header, r)) for r in raw[1:]]


def profile(header, rows):
    fields = {}
    for k in header:
        values = [r[k] for r in rows if r[k] not in MISSING]
        nums = [number(v) for v in values]
        numeric = bool(values) and all(v is not None for v in nums)
        fields[k] = {'missing': len(rows)-len(values), 'missing_pct': round(100*(len(rows)-len(values))/len(rows), 2), 'distinct': len(set(values)), 'numeric': numeric}
        if numeric:
            fields[k].update(min=min(nums), median=S.median(nums), max=max(nums))
    return {'rows': len(rows), 'columns': len(header), 'exact_duplicate_rows': len(rows)-len({tuple(r[k] for k in header) for r in rows}), 'fields': fields}


def temporal(rows, idkey, timekey):
    groups = C.defaultdict(list)
    for r in rows:
        n = number(r[timekey])
        if n is not None:
            groups[ident(r[idkey])].append(n)
    gaps = []; spans = []; counts = []; dup = 0
    for times in groups.values():
        t = sorted(times); gaps.extend(b-a for a,b in zip(t,t[1:]) if b>a); spans.append(t[-1]-t[0]);counts.append(len(t));dup += len(t)-len(set(t))
    alltimes = [v for values in groups.values() for v in values]
    return {'patients': len(groups), 'range': [min(alltimes),max(alltimes)], 'observations_per_patient': [min(counts),S.median(counts),max(counts)], 'span_min_median_max': [min(spans),S.median(spans),max(spans)], 'positive_gap_min_median_max': [min(gaps),S.median(gaps),max(gaps)] if gaps else None, 'duplicate_patient_time_excess': dup}


def frequencies(rows, keys):
    return {k: dict(C.Counter(r[k] for r in rows)) for k in keys}


def outliers(rows, bounds):
    # Broad screening bounds, not validated clinical thresholds; no automatic removal.
    return {k: sum((n:=number(r[k])) is not None and not lo <= n <= hi for r in rows) for k,(lo,hi) in bounds.items()}


def audit(root):
    result = {'inventory': [], 'screening_note': 'Bounds are broad plausibility screens, not diagnoses; values require source review.'}
    for p in sorted(root.iterdir()):
        item = {'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        if p.suffix=='.zip':
            with zipfile.ZipFile(p) as z:
                item['members'] = [{'path':i.filename,'bytes':i.file_size} for i in z.infolist()] if p.name != 'synthea_sample_data_fhir_latest.zip' else {'count':len(z.infolist()),'uncompressed_bytes':sum(i.file_size for i in z.infolist())}
        result['inventory'].append(item)
    with zipfile.ZipFile(root/ZIGONG) as z:
        read = lambda suffix: z.read(next(n for n in z.namelist() if n.endswith('/'+suffix))).decode('utf-8-sig')
        h,rows=table(read('dat.csv')); mh,meds=table(read('dat_md.csv')); dh,dictionary=table(read('dataDictionary.csv'))
        a=profile(h,rows); a['duplicate_rows_excluding_export_index']=len(rows)-len({tuple(r[k] for k in h if not k.startswith('_unnamed_')) for r in rows}); a['patients']=len({r['inpatient.number'] for r in rows});a['dictionary']={r['Variables']:r['Description'] for r in dictionary};a['dictionary_unmatched']=sorted(set(h)-set(a['dictionary']));a['archive_checksums']={}
        for line in read('SHA256SUMS.txt').splitlines():
            checksum, filename = line.split()
            a['archive_checksums'][filename] = hashlib.sha256(z.read(next(n for n in z.namelist() if n.endswith('/'+filename)))).hexdigest() == checksum
        a['license_title']=read('LICENSE.txt').splitlines()[0]
        outcomes=[k for k in h if 'within' in k or k=='outcome.during.hospitalization']
        a['frequencies']=frequencies(rows,outcomes+['gender','ageCat','type.of.heart.failure','NYHA.cardiac.function.classification','DestinationDischarge','congestive.heart.failure'])
        a['drug_names']=sorted({r['Drug_name'] for r in meds});a['medications']=profile(mh,meds); a['medications'].update(patients=len({r['inpatient.number'] for r in meds}),drug_names=len({r['Drug_name'] for r in meds}),duplicate_patient_drug_excess=len(meds)-len({(r['inpatient.number'],r['Drug_name']) for r in meds}),orphan_rows=sum(r['inpatient.number'] not in {x['inpatient.number'] for x in rows} for r in meds))
        a['outlier_bounds']={'body.temperature':[30,43],'pulse':[20,250],'respiration':[5,70],'systolic.blood.pressure':[50,300],'diastolic.blood.pressure':[20,180],'weight':[20,300],'height':[1,2.5],'BMI':[10,80],'LVEF':[0,100],'oxygen.saturation':[0,100],'pH':[6.5,8]}
        a['outliers']=outliers(rows,a['outlier_bounds']);a['bp_sys_le_dia']=sum(number(r['systolic.blood.pressure'])<=number(r['diastolic.blood.pressure']) for r in rows)
        a['consistency']={}
        for prefix in ['death','re.admission']:
            labels=[prefix+'.within.'+s for s in ['28.days','3.months','6.months']]
            timekey='time.of.death..days.from.admission.' if prefix=='death' else 're.admission.time..days.from.admission.'
            a['consistency'][prefix]={'nonmonotone_labels':sum(r[labels[0]]>r[labels[1]] or r[labels[1]]>r[labels[2]] for r in rows),'positive_6m_missing_event_time':sum(r[labels[2]]=='1' and number(r[timekey]) is None for r in rows),'negative_6m_has_event_time':sum(r[labels[2]]=='0' and number(r[timekey]) is not None for r in rows),'event_before_discharge':sum((t:=number(r[timekey])) is not None and t<number(r['dischargeDay']) for r in rows),'horizon_mismatches_admission_clock':{},'horizon_mismatches_discharge_clock':{}}
            for key,days in zip(labels,[28,90,180]):
                for clock,offset in [('admission',False),('discharge',True)]:
                    a['consistency'][prefix]['horizon_mismatches_'+clock+'_clock'][key]=sum((t:=number(r[timekey])) is not None and (r[key]=='1') != (0<=t-(number(r['dischargeDay']) if offset else 0)<=days) for r in rows)
        a['readmission_positive_inhospital_dead']={k:sum(r[k]=='1' and r['outcome.during.hospitalization']=='Dead' for r in rows) for k in outcomes if k.startswith('re.admission')}
        a['alive_readmission_labels']=frequencies([r for r in rows if r['outcome.during.hospitalization']=='Alive'],[k for k in outcomes if k.startswith('re.admission')])
        result['zigong']=a
    h,rows=table((root/'pone.0190323.s017.csv').read_text(),';');a=profile(h,rows);a['decimal_comma_cells']=sum(',' in r[k] and number(r[k]) is not None for r in rows for k in h);a['temporal']=temporal(rows,'Patient_ID','Day');a['frequencies']=frequencies(rows,['Feel']);a['blank_columns']=[k for k in h if k.startswith('_unnamed_')];a['nonconsecutive_next_days']=sum(1 for times in ({ident(r['Patient_ID']):sorted(number(x['Day']) for x in rows if x['Patient_ID']==r['Patient_ID']) for r in rows}).values() for x,y in zip(times,times[1:]) if y-x!=1)
    a['outlier_bounds']={'HR_avg':[20,250],'SystolicBP':[50,300],'DiastolicBP':[20,180],'Weight':[20,300],'SpO2':[0,100],'Duration':[0,1440]};a['outliers']=outliers(rows,a['outlier_bounds']);result['chiron']=a
    h,q=table((root/'personal_questionnaires.csv').read_text());a={'questionnaires':profile(h,q),'questionnaire_patients':len({ident(r['user_id']) for r in q}),'demographics':frequencies(q,['sex']),'recent_hf_nonmissing':{k:sum(r[k] not in MISSING for r in q) for k in ['recent_hf_1','recent_hf_2','recent_hf_3']},'recent_hf_hospitalized_values':frequencies(q,['recent_hf_3']),'medical':{}}
    allids=set(); allrows=[]
    with zipfile.ZipFile(root/'medical.zip') as z:
        for member in z.namelist():
            if not member.endswith('.csv'): continue
            h,rows=table(z.read(member).decode()); p=profile(h,rows);p['temporal']=temporal(rows,'user_id','ts');p['temporal']['utc_range']=[dt.datetime.fromtimestamp(v,dt.timezone.utc).isoformat() for v in p['temporal']['range']];p['orphan_rows']=sum(ident(r['user_id']) not in {ident(x['user_id']) for x in q} for r in rows)
            bounds={'bp_sys':[50,300],'bp_dia':[20,180],'hr':[20,250],'sat':[0,100],'perf':[0,100]}
            if 'value' in h: bounds['value']=[20,300] if 'body_mass' in member else [1,40] if 'glucose' in member else [30,43]
            p['outlier_bounds']={k:v for k,v in bounds.items() if k in h};p['outliers']=outliers(rows,p['outlier_bounds']); a['medical'][member]=p; allids.update(ident(r['user_id']) for r in rows);allrows.extend(rows)
    a['medical_patients']=len(allids);a['medical_rows']=len(allrows);a['combined_temporal']=temporal(allrows,'user_id','ts'); result['perheart']=a
    result['synthea']=audit_fhir(root/'synthea_sample_data_fhir_latest.zip')
    return result


def audit_fhir(path):
    counts=C.Counter();bundles=C.Counter();sex=C.Counter();dead=C.Counter();codes={k:C.Counter() for k in ['Condition','Observation','MedicationRequest']}; units=C.Counter();missing=C.Counter(); keys=C.Counter();seen={};duplicates=0;conflicts=0;patientids=set();births=[];dates=[];obsdates=[];subjectids=set();refcount=0;unresolved=0;hfpatients=set();hfconditions=0; bpcomponents=0;nonfinite=0
    identifier_index=set(); global_urls=set(); external_refs=C.Counter(); patient_times=C.defaultdict(list); categories=C.Counter(); demographic_missing=C.Counter(); fhir_versions=C.Counter()
    hf_bounds={'8867-4':(20,250),'8480-6':(50,300),'8462-4':(20,180),'2708-6':(0,100),'29463-7':(20,300),'8310-5':(30,43)};screens=C.Counter(); screened=C.Counter(); resources=0
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith('.json'):continue
            b=json.loads(z.read(member));bundles[b.get('type','missing')]+=1
            for entry in b.get('entry', []):
                rr=entry['resource'];
                for identifier in rr.get('identifier', []):
                    identifier_index.add((rr['resourceType'], identifier.get('system',''), identifier.get('value','')))
                global_urls.add(entry.get('fullUrl'));global_urls.add(rr.get('resourceType','')+'/'+rr.get('id',''))
            entries=b.get('entry',[]); local={e.get('fullUrl') for e in entries};local.update(e['resource'].get('resourceType','')+'/'+e['resource'].get('id','') for e in entries)
            for e in entries:
                r=e['resource']; t=r['resourceType'];
                if t == 'Provenance':
                    for entity in r.get('entity', []):
                        version = entity.get('what', {}).get('display', '')
                        if version: fhir_versions[version] += 1
                counts[t]+=1;resources+=1
                key=(t,r.get('id'));keys[t]+=bool(r.get('id'));finger=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
                if key in seen:duplicates+=1;conflicts+=seen[key]!=finger
                else:seen[key]=finger
                if t=='Patient':
                    for field in ['gender', 'birthDate', 'address', 'extension']:
                        demographic_missing[field] += not bool(r.get(field))
                    patientids.add(r.get('id'));sex[r.get('gender','missing')]+=1;births.append(r.get('birthDate',''));dead['dateTime' if r.get('deceasedDateTime') else str(r.get('deceasedBoolean','absent'))]+=1
                subject=r.get('subject',{}).get('reference','')
                if subject:subjectids.add(subject.split(':')[-1].split('/')[-1])
                concept = r.get('medicationCodeableConcept', {}) if t == 'MedicationRequest' else r.get('code', {})
                coding = concept.get('coding', []) if isinstance(concept, dict) else []
                for c in coding:
                    if t in codes:codes[t][(c.get('system',''),c.get('code',''),c.get('display',''))]+=1
                if t=='Condition' and any('heart failure' in (c.get('display','').lower()) for c in coding):hfconditions+=1;hfpatients.add(subject)
                temporalvalue = r.get('effectiveDateTime') or r.get('effectivePeriod',{}).get('start') or r.get('onsetDateTime') or r.get('period',{}).get('start') or r.get('authoredOn')
                if temporalvalue:dates.append(temporalvalue)
                if t=='Observation':
                    if temporalvalue:
                        obsdates.append(temporalvalue)
                        patient_times[subject].append(dt.datetime.fromisoformat(temporalvalue.replace('Z','+00:00')).timestamp())
                    else:missing['observation_time']+=1
                    for cat in r.get('category', []):
                        for c in cat.get('coding', []): categories[c.get('code', 'missing')] += 1
                    if not subject:missing['observation_subject']+=1
                    if not coding:missing['observation_code']+=1
                    if not any(k.startswith('value') or k in ('component','dataAbsentReason') for k in r):missing['observation_value_component_or_absent_reason']+=1
                    quantities=[(coding,r.get('valueQuantity'))]+[(c.get('code',{}).get('coding',[]),c.get('valueQuantity')) for c in r.get('component',[])]
                    bpcomponents+=bool(r.get('component'))
                    for cs,q in quantities:
                        if q is None:continue
                        units[(q.get('system',''),q.get('code',''),q.get('unit',''))]+=1
                        if 'unit' not in q:missing['quantity_display_unit']+=1
                        n=number(q.get('value')); nonfinite+=n is None
                        for c in cs:
                            code=c.get('code')
                            if code in hf_bounds and n is not None:
                                screened[code]+=1;lo,hi=hf_bounds[code];screens[code]+=not lo<=n<=hi
                def walk(o):
                    nonlocal refcount,unresolved
                    if isinstance(o,dict):
                        if 'reference' in o:
                            refcount+=1;ref=o['reference'];unresolved+=ref not in local and not ref.startswith('#')
                            if ref not in local and not ref.startswith('#'): external_refs[ref] += 1
                        for v in o.values():walk(v)
                    elif isinstance(o,list):
                        for v in o:walk(v)
                walk(r)
    query_refs = 0; query_resolved = 0
    for ref, n in external_refs.items():
        if '?identifier=' in ref:
            query_refs += n
            resource_type, token = ref.split('?identifier=', 1)
            system, sep, value = unquote(token).partition('|')
            if sep and (resource_type, system, value) in identifier_index: query_resolved += n
    gaps=[b-a for times in patient_times.values() for a,b in zip(sorted(set(times)), sorted(set(times))[1:])]; spans=[max(v)-min(v) for v in patient_times.values()]; obs_counts=[len(v) for v in patient_times.values()]
    return {'conditional_identifier_references':query_refs,'conditional_identifier_references_resolvable':query_resolved,'references_unresolved_after_identifier_lookup':sum(n for ref,n in external_refs.items() if ref not in global_urls)-query_resolved,'observation_categories':dict(categories),'demographic_missing':dict(demographic_missing),'observation_cadence_seconds':{'positive_gap_min_median_max':[min(gaps),S.median(gaps),max(gaps)],'patient_span_min_median_max':[min(spans),S.median(spans),max(spans)],'counts_per_patient_min_median_max':[min(obs_counts),S.median(obs_counts),max(obs_counts)]},'references_unresolved_across_all_bundles':sum(n for ref,n in external_refs.items() if ref not in global_urls),'bundles':dict(bundles),'resources':resources,'resource_counts':dict(counts),'patients':len(patientids),'sex':dict(sex),'deceased':dict(dead),'birthdate_range':[min(births),max(births)],'resource_time_lexical_range':[min(dates),max(dates)],'observation_time_lexical_range':[min(obsdates),max(obsdates)],'duplicate_resource_key_excess':duplicates,'conflicting_resource_keys':conflicts,'missing':dict(missing),'references':refcount,'references_not_resolved_within_own_bundle':unresolved,'unique_subject_ids':len(subjectids),'subject_ids_not_in_patient_set':len(subjectids-patientids),'hf_condition_display_matches':hfconditions,'hf_patients_display_matches':len(hfpatients),'component_observations':bpcomponents,'invalid_numeric_quantities':nonfinite,'screened_quantities':dict(screened),'outlier_quantities':dict(screens),'outlier_bounds':hf_bounds,'codes':{k:[{'system':c[0],'code':c[1],'display':c[2],'count':n} for c,n in v.most_common()] for k,v in codes.items()},'units':[{'system':u[0],'code':u[1],'unit':u[2],'count':n} for u,n in units.most_common()]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('Datasets'));p.add_argument('--output',type=Path,default=Path('data/audit/report.json'));args=p.parse_args();r=audit(args.input);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('Audit complete:',args.output)

if __name__=='__main__':main()
