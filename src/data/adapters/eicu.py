"""Source inspection and compact, provenance-linked eICU series (no model fitting)."""
import csv
import gzip
import json
import re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from src.data.io import file_hash,strict_json_loads
from src.data.quality import SchemaError
from src.data.eicu_spec import PERIODIC,APERIODIC,LAB_SCREENS

SCHEMAS = strict_json_loads(Path(__file__).parents[1].joinpath('eicu_source_schemas.json').read_text())


def table_rows(root,relative):
    with gzip.open(Path(root)/relative,'rt',newline='') as f:
        reader=csv.DictReader(f)
        expected=SCHEMAS[relative]
        if reader.fieldnames!=expected:raise SchemaError('Unexpected Phase 3.5 source schema: '+relative)
        for number,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()):raise SchemaError('Ragged source row')
            yield number,row


def number(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else None
    except (ValueError,TypeError):return None


def conditions(text,codes=''):
    """Explicit HF only; cardiomyopathy alone is not HF. Negated mentions excluded."""
    t=text.lower();c=[x.strip().upper().replace('.','') for x in codes.split(',')]
    neg=any(x in t for x in ['without chf','not due to chf','no chf','without heart failure','no heart failure','rule out','possible','suspected','not obtainable','no health problems'])
    hf_code=any(x.startswith(('428','I50')) or x in {'I110','I130','I132'} or
                (len(x)==5 and x.startswith(('402','404')) and x[-1] in ({'1'} if x.startswith('402') else {'1','3'})) for x in c)
    hf=not neg and (hf_code or 'congestive heart failure' in t or 'heart failure' in t or re.search(r'\bchf\b',t) is not None)
    htn=not neg and ('hypertension' in t and 'pulmonary' not in t and 'portal' not in t or any(x.startswith(('401','I10')) for x in c))
    dm=not neg and ('diabetes' in t or any(x.startswith(('250','E10','E11','E13')) for x in c))
    ckd=not neg and (any(x in t for x in ['chronic kidney disease','chronic renal failure','chronic renal insufficiency','end stage renal','end-stage renal']) or any(x.startswith(('585','N18')) for x in c))
    cv=not neg and (hf or htn or t.startswith('cardiovascular|') or '/cardiovascular' in t or '|cardiovascular|' in t)
    return {'hf':bool(hf),'hypertension':bool(htn),'diabetes':bool(dm),'ckd':bool(ckd),'cardiovascular':bool(cv)}


def read_patients(root):
    patients={};hospital_map={}
    for _,r in table_rows(root,'eicu_demo/patient.csv.gz'):
        if not all(r[k] for k in ['patientunitstayid','patienthealthsystemstayid','uniquepid']):raise SchemaError('Missing eICU patient/stay identity')
        stay=int(r['patientunitstayid']);pid='eicu:'+r['uniquepid'];hospital=r['patienthealthsystemstayid']
        if stay in patients:raise SchemaError('Duplicate ICU stay')
        if hospital in hospital_map and hospital_map[hospital]!=pid:raise SchemaError('Hospital stay crosses patient identities')
        hospital_map[hospital]=pid
        age=number(r['age']);adult=(age is not None and age>=18) or r['age'].strip()=='> 89'
        end=number(r['unitdischargeoffset'])
        if end is None or end<0:raise SchemaError('Invalid ICU discharge boundary')
        patients[stay]={'patient_id':pid,'icu_stay_id':'eicu:icu:'+str(stay),'hospital_stay_id':'eicu:hospital:'+hospital,
                        'adult':adult,'end':end,'raw':r,'conditions':[],'ever':{k:False for k in ['hf','hf_history','cardiovascular','hypertension','diabetes','ckd']}}
    return patients


def context_audit(root,patients):
    tables={};extra={};lab_events=defaultdict(list);pressor=defaultdict(list)
    condition_config={'diagnosis':(['diagnosisstring'],'diagnosisoffset',None,'icd9code'),
                      'pastHistory':(['pasthistorypath','pasthistoryvalue','pasthistoryvaluetext'],'pasthistoryoffset','pasthistoryenteredoffset',None),
                      'admissionDx':(['admitdxpath','admitdxname','admitdxtext'],'admitdxenteredoffset',None,None)}
    names=['admissionDx','diagnosis','pastHistory','lab','medication','admissiondrug','infusiondrug','intakeOutput','apacheApsVar','apachePredVar','apachePatientResult','treatment']
    for name in names:
        ids=set();stays=set();count=0;duplicates=0;orphans=0;late=0;negative=0;missing_time=0;labels=Counter();units=Counter();lags=[];future_state_fields=[]
        pk=SCHEMAS[f'eicu_demo/{name}.csv.gz'][0]
        seen_payloads=set()
        for _,r in table_rows(root,f'eicu_demo/{name}.csv.gz'):
            count+=1;record_id=r[pk]
            if not record_id or record_id in ids:raise SchemaError('Missing/duplicate source primary key in '+name)
            ids.add(record_id);stay=int(r['patientunitstayid'])
            if stay not in patients:orphans+=1;continue
            stays.add(stay);p=patients[stay]
            if name in condition_config:
                fields,tfield,entered,code=condition_config[name];text=' | '.join(r[k] for k in fields)
                flags=conditions(text,r[code] if code else '')
                flags['hf_history']=name=='pastHistory' and flags['hf']
                observed=number(r[tfield]);available=number(r[entered]) if entered else observed
                if observed is None or available is None:missing_time+=1;continue
                available=max(observed,available)
                signature=(stay,text,r[code] if code else '',observed,available)
                if signature in seen_payloads:duplicates+=1;continue
                seen_payloads.add(signature)
                entry={'available_offset':available,'observed_offset':observed,'flags':flags,'source_table':name,'source_record':record_id,
                       'text':text,'codes':r[code] if code else ''}
                p['conditions'].append(entry)
                for k,v in flags.items():p['ever'][k]|=v
                labels.update([text]);late+=available>p['end'];negative+=available<0
            elif name=='lab':
                labels[r['labname']]+=1;unit=r['labmeasurenamesystem'] or r['labmeasurenameinterface'];units[unit]+=1
                drawn=number(r['labresultoffset']);revised=number(r['labresultrevisedoffset']);value=number(r['labresult'])
                if drawn is None or revised is None:missing_time+=1;continue
                lags.append(revised-drawn);late+=max(drawn,revised)>p['end'];negative+=revised<drawn
                concept=r['labname'].lower()
                screen=LAB_SCREENS.get(concept)
                if screen and value is not None and revised>=drawn and unit.casefold().strip() in screen[0] and screen[1]<=value<=screen[2]:
                    lab_events[stay].append({'concept':r['labname'].lower(),'value':value,'unit':unit,'observed_offset':drawn,
                                            'available_offset':revised,'source_table':name,'source_record':record_id})
            elif name in ['medication','admissiondrug','infusiondrug']:
                labels[r['drugname']]+=1
                timefield={'medication':'drugorderoffset','admissiondrug':'drugenteredoffset','infusiondrug':'infusionoffset'}[name]
                t=number(r[timefield]);late+=t is not None and t>p['end'];missing_time+=t is None
                if name=='infusiondrug' and t is not None and re.search(r'\b(norepinephrine|levophed|epinephrine|phenylephrine|neosynephrine|vasopressin|dopamine)\b',r['drugname'],re.I):
                    rate=number(r['drugrate']);inf=number(r['infusionrate'])
                    if (rate is not None and rate>0) or (inf is not None and inf>0):pressor[stay].append(t)
            elif name=='intakeOutput':
                labels[r['celllabel']]+=1;t=number(r['intakeoutputoffset']);entry=number(r['intakeoutputentryoffset'])
                if t is None or entry is None:missing_time+=1
                else:lags.append(entry-t);late+=max(t,entry)>p['end'];negative+=entry<t
        tables[name]={'rows':count,'stays':len(stays),'patients':len({patients[s]['patient_id'] for s in stays}),
                      'orphan_stay_rows':orphans,'duplicate_payloads_removed_from_condition_context':duplicates,
                      'after_discharge_entries':late,'negative_or_inverted_time_count':negative,'missing_timestamps':missing_time,
                      'top_terms':labels.most_common(25),'units':dict(units),'entry_lag_minutes_quantiles':quantiles(lags)}
        if orphans:raise SchemaError('Orphan clinical rows in '+name)
    for stay,events in lab_events.items():
        groups=defaultdict(list)
        for event in events:groups[(event['concept'],event['observed_offset'],event['available_offset'],event['unit'].casefold().strip())].append(event)
        kept=[]
        for group in groups.values():
            if len({r['value'] for r in group})==1:
                kept.append({**group[0],'source_records':[r['source_record'] for r in group]})
        lab_events[stay]=sorted(kept,key=lambda r:(r['observed_offset'],r['available_offset'],r['concept']))
    for p in patients.values():p['conditions'].sort(key=lambda r:(r['available_offset'],r['source_table'],r['source_record']))
    return tables,lab_events,pressor


def quantiles(values):
    return np.quantile(values,[0,.01,.25,.5,.75,.99,1]).tolist() if len(values) else None


def vital_arrays(root,name,patients):
    spec=PERIODIC if name=='vitalPeriodic' else APERIODIC;fields=list(spec)
    dtype=[('stay','i8'),('offset','i8'),('source_id','i8')]+[(f,'f8') for f in fields]
    broken=Counter();missing=Counter();broken_indices=defaultdict(list)
    def records():
        for index,(_,r) in enumerate(table_rows(root,f'eicu_demo/{name}.csv.gz')):
            try:stay=int(r['patientunitstayid']);t=int(r['observationoffset']);rid=int(r[SCHEMAS[f'eicu_demo/{name}.csv.gz'][0]])
            except ValueError:raise SchemaError('Missing/invalid vital ID or timestamp')
            if stay not in patients:raise SchemaError('Vital references unknown ICU stay')
            values=[]
            for f in fields:
                v=number(r[f]);values.append(np.nan if v is None else v)
                if v is None:
                    (missing if not r[f] else broken)[f]+=1
                    if r[f]:broken_indices[f].append(rid)
            yield (stay,t,rid,*values)
    raw=np.fromiter(records(),dtype=dtype)
    if len(np.unique(raw['source_id']))!=len(raw):raise SchemaError('Duplicate vital primary key')
    raw.sort(order=['stay','offset','source_id'])
    status={};usable={};audit={};within=(raw['offset']>=0)&(raw['offset']<=np.array([patients[int(s)]['end'] for s in raw['stay']]))
    for f,(_,unit,lo,hi) in spec.items():
        value=raw[f];finite=np.isfinite(value);quality=np.where(finite,0,1).astype('u1') # 0 valid,1 missing,2 suspect,3 invalid
        hard=(finite&((value<=0)|(value>400))) if f not in ['cvp','icp','st1','st2','st3','padiastolic','pamean','paop'] else finite&(abs(value)>1000)
        if f=='sao2':hard=finite&((value<0)|(value>100))
        quality[np.isin(raw['source_id'],broken_indices[f])]=3
        quality[hard]=3;quality[finite&~hard&((value<lo)|(value>hi))]=2
        quality[finite&~within]=2
        if unit=='unknown':quality[finite&~hard]=2
        v=value.copy();v[quality!=0]=np.nan;status[f]=quality;usable[f]=v
    bp=['systemicsystolic','systemicdiastolic','systemicmean'] if name=='vitalPeriodic' else ['noninvasivesystolic','noninvasivediastolic','noninvasivemean']
    s,d,m=[raw[f] for f in bp]
    inconsistent=(np.isfinite(s)&np.isfinite(d)&(s<=d))|(np.isfinite(m)&np.isfinite(d)&(m<d-5))|(np.isfinite(m)&np.isfinite(s)&(m>s+5))
    for f in bp:
        status[f][inconsistent&np.isfinite(raw[f])&(status[f]!=3)]=2;usable[f][inconsistent]=np.nan
    # Same-time records are consolidated only when every nonmissing value agrees;
    # conflicting values are blocked per signal, never averaged or arbitrarily selected.
    starts=np.r_[0,1+np.flatnonzero((raw['stay'][1:]!=raw['stay'][:-1])|(raw['offset'][1:]!=raw['offset'][:-1]))]
    ends=np.r_[starts[1:],len(raw)];duplicates=int(len(raw)-len(starts));conflicts=Counter();identical_groups=0
    for a,b in zip(starts,ends):
        if b-a<2:continue
        equal=True
        for f in fields:
            vals=raw[f][a:b];nonmissing=vals[np.isfinite(vals)]
            if len(np.unique(nonmissing))>1:
                usable[f][a:b]=np.nan;status[f][a:b]=np.where(status[f][a:b]==3,3,2);conflicts[f]+=1;equal=False
            elif len(nonmissing):
                # Keep one raw source position having this field; preserve all raw rows.
                idx=a+int(np.flatnonzero(np.isfinite(vals))[0]);usable[f][a:b]=np.nan
                if status[f][idx]==0:usable[f][idx]=raw[f][idx]
        identical_groups+=equal
    series={}
    compact_stay=raw['stay'][starts];compact_time=raw['offset'][starts]
    compact_values={}
    for f in fields:
        valid=np.isfinite(usable[f])
        count=np.add.reduceat(valid.astype(int),starts)
        compact_values[f]=np.where(count>0,np.add.reduceat(np.nan_to_num(usable[f],nan=0),starts),np.nan)
    unique_stays=np.unique(compact_stay);split=np.searchsorted(compact_stay,unique_stays)
    for index,stay in enumerate(unique_stays):
        a=split[index];b=split[index+1] if index+1<len(split) else len(compact_stay)
        series[int(stay)]={'offset':compact_time[a:b],**{f:compact_values[f][a:b] for f in fields}}
    for f,(concept,unit,lo,hi) in spec.items():
        v=usable[f];idx=np.flatnonzero(np.isfinite(v));valid_stays=set(map(int,raw['stay'][idx]));gaps=[];duration=0.;spans=[]
        for stay in valid_stays:
            times=series[stay]['offset'][np.isfinite(series[stay][f])];diff=np.diff(times);gaps.extend(diff[diff>0].tolist())
            spans.append(float(times[-1]-times[0])/60);duration+=float(sum(diff[(diff>0)&(diff<=10)]))/60
        audit[f]={'concept':concept,'unit':unit,'raw_rows':len(raw),'nonmissing':int(sum(np.isfinite(raw[f]))),
                  'missing':missing[f],'unparseable':broken[f],'missing_fraction':float(missing[f]/len(raw)),
                  'quality_counts':{k:int(sum(status[f]==c)) for k,c in [('VALID',0),('MISSING',1),('SUSPECT',2),('INVALID',3)]},
                  'usable_values':len(idx),'patients':len({patients[s]['patient_id'] for s in valid_stays}),'stays':len(valid_stays),
                  'raw_quantiles':quantiles(raw[f][np.isfinite(raw[f])]),'usable_quantiles':quantiles(v[idx]),
                  'cadence_minutes_quantiles':quantiles(gaps),'cadence_counts':dict(Counter(gaps).most_common(15)),
                  'usable_connected_hours_max_gap_10min':duration,'span_hours_quantiles':quantiles(spans),
                  'conflicting_same_time_groups':conflicts[f],'screen':[lo,hi],
                  'raw_outside_screen':int(sum(np.isfinite(raw[f])&((raw[f]<lo)|(raw[f]>hi))))}
    general={'raw_rows':len(raw),'same_time_extra_rows':duplicates,'identical_nonmissing_same_time_groups':identical_groups,
             'bp_inconsistent_rows':int(sum(inconsistent)),'pre_admission_rows':int(sum(raw['offset']<0)),
             'post_discharge_rows':int(sum(~within&(raw['offset']>=0))),'fields':audit}
    return raw,status,usable,series,general
