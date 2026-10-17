"""Bounded deterministic PDF/text/JSON/FHIR parsing. Never reads fixture metadata."""
from dataclasses import dataclass
from datetime import date
import io
import json
import re
from .schema import Fact, display, IngestionError, validate_value

MAX_TEXT=250000


def strict_json(raw):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    def invalid(_):raise ValueError('Non-finite JSON')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
    def bounded(node,depth=0):
        if depth>20:raise ValueError('JSON too deeply nested')
        if isinstance(node,dict):
            if len(node)>1000:raise ValueError('Too many JSON properties')
            for nested in node.values():bounded(nested,depth+1)
        elif isinstance(node,list):
            if len(node)>1000:raise ValueError('Too many JSON entries')
            for nested in node:bounded(nested,depth+1)
    bounded(value)
    return value


def medication(text,status=None):
    text=text.strip(' -•')
    match=re.match(r'([A-Za-z][A-Za-z -]*?)(?=\s+(?:\d|discontinued|stopped)|$)',text,re.I)
    if not match:return None
    name=match.group(1).strip()
    dose=re.search(r'\b(\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|units?))\b',text,re.I)
    frequency=None
    for pattern,value in [(r'twice daily|\bBD\b|\bBID\b','twice daily'),(r'three times daily|\bTID\b','three times daily'),(r'nightly|at night','nightly'),(r'each morning|every morning','each morning'),(r'\bdaily\b|\bOD\b','daily')]:
        if re.search(pattern,text,re.I):frequency=value;break
    if status is None:
        status='discontinued' if re.search(r'\bdiscontinued|\bstopped',text,re.I) else 'active' if dose or re.search(r'\bstarted',text,re.I) else 'unknown'
    route='subcutaneous' if re.search('subcutaneously|subcutaneous',text,re.I) else 'oral' if re.search('orally|oral',text,re.I) else None
    start=re.search(r'started(?: on)?\s+(\d{4}-\d{2}-\d{2})',text,re.I)
    stop=re.search(r'(?:discontinued|stopped)(?: on)?\s+(\d{4}-\d{2}-\d{2})',text,re.I)
    return {'name':name,'dose':dose.group(1) if dose else None,'frequency':frequency,'route':route,
            'status':status,'start_date':start.group(1) if start else None,'stop_date':stop.group(1) if stop else None,'original_text':text}


def category(meds):
    active=[m for m in meds if m['status']=='active']
    if not active:return None
    return 'explicit_insulin_term' if any(re.search(r'insulin|lispro|aspart|glargine|detemir|degludec|novolin|humalog|lantus',m['name'],re.I) for m in active) else 'recorded_without_explicit_insulin_term'


def alcohol(text):
    if re.search(r'no current alcohol|does not drink|no alcohol|non[- ]drinker|abstinent|never',text,re.I):return 'non-drinker'
    if re.search(r'occasion|drinker|drinks|social alcohol',text,re.I):return 'drinker'
    return 'unknown'


def canonical_diagnosis(value):
    if re.search(r'possible|suspected|rule out|no evidence|family history',value,re.I):return value.strip()
    return 'Type 2 Diabetes' if re.search(r'type\s*(?:2|ii)\s+diabetes',value,re.I) else value.strip()


def text_extract(pages,document_id,filename):
    chunks=[];offsets=[];position=0
    for text,page in pages:
        offsets.append((position,position+len(text),page))
        chunks.append(text);position+=len(text)+1
    text='\n'.join(chunks)
    titles={'outpatient follow up':'outpatient_follow_up','medication history':'medication_history','lab report':'lab_report',
            'clinical history':'clinical_history','referral follow up':'referral_follow_up','discharge summary':'discharge_summary',
            'medication change conflict':'medication_change_conflict'}
    document_type=titles.get(text.splitlines()[0].strip().lower(),'clinical_record') if text.splitlines() else 'clinical_record'
    facts=[]
    def emit(name,value,start,end,section,confidence=.95,unit=None):
        if value is None:return
        page=next((p for low,high,p in offsets if low<=start<=high),None)
        snippet=text[start:end].strip()[:1800]
        fact=Fact(field_name=name,normalized_value=value,original_extracted_value=value,
                  display_value=display(name,value),confidence=confidence,source_document_id=document_id,
                  source_filename=filename,source_text_snippet=snippet,source_page=page,source_section=section,unit=unit,
                  status='EXTRACTED' if confidence>=.85 else 'LOW_CONFIDENCE')
        facts.append(fact.view())
    def match(name,pattern,convert=lambda x:x,group=1,section='Patient',unit=None):
        found=re.search(pattern,text,re.I|re.M)
        if found:
            emit(name,convert(found.group(group)),found.start(),found.end(),section,unit=unit)
        return found
    dated=re.search(r'Document date:\s*(\d{4}-\d{2}-\d{2})',text,re.I)
    document_date=dated.group(1) if dated else None
    match('display_name',r'^Patient(?: name)?:\s*([^\n]+)')
    match('date_of_birth',r'\b(?:DOB|Date of birth):\s*(\d{4}-\d{2}-\d{2})')
    match('age_years',r'\bAge(?: on document date)?:\s*(\d+(?:\.\d+)?)\s*(?:years|y)',float,unit='years')
    sex=match('sex',r'\bSex:\s*(female|male|unknown|other|F\b|M\b)',lambda x:{'m':'male','f':'female'}.get(x.lower(),x.lower()))
    primary=re.search(r'Primary diagnosis:\s*([^\n]+)',text,re.I)
    diagnoses=[]
    if primary:diagnoses.append(canonical_diagnosis(primary.group(1).split(',')[0].split(';')[0]))
    else:
        explicit=re.search(r'^([^\n]*(?:type\s*(?:2|ii)\s+diabetes)[^\n]*)$',text,re.I|re.M)
        if explicit and not re.search(r'family history|sibling|no evidence|denies',explicit.group(1),re.I):
            diagnoses.append(canonical_diagnosis(explicit.group(1)))
    relevant=re.search(r'Relevant history:\s*([^\n]+)',text,re.I)
    if relevant:diagnoses.extend(v.strip() for v in relevant.group(1).split(',') if v.strip())
    if diagnoses:
        found=primary or re.search(r'type\s*(?:2|ii)\s+diabetes',text,re.I)
        emit('diagnoses',list(dict.fromkeys(diagnoses)),found.start(),relevant.end() if relevant else found.end(),'Diabetes history',
             .60 if any(re.search(r'possible|suspected|rule out',d,re.I) for d in diagnoses) else .95)
    match('diabetes_duration_years',r'\b(?:diabetes duration|duration)(?: of diabetes)?:?\s*(\d+(?:\.\d+)?)\s*years',float,section='Diabetes history',unit='years')
    match('diagnosis_date',r'(?:diabetes[^\n]*?recorded in|diagnosed(?: in)?|recorded in)\s*(\d{4}(?:-\d{2}-\d{2})?)',section='Diabetes history')
    section=re.search(r'MEDICATION RECONCILIATION\s*\n(.*?)(?=\n\s*SOCIAL HISTORY|\n\s*[A-Z][A-Z /-]{6,}\n|\Z)',text,re.S)
    meds=[]
    if section:
        for line in section.group(1).splitlines():
            parsed=medication(line)
            if parsed:meds.append(parsed)
    else:
        for found in re.finditer(r'^([A-Z][A-Za-z -]+?\s+\d+(?:\.\d+)?\s*(?:mg|mcg|units?)\b[^\n]*)$',text,re.M):
            parsed=medication(found.group(1))
            if parsed:meds.append(parsed)
            if section is None:section=found
    if meds:
        emit('medications',meds,section.start(),section.end(),'Medications',.65 if any(m['status']=='unknown' or (m['status']=='active' and not m['frequency']) for m in meds) else .95)
        cat=category(meds)
        if cat:emit('medication_context',cat,section.start(),section.end(),'Medications',.80)
        # The input never identifies a trained hypoglycemic-vs-other grouping reliably.
        # Do not invent other_medication_context from an arbitrary mixed list.
    smoking=re.search(r'^([^\n]*(?:Former smoker|Never smoked|Current smoker|Non-smoker)[^\n]*)',text,re.I|re.M)
    if smoking:emit('smoking_history',smoking.group(1).strip(),smoking.start(),smoking.end(),'Lifestyle history')
    match('smoking_pack_years',r'(\d+(?:\.\d+)?)\s*pack[- ]years',float,section='Lifestyle history',unit='pack-years')
    if smoking and re.search('never smoked',smoking.group(1),re.I):
        emit('smoking_pack_years',0,smoking.start(),smoking.end(),'Lifestyle history',.90,'pack-years')
    drinking=re.search(r'^([^\n]*(?:alcohol|drinks per month|drinker)[^\n]*)',text,re.I|re.M)
    if drinking:emit('alcohol_history',alcohol(drinking.group(1)),drinking.start(),drinking.end(),'Lifestyle history',.90)
    labsection=re.search(r'SELECTED LABORATORY CONTEXT\s*\n(.*?)(?=\n\s*[A-Z][A-Z /-]{6,}\n|\Z)',text,re.S)
    labs=[]
    collected=re.search(r'Samples collected\s*(\d{4}-\d{2}-\d{2})\s*at\s*(\d{2}:\d{2})\s*([+-]\d{2}:\d{2});\s*results available at\s*(\d{2}:\d{2})\s*([+-]\d{2}:\d{2})',text,re.I)
    observed=collected.group(1)+'T'+collected.group(2)+':00'+collected.group(3) if collected else None
    available=collected.group(1)+'T'+collected.group(4)+':00'+collected.group(5) if collected else None
    labtext=labsection.group(1) if labsection else text
    for found in re.finditer(r'^([A-Za-z][A-Za-z0-9 /()]+):\s*(-?\d+(?:\.\d+)?)\s+([^\n]+)',labtext,re.M):
        test,value,unit=found.groups()
        if test.lower().strip() in ('age','duration'):continue
        if not labsection and not re.search(r'hba1c|glucose|creatinine|egfr|cholesterol|albumin|tsh|triglyceride|sodium|potassium|hemoglobin',test,re.I):continue
        labs.append({'test':test.strip(),'value':float(value),'unit':unit.strip(),'observed_time':observed,'available_time':available,'original_text':found.group(0)})
    if labs:
        section=labsection or re.search(r'^.*(?:HbA1c|glucose).*$',text,re.I|re.M)
        emit('labs',labs,section.start() if section else 0,section.end() if section else min(1800,len(text)),'Laboratory context')
    narrative=re.search(r'\n(?:DIABETES FOLLOW-UP|LONGITUDINAL HISTORY|CURRENT PATTERN|CLINICAL CONTEXT|SHORT-STAY SUMMARY|DATED MEDICATION CHANGE)\s*\n(.*?)(?=\n\s*[A-Z][A-Z /-]{6,}\n|\Z)',text,re.S)
    if narrative:
        emit('notes',' '.join(narrative.group(1).split()),narrative.start(),narrative.end(),'Clinical notes',.80)
    for fact in facts:
        fact['source_document_date']=document_date
        fact['source_document_type']=document_type
        fact['source_evidence']=[{'source_document_id':document_id,'source_filename':filename,'source_page':page,
                                 'source_section':fact['source_section'],'source_text_snippet':block[:1800]}
                                for block,page in pages if fact['source_text_snippet'][:30] in block or fact['field_name'] in ('medications','labs')]
    return {'facts':facts,'document_date':document_date,'document_type': document_type,
            'source_type':'text_rules','synthetic': 'SYNTHETIC TEST RECORD' in text,
            'pages':len(pages),'warnings':[]}


def fhir_extract(bundle,document_id,filename):
    if bundle.get('resourceType')!='Bundle' or bundle.get('type') not in ('collection','document','searchset') or not isinstance(bundle.get('entry'),list):
        raise ValueError('Supported FHIR Bundle required')
    entries=bundle['entry']
    if not entries or len(entries)>200:raise ValueError('Invalid FHIR entries')
    resources=[]
    for index,entry in enumerate(entries):
        resource=entry.get('resource') if isinstance(entry,dict) else None
        if not isinstance(resource,dict) or not resource.get('resourceType'):raise ValueError('Invalid FHIR resource')
        resources.append((index,resource))
    patients=[(index,r) for index,r in resources if r['resourceType']=='Patient']
    if len(patients)!=1:raise ValueError('Single-patient FHIR Bundle required')
    pi,patient=patients[0]
    refs={entries[pi].get('fullUrl'),'Patient/'+str(patient.get('id')),'urn:uuid:'+str(patient.get('id'))}
    for _,resource in resources:
        subject=resource.get('subject')
        if subject is not None and (not isinstance(subject,dict) or subject.get('reference') not in refs):
            raise ValueError('Mixed patient FHIR reference')
    facts=[]
    def emit(name,value,index,resource,confidence=.98):
        if value is None:return
        fact=Fact(field_name=name,normalized_value=value,original_extracted_value=value,display_value=display(name,value),
                  confidence=confidence,source_document_id=document_id,source_filename=filename,
                  source_section=f'/entry/{index}/resource',source_text_snippet=json.dumps(resource,ensure_ascii=False)[:1800],
                  status='EXTRACTED' if confidence>=.85 else 'LOW_CONFIDENCE')
        facts.append(fact.view())
    names=patient.get('name',[])
    if names:
        first=names[0];name=first.get('text') or ' '.join([*first.get('given',[]),first.get('family','')]).strip()
        if name:emit('display_name',name,pi,patient)
    emit('date_of_birth',patient.get('birthDate'),pi,patient)
    if patient.get('gender') is not None:emit('sex',patient['gender'],pi,patient)
    diagnoses=[];meds=[];labs=[];docdates=[];notes=[];note_sources=[]
    for index,resource in resources:
        kind=resource['resourceType']
        if kind=='Condition':
            code=resource.get('code',{}).get('text')
            if code:diagnoses.append(canonical_diagnosis(code))
            onset=resource.get('onsetDateTime') or resource.get('onsetString')
            if onset:
                found=re.search(r'\b\d{4}(?:-\d{2}-\d{2})?\b',onset)
                if found:emit('diagnosis_date',found.group(0),index,resource)
        elif kind=='MedicationRequest':
            if resource.get('status') is None or resource.get('intent') is None:raise ValueError('Incomplete MedicationRequest')
            if resource.get('authoredOn'):docdates.append(resource['authoredOn'][:10])
            medication_text=resource.get('medicationCodeableConcept',{}).get('text')
            instructions=resource.get('dosageInstruction',[])
            dose_text=instructions[0].get('text','') if instructions else ''
            if medication_text:
                text=medication_text if medication_text in dose_text else medication_text+' '+dose_text
                status='active' if resource['status']=='active' else 'discontinued' if resource['status'] in ('stopped','cancelled') else 'unknown'
                item=medication(text,status)
                if item:meds.append(item)
        elif kind=='Observation':
            if resource.get('status') is None or not isinstance(resource.get('code'),dict):raise ValueError('Incomplete Observation')
            label=resource['code'].get('text')
            if not label:continue
            value=resource.get('valueString')
            quantity=resource.get('valueQuantity')
            if quantity:
                if quantity.get('value') is None or not quantity.get('unit'):raise ValueError('Lab value/unit required')
                labs.append({'test':label,'value':quantity['value'],'unit':quantity['unit'],'observed_time':resource.get('effectiveDateTime'),
                             'available_time':resource.get('issued'),'original_text':json.dumps(quantity)})
            elif value is not None:
                if re.search('diabetes duration',label,re.I):
                    try:emit('diabetes_duration_years',float(value),index,resource)
                    except ValueError:pass
                elif re.search('smoking',label,re.I):
                    emit('smoking_history',value,index,resource)
                    found=re.search(r'(\d+(?:\.\d+)?)\s*pack[- ]years',value,re.I)
                    if found:emit('smoking_pack_years',float(found.group(1)),index,resource)
                    elif re.search('never smoked',value,re.I):emit('smoking_pack_years',0,index,resource,.90)
                elif re.search('alcohol',label,re.I):emit('alcohol_history',alcohol(value),index,resource,.90)
                elif re.search('history|pattern|notes',label,re.I):
                    notes.append(value);note_sources.append((index,resource))
    if diagnoses:emit('diagnoses',list(dict.fromkeys(diagnoses)),next(i for i,r in resources if r['resourceType']=='Condition'),next(r for i,r in resources if r['resourceType']=='Condition'))
    if meds:
        index,resource=next((i,r) for i,r in resources if r['resourceType']=='MedicationRequest')
        emit('medications',meds,index,resource)
        cat=category(meds)
        if cat:emit('medication_context',cat,index,resource,.80)
    if labs:
        index,resource=next((i,r) for i,r in resources if r.get('valueQuantity'))
        emit('labs',labs,index,resource)
    if notes:emit('notes','\n'.join(notes),note_sources[0][0],note_sources[0][1],.75)
    docdate=min(docdates) if docdates else None
    for fact in facts:
        fact['source_document_date']=docdate
        fact['source_document_type']='fhir_bundle'
        if fact['field_name'] in ('medications','labs','diagnoses','notes'):
            kinds={'medications':'MedicationRequest','labs':'Observation','diagnoses':'Condition','notes':'Observation'}
            sources=[(i,r) for i,r in resources if r['resourceType']==kinds[fact['field_name']]]
            if fact['field_name']=='labs':sources=[(i,r) for i,r in sources if r.get('valueQuantity')]
            if fact['field_name']=='notes':sources=note_sources
            fact['source_evidence']=[{'source_document_id':document_id,'source_filename':filename,'source_section':f'/entry/{i}/resource',
                                     'source_text_snippet':json.dumps(r)[:1800],'source_page':None} for i,r in sources]
    return {'facts':facts,'document_date':docdate,'document_type':'fhir_bundle','source_type':'fhir_structured',
            'synthetic':any(t.get('code')=='synthetic-test-record' for t in bundle.get('meta',{}).get('tag',[])),
            'pages':0,'warnings':[]}


def generic_extract(value,document_id,filename):
    if not isinstance(value,dict) or set(value)-{'schema_version','synthetic','document_date','fields'} or value.get('schema_version')!='twincare-clinical/1' or not isinstance(value.get('fields'),dict):
        raise ValueError('Supported clinical JSON schema required')
    facts=[]
    for name,raw in value['fields'].items():
        if not isinstance(raw,dict) or set(raw)-{'value','unit','observed_time','available_time'} or 'value' not in raw:
            raise ValueError('Typed clinical JSON fields required')
        validate_value(name,raw['value'])
        fact=Fact(field_name=name,normalized_value=raw['value'],original_extracted_value=raw['value'],
                  display_value=display(name,raw['value']),confidence=.97,source_document_id=document_id,source_filename=filename,
                  source_section='/fields/'+name,source_text_snippet=json.dumps(raw)[:1800],unit=raw.get('unit'),
                  source_document_date=value.get('document_date'),source_document_type='clinical_json',
                  observed_time=raw.get('observed_time'),available_time=raw.get('available_time'))
        facts.append(fact.view())
    if value.get('document_date'):date.fromisoformat(value['document_date'])
    if type(value.get('synthetic',False)) is not bool:raise ValueError('Synthetic must be boolean')
    return {'facts':facts,'document_date':value.get('document_date'),'document_type':'clinical_json',
            'source_type':'json_structured','synthetic':value.get('synthetic',False),'pages':0,'warnings':[]}


def parse(raw,format,document_id,filename):
    if format=='pdf':
        from pypdf import PdfReader
        if not raw.startswith(b'%PDF-'):raise ValueError('Invalid PDF signature')
        reader=PdfReader(io.BytesIO(raw),strict=True)
        if reader.is_encrypted or len(reader.pages)>40:raise ValueError('Unsupported PDF')
        pages=[]
        for number,page in enumerate(reader.pages,1):
            contents=page.get_contents()
            if contents and len(contents.get_data())>8*1024*1024:raise ValueError('PDF content stream too large')
            text=page.extract_text() or ''
            pages.append((text,number))
            if sum(len(t) for t,_ in pages)>MAX_TEXT:raise ValueError('Extracted text too large')
        if not any(t.strip() for t,_ in pages):
            raise IngestionError('NO_TEXT_LAYER','This PDF has no readable text layer. Try a text-based record or enter the information manually.')
        return text_extract(pages,document_id,filename)
    if format=='text':
        text=raw.decode('utf-8-sig')
        if len(text)>MAX_TEXT or '\x00' in text:raise ValueError('Invalid plain text')
        return text_extract([(text,None)],document_id,filename)
    value=strict_json(raw.decode('utf-8-sig'))
    if format=='fhir' and (not isinstance(value,dict) or value.get('resourceType')!='Bundle'):
        raise ValueError('FHIR MIME requires a FHIR Bundle')
    return fhir_extract(value,document_id,filename) if isinstance(value,dict) and 'resourceType' in value else generic_extract(value,document_id,filename)
