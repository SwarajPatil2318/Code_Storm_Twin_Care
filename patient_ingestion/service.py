"""Patient-owned document lifecycle, review revisions and atomic confirmation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from src.patient_domain.models import (UploadedDocument, ClinicalField, ClinicalProfile, AuditEvent,
                                       identifier, now, json_safe, owned_view, MODEL_ELIGIBLE, PROFILE_FIELDS)
from src.patient_domain.repository import Repository, NotFound
from .schema import Fact, FIELDS, IngestionError, display, validate_value
from .parsers import medication

MAX_UPLOAD=4*1024*1024
FORMATS={'.pdf':('pdf',{'application/pdf'}),'.txt':('text',{'text/plain'}),
         '.md':('text',{'text/plain','text/markdown'}),'.json':('json',{'application/json','application/fhir+json'})}


def canonical(name,value):
    if name=='medications':
        result=[]
        for item in value or []:
            item=medication(item) if isinstance(item,str) else item
            if item:
                result.append({k:item.get(k) for k in ('name','dose','frequency','status')})
        return sorted(result,key=lambda m:m['name'].lower())
    if name=='labs':
        return [{k:item.get(k) for k in ('test','value','unit','observed_time','available_time')} for item in value or []]
    return value


class IngestionService:
    def __init__(self,repository=None,storage=Path('data/patient_domain/uploads'),time=None):
        self.time=time
        self.repo=repository or Repository()
        self.storage=Path(storage).resolve()
        self.storage.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.storage.chmod(0o700)
        with self.repo.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS ingestion_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)')
            versions=[r[0] for r in db.execute('SELECT version FROM ingestion_migrations')]
            if any(v not in (1,2) for v in versions):raise ValueError('Unsupported ingestion schema')
            if not versions:
                db.execute('''CREATE TABLE ingestion_documents(
                    document_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,content_hash TEXT NOT NULL,
                    data TEXT NOT NULL CHECK(json_valid(data)),UNIQUE(patient_id,content_hash),
                    FOREIGN KEY(patient_id,document_id) REFERENCES documents(patient_id,entity_id) ON DELETE CASCADE)''')
                db.execute('''CREATE TABLE extraction_candidates(
                    document_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,revision INTEGER NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),
                    FOREIGN KEY(patient_id,document_id) REFERENCES documents(patient_id,entity_id) ON DELETE CASCADE)''')
                db.execute('''CREATE TABLE extraction_revisions(
                    revision_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,document_id TEXT NOT NULL,revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),
                    FOREIGN KEY(patient_id,document_id) REFERENCES documents(patient_id,entity_id) ON DELETE CASCADE)''')
                db.execute('INSERT INTO ingestion_migrations VALUES(1,?)',(self._stamp(),))
                for table in ('ingestion_documents','extraction_candidates','extraction_revisions'):
                    db.execute(f'CREATE INDEX {table}_owner ON {table}(patient_id,document_id)')
            if 2 not in versions:
                for table in ('ingestion_documents','extraction_candidates','extraction_revisions'):
                    prefix='$.candidate.' if table=='extraction_revisions' else '$.'
                    for operation in ('INSERT','UPDATE'):
                        db.execute(f"""CREATE TRIGGER {table}_ownership_{operation.lower()}
                            BEFORE {operation} ON {table}
                            WHEN coalesce(json_extract(NEW.data,'{prefix}patient_id')=NEW.patient_id,0)=0
                            OR coalesce(json_extract(NEW.data,'{prefix}document_id')=NEW.document_id,0)=0
                            BEGIN SELECT RAISE(ABORT,'Ingestion ownership mismatch'); END""")
                db.execute('INSERT INTO ingestion_migrations VALUES(2,?)',(self._stamp(),))


    def _stamp(self):return self.time() if self.time else now()
    def _audit(self,db,pid,document_id,action,fields=(),revision_id=None):
        event=AuditEvent(patient_id=pid,entity_id=identifier('entity'),action=action,entity_type='document',target_id=document_id,
                         synthetic=self.repo._require(db,pid)['synthetic'],
                         details={'document_id':document_id,'changed_fields':sorted(fields),'revision_id':revision_id})
        db.execute('INSERT INTO audit_events(entity_id,patient_id,data) VALUES(?,?,?)',(event.entity_id,pid,json_safe(owned_view(event))))

    def _document(self,db,pid,did):
        self.repo._require(db,pid)
        row=db.execute('SELECT data FROM ingestion_documents WHERE patient_id=? AND document_id=?',(pid,did)).fetchone()
        if not row:raise NotFound('Document not found')
        return json.loads(row[0])

    def get_document(self,pid,did):
        with self.repo.connection() as db:return self._document(db,pid,did)

    def list_documents(self,pid):
        self.repo.get_patient(pid)
        with self.repo.connection() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT data FROM ingestion_documents WHERE patient_id=? ORDER BY rowid DESC',(pid,))]

    def _save_document(self,db,doc):
        db.execute('UPDATE ingestion_documents SET data=? WHERE patient_id=? AND document_id=?',
                   (json_safe(doc),doc['patient_id'],doc['document_id']))
        row=db.execute('SELECT data FROM documents WHERE patient_id=? AND entity_id=?',(doc['patient_id'],doc['document_id'])).fetchone()
        if row:
            base=json.loads(row[0])
            for name in ('document_date','document_type','synthetic','processing_status','parse_status','extraction_status'):
                base[name]=doc[name]
            base['metadata'].update({'ingestion_schema_version':2,'document_date_known':doc['document_date'] is not None})
            db.execute('UPDATE documents SET data=? WHERE patient_id=? AND entity_id=?',(json_safe(base),doc['patient_id'],doc['document_id']))

    def upload(self,pid,filename,mime,raw):
        self.repo.get_patient(pid)
        if not isinstance(filename,str) or not filename or len(filename)>180 or '/' in filename or '\\' in filename or any(ord(c)<32 for c in filename):
            raise IngestionError('UNSAFE_FILENAME','Choose a file with a simple filename.')
        ext=Path(filename).suffix.lower()
        if ext not in FORMATS:raise IngestionError('UNSUPPORTED_FILE','Use a PDF, text or supported clinical JSON document.',415)
        format,mimes=FORMATS[ext]
        if mime.split(';')[0].strip().lower() not in mimes:raise IngestionError('TYPE_MISMATCH','The file type does not match its extension.',415)
        if format=='json' and mime.split(';')[0].strip().lower()=='application/fhir+json':format='fhir'
        if not raw or len(raw)>MAX_UPLOAD:raise IngestionError('FILE_SIZE','Choose a nonempty document smaller than 4 MB.',413)
        if format=='pdf' and not raw.startswith(b'%PDF-'):raise IngestionError('CORRUPT_PDF','This file does not appear to be a readable PDF.')
        digest=hashlib.sha256(raw).hexdigest()
        did=identifier('entity');path=self.storage/(did.split(':')[1]+ext)
        doc={'document_id':did,'patient_id':pid,'filename':filename,'document_type':'clinical_record','mime_type':mime,
             'uploaded_at':self._stamp(),'document_date':None,'processing_status':'UPLOADED','source_type':format,
             'synthetic':False,'content_hash':digest,'parse_status':'pending','extraction_status':'pending',
             'format':format,'storage_path':str(path),'error':None}
        base=UploadedDocument(patient_id=pid,entity_id=did,filename=filename,document_type='clinical_record',document_date=doc['uploaded_at'][:10],
                              media_type=mime,storage_path=str(path),sha256=digest,metadata={'patient_id':pid,'source_type':'uploaded'},synthetic=False)
        with self.repo.connection() as db:
            self.repo._require(db,pid)
            duplicate=db.execute('SELECT document_id FROM ingestion_documents WHERE patient_id=? AND content_hash=?',(pid,digest)).fetchone()
            if duplicate:raise IngestionError('DUPLICATE_UPLOAD','This record has already been added to this patient.',409)
            try:
                with path.open('xb') as file:file.write(raw)
                path.chmod(0o600)
                # P2 extends the serialized document schema with an honest unknown date.
                # The P1 constructor date above is only validated before this explicit extension.
                base_value=owned_view(base);base_value['document_date']=None
                base_value['metadata'].update({'ingestion_schema_version':2,'document_date_known':False})
                db.execute('INSERT INTO documents(entity_id,patient_id,data) VALUES(?,?,?)',(did,pid,json_safe(base_value)))
                db.execute('INSERT INTO ingestion_documents VALUES(?,?,?,?)',(did,pid,digest,json_safe(doc)))
                self._audit(db,pid,did,'DOCUMENT_UPLOADED')
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return doc

    def _candidate(self,db,pid,did):
        self._document(db,pid,did)
        row=db.execute('SELECT data FROM extraction_candidates WHERE patient_id=? AND document_id=?',(pid,did)).fetchone()
        if not row:raise NotFound('Extraction not started')
        return json.loads(row[0])

    def get_extraction(self,pid,did):
        with self.repo.connection() as db:return self._candidate(db,pid,did)

    @staticmethod
    def _profile(db,pid):
        return json.loads(db.execute('SELECT data FROM clinical_profiles WHERE patient_id=?',(pid,)).fetchone()[0])

    @staticmethod
    def _previous(profile,name):
        if name in profile.get('confirmed_information',{}):return profile['confirmed_information'][name]['value']
        return profile.get('fields',{}).get(name,{}).get('value')

    def _conflicts(self,facts,profile,patient):
        for fact in facts:
            name=fact['field_name'];previous=self._previous(profile,name)
            if previous is None and name in ('date_of_birth','sex'):
                previous=patient.get(name)
                if previous=='unknown':previous=None
            if previous is not None and canonical(name,previous)!=canonical(name,fact['normalized_value']):
                if name=='labs':
                    conflicting=any(a['test'].casefold()==b['test'].casefold() and (a.get('observed_time') or a.get('observed_date')) is not None
                                    and (a.get('observed_time') or a.get('observed_date'))==(b.get('observed_time') or b.get('observed_date')) and (a['unit']!=b['unit'] or a['value']!=b['value'])
                                    for a in previous for b in fact['normalized_value'])
                    if not conflicting:continue  # Distinct dated measurements remain historical facts.
                fact.update({'previous_value':previous,'status':'CONFLICT','decision':'pending'})

    def extract(self,pid,did):
        with self.repo.connection() as db:
            doc=self._document(db,pid,did)
            existing=db.execute('SELECT 1 FROM extraction_candidates WHERE patient_id=? AND document_id=?',(pid,did)).fetchone()
            if existing:return self._candidate(db,pid,did)
            if doc['processing_status'] in ('PARSING','EXTRACTING','VALIDATING'):
                raise IngestionError('PROCESSING','This record is still being read.',409)
            doc['processing_status']='PARSING';self._save_document(db,doc)
        try:
            source=Path(doc['storage_path']).resolve()
            if source.parent!=self.storage or source.stem!=did.split(':')[1] or hashlib.sha256(source.read_bytes()).hexdigest()!=doc['content_hash']:
                raise IngestionError('SOURCE_CHANGED','The source file did not pass its integrity check. Upload the original again.')
            result=subprocess.run([sys.executable,'-m','patient_ingestion.worker',doc['storage_path'],doc['format'],did,doc['filename']],
                                  capture_output=True,timeout=15)
            if result.returncode!=0:raise IngestionError('PARSE_LIMIT','This document could not be read within safe processing limits. Try another file or enter information manually.')
            message=json.loads(result.stdout)
            if not message['ok']:raise IngestionError(message['code'],message['message'])
            parsed=message['result']
            with self.repo.connection() as db:
                doc=self._document(db,pid,did);doc.update({'processing_status':'EXTRACTING','parse_status':'parsed'});self._save_document(db,doc)
            if not parsed['facts']:raise IngestionError('NO_INFORMATION',"We couldn't reliably extract health information from this file. Try another document or enter it manually.")
            with self.repo.connection() as db:
                doc=self._document(db,pid,did)
                patient=self.repo._require(db,pid);profile=self._profile(db,pid)
                facts=parsed['facts']
                for fact in facts:
                    if fact['source_document_id']!=did:raise ValueError('Extraction provenance mismatch')
                    validate_value(fact['field_name'],fact['normalized_value'])
                self._conflicts(facts,profile,patient)
                candidate={'patient_id':pid,'document_id':did,'revision':1,'status':'REVIEW_REQUIRED',
                           'created_at':self._stamp(),'facts':facts,'profile_snapshot':profile,'confirmed_at':None,'active_model_input':False}
                doc.update({'processing_status':'REVIEW_REQUIRED','parse_status':'parsed','extraction_status':'review_required',
                            'document_date':parsed['document_date'],'source_type':parsed['source_type'],'document_type':parsed['document_type'],
                            'synthetic':parsed['synthetic'],'error':None})
                self._save_document(db,doc)
                db.execute('INSERT INTO extraction_candidates VALUES(?,?,?,?)',(did,pid,1,json_safe(candidate)))
                self._revision(db,candidate,'extracted')
                self._audit(db,pid,did,'DOCUMENT_PARSED')
                self._audit(db,pid,did,'EXTRACTION_COMPLETED',[f['field_name'] for f in facts])
            return candidate
        except Exception as error:
            if isinstance(error,IngestionError):code,message=error.code,error.message
            else:code,message='PROCESSING_FAILED',"We couldn't reliably extract health information from this file. Try another document or enter it manually."
            with self.repo.connection() as db:
                doc=self._document(db,pid,did)
                doc.update({'processing_status':'FAILED','parse_status':'failed','extraction_status':'failed','error':{'code':code,'message':message}})
                self._save_document(db,doc)
            raise IngestionError(code,message)

    def _revision(self,db,candidate,operation):
        rid=identifier('entity')
        db.execute('INSERT INTO extraction_revisions VALUES(?,?,?,?,?,?)',(rid,candidate['patient_id'],candidate['document_id'],
                   candidate['revision'],self._stamp(),json_safe({'operation':operation,'candidate':candidate})))
        return rid

    def edit(self,pid,did,revision,edits):
        with self.repo.connection() as db:
            candidate=self._candidate(db,pid,did)
            if type(revision) is not int or not isinstance(edits,list) or not 1<=len(edits)<=30:raise ValueError('Invalid review edits')
            if candidate['status']=='CONFIRMED':raise IngestionError('ALREADY_CONFIRMED','This review has already been confirmed.',409)
            if candidate['revision']!=revision:raise IngestionError('STALE_REVIEW','This review changed. Reload it before continuing.',409)
            seen=set();changed=[];resolved=[]
            for edit in edits:
                if not isinstance(edit,dict) or set(edit)-{'fact_id','field_name','value','decision','resolution'}:raise ValueError('Invalid edit schema')
                key=edit.get('fact_id')
                marker=key if key is not None else 'new:'+str(edit.get('field_name'))
                if marker in seen:raise ValueError('Duplicate edit')
                seen.add(marker)
                fact=next((f for f in candidate['facts'] if f['fact_id']==key),None)
                if fact is None:
                    if key is not None or 'field_name' not in edit or 'value' not in edit:raise ValueError('Unknown fact')
                    name=edit['field_name'];validate_value(name,edit['value'])
                    if isinstance(edit['value'],list) and any(isinstance(item,dict) and 'provenance' in item for item in edit['value']):
                        raise ValueError('Manual values cannot supply source provenance')
                    if any(f['field_name']==name for f in candidate['facts']):raise ValueError('Edit the existing field')
                    fact=Fact(field_name=name,normalized_value=edit['value'],display_value=display(name,edit['value']),confidence=0,
                              source_document_id=did,source_filename='User-entered information',source_text_snippet='User entry: '+display(name,edit['value']),source_section='Manual review',
                              status='USER_EDITED',original_extracted_value=None).view()
                    self._conflicts([fact],candidate['profile_snapshot'],self.repo._require(db,pid))
                    candidate['facts'].append(fact)
                original=fact['normalized_value']
                decision=edit.get('decision',fact['decision'])
                if decision not in ('pending','accept','reject'):raise ValueError('Invalid decision')
                resolution=edit.get('resolution')
                if fact['previous_value'] is not None and decision=='accept':
                    if resolution not in ('keep_previous','use_new','use_edited'):raise IngestionError('UNRESOLVED_CONFLICT','Choose which information to keep before confirming.',409)
                    if resolution=='keep_previous':
                        edit={**edit,'value':fact['previous_value']}
                    fact['conflict_resolution']=resolution;resolved.append(fact['field_name'])
                if 'value' in edit:
                    if not (resolution=='keep_previous' and fact['previous_value'] is not None) and isinstance(edit['value'],list) and any(isinstance(item,dict) and 'provenance' in item for item in edit['value']):
                        raise ValueError('Source provenance cannot be supplied through a value edit')
                    validate_value(fact['field_name'],edit['value'])
                    fact['normalized_value']=edit['value'];fact['display_value']=display(fact['field_name'],edit['value'])
                    fact['status']='USER_EDITED'
                if decision=='reject':fact['status']='REJECTED'
                elif fact['status']=='CONFLICT' and resolution:fact['status']='USER_EDITED'
                fact['decision']=decision
                fact['edit_history'].append({'original_value':original,'new_value':fact['normalized_value'],'decision':decision,
                                             'resolution':resolution,'edited_at':self._stamp(),'actor':'local_reviewer'})
                changed.append(fact['field_name'])
            candidate['revision']+=1
            rid=self._revision(db,candidate,'reviewed')
            db.execute('UPDATE extraction_candidates SET revision=?,data=? WHERE patient_id=? AND document_id=?',
                       (candidate['revision'],json_safe(candidate),pid,did))
            self._audit(db,pid,did,'EXTRACTION_EDITED',changed,rid)
            if resolved:self._audit(db,pid,did,'CONFLICT_RESOLVED',resolved,rid)
            return candidate

    def confirm(self,pid,did,revision,explicit):
        if type(revision) is not int:raise ValueError('Integer review revision required')
        if explicit is not True:raise IngestionError('CONFIRM_REQUIRED','Explicit confirmation is required.',409)
        with self.repo.connection() as db:
            candidate=self._candidate(db,pid,did)
            if candidate['status']=='CONFIRMED':raise IngestionError('ALREADY_CONFIRMED','This review has already been confirmed.',409)
            if candidate['revision']!=revision:raise IngestionError('STALE_REVIEW','This review changed. Reload it before continuing.',409)
            facts=candidate['facts']
            if not facts or any(f['decision']=='pending' for f in facts):raise IngestionError('REVIEW_REQUIRED','Accept or exclude every item before confirming.',409)
            if any(f['decision']=='accept' and f['previous_value'] is not None and not f['conflict_resolution'] for f in facts):
                raise IngestionError('UNRESOLVED_CONFLICT','Resolve the information changes before confirming.',409)
            accepted=[f for f in facts if f['decision']=='accept']
            if not accepted:raise IngestionError('NO_ACCEPTED_FIELDS','Accept at least one item to update the profile.',409)
            profile=self._profile(db,pid)
            if profile!=candidate['profile_snapshot']:raise IngestionError('PROFILE_CHANGED','The clinical profile changed during this review. Start a fresh review before confirming.',409)
            info=profile.setdefault('confirmed_information',{})
            history=profile.setdefault('confirmation_history',[])
            for fact in accepted:
                name=fact['field_name'];value=fact['normalized_value'];validate_value(name,value)
                previous=self._previous(profile,name)
                if name in ('medications','labs'):
                    def with_source(items,record_id,evidence):
                        result=[]
                        for item in items or []:
                            item=medication(item) if isinstance(item,str) else dict(item)
                            if item:
                                item.setdefault('provenance',{'source_document_id':record_id,'evidence':evidence})
                                result.append(item)
                        return result
                    prior_info=info.get(name,profile.get('fields',{}).get(name,{}))
                    previous=with_source(previous,prior_info.get('source_document_id'),prior_info.get('provenance',prior_info.get('source_provenance',{})))
                    value=with_source(value,did,{'source_filename':fact['source_filename'],'source_page':fact['source_page'],
                                               'source_section':fact['source_section'],'source_text_snippet':fact['source_text_snippet'],
                                               'source_evidence':fact.get('source_evidence',[])})
                if name=='labs':
                    value=self._merge_labs(previous,value)
                elif name=='medications':
                    value=self._merge_medications(previous,value)
                    med_history=profile.setdefault('medication_history',[])
                    if not med_history or canonical('medications',previous)!=canonical('medications',value):
                        med_history.append({'document_id':did,'document_date':self._document(db,pid,did)['document_date'],
                            'confirmed_at':self._stamp(),'previous':previous,'selected':value,'source_fact_id':fact['fact_id']})
                    else:
                        med_history[-1].setdefault('corroborating_documents',[]).append({'document_id':did,'confirmed_at':self._stamp()})
                provenance={'kind':'explicitly_confirmed_record','fact_id':fact['fact_id'],'source_filename':fact['source_filename'],
                            'source_page':fact['source_page'],'source_section':fact['source_section'],
                            'source_text_snippet':fact['source_text_snippet'],'source_evidence':fact.get('source_evidence',[]),
                            'source_document_date':fact.get('source_document_date'),'original_extracted_value':fact['original_extracted_value'],
                            'edit_history':fact['edit_history'],'confidence':fact['confidence'],'confirmed_at':self._stamp()}
                info[name]={'value':value,'source_document_id':did,'provenance':provenance,'model_input_eligible':name in MODEL_ELIGIBLE,'active_model_input':False}
                if name in PROFILE_FIELDS:
                    profile['fields'][name]=ClinicalField(patient_id=pid,name=name,value=value,unit=fact['unit'],
                        observed_time=fact.get('observed_time'),available_time=fact.get('available_time'),
                        source_document_id=did,source_provenance=provenance,review_status='confirmed').view()
                fact['status']='CONFIRMED'
            profile['updated_at']=self._stamp();profile['active_model_input']=False
            history.append({'document_id':did,'confirmed_at':profile['updated_at'],'accepted_fields':[f['field_name'] for f in accepted]})
            db.execute('UPDATE clinical_profiles SET data=? WHERE patient_id=?',(json_safe(profile),pid))
            candidate.update({'status':'CONFIRMED','confirmed_at':self._stamp(),'revision':candidate['revision']+1})
            db.execute('UPDATE extraction_candidates SET revision=?,data=? WHERE patient_id=? AND document_id=?',
                       (candidate['revision'],json_safe(candidate),pid,did))
            rid=self._revision(db,candidate,'confirmed')
            doc=self._document(db,pid,did);doc.update({'processing_status':'CONFIRMED','extraction_status':'confirmed'});self._save_document(db,doc)
            self._audit(db,pid,did,'PROFILE_CONFIRMED',[f['field_name'] for f in accepted],rid)
            return profile

    @staticmethod
    def _merge_labs(old,new):
        result=list(old or [])
        for item in new:
            key=(item['test'].casefold(),item.get('observed_time') or item.get('observed_date'),item['unit'],item['value'])
            same=next((i for i,x in enumerate(result) if (x['test'].casefold(),x.get('observed_time') or x.get('observed_date'),x['unit'],x['value'])==key),None)
            if same is not None:continue
            replacement=next((i for i,x in enumerate(result) if (item.get('observed_time') or item.get('observed_date')) is not None and
                              (x['test'].casefold(),x.get('observed_time') or x.get('observed_date'))==key[:2]),None)
            if replacement is None:result.append(item)
            else:result[replacement]=item
        return result

    @staticmethod
    def _merge_medications(old,new):
        result=[medication(m) if isinstance(m,str) else m for m in old or []]
        result=[m for m in result if m]
        for item in new:
            position=next((i for i,m in enumerate(result) if m['name'].casefold()==item['name'].casefold()),None)
            if position is None:result.append(item)
            else:result[position]=item
        return result

    def manual(self,pid):
        doc=self.upload(pid,'manual-entry.txt','text/plain',('Manual clinical entry '+self._stamp()).encode())
        with self.repo.connection() as db:
            candidate={'patient_id':pid,'document_id':doc['document_id'],'revision':1,'status':'REVIEW_REQUIRED','created_at':self._stamp(),
                       'facts':[],'profile_snapshot':self._profile(db,pid),'confirmed_at':None,'active_model_input':False}
            db.execute('INSERT INTO extraction_candidates VALUES(?,?,?,?)',(doc['document_id'],pid,1,json_safe(candidate)))
            doc.update({'processing_status':'REVIEW_REQUIRED','source_type':'manual','document_type':'manual_entry','document_date':doc['uploaded_at'][:10],
                        'extraction_status':'review_required','parse_status':'manual'})
            self._save_document(db,doc);self._revision(db,candidate,'manual_entry')
        return candidate

    def refresh(self,pid,did):
        with self.repo.connection() as db:
            candidate=self._candidate(db,pid,did)
            if candidate['status']=='CONFIRMED':raise IngestionError('ALREADY_CONFIRMED','This review is already confirmed.',409)
            profile=self._profile(db,pid)
            for fact in candidate['facts']:
                fact.update({'decision':'pending','conflict_resolution':None,'previous_value':None,'status':'USER_EDITED' if fact['edit_history'] else 'EXTRACTED'})
            self._conflicts(candidate['facts'],profile,self.repo._require(db,pid))
            candidate['profile_snapshot']=profile;candidate['revision']+=1
            self._revision(db,candidate,'review_refreshed')
            db.execute('UPDATE extraction_candidates SET revision=?,data=? WHERE patient_id=? AND document_id=?',(candidate['revision'],json_safe(candidate),pid,did))
            self._audit(db,pid,did,'EXTRACTION_EDITED',[f['field_name'] for f in candidate['facts']])
            return candidate

    def revisions(self,pid,did):
        with self.repo.connection() as db:
            self._document(db,pid,did)
            return [{'revision_id':r['revision_id'],'created_at':r['created_at'],**json.loads(r['data'])}
                    for r in db.execute('SELECT * FROM extraction_revisions WHERE patient_id=? AND document_id=? ORDER BY rowid',(pid,did))]
