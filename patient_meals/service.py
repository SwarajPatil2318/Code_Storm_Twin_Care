"""Meal source of truth, review gate, event projection and causal version queries."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import re
from src.patient_domain.models import (MealEvent,Observation,AuditEvent,identifier,owned_view,json_safe,now)
from src.patient_domain.repository import Repository,NotFound
from patient_connections.contracts import clock
from .schema import MealAnalysisResult,meal_input,canonical_meal,MealError,MEAL_FIELDS,NUTRITION_FIELDS
from .providers import RESULT_FIELDS,configured_provider,valid_result_fields
from .images import validated_image


class Meals:
    def __init__(self,repository=None,storage=Path('data/patient_domain/meal_photos'),provider=None,time=None,engineering_test=False):
        self.repo=repository or Repository();self.storage=Path(storage).resolve();self.storage.mkdir(parents=True,exist_ok=True,mode=0o700);self.storage.chmod(0o700)
        self.provider=provider or configured_provider();self.time=time or (lambda:datetime.now(timezone.utc))
        if self.provider.engineering_only and not engineering_test:raise ValueError('Fixture provider is restricted to engineering tests')
        self.engineering_test=engineering_test
        with self.repo.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS meal_migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL)')
            versions=[r[0] for r in db.execute('SELECT version FROM meal_migrations')]
            if any(v!=1 for v in versions):raise ValueError('Unknown meal schema')
            if not versions:
                db.execute('''CREATE TABLE meal_photo_assets(asset_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    content_hash TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),UNIQUE(patient_id,asset_id),UNIQUE(patient_id,content_hash))''')
                db.execute('''CREATE TABLE meal_drafts(draft_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    asset_id TEXT,revision INTEGER NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),UNIQUE(patient_id,draft_id),
                    FOREIGN KEY(patient_id,asset_id) REFERENCES meal_photo_assets(patient_id,asset_id))''')
                db.execute('''CREATE TABLE meal_analysis_runs(analysis_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,draft_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),
                    FOREIGN KEY(patient_id,draft_id) REFERENCES meal_drafts(patient_id,draft_id) ON DELETE CASCADE)''')
                db.execute('''CREATE TABLE meal_revisions(revision_id TEXT PRIMARY KEY,patient_id TEXT NOT NULL,meal_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,available_at TEXT NOT NULL,data TEXT NOT NULL CHECK(json_valid(data)),
                    FOREIGN KEY(patient_id,meal_id) REFERENCES meal_events(patient_id,entity_id) ON DELETE CASCADE,
                    UNIQUE(patient_id,meal_id,revision))''')
                for table,key in [('meal_photo_assets','asset_id'),('meal_drafts','draft_id'),('meal_analysis_runs','analysis_id'),('meal_revisions','revision_id')]:
                    db.execute(f'CREATE INDEX {table}_owner ON {table}(patient_id)')
                    for op in ('INSERT','UPDATE'):
                        db.execute(f"""CREATE TRIGGER {table}_owner_{op.lower()} BEFORE {op} ON {table}
                            WHEN coalesce(json_extract(NEW.data,'$.patient_id')=NEW.patient_id,0)=0
                            OR coalesce(json_extract(NEW.data,'$.{key}')=NEW.{key},0)=0
                            BEGIN SELECT RAISE(ABORT,'Meal ownership mismatch'); END""")
                db.execute('INSERT INTO meal_migrations VALUES(1,?)',(now(),))
    def _stamp(self):return self.time().isoformat()
    def _audit(self,db,pid,target,action,fields=()):
        value=owned_view(AuditEvent(patient_id=pid,entity_id=identifier('entity'),action=action,entity_type='meal',target_id=target,
            synthetic=self.repo._require(db,pid)['synthetic'],details={'field_names':sorted(fields)}))
        db.execute('INSERT INTO audit_events VALUES(?,?,?)',(value['entity_id'],pid,json_safe(value)))
    def _asset(self,db,pid,asset_id):
        self.repo._require(db,pid);row=db.execute('SELECT data FROM meal_photo_assets WHERE patient_id=? AND asset_id=?',(pid,asset_id)).fetchone()
        if not row:raise NotFound('Photo not found')
        return json.loads(row[0])
    def photo(self,pid,asset_id):
        with self.repo.connection() as db:asset=self._asset(db,pid,asset_id)
        p=Path(asset['storage_path']).resolve()
        if p.parent!=self.storage or p.stem!=asset_id.split(':')[1]:raise NotFound('Photo not found')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=asset['content_hash']:raise MealError('The stored image failed its integrity check.',409)
        return raw,asset['mime_type']
    def _draft(self,db,pid,did):
        self.repo._require(db,pid);row=db.execute('SELECT data FROM meal_drafts WHERE patient_id=? AND draft_id=?',(pid,did)).fetchone()
        if not row:raise NotFound('Meal draft not found')
        return json.loads(row[0])
    def draft(self,pid,did):
        with self.repo.connection() as db:
            d=self._draft(db,pid,did)
            if d['status']=='ANALYSING' and self.time()-clock(d['analysis_started_at'])>timedelta(minutes=2):
                d.update({'status':'FAILED','error':'The previous analysis did not finish. Try again or enter the meal manually.',
                    'revision':d['revision']+1,'updated_at':self._stamp(),'analysis_running':False})
                self._save_draft(db,d);self._audit(db,pid,did,'MEAL_ANALYSIS_FAILED')
            return d
    def _save_draft(self,db,value):
        db.execute('INSERT INTO meal_drafts VALUES(?,?,?,?,?) ON CONFLICT(draft_id) DO UPDATE SET revision=excluded.revision,data=excluded.data WHERE patient_id=excluded.patient_id',
            (value['draft_id'],value['patient_id'],value['photo_asset_reference'],value['revision'],json_safe(value)))
    def _new_draft(self,pid,method,asset=None):
        return {'draft_id':identifier('entity'),'patient_id':pid,'status':'DRAFT','entry_method':method,'photo_asset_reference':asset,
            'analysis_id':None,'original_analysis':None,'revision':1,'created_at':self._stamp(),'updated_at':self._stamp(),
            'reviewed_result':{'title':'Meal','foods':[],'meal_occurred_at':self._stamp(),'carbohydrate_category':'UNKNOWN',
                'glycemic_impact_category':'UNKNOWN','estimated_carb_range_g':None,'user_notes':''},'edit_history':[],'error':None}
    def manual_draft(self,pid,source_draft_id=None):
        with self.repo.connection() as db:
            self.repo._require(db,pid)
            source=self._draft(db,pid,source_draft_id) if source_draft_id else None
            d=self._new_draft(pid,'MANUAL',source['photo_asset_reference'] if source else None)
            self._save_draft(db,d);return d
    def upload(self,pid,filename,mime,raw):
        self.repo.get_patient(pid)
        if not isinstance(filename,str) or len(filename)>180 or not filename or '/' in filename or '\\' in filename or re.search(r'[\x00-\x1f\x7f]',filename):raise MealError('Choose a valid image filename.')
        if Path(filename).suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):raise MealError('Choose a JPEG, PNG or WEBP photo.')
        try:info=validated_image(raw,mime)
        except Exception:raise MealError('This image could not be read. Choose a valid JPEG, PNG or WEBP photo smaller than 8 MB.') from None
        digest=hashlib.sha256(raw).hexdigest();created=None
        try:
            with self.repo.connection() as db:
                self.repo._require(db,pid);row=db.execute('SELECT data FROM meal_photo_assets WHERE patient_id=? AND content_hash=?',(pid,digest)).fetchone()
                if row:asset=json.loads(row[0])
                else:
                    aid=identifier('entity');extension={ 'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}[info['format']];p=self.storage/(aid.split(':')[1]+extension)
                    with p.open('xb') as f:f.write(raw)
                    p.chmod(0o600);created=p
                    asset={'asset_id':aid,'patient_id':pid,'filename':filename,'mime_type':mime,'content_hash':digest,'storage_path':str(p),
                        'width':info['width'],'height':info['height'],'uploaded_at':self._stamp()}
                    db.execute('INSERT INTO meal_photo_assets VALUES(?,?,?,?)',(aid,pid,digest,json_safe(asset)))
                d=self._new_draft(pid,'PHOTO_AI',asset['asset_id']);self._save_draft(db,d)
                self._audit(db,pid,d['draft_id'],'MEAL_PHOTO_UPLOADED');return d
        except BaseException:
            if created:created.unlink(missing_ok=True)
            raise
    def start_analysis(self,pid,did,revision):
        with self.repo.connection() as db:
            d=self._draft(db,pid,did)
            if type(revision) is not int or d['revision']!=revision:raise MealError('This review has changed. Reload before continuing.',409)
            if d['entry_method']!='PHOTO_AI' or d['status'] in ('CONFIRMED','ANALYSING'):raise MealError('This draft cannot be analysed now.',409)
            attempts=db.execute('SELECT count(*) FROM meal_analysis_runs WHERE patient_id=? AND draft_id=?',(pid,did)).fetchone()[0]
            if attempts>=5:raise MealError('Start a new photo draft or enter the meal manually.',409)
            d['status']='ANALYSING';d['revision']+=1;d['updated_at']=self._stamp();d['analysis_started_at']=self._stamp();d['analysis_running']=False;self._save_draft(db,d)
            return d
    def analyze(self,pid,did,revision):
        with self.repo.connection() as db:
            d=self._draft(db,pid,did)
            if type(revision) is not int or d['revision']!=revision or d['status']!='ANALYSING' or d.get('analysis_running'):
                raise MealError('Start or reload this photo analysis before continuing.',409)
            d['analysis_running']=True;d['revision']+=1;self._save_draft(db,d)
        try:
            raw,mime=self.photo(pid,d['photo_asset_reference']);info=validated_image(raw,mime)
            output=self.provider.analyze(info['provider_pixels'],info['provider_mime'],hashlib.sha256(raw).hexdigest())
            if not valid_result_fields(output):raise ValueError('Provider schema mismatch')
            analysis=MealAnalysisResult(analysis_id=identifier('entity'),patient_id=pid,provider=self.provider.name,
                created_at=self._stamp(),**output).view()
            with self.repo.connection() as db:
                current=self._draft(db,pid,did)
                if current['status']!='ANALYSING' or current['revision']!=d['revision']:raise MealError('Meal review changed during analysis.',409)
                db.execute('INSERT INTO meal_analysis_runs VALUES(?,?,?,?,?)',(analysis['analysis_id'],pid,did,analysis['created_at'],json_safe(analysis)))
                current.update({'status':'REVIEW_REQUIRED','original_analysis':analysis,'analysis_id':analysis['analysis_id'],
                    'updated_at':self._stamp(),'error':None,'revision':current['revision']+1})
                current['reviewed_result'].update({k:analysis[k] for k in ('foods','carbohydrate_category','glycemic_impact_category','estimated_carb_range_g')})
                current['reviewed_result'].update({k:analysis[k] for k in NUTRITION_FIELDS})
                self._save_draft(db,current);self._audit(db,pid,did,'MEAL_ANALYSIS_COMPLETED');return current
        except Exception:
            message=("AI analysis unavailable. Enter the meal manually or configure a meal-vision provider."
                     if not self.provider.available else "We couldn't confidently analyse this meal photo. Try again or enter the meal manually.")
            with self.repo.connection() as db:
                current=self._draft(db,pid,did)
                if current['status']=='ANALYSING' and current['revision']==d['revision']:
                    current.update({'status':'FAILED','updated_at':self._stamp(),'revision':current['revision']+1,
                        'error':message})
                    self._save_draft(db,current);self._audit(db,pid,did,'MEAL_ANALYSIS_FAILED')
            raise MealError(message,503) from None
    def review(self,pid,did,revision,value):
        with self.repo.connection() as db:
            d=self._draft(db,pid,did)
            if type(revision) is not int or revision!=d['revision'] or d['status'] in ('CONFIRMED','ANALYSING'):raise MealError('Reload this meal review before editing.',409)
            if d['entry_method']=='PHOTO_AI' and (d['status']!='REVIEW_REQUIRED' or not d['original_analysis']):
                raise MealError('Analyse the photo or choose Enter meal manually before reviewing.',409)
            reviewed=meal_input(value,self.time(),manual=d['entry_method']=='MANUAL')
            reviewed={k:reviewed[k] for k in MEAL_FIELDS|NUTRITION_FIELDS}
            # Keep the reviewed offset representation until confirmation; normalization alone would erase it.
            reviewed['meal_occurred_at']=value['meal_occurred_at']
            if d['entry_method']=='PHOTO_AI' and d['status']=='FAILED':
                # Manual fallback is a new explicit draft; never relabel a failed photo as successful AI.
                raise MealError('Use Enter meal manually to continue without photo analysis.',409)
            d['edit_history'].append({'edited_at':self._stamp(),'previous':d['reviewed_result'],'new':reviewed})
            d.update({'reviewed_result':reviewed,'revision':d['revision']+1,'updated_at':self._stamp(),'status':'REVIEW_REQUIRED'})
            self._save_draft(db,d);self._audit(db,pid,did,'MEAL_EDITED',MEAL_FIELDS);return d
    def _get_meal(self,db,pid,mid,include_deleted=False):
        self.repo._require(db,pid);row=db.execute("SELECT data FROM meal_events WHERE patient_id=? AND entity_id=? AND json_extract(data,'$.meal_schema_version')=4",(pid,mid)).fetchone()
        if not row:raise NotFound('Meal not found')
        value=json.loads(row[0])
        if value['status']=='DELETED' and not include_deleted:raise NotFound('Meal not found')
        return value
    def get(self,pid,mid):
        with self.repo.connection() as db:return self._get_meal(db,pid,mid)
    def list(self,pid):
        with self.repo.connection() as db:
            self.repo._require(db,pid)
            items=[json.loads(r[0]) for r in db.execute("SELECT data FROM meal_events WHERE patient_id=? AND json_extract(data,'$.meal_schema_version')=4 AND json_extract(data,'$.status')='CONFIRMED' ORDER BY json_extract(data,'$.meal_occurred_at') DESC,entity_id",(pid,))]
        return {'patient_id':pid,'items':items}
    def _persist_meal(self,db,meal,action):
        pid=meal['patient_id'];mid=meal['meal_id'];stamp=meal['updated_at']
        db.execute('INSERT INTO meal_events VALUES(?,?,?) ON CONFLICT(entity_id) DO UPDATE SET data=excluded.data WHERE patient_id=excluded.patient_id',(mid,pid,json_safe(meal)))
        rid=identifier('entity');snapshot={'revision_id':rid,'patient_id':pid,'meal_id':mid,'meal':meal}
        db.execute('INSERT INTO meal_revisions VALUES(?,?,?,?,?,?)',(rid,pid,mid,meal['revision'],stamp,json_safe(snapshot)))
        event=canonical_meal(meal)
        if meal['status']=='DELETED':db.execute('DELETE FROM observations WHERE patient_id=? AND entity_id=?',(pid,event['event_id']))
        else:
            observation=owned_view(Observation(patient_id=pid,entity_id=event['event_id'],concept='MEAL_EVENT',raw_value=event['payload'],
                original_unit=None,observed_time=event['observed_at'],available_time=event['received_at'],source_provenance={'canonical_event':event},
                quality_status='GOOD',time_basis='user_recorded',synthetic=meal['synthetic']))
            db.execute('INSERT INTO observations VALUES(?,?,?) ON CONFLICT(entity_id) DO UPDATE SET data=excluded.data WHERE patient_id=excluded.patient_id',(event['event_id'],pid,json_safe(observation)))
        self._audit(db,pid,mid,action,MEAL_FIELDS)
    def confirm(self,pid,did,revision,explicit):
        if explicit is not True:raise MealError('Review and explicitly confirm this meal before saving.',409)
        with self.repo.connection() as db:
            d=self._draft(db,pid,did)
            if type(revision) is not int or revision!=d['revision'] or d['status']!='REVIEW_REQUIRED':raise MealError('Finish reviewing this meal before confirming.',409)
            data=meal_input(d['reviewed_result'],self.time(),manual=d['entry_method']=='MANUAL');stamp=self._stamp();mid=identifier('entity')
            patient=self.repo._require(db,pid);analysis=d['original_analysis'];engineering=bool(analysis and analysis['provider']=='engineering_fixture_provider');synthetic=patient['synthetic'] or engineering
            base=owned_view(MealEvent(patient_id=pid,entity_id=mid,observed_time=data['meal_occurred_at'],available_time=stamp,
                description=data['title'],source_provenance={'origin':'user_confirmed_meal','context_only':True},synthetic=synthetic))
            meal={**base,**data,'meal_id':mid,'meal_schema_version':4,'logged_at':stamp,'confirmed_at':stamp,'created_at':stamp,'updated_at':stamp,
                'entry_method':d['entry_method'],'photo_asset_reference':d['photo_asset_reference'],'analysis_id':d['analysis_id'],
                'analysis_confidence':analysis['overall_confidence'] if analysis else None,'original_analysis':analysis,
                'draft_edit_history':d['edit_history'],'user_confirmed':True,'revision':1,'status':'CONFIRMED',
                'provenance':{'provider':analysis['provider'] if analysis else None,'model_version':analysis['model_version'] if analysis else None,
                    'draft_id':did,'confirmation_required':True,'context_only':True,'engineering_fixture':engineering},
                'active_model_input':False}
            self._persist_meal(db,meal,'MEAL_CONFIRMED');d.update({'status':'CONFIRMED','meal_id':mid,'revision':d['revision']+1,'updated_at':stamp});self._save_draft(db,d)
            return meal
    def edit(self,pid,mid,revision,value,explicit):
        if explicit is not True:raise MealError('Explicit confirmation is required to save meal changes.',409)
        with self.repo.connection() as db:
            meal=self._get_meal(db,pid,mid)
            if type(revision) is not int or meal['revision']!=revision:raise MealError('This meal changed. Reload before editing.',409)
            data=meal_input(value,self.time(),manual=meal['entry_method']=='MANUAL')
            stamp=self._stamp();meal.update(data);meal.update({'observed_time':data['meal_occurred_at'],'description':data['title'],'available_time':stamp,'updated_at':stamp,'revision':meal['revision']+1})
            self._persist_meal(db,meal,'MEAL_EDITED');return meal
    def delete(self,pid,mid,revision):
        with self.repo.connection() as db:
            meal=self._get_meal(db,pid,mid)
            if type(revision) is not int or meal['revision']!=revision:raise MealError('This meal changed. Reload before removing.',409)
            meal.update({'status':'DELETED','updated_at':self._stamp(),'revision':meal['revision']+1})
            self._persist_meal(db,meal,'MEAL_DELETED');return {'removed':True,'meal_id':mid}
    def history(self,pid,mid):
        with self.repo.connection() as db:
            self._get_meal(db,pid,mid,True)
            return {'patient_id':pid,'items':[json.loads(r[0]) for r in db.execute('SELECT data FROM meal_revisions WHERE patient_id=? AND meal_id=? ORDER BY revision',(pid,mid))]}
    def recent_context(self,pid,reference_time,lookback_minutes=120):
        reference=clock(reference_time)
        if type(lookback_minutes) is not int or not 1<=lookback_minutes<=10080:raise ValueError('Lookback must be 1 minute to 7 days')
        if reference>self.time()+timedelta(minutes=5):raise ValueError('Future reference time is not available')
        lower=reference-timedelta(minutes=lookback_minutes)
        with self.repo.connection() as db:
            self.repo._require(db,pid)
            rows=db.execute('SELECT data FROM meal_revisions WHERE patient_id=? AND available_at<=? ORDER BY meal_id,revision',(pid,reference.isoformat()))
            versions={}
            for row in rows:
                value=json.loads(row[0]);versions[value['meal_id']]=value['meal']
        meals=[m for m in versions.values() if m['status']=='CONFIRMED' and m['user_confirmed'] is True and lower<=clock(m['meal_occurred_at'])<=reference]
        meals.sort(key=lambda m:clock(m['meal_occurred_at']))
        return {'patient_id':pid,'reference_time':reference.isoformat(),'lookback_minutes':lookback_minutes,'context_only':True,
            'meals':meals,'evidence':[{'kind':'HIGH_CARB_MEAL' if m['carbohydrate_category']=='HIGH' else 'RECENT_MEAL',
                'meal_id':m['meal_id'],'minutes_since_meal':round((reference-clock(m['meal_occurred_at'])).total_seconds()/60,1),
                'carbohydrate_category':m['carbohydrate_category'],'context_only':True,'causal_claim_allowed':False} for m in meals]}
