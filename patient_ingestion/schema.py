"""Typed review facts. Confidence is a rule strength, never a calibrated probability."""
from dataclasses import asdict, dataclass, field
from datetime import date
import math
from src.patient_domain.models import MODEL_ELIGIBLE, identifier, now, json_safe

NUMERIC={'age_years','diabetes_duration_years','smoking_pack_years'}
TEXT={'display_name','date_of_birth','sex','diagnosis_date','smoking_history','alcohol_history',
      'medication_context','other_medication_context','notes'}
LISTS={'diagnoses','medications','labs'}
FIELDS=NUMERIC|TEXT|LISTS
GROUPS={'display_name':'Patient','date_of_birth':'Patient','age_years':'Patient','sex':'Patient',
        'diagnoses':'Diabetes history','diabetes_duration_years':'Diabetes history','diagnosis_date':'Diabetes history',
        'medications':'Medications','medication_context':'Medications','other_medication_context':'Medications',
        'smoking_history':'Lifestyle history','smoking_pack_years':'Lifestyle history','alcohol_history':'Lifestyle history',
        'labs':'Laboratory context','notes':'Other clinical information'}
STATUSES={'EXTRACTED','LOW_CONFIDENCE','CONFLICT','USER_EDITED','CONFIRMED','REJECTED'}


class IngestionError(ValueError):
    def __init__(self, code, message, status=400):
        self.code,self.message,self.status=code,message,status
        super().__init__(message)


def validate_value(name, value):
    if name not in FIELDS:
        raise ValueError('Unsupported clinical field')
    json_safe(value)
    if value is None:
        return
    if name in NUMERIC:
        if type(value) not in (float,int) or not math.isfinite(value) or value<0 or value>150:
            raise ValueError('Invalid numeric clinical value')
    elif name in TEXT:
        if not isinstance(value,str) or len(value)>12000 or not value.strip():
            raise ValueError('Invalid text clinical value')
        if name=='date_of_birth':
            if date.fromisoformat(value)>date.today():
                raise ValueError('Invalid birth date')
        if name=='diagnosis_date':
            if len(value)==4:
                if not value.isdigit() or not 1900<=int(value)<=date.today().year:
                    raise ValueError('Invalid diagnosis year')
            else:
                date.fromisoformat(value)
        if name=='sex' and value not in ('female','male','other','unknown'):
            raise ValueError('Invalid sex')
        if name=='alcohol_history' and value not in ('drinker','non-drinker','unknown'):
            raise ValueError('Invalid alcohol category')
        if name in ('medication_context','other_medication_context') and value not in ('explicit_insulin_term','recorded_without_explicit_insulin_term'):
            raise ValueError('Unsupported medication category')
    elif name in LISTS:
        if not isinstance(value,list) or len(value)>100:
            raise ValueError('Expected bounded clinical list')
        for item in value:
            if name=='diagnoses':
                if not isinstance(item,str) or not item.strip() or len(item)>200:
                    raise ValueError('Invalid diagnosis')
            elif name=='medications':
                if not isinstance(item,dict) or not isinstance(item.get('name'),str) or not item['name'].strip():
                    raise ValueError('Medication name required')
                if set(item)-{'name','dose','frequency','route','status','start_date','stop_date','original_text','provenance'}:
                    raise ValueError('Unsupported medication properties')
                if item.get('status') not in ('active','discontinued','unknown'):
                    raise ValueError('Explicit medication status required')
                for key,val in item.items():
                    if key=='provenance':
                        if not isinstance(val,dict):raise ValueError('Invalid item provenance')
                        continue
                    if val is not None and (not isinstance(val,str) or len(val)>1500):
                        raise ValueError('Invalid medication property')
            else:
                if not isinstance(item,dict) or not isinstance(item.get('test'),str) or not item.get('unit'):
                    raise ValueError('Lab test and original unit required')
                if set(item)-{'test','value','unit','observed_time','available_time','observed_date','original_text','provenance'}:
                    raise ValueError('Unsupported laboratory properties')
                if type(item.get('value')) not in (int,float) or not math.isfinite(item['value']):
                    raise ValueError('Numeric laboratory value required')
                if item.get('observed_date') is not None:date.fromisoformat(item['observed_date'])
                for key in ('observed_time','available_time'):
                    if item.get(key) is not None:
                        from src.patient_domain.models import timestamp
                        timestamp(item[key])


@dataclass(kw_only=True)
class Fact:
    field_name: str
    normalized_value: object
    display_value: str
    confidence: float
    source_document_id: str
    source_filename: str
    source_text_snippet: str
    fact_id: str = field(default_factory=lambda:identifier('entity'))
    unit: str | None = None
    source_page: int | None = None
    source_section: str | None = None
    source_document_date: str | None = None
    source_document_type: str | None = None
    source_evidence: list = field(default_factory=list)
    observed_time: str | None = None
    available_time: str | None = None
    status: str = 'EXTRACTED'
    previous_value: object = None
    decision: str = 'pending'
    conflict_resolution: str | None = None
    original_extracted_value: object = None
    edit_history: list = field(default_factory=list)

    def view(self):
        validate_value(self.field_name,self.normalized_value)
        if self.status not in STATUSES or self.decision not in ('pending','accept','reject'):
            raise ValueError('Invalid review state')
        if type(self.confidence) not in (float,int) or not 0<=self.confidence<=1:
            raise ValueError('Invalid extraction confidence')
        if self.source_page is not None and (type(self.source_page) is not int or self.source_page<1):
            raise ValueError('Invalid source page')
        for key in ('source_document_id','source_filename','source_text_snippet','display_value'):
            if not isinstance(getattr(self,key),str):
                raise ValueError('Invalid provenance')
        result=asdict(self)
        result.update({'model_input_eligible':self.field_name in MODEL_ELIGIBLE,'active_model_input':False,
                       'group':GROUPS[self.field_name],'confidence_label':'High confidence' if self.confidence>=.85 else 'Needs review'})
        json_safe(result)
        return result


def display(name,value):
    if name=='medications':
        return '\n'.join(' · '.join(str(m.get(k)) for k in ('name','dose','frequency','status') if m.get(k)) for m in value)
    if name=='labs':
        return '\n'.join(f"{m['test']}: {m['value']} {m['unit']}"+(f" · {m['observed_time'][:10]}" if m.get('observed_time') else '') for m in value)
    if isinstance(value,list):
        return ', '.join(str(v) for v in value)
    return str(value) if value is not None else 'Not recorded'
