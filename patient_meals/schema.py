"""Strict meal estimates and confirmed context with separate provenance."""
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
import math
import re
from uuid import uuid5, NAMESPACE_URL
from src.patient_domain.models import identifier, validate_id, json_safe
from patient_connections.contracts import clock, check_owner

CATEGORIES=('LOW','MODERATE','HIGH','UNKNOWN')
MEAL_FIELDS=frozenset(('title','foods','meal_occurred_at','carbohydrate_category','glycemic_impact_category',
                      'estimated_carb_range_g','user_notes'))
NUTRITION_FIELDS=frozenset(('estimated_gi_range','glycemic_index_category','glycemic_load_category'))


class MealError(ValueError):
    def __init__(self,message,status=400):self.status=status;super().__init__(message)


def text(value,limit=500,required=False):
    if not isinstance(value,str) or len(value)>limit or (required and not value.strip()) or '\x00' in value:
        raise ValueError('Invalid meal text')
    # Neither the provider nor user meal notes may become fabricated prediction statements.
    if re.search(r'\b(?:exactly\s+\d|(?:glucose|blood sugar)\s+(?:will|shall)|(?:caused|causes)\s+(?:the\s+)?(?:glucose\s+)?spike)',value,re.I):
        raise ValueError('Meal context cannot assert precise predictions or causation')
    return value.strip()


def confidence(value,nullable=False):
    if value is None and nullable:return None
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid confidence')
    return value


def category(value):
    if value not in CATEGORIES:raise ValueError('Invalid category')
    return value


def carb_range(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'low','high'} or any(type(v) is not int for v in value.values()):
        raise ValueError('An approximate whole-gram range is required')
    if not 0<=value['low']<value['high']<=1000 or value['high']-value['low']<10:raise ValueError('Range must express uncertainty')
    return dict(value)


def gi_range(value):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'low','high'} or any(type(v) is not int for v in value.values()):
        raise ValueError('Approximate whole-number GI range required')
    if not 0<=value['low']<value['high']<=100 or value['high']-value['low']<5:
        raise ValueError('GI range must express uncertainty within 0–100')
    return dict(value)


def foods(value,manual=False):
    if not isinstance(value,list) or len(value)>20:raise ValueError('Maximum 20 food items')
    result=[]
    for food in value:
        if not isinstance(food,dict) or set(food)-{'name','estimated_portion_description','confidence','carb_relevance','estimated_gi_range'}:raise ValueError('Invalid food item')
        if manual and food.get('confidence') is not None:raise ValueError('Manual entry has no AI confidence')
        result.append({'name':text(food.get('name'),120,True),'estimated_portion_description':text(food.get('estimated_portion_description',''),250),
            'confidence':None if manual else confidence(food.get('confidence'),True),'carb_relevance':category(food.get('carb_relevance','UNKNOWN')),
            **({'estimated_gi_range':gi_range(food['estimated_gi_range'])} if 'estimated_gi_range' in food else {})})
    return result


@dataclass(frozen=True,kw_only=True)
class MealAnalysisResult:
    analysis_id:str
    patient_id:str
    foods:list
    overall_confidence:float
    carbohydrate_category:str
    glycemic_impact_category:str
    notes:str
    uncertainties:list
    provider:str
    created_at:str
    estimated_carb_range_g:dict|None=None
    model_version:str|None=None
    estimated_gi_range:dict|None=None
    glycemic_index_category:str='UNKNOWN'
    glycemic_load_category:str='UNKNOWN'

    def view(self):
        validate_id(self.patient_id);validate_id(self.analysis_id,'entity');clock(self.created_at)
        result=asdict(self);result['foods']=foods(self.foods)
        confidence(self.overall_confidence);category(self.carbohydrate_category);category(self.glycemic_impact_category)
        result['notes']=text(self.notes,1000);text(self.provider,80,True)
        if self.model_version is not None:text(self.model_version,100,True)
        if not isinstance(self.uncertainties,list) or len(self.uncertainties)>20:raise ValueError('Invalid uncertainties')
        result['uncertainties']=[text(s,300,True) for s in self.uncertainties]
        result['estimated_carb_range_g']=carb_range(self.estimated_carb_range_g)
        result['estimated_gi_range']=gi_range(self.estimated_gi_range)
        category(self.glycemic_index_category);category(self.glycemic_load_category)
        if self.glycemic_index_category=='UNKNOWN' and self.estimated_gi_range is not None:
            raise ValueError('Unknown GI cannot carry a numerical estimate')
        if self.overall_confidence<.5 or not result['foods']:
            if self.carbohydrate_category!='UNKNOWN' or self.glycemic_impact_category!='UNKNOWN' or self.estimated_carb_range_g is not None:
                raise ValueError('Uncertain image result must retain UNKNOWN and no precise range')
            if self.glycemic_index_category!='UNKNOWN' or self.glycemic_load_category!='UNKNOWN' or self.estimated_gi_range is not None:
                raise ValueError('Uncertain image result must retain unknown GI/GL')
        if self.carbohydrate_category=='UNKNOWN' and self.estimated_carb_range_g is not None:raise ValueError('Unknown category cannot carry a numerical estimate')
        check_owner(result,self.patient_id);json_safe(result);return result


def meal_input(value,as_of=None,manual=False):
    if not isinstance(value,dict) or not MEAL_FIELDS<=set(value) or set(value)-(MEAL_FIELDS|NUTRITION_FIELDS):raise ValueError('Complete meal fields required')
    occurred=clock(value['meal_occurred_at']);wall=as_of or datetime.now(timezone.utc)
    if occurred>wall+timedelta(minutes=5):raise ValueError('Meal time cannot be more than five minutes in the future')
    result={**value,'title':text(value['title'],120,True),'foods':foods(value['foods'],manual),
        'carbohydrate_category':category(value['carbohydrate_category']),'glycemic_impact_category':category(value['glycemic_impact_category']),
        'user_notes':text(value['user_notes'],2000),'estimated_carb_range_g':carb_range(value['estimated_carb_range_g']),
        'meal_occurred_at':occurred.isoformat(),'meal_original_timestamp':value['meal_occurred_at'],
        'meal_timezone_offset_minutes':int(datetime.fromisoformat(value['meal_occurred_at'].replace('Z','+00:00')).utcoffset().total_seconds()/60)}
    result.update(estimated_gi_range=gi_range(value.get('estimated_gi_range')),
                  glycemic_index_category=category(value.get('glycemic_index_category','UNKNOWN')),
                  glycemic_load_category=category(value.get('glycemic_load_category','UNKNOWN')))
    if result['glycemic_index_category']=='UNKNOWN' and result['estimated_gi_range'] is not None:raise ValueError('Unknown GI has no numeric range')
    if result['carbohydrate_category']=='UNKNOWN' and result['estimated_carb_range_g'] is not None:raise ValueError('Unknown category has no numeric range')
    if not result['foods']:raise ValueError('Enter at least one food before confirming')
    json_safe(result);return result


def canonical_meal(meal):
    """P3-compatible MEAL_EVENT extension; a projection referencing the P1 meal."""
    pid=meal['patient_id'];validate_id(pid);clock(meal['meal_occurred_at']);clock(meal['updated_at'])
    event_id='twincare-entity:'+str(uuid5(NAMESPACE_URL,'twincare-meal-event:'+meal['meal_id']))
    result={'event_id':event_id,'patient_id':pid,'source_type':'MEAL_LOGGER','provider_type':'meal_logger',
        'event_type':'MEAL_EVENT','observed_at':meal['meal_occurred_at'],'received_at':meal['updated_at'],
        'schema_version':'twincare-event/1','sequence_id':meal['meal_id'],'device_id':None,'raw_reference':None,'ingestion_trace_id':None,
        'payload':{'meal_id':meal['meal_id'],'carbohydrate_category':meal['carbohydrate_category'],
            'glycemic_impact_category':meal['glycemic_impact_category'],'foods_summary':[f['name'] for f in meal['foods']],
            'meal_occurred_at':meal['meal_occurred_at'],'entry_method':meal['entry_method']},
        'quality':{'status':'GOOD','reasons':['User-confirmed contextual estimate; not a validated model input']},
        'provenance':{'origin':'confirmed_P1_MealEvent','synthetic':meal['synthetic'],'meal_id':meal['meal_id'],
            'analysis_provider':meal['provenance'].get('provider'),'confirmation_required':True,'context_only':True},
        'active_model_input':False}
    check_owner(result,pid);json_safe(result);return result
