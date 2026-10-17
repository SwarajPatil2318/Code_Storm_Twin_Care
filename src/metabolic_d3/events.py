"""Strict canonical event adapters preserving raw units, two clocks and evidence origin."""
from dataclasses import dataclass,asdict
from functools import cached_property
import datetime as dt
import math
import numpy as np

CATEGORIES={'OBSERVED','MODEL PREDICTED','PERSONAL EXPECTED','STATIC CLINICAL','SIMULATED LIVE STREAM'}

def timestamp(value):
    if not isinstance(value,str) or not value:raise ValueError('Explicit observation and receipt timestamps are required')
    parsed=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    basis='UTC' if parsed.tzinfo is not None else 'naive_source_clock'
    if parsed.tzinfo is not None:
        offset=int(parsed.utcoffset().total_seconds()/60)
        native=parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    else:offset=None;native=parsed
    ns=int(np.datetime64(native.isoformat(),'ns').astype('int64'))
    return ns,basis,offset

@dataclass(frozen=True)
class Event:
    event_id:str
    patient_id:str
    source:str
    concept:str
    observed_time:str
    available_time:str
    value:float|None
    unit:str|None
    original_value:float|None
    original_unit:str|None
    synthetic:bool=False
    context:dict|None=None
    native_period_minutes:int|None=None
    def __post_init__(self):
        if self.concept not in ['glucose','heart_rate','steps','activity','meal','ehr']:raise ValueError('Unsupported canonical concept')
        if ':' not in self.patient_id or not self.event_id or not self.source:raise ValueError('Namespaced patient and source/event identities required')
        observed,basis,_=timestamp(self.observed_time);available,receipt_basis,_=timestamp(self.available_time)
        if basis!=receipt_basis:raise ValueError('Observation and receipt clock bases differ')
        if observed>available:raise ValueError('Observation cannot occur after its receipt')
        if self.value is not None and not math.isfinite(self.value):raise ValueError('Nonfinite event value')
        if self.concept=='glucose' and (self.unit!='mg/dL' or (self.value is not None and self.value<=0)):raise ValueError('CGM requires positive mg/dL or explicit missingness')
        if self.concept=='glucose' and self.native_period_minutes!=15:raise ValueError('D2 accepts native 15-minute CGM; other cadences require a separately validated adapter')
        if self.synthetic and not self.patient_id.startswith('metabolic-demo:'):raise ValueError('Synthetic context cannot be attached to real participant identity')
    @cached_property
    def observed_ns(self):return timestamp(self.observed_time)[0]
    @cached_property
    def available_ns(self):return timestamp(self.available_time)[0]
    @cached_property
    def time_basis(self):return timestamp(self.observed_time)[1]
    def payload(self):return asdict(self)

class CGMAdapter:
    def adapt(self,**item):
        original=item['value'];unit=item['unit']
        if unit not in ['mg/dL','mmol/L']:raise ValueError('Unsupported glucose unit')
        value=original*18.0182 if original is not None and unit=='mmol/L' else original
        return Event(**{**item,'concept':'glucose','value':value,'unit':'mg/dL','original_value':original,'original_unit':unit})

class SmartwatchAdapter:
    UNITS={'heart_rate':'bpm','steps':'count','activity':'MET'}
    def adapt(self,**item):
        if self.UNITS.get(item['concept'])!=item['unit']:raise ValueError('Unsupported wearable concept/unit')
        if item['value'] is not None and item['value']<0:raise ValueError('Negative wearable value')
        return Event(**item,original_value=item['value'],original_unit=item['unit'])

class MealAdapter:
    def adapt(self,**item):
        return Event(**{**item,'concept':'meal','value':None,'unit':None,'original_value':None,'original_unit':None})
