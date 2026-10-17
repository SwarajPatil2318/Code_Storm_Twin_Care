"""Receipt-safe bounded stream memory, explicit conflicts, and native window alignment."""
import numpy as np
from src.metabolic.windows import MINUTE,nearest_indices
from src.metabolic_d2.data import cgm_features,personal_features

class Buffer:
    def __init__(self,patient_id,clock_offset_minutes=0):
        self.patient_id=patient_id;self.clock_offset_minutes=clock_offset_minutes
        self.events={};self.as_of=None;self.time_basis=None;self.cgm_source=None
    def ingest(self,event):
        if event.patient_id!=self.patient_id:raise ValueError('Cross-patient stream event')
        if self.time_basis is not None and event.time_basis!=self.time_basis:raise ValueError('Mixed stream clock bases')
        if event.event_id in self.events:
            if self.events[event.event_id]!=event:raise ValueError('Conflicting event ID')
            return False
        if event.concept=='glucose':
            if self.cgm_source is not None and self.cgm_source!=event.source:raise ValueError('Multiple CGM sources cannot be silently merged')
            self.cgm_source=event.source
        self.time_basis=event.time_basis;self.events[event.event_id]=event
        self.as_of=max(event.available_ns,self.as_of or event.available_ns)
        # Keep eight days for seven-day prefix plus current-window embargo.
        boundary=self.as_of-8*1440*MINUTE
        self.events={k:v for k,v in self.events.items() if v.observed_ns>=boundary or v.concept=='ehr'}
        return True
    def visible(self,cutoff=None,concept=None):
        cutoff=self.as_of if cutoff is None else cutoff
        return sorted([v for v in self.events.values() if v.observed_ns<=cutoff and v.available_ns<=cutoff and (concept is None or v.concept==concept)],key=lambda v:(v.observed_ns,v.available_ns,v.event_id))
    def glucose(self,cutoff=None):
        values={};conflicts=set()
        for event in self.visible(cutoff,'glucose'):
            if event.value is None:continue
            key=event.observed_ns
            if key in values and values[key].value!=event.value:conflicts.add(key)
            else:values.setdefault(key,event)
        usable=[v for k,v in sorted(values.items()) if k not in conflicts]
        return np.array([v.observed_ns for v in usable],dtype='int64'),np.array([v.value for v in usable],float),usable,conflicts
    def features(self,cutoff=None):
        cutoff=self.as_of if cutoff is None else cutoff
        t,g,events,conflicts=self.glucose(cutoff)
        if not len(t):return None,{'status':'No usable CGM','conflicts':len(conflicts)}
        age={'conflicts':len(conflicts),'last_observed_time':events[-1].observed_time,
            'receipt_age_minutes':float((cutoff-events[-1].available_ns)/MINUTE),
            'observation_age_minutes':float((cutoff-t[-1])/MINUTE),
            'receipt_policy':'Both observation and receipt must be at or before the cutoff'}
        if cutoff-t[-1]>7.5*MINUTE:return None,{'status':'Latest CGM is stale',**age}
        if (np.diff(t)<7.5*MINUTE).any():return None,{'status':'Unsupported dense CGM cadence',**age}
        wanted=cutoff-np.arange(120,-1,-15)*MINUTE;indices,found=nearest_indices(t,wanted,7.5*MINUTE)
        if not found.all() or (np.diff(indices)<=0).any() or (np.diff(t[indices])>22.5*MINUTE).any():
            return None,{'status':'A complete native 120-minute history is required',**age}
        sequence=g[indices][None,:].astype('float32');shift=self.clock_offset_minutes*MINUTE
        x=cgm_features(sequence,np.array([cutoff+shift]))
        personal,warm=personal_features(t+shift,g,np.array([cutoff+shift]),sequence[:,-1],x[:,10])
        data={'sequence':sequence,'x':x,'personal':personal,'warm':warm,'time':np.array([cutoff+shift])}
        quality={'status':'Ready','conflicts':len(conflicts),'last_observed_time':events[-1].observed_time,
            'receipt_age_minutes':float((cutoff-events[-1].available_ns)/MINUTE),'observation_age_minutes':float((cutoff-t[-1])/MINUTE),
            'history_observations':9,'warm_prefix':bool(warm[0]),'receipt_policy':'Both observation and receipt must be at or before the cutoff'}
        return data,quality
