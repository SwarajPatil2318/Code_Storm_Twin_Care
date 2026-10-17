"""Read-only adapter to the existing causal Buffer/Shadow feature contracts."""
from dataclasses import dataclass
from datetime import datetime,timezone
from functools import cached_property
import inspect,json,math
from pathlib import Path
from src.metabolic_d3.buffer import Buffer
from src.metabolic_d3.shadow import Shadow,CONFIG
from src.metabolic_d3.events import timestamp
from src.metabolic.windows import MINUTE,nearest_indices
import numpy as np

STATES=('PROFILE_INCOMPLETE','CGM_NOT_CONNECTED','COLLECTING_HISTORY',
    'PERSONALIZATION_BUILDING','PERSONALIZED_TWIN_READY','DATA_STALE','SOURCE_ERROR')


def discovered_contract():
    # Literal assertions detect drift in runtime rules not exported as constants.
    source=inspect.getsource(Buffer.features)
    shadow_source=inspect.getsource(Shadow.predict)
    for rule in ['np.arange(120,-1,-15)','7.5*MINUTE','22.5*MINUTE']:
        if rule not in source:raise RuntimeError('Forecast readiness contract requires review')
    if '50*60*MINUTE' not in shadow_source:raise RuntimeError('Personalization readiness contract requires review')
    d2=json.loads(Path('configs/metabolic_d2.json').read_text())
    if CONFIG['native_cgm_period_minutes']!=15 or CONFIG['alignment_tolerance_minutes']!=7.5 or d2['history_minutes']!=120:
        raise RuntimeError('Native history contract requires review')
    return {'nominal_history_minutes':d2['history_minutes'],'native_cadence_minutes':CONFIG['native_cgm_period_minutes'],
        'history_slots':9,'alignment_tolerance_minutes':CONFIG['alignment_tolerance_minutes'],'minimum_spacing_minutes':7.5,
        'maximum_selected_gap_minutes':22.5,'stale_after_minutes':7.5,'required_slot_coverage':1.0,
        'personal_prefix_minimum_observations':CONFIG['shadow_minimum_observations'],'personal_minimum_elapsed_hours':50,
        'personal_embargo_minutes':CONFIG['shadow_embargo_minutes'],'personal_lookback_days':CONFIG['shadow_lookback_days'],
        'hour_minimum_samples':CONFIG['shadow_hour_minimum_samples'],'hour_minimum_days':CONFIG['shadow_hour_minimum_days'],
        'hour_radius_minutes':CONFIG['shadow_hour_radius_minutes'],'meal_analogs_optional':True,
        'warm_prefix_observations':d2['warm_prefix_equivalent_hours']*4,'source_files':['src/metabolic_d3/buffer.py',
            'src/metabolic_d3/shadow.py','src/metabolic_d2/data.py','configs/metabolic_d2.json','configs/metabolic_d3.json']}


@dataclass(frozen=True)
class ReadinessObservation:
    """Preserve P1 identity and synthetic provenance without relabeling as a demo cohort.

    Buffer accepts this event protocol. The old Event constructor is intentionally
    restricted to its original demo namespace; P5 does not loosen that restriction.
    """
    event_id:str
    patient_id:str
    source:str
    observed_time:str
    available_time:str
    value:float|None
    synthetic:bool
    concept:str='glucose'
    context:dict|None=None
    @cached_property
    def observed_ns(self):return timestamp(self.observed_time)[0]
    @cached_property
    def available_ns(self):return timestamp(self.available_time)[0]
    @cached_property
    def time_basis(self):return timestamp(self.observed_time)[1]


def build_buffer(pid,events):
    buffer=Buffer(pid);rejected=0;unknown_cadence=False
    for event in sorted(events,key=lambda e:(timestamp(e['received_at'])[0],timestamp(e['observed_at'])[0],e['event_id'])):
        value=event['payload']['glucose_mg_dl'];good=event['quality']['status']=='GOOD'
        if not good or type(value) not in (float,int) or not math.isfinite(value) or value<=0:value=None;rejected+=1
        cadence=event['provenance'].get('native_period_minutes')
        # The supplied P3 replay adapter declares cadence; undeclared imports do not qualify.
        if cadence!=15:unknown_cadence=True
        buffer.ingest(ReadinessObservation(event['event_id'],pid,event['provider_type']+':'+str(event.get('device_id') or ''),
            event['observed_at'],event['received_at'],value,event['provenance']['synthetic'],context=event['provenance']))
    return buffer,rejected,unknown_cadence


def feature_check(buffer,cutoff,unknown_cadence=False):
    data,quality=buffer.features(cutoff)
    if unknown_cadence:return None,{**quality,'status':'Declared native 15-minute cadence required'}
    if data is not None and (not np.isfinite(data['sequence']).all() or not np.isfinite(data['x']).all()):
        return None,{**quality,'status':'Nonfinite feature values'}
    return data,quality
