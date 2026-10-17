"""Sparse canonical timelines; unsupported/absent modalities stay explicit."""
from collections import defaultdict

from ..features.temporal import time_number

TWIN_MODALITIES = ('heart_rate','hrv','systolic_bp','diastolic_bp','spo2','weight',
                  'temperature','activity','sleep','ecg_waveform','scg_waveform')


def timelines(adapter):
    by_patient = defaultdict(lambda: defaultdict(list))
    supported = {item.concept for item in adapter.observations}
    for item in adapter.observations:
        if item.value.quality_status != 'DUPLICATE':
            by_patient[item.patient_id][item.concept].append(item)
    for patient in sorted(adapter.patients):
        modalities = {}
        for concept in sorted(supported | set(TWIN_MODALITIES)):
            items = by_patient[patient].get(concept,[])
            items.sort(key=lambda item:(item.observed_time is None,
                                        time_number(item.observed_time,item.time_basis) if item.observed_time is not None else 0,
                                        item.observation_id))
            coverage = ('unsupported' if concept not in supported else 'not_collected' if not items else
                        'all_source_missing' if all(item.value.quality_status == 'MISSING' for item in items) else
                        'observed_unusable' if all(item.value.usable_value is None for item in items) else 'observed')
            modalities[concept] = {'support':'supported' if concept in supported else 'unsupported',
                                   'patient_coverage':coverage,
                                   'timestamps':[item.observed_time for item in items],
                                   'available_times':[item.available_time for item in items],
                                   'observation_ids':[item.observation_id for item in items],
                                   'values':[item.value.usable_value for item in items],
                                   'missingness_mask':[item.value.usable_value is None for item in items],
                                   'quality_mask':[item.value.quality_status for item in items],
                                   'units':[item.value.unit for item in items]}
        yield {'patient_id':patient,'dataset':adapter.dataset,'modalities':modalities,
               'clinical_context_reference':{'table':'clinical_context.jsonl','patient_id':patient},
               'outcomes_reference':{'table':'outcomes.jsonl','patient_id':patient},
               'personal_baseline':{'status':'causal_reference_features' if adapter.dataset in {'chiron','perheart'} else 'not_computed',
                                    'table':'temporal_features.jsonl' if adapter.dataset in {'chiron','perheart'} else None},
               'twin_state':{'risk':'not_computed','shadow':'not_computed','divergence':'not_computed',
                             'hemodynamics':'unsupported'}}
