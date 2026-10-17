"""Product schema v2 adapter preserving the source's unspecified timezone.

The frozen D2 scientific package and its one-time predictions remain immutable.
This presentation adapter corrects a v1 timestamp serialization assumption; it
does not fit models, change predictions, evaluate targets or alter calibration.
"""
import datetime as dt
import json
from pathlib import Path
import numpy as np
from src.metabolic_d2.product import MetabolicTwinState,make_state as frozen_make_state,validate_state as frozen_validate_state
from src.metabolic_d2.protocol import ROOT,digest,verify_frozen,write_json


def native_timestamp(nanoseconds):
    return str(np.datetime_as_string(np.datetime64(int(nanoseconds),'ns'),unit='ns'))


def apply_native_time(state,nanoseconds):
    state={**state,'timestamp':native_timestamp(nanoseconds),'data_sources':{**state['data_sources'],
        'source_timezone':None,'timestamp_basis':'naive source wall-clock time; timezone not supplied',
        'product_schema_version':2}}
    validate_state(state);return state


def make_state(data,index,prediction,probability,threshold,model_hash):
    return apply_native_time(frozen_make_state(data,index,prediction,probability,threshold,model_hash),data['time'][index])


def validate_state(state):
    frozen_validate_state(state)
    if dt.datetime.fromisoformat(state['timestamp']).tzinfo is not None:raise ValueError('Unknown source timezone must not be asserted')
    if state['data_sources'].get('source_timezone','missing') is not None:raise ValueError('Source timezone uncertainty missing')
    if state['data_sources'].get('product_schema_version')!=2:raise ValueError('Native-time schema version missing')
    return state


def correct_saved_states():
    verify_frozen()
    if not (ROOT/'HOLDOUT_EVALUATION_COMPLETED.json').exists():raise RuntimeError('Frozen prediction event must finish first')
    source=ROOT/'product_states_v1_private.json';target=ROOT/'product_states_private.json'
    if source.exists():
        record=json.loads((ROOT/'product_schema_correction.json').read_text())
        if record['corrected_state_sha256']!=digest(target):raise RuntimeError('Corrected state hash mismatch')
        return record
    # Preserve the exact event-generated v1 records as evidence rather than rewriting history.
    target.rename(source);states=json.loads(source.read_text())
    examples=json.loads((ROOT/'representative_examples_private.json').read_text())
    with np.load(ROOT/'cache/holdout.npz',allow_pickle=False) as data:
        for item in states:
            index=examples[item['example']]['landmark_index']
            item['state']=apply_native_time(item['state'],data['time'][index])
    write_json(target,states)
    record={'schema_version':2,'reason':'Source timezone is unspecified; v1 used an unsupported UTC suffix.',
        'scope':'Product timestamp representation only; no model, preprocessing, predictions, target labels, risk mappings or metrics changed.',
        'original_state_sha256':digest(source),'corrected_state_sha256':digest(target),
        'adapter_sha256':digest(__file__),'holdout_prediction_sha256':digest(ROOT/'holdout_predictions.npz'),
        'frozen_model_manifest_sha256':digest(ROOT/'FROZEN_MODEL_MANIFEST.json')}
    write_json(ROOT/'product_schema_correction.json',record,exclusive=True)
    print(json.dumps({'product_schema_version':2,'source_timezone':'unspecified','holdout_evaluation_count':1}))
    return record

if __name__=='__main__':correct_saved_states()
