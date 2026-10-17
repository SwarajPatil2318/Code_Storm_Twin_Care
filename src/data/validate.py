"""Validate canonical artifacts, feature causality, outcome separation and splits."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .adapters.zigong import ADMISSION_EVENT_TIMES, APPROVED_FEATURES, TARGET, is_outcome, validate_feature_matrix
from .io import dumps, file_hash, read_jsonl
from .quality import SchemaError
from .schema import DATASETS, SCHEMA_VERSION, Observation, Provenance, QualityStatus, SourceValue
from .splits import validate_splits
from ..features.temporal import causal_features, time_number


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def validate_value(value, dataset, input_hashes):
    required={'raw_value','usable_value','quality_status','reasons','unit','provenance'}
    require(required <= set(value),'SourceValue schema is incomplete')
    require(value['quality_status'] in {s.value for s in QualityStatus},'Unknown quality state')
    if value['quality_status'] in {'INVALID','MISSING','DUPLICATE'}:
        require(value['usable_value'] is None,'Unusable quality state exposes a usable value')
    prov=value['provenance']
    require(set(Provenance.__dataclass_fields__)<=set(prov),'Provenance schema is incomplete')
    require(prov['dataset']==dataset,'Cross-dataset provenance')
    require(prov['source_sha256'] in input_hashes.values(),'Unrecognized input provenance hash')


def reconstruct_observation(item):
    value=item['value'].copy()
    value['provenance']=Provenance(**value['provenance'])
    data=item.copy()
    data['value']=SourceValue(**value)
    return Observation(**data)


def validate_dataset(directory, check_manifest=True):
    root=Path(directory)
    report=json.loads((root/'report.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    dataset=report['dataset']
    require(dataset in DATASETS,'Unknown dataset')
    require(manifest['schema_version']==SCHEMA_VERSION,'Unsupported canonical schema version')
    if check_manifest:
        actual={p.name for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}
        require(actual==set(manifest['artifacts']),'Artifact inventory mismatch')
        for filename,meta in manifest['artifacts'].items():
            require(Path(filename).name==filename,'Unsafe manifest artifact path')
            path=root/filename
            require(path.stat().st_size==meta['bytes'] and file_hash(path)==meta['sha256'],'Artifact checksum mismatch')
    patients={}
    for patient in read_jsonl(root/'patients.jsonl'):
        require(patient.get('dataset')==dataset and patient.get('schema_version')==SCHEMA_VERSION,'Invalid Patient schema')
        identifier=patient.get('patient_id','')
        require(identifier.startswith(dataset+':'),'Patient namespace mismatch')
        require(identifier not in patients,'Repeated patient identifier')
        patients[identifier]=patient
    require(len(patients)==report['patients_retained'],'Patient count mismatch')
    observations={}
    for item in read_jsonl(root/'observations.jsonl'):
        require(set(Observation.__dataclass_fields__)<=set(item),'Observation schema is incomplete')
        require(item['patient_id'] in patients and item['dataset']==dataset,'Inconsistent observation patient reference')
        key=item['observation_id']
        require(key not in observations,'Duplicate observation identity')
        validate_value(item['value'],dataset,manifest['inputs'])
        if item['observed_time'] is None:
            require(item['value']['usable_value'] is None,'Untimed observation is usable')
        if item['concept']=='spo2' and item['value']['usable_value'] is not None:
            require(0<=item['value']['usable_value']<=100,'Impossible SpO2 escaped quarantine')
        observations[key]=item
    require(len(observations)==report['canonical_observations'],'Observation count mismatch')
    expected_usable=sum(item['value']['usable_value'] is not None for item in observations.values())
    require(expected_usable==report['usable_observations'],'Usable observation count mismatch')
    approved_contexts={}
    for table in ('clinical_context','source_features'):
        for context in read_jsonl(root/(table+'.jsonl')):
            require(context['patient_id'] in patients and context['dataset']==dataset,'Inconsistent clinical context patient reference')
            values=context.get('attributes',context.get('features',{}))
            if dataset=='zigong' and table=='clinical_context':
                require(context['patient_id'] not in approved_contexts,'Repeated Zigong clinical context')
                require(set(APPROVED_FEATURES)<=set(values),'Missing approved clinical attribute')
                approved_contexts[context['patient_id']]={field:values[field]['usable_value'] for field in APPROVED_FEATURES}
            for field,value in values.items():
                validate_value(value,dataset,manifest['inputs'])
                if dataset=='zigong':
                    require(not is_outcome(field),'Outcome appears in clinical context')
                    if field not in APPROVED_FEATURES:
                        require(value['usable_value'] is None,'Untimed clinical field escaped exclusion')
                if dataset=='chiron':
                    require(value['usable_value'] is None,'Unapproved CHIRON source feature escaped exclusion')
    outcome_counts=Counter()
    primary_labels={}
    for outcome in read_jsonl(root/'outcomes.jsonl'):
        require(outcome['patient_id'] in patients and outcome['dataset']==dataset,'Inconsistent outcome reference')
        validate_value(outcome['value'],dataset,manifest['inputs'])
        if dataset=='zigong' and outcome['endpoint'] in ADMISSION_EVENT_TIMES:
            require(outcome['time_basis']=='days_from_admission' and outcome['time_origin']=='admission','Explicit event clock was lost')
            require(outcome['event_time']==outcome['value']['usable_value'],'Event day differs from recorded source value')
        outcome_counts[outcome['endpoint']]+=1
        if dataset=='zigong' and outcome['endpoint']==TARGET:
            require(outcome['patient_id'] not in primary_labels,'Repeated primary patient outcome')
            primary_labels[outcome['patient_id']]=int(outcome['value']['usable_value'])
    for table in ('conditions','medications','encounters','procedures'):
        for event in read_jsonl(root/(table+'.jsonl')):
            require(event['patient_id'] in patients and event['dataset']==dataset,'Inconsistent event patient reference')
            if dataset=='zigong':
                require(event['value']['usable_value'] is None and event['available_time'] is None,'Untimed medication exposure is usable')
                validate_value(event['value'],dataset,manifest['inputs'])
    splits=json.loads((root/'splits.json').read_text())
    try:
        validate_splits(splits,set(patients))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if dataset=='zigong':
        contract=json.loads((root/'feature_availability.json').read_text())
        with (root/'predictors.csv').open(newline='') as source:
            reader=csv.reader(source)
            validate_feature_matrix(next(reader),contract)
            predictor_ids=[]
            for row in reader:
                require(len(row)==len(APPROVED_FEATURES)+1,'Ragged predictor matrix')
                require(row[0] in approved_contexts,'Predictor lacks clinical context')
                expected=[str(approved_contexts[row[0]][field]) if approved_contexts[row[0]][field] is not None else '' for field in APPROVED_FEATURES]
                require(row[1:]==expected,'Predictor values differ from approved context')
                predictor_ids.append(row[0])
        with (root/'labels.csv').open(newline='') as source:
            reader=csv.reader(source)
            require(next(reader)==['patient_id',TARGET],'Invalid label matrix schema')
            label_rows=list(reader)
            require(all(len(row)==2 for row in label_rows),'Ragged label matrix')
        require(len(predictor_ids)==len(set(predictor_ids)) and set(predictor_ids)==set(patients),'Predictor patients mismatch')
        require(len(label_rows)==len(patients) and {row[0]:int(row[1]) for row in label_rows}==primary_labels,'Labels differ from canonical outcomes')
        class_counts={part:dict(sorted(Counter(str(primary_labels[p]) for p,v in splits['assignments'].items() if v==part).items())) for part in ('train','validation','test')}
        require(class_counts==splits['class_counts'],'Split class-count mismatch')
    timeline_ids=set()
    for timeline in read_jsonl(root/'timelines.jsonl'):
        require(timeline['patient_id'] in patients and timeline['dataset']==dataset,'Invalid timeline patient reference')
        require(timeline['patient_id'] not in timeline_ids,'Repeated patient timeline')
        timeline_ids.add(timeline['patient_id'])
        for modality in timeline['modalities'].values():
            names=('timestamps','available_times','observation_ids','values','missingness_mask','quality_mask','units')
            require(len({len(modality[name]) for name in names})==1,'Timeline arrays are misaligned')
            for index,key in enumerate(modality['observation_ids']):
                require(key in observations,'Timeline contains unknown observation')
                item=observations[key]
                require(item['patient_id']==timeline['patient_id'],'Timeline includes another patient')
                require(item['value']['quality_status']!='DUPLICATE','Duplicate remains in timeline')
                require(modality['values'][index]==item['value']['usable_value'],'Timeline changes usable value')
                require(modality['missingness_mask'][index]==(modality['values'][index] is None),'Incorrect missingness mask')
    require(timeline_ids==set(patients),'Timeline patient set mismatch')
    features=list(read_jsonl(root/'temporal_features.jsonl'))
    if dataset in {'chiron','perheart'}:
        expected=causal_features(reconstruct_observation(item) for item in observations.values())
        count=0
        for actual,recomputed in zip(features,expected,strict=True):
            require(dumps(actual)==dumps(recomputed),'Temporal features do not match causal recomputation')
            cutoff=actual['provenance']['cutoff_in_days']
            for key in actual['provenance']['prior_observation_ids']:
                require(key in observations,'Unknown prior observation')
                prior=observations[key]
                require(prior['patient_id']==actual['patient_id'] and prior['concept']==actual['concept'],'Cross-patient/concept rolling input')
                require(time_number(prior['observed_time'],prior['time_basis'])<cutoff,'Future or current observation entered personal baseline')
                if prior['available_time'] is not None:
                    require(time_number(prior['available_time'],prior['time_basis'])<=cutoff,'Unavailable delayed observation entered baseline')
            count+=1
        require(count==report['temporal_feature_rows'],'Temporal feature count mismatch')
    else:
        require(not features,'Temporal physiology features are unsupported for this source')
    for table,count in report['canonical_table_rows'].items():
        require(sum(1 for _ in read_jsonl(root/(table+'.jsonl')))==count,'Canonical table row count mismatch')
    return {'dataset':dataset,'schema_version':SCHEMA_VERSION,'status':'PASS',
            'patients':len(patients),'observations':len(observations),'temporal_features':len(features),
            'checks':['artifact_checksums','schemas','provenance','patient_namespaces',
                      'outcome_separation','feature_allowlist','patient_split_isolation',
                      'quality_masks','physiological_quarantine','duplicate_exclusion','causal_recomputation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=(*DATASETS,'all'),default='all')
    parser.add_argument('--root',type=Path,default=Path('data/processed'))
    args=parser.parse_args()
    for dataset in DATASETS if args.dataset=='all' else (args.dataset,):
        result=validate_dataset(args.root/dataset)
        print(f'{dataset}: {result["status"]}; {result["patients"]} patients')

if __name__=='__main__':main()
