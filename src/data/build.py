"""Reproducible preprocessing CLI; private staged outputs and immutable inputs."""
import argparse
from collections import Counter
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

from .adapters.chiron import ChironAdapter
from .adapters.perheart import PerheartAdapter
from .adapters.synthea import SyntheaAdapter
from .adapters.zigong import ZigongAdapter
from .io import ArtifactWriter, file_hash
from .schema import DATASETS, SCHEMA_VERSION
from .splits import grouped_cv, stratified_split
from .timelines import timelines
from ..features.temporal import TEMPORAL_FEATURES, causal_features

ADAPTERS = {'zigong':ZigongAdapter,'chiron':ChironAdapter,'perheart':PerheartAdapter,'synthea':SyntheaAdapter}
TABLES = ('patients','clinical_context','observations','outcomes','quality','quarantine',
          'source_records','source_features','duplicates','conditions','medications',
          'encounters','procedures','timelines','temporal_features')


def implementation_hash():
    root = Path(__file__).parents[1]
    files = sorted([*root.rglob('*.py'), root/'data/source_schemas.json'])
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_dataset(dataset, input_root=Path('Datasets'), output_root=Path('data/processed'), seed=2026):
    """Publish a completed dataset generation only after successful validation."""
    from .validate import validate_dataset
    if dataset not in ADAPTERS:
        raise ValueError('Unknown source dataset')
    source, output = Path(input_root).resolve(), Path(output_root).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Raw input and processed output roots must be disjoint')
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    destination = output/dataset
    if destination.is_symlink():
        raise ValueError('Refusing to replace a symlinked dataset destination')
    if destination.exists() and not (destination/'manifest.json').is_file():
        raise ValueError('Refusing to replace a directory without a pipeline manifest')
    if destination.exists():
        prior_manifest=json.loads((destination/'manifest.json').read_text())
        managed=set(prior_manifest.get('artifacts',{})) | {'manifest.json'}
        if {p.name for p in destination.iterdir()} != managed:
            raise ValueError('Refusing to replace dataset directory containing unmanaged files')
    code_hash = implementation_hash()
    stage = Path(tempfile.mkdtemp(prefix=f'.{dataset}-build-',dir=output))
    writer = ArtifactWriter(stage)
    try:
        adapter = ADAPTERS[dataset](source,writer)
        adapter.build()
        features = 0
        for feature in causal_features(adapter.observations):
            writer.emit('temporal_features',feature)
            features += 1
        for timeline in timelines(adapter):
            writer.emit('timelines',timeline)
        if dataset=='zigong':
            splits=stratified_split(adapter.labels,seed)
        elif dataset in {'chiron','perheart'}:
            splits=grouped_cv(adapter.patients,seed)
        else:
            splits={'strategy':'integration_only_no_clinical_evaluation','seed':seed,
                    'patients':sorted(adapter.patients),'assignments':{}}
        writer.json('splits',splits)
        report=adapter.report()
        report['schema_version']=SCHEMA_VERSION
        report['split_strategy']=splits['strategy']
        report['split_counts'] = {part:sum(value==part for value in splits['assignments'].values()) for part in ('train','validation','test')} if dataset=='zigong' else [
            {'fold':fold['fold'],**{part:sum(value==part for value in fold['assignments'].values()) for part in ('train','validation','test')}} for fold in splits.get('folds',[])]
        if dataset=='zigong': report['split_class_counts']=splits['class_counts']
        if dataset=='chiron':
            report['pro_worsening_counts_by_fold']=[]
            for fold in splits['folds']:
                counts={part:Counter() for part in ('train','validation','test')}
                for patient,label in adapter.feel_labels:
                    name='unknown' if label not in {'1','2','3','4','5'} else 'worse' if label in {'1','2'} else 'nonworse'
                    counts[fold['assignments'][patient]][name]+=1
                report['pro_worsening_counts_by_fold'].append({'fold':fold['fold'],'counts':{part:dict(sorted(value.items())) for part,value in counts.items()}})
        report['temporal_feature_rows']=features
        report['temporal_feature_names']=list(TEMPORAL_FEATURES) if features else []
        report['rows_quarantined_definition']='Nonmissing observations with no usable value, excluding exact duplicates; excluded source features/clinical attributes reported separately.'
        writer.close()
        writer.ensure_tables(TABLES)
        report['canonical_table_rows']=dict(sorted(writer.counts.items()))
        writer.json('report',report)
        for filename,digest in adapter.input_hashes.items():
            if file_hash(source/filename)!=digest:
                raise RuntimeError('Raw input changed during preprocessing')
        writer.manifest(adapter.input_hashes,seed,{'dataset':dataset,'temporal_windows_days':[7,30],
                        'longitudinal_availability':'explicit_retrospective_assumption',
                        'no_imputation':True},code_hash)
        validation=validate_dataset(stage,check_manifest=True)
        writer.json('validation',validation)
        writer.manifest(adapter.input_hashes,seed,{'dataset':dataset,'temporal_windows_days':[7,30],
                        'longitudinal_availability':'explicit_retrospective_assumption',
                        'no_imputation':True},code_hash)
        if implementation_hash() != code_hash:
            raise RuntimeError('Pipeline source code changed during preprocessing')
        # A fixed destination is replaced only after a validated generation exists.
        backup=None
        if destination.exists():
            backup=Path(tempfile.mkdtemp(prefix=f'.{dataset}-previous-',dir=output))
            backup.rmdir()
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup is not None: backup.rename(destination)
            raise
        if backup is not None: shutil.rmtree(backup)
        (output/f'{dataset}_build_failure.json').unlink(missing_ok=True)
        return report
    except BaseException as exc:
        writer.close()
        failure = {'dataset':dataset,'status':'FAIL','error_type':type(exc).__name__,
                   'reason':'Build rejected; previous validated generation preserved. See exception for schema/validation rule.',
                   'inputs':getattr(locals().get('adapter'), 'input_hashes', {})}
        path=output/f'{dataset}_build_failure.json'
        path.write_text(json.dumps(failure,sort_keys=True)+'\n')
        path.chmod(0o600)
        shutil.rmtree(stage,ignore_errors=True)
        raise


def build_dataset(dataset, input_root=Path('Datasets'), output_root=Path('data/processed'), seed=2026):
    source, output = Path(input_root).resolve(), Path(output_root).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Raw input and processed output roots must be disjoint')
    if dataset not in ADAPTERS:
        raise ValueError('Unknown source dataset')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = output / f'.{dataset}.lock'
    with path.open('a') as lock:
        path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another build already holds this dataset lock') from exc
        try:
            return _build_dataset(dataset, source, output, seed)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=(*DATASETS,'all'),required=True)
    parser.add_argument('--input',type=Path,default=Path('Datasets'))
    parser.add_argument('--output',type=Path,default=Path('data/processed'))
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    datasets=DATASETS if args.dataset=='all' else (args.dataset,)
    for dataset in datasets:
        report=build_dataset(dataset,args.input,args.output,args.seed)
        print(f'{dataset}: {report["patients_retained"]} patients; {report["canonical_observations"]} observations; validation passed')

if __name__=='__main__':main()
