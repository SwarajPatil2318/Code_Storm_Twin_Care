"""Immutable source-ID-only split, stage guards and private reproducibility records."""
import datetime as dt
import hashlib,json,os,zipfile
from pathlib import Path
import numpy as np

ROOT=Path('data/metabolic_d2')
CONFIG=Path('configs/metabolic_d2.json')
SOURCE=Path('Datasets/diabetes_datasets.zip')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,obj,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    text=json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n'
    if exclusive:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(text)
    else:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.chmod(0o600);tmp.replace(path)
    path.chmod(0o600)


def config():return json.loads(CONFIG.read_text())


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def code_hash():
    h=hashlib.sha256()
    paths=sorted(Path('src/metabolic_d2').glob('*.py'))+[Path('src/metabolic/windows.py'),CONFIG]
    for p in paths:h.update(p.as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def freeze_split():
    ROOT.mkdir(exist_ok=True,mode=0o700)
    path=ROOT/'participant_manifest.json';c=config()
    if path.exists():
        m=json.loads(path.read_text())
        if m['config_sha256']!=digest(CONFIG) or m['raw_archive_sha256']!=digest(SOURCE):
            raise RuntimeError('Frozen split source/config mismatch; refusing reselection')
        return m
    with zipfile.ZipFile(SOURCE) as z:
        ids=sorted({'ShanghaiT2D:'+Path(n).stem.split('_')[0] for n in z.namelist()
                    if n.startswith('Shanghai_T2DM/') and n.endswith(('.xls','.xlsx'))})
    assert len(ids)==100
    order=np.random.default_rng(c['seed']).permutation(len(ids));holdout=[ids[i] for i in order[:20]]
    development=[ids[i] for i in order[20:]]
    calibration=development[:16];training=development[16:]
    folds={pid:i%5 for i,pid in enumerate(training)}
    m={'created_at_utc':utc(),'seed':c['seed'],'method':'sorted source IDs, seeded permutation; no clinical/target stratification',
        'raw_archive_sha256':digest(SOURCE),'config_sha256':digest(CONFIG),
        'development':development,'model_development':training,'calibration':calibration,
        'holdout':holdout,'folds':folds,'holdout_reads':0}
    assert len(set(development)&set(holdout))==0
    assert set(training).isdisjoint(calibration) and len(training)==64
    write_json(path,m,exclusive=True)
    write_json(ROOT/'access_ledger.json',{'split_frozen_at_utc':m['created_at_utc'],'events':[]},exclusive=True)
    print(json.dumps({'split_frozen':True,'development':80,'model_development':64,'calibration':16,'holdout':20,'source_IDs_logged':False}),flush=True)
    return m


def assert_stage(stage):
    if stage not in ['development','calibration','holdout']:raise ValueError('Unknown data stage')
    m=freeze_split()
    if stage in ['development','calibration'] and (ROOT/'HOLDOUT_EVALUATION_STARTED.json').exists():
        raise PermissionError('Holdout already opened; post-evaluation development is forbidden')
    if stage=='holdout':
        if not (ROOT/'FROZEN_MODEL_MANIFEST.json').exists():raise PermissionError('Model choices must freeze before holdout access')
        if not (ROOT/'HOLDOUT_EVALUATION_STARTED.json').exists():raise PermissionError('One-time evaluation event must be recorded first')
        verify_frozen()
    return m


def verify_frozen():
    frozen=json.loads((ROOT/'FROZEN_MODEL_MANIFEST.json').read_text())
    if frozen['code_sha256']!=code_hash():raise RuntimeError('Frozen scientific code changed')
    for path,expected in frozen['artifact_hashes'].items():
        if digest(path)!=expected:raise RuntimeError('Frozen artifact integrity mismatch')
    return frozen


def record_access(stage,subject_count,recording_count):
    path=ROOT/'access_ledger.json';ledger=json.loads(path.read_text())
    ledger['events'].append({'stage':stage,'participants':subject_count,'recordings':recording_count,'at_utc':utc()})
    write_json(path,ledger)

if __name__=='__main__':freeze_split()
