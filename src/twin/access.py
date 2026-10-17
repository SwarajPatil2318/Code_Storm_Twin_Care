"""Role filtering before clinical deserialization, including mixed NPZ blocks.

Compressed archives necessarily stream discarded bytes. Excluded patient rows
are never parsed into clinical JSON or NumPy values. Patient ID/header metadata
alone route blocks. Private oracle access is a separate sealed evaluation path.
"""
import gzip,json,re,zipfile,datetime as dt
from pathlib import Path
import numpy as np
from numpy.lib import format as npformat
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json
from src.twin.protocol import require_role,EVIDENCE,code_hash

PID=re.compile(rb'"patient_id"\s*:\s*"([^"\\]+)"')
OBS_KEYS=['value','raw_value','quality_status','missingness_mask','scheduled','available_hour','reason_code']

def metadata_patient(line):
    match=PID.search(line)
    if not match:raise ValueError('Missing namespace routing metadata')
    return match.group(1).decode('ascii')

def _decode_row(line):return json.loads(line)

def selective_json(path,allowed):
    with gzip.open(path,'rb') as stream:
        for line in stream:
            pid=metadata_patient(line)
            if pid in allowed:yield _decode_row(line)

def selective_npz(path,indices,keys):
    """Read approved patient slabs only; seek over others without numeric decode."""
    indices=sorted(set(indices));out={}
    with zipfile.ZipFile(path) as archive:
        for key in keys:
            with archive.open(key+'.npy') as stream:
                version=npformat.read_magic(stream)
                shape,fortran,dtype=npformat._read_array_header(stream,version)
                if fortran or dtype.hasobject or len(shape)<2:raise ValueError('Unsafe slab layout')
                slab_bytes=int(np.prod(shape[1:]))*dtype.itemsize;header=stream.tell();slabs=[]
                for i in indices:
                    if not 0<=i<shape[0]:raise ValueError('Slab index out of bounds')
                    stream.seek(header+i*slab_bytes);raw=stream.read(slab_bytes)
                    if len(raw)!=slab_bytes:raise ValueError('Truncated approved slab')
                    slabs.append(np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).copy())
                out[key]=np.stack(slabs) if slabs else np.empty((0,*shape[1:]),dtype=dtype)
    return out

class Cohort:
    def __init__(self,root,output,config):
        self.root=Path(root);self.output=Path(output);self.config=config
        self.manifest=json.loads((self.root/'manifest.json').read_text());self.roles=json.loads((self.root/'splits.json').read_text())['assignments']
        self.certificate=json.loads((self.root/'large_validation.json').read_text())
        if not self.certificate['all_gates_passed'] or self.manifest['evidence']!='SYNTHETIC POC':raise ValueError('Validated synthetic cohort required')
        if len(self.roles)!=self.manifest['patients'] or set(self.roles.values())!={'train','validation','test'}:raise ValueError('Partition metadata mismatch')
        self.metadata_sha=file_hash(self.root/'manifest.json');self.accesses=[]
    def authorize(self,role,oracle=False):
        if role not in {'train','validation','test','stress'}:raise ValueError('Unknown role')
        if role=='test' or oracle:
            seal_path=self.output/'selection_seal.json'
            if not seal_path.exists():raise ValueError('Clinical test/oracle data require frozen selection')
            seal=json.loads(seal_path.read_text())
            if seal['status']!='FROZEN' or seal['model_code_sha256']!=code_hash() or seal['config_sha256']!=self.config['_sha256']:raise ValueError('Stale selection seal')
            if role=='test' and not (self.output/'evaluation_started.json').exists():raise ValueError('Test transaction has not started')
            for p,h in seal['artifacts'].items():
                if file_hash(self.output/p)!=h:raise ValueError('Frozen pipeline artifact changed')
            if seal['cohort_manifest_sha256']!=self.metadata_sha:raise ValueError('Cohort identity changed')
    def patients(self,role):
        self.authorize(role);return {p for p,r in self.roles.items() if r==role}
    def rows(self,table,role,oracle=False):
        self.authorize(role,oracle)
        if table=='ground_truth' and not oracle:raise ValueError('Oracle truth is not a model interface')
        allowed={p for p,r in self.roles.items() if r==role}
        path=self.root/table/'records.jsonl.gz';self.check_original(path)
        for row in selective_json(path,allowed):
            if table=='ehr' and row['available_hour']>=0:raise ValueError('EHR was not historical at monitoring start')
            self.accesses.append({'table':table,'role':role,'patient_id':row['patient_id'],'oracle':oracle});yield row
    def check_original(self,path):
        name=str(Path(path).relative_to(self.root));meta=self.manifest['artifacts'][name]
        if file_hash(path)!=meta['sha256']:raise ValueError('Frozen Phase 5 artifact changed')
    def streams(self,role,oracle=False):
        self.authorize(role,oracle);wanted={p for p,r in self.roles.items() if r==role}
        directory='ground_truth' if oracle else 'observations';keys=['state','stable_daily','true_physiology'] if oracle else OBS_KEYS
        for name in self.manifest['batches']:
            path=self.root/directory/name;self.check_original(path)
            with np.load(path,allow_pickle=False) as z:ids=z['patient_ids'].tolist()
            indices=[i for i,p in enumerate(ids) if p in wanted]
            if not indices:continue
            slabs=selective_npz(path,indices,keys)
            for k,i in enumerate(indices):
                self.accesses.append({'table':directory,'role':role,'patient_id':ids[i],'oracle':oracle})
                yield ids[i],{key:a[k] for key,a in slabs.items()}
    def save_access_log(self,name):
        private_json(self.output/name,{'clinical_rows_decoded':self.accesses,'compressed_bytes_skipped_without_clinical_deserialization':True,**EVIDENCE})
