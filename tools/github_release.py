"""Build a source-only GitHub checkout using an explicit allowlist; originals stay local."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE_DIRS={'src','patient_alerts','patient_connections','patient_experience','patient_inference',
             'patient_ingestion','patient_meals','patient_readiness','live_simulation','judge_scenarios'}
UI_DIRS={'alerts_ui','connections_ui','experience_ui','inference_ui','judge_ui','meal_logger_ui',
         'patient_product_ui','readiness_ui'}
TOOLS={'runtime_bundle.py','run.py','doctor.py','create_sample_records.py','github_release.py'}
DOCS={'ARCHITECTURE.md','SETUP.md','MANUAL_TESTING.md','GITHUB_PREPARATION.md','SUBMISSION_EVIDENCE.md'}
FRONTEND={'package.json','package-lock.json','index.html','tsconfig.json','vite.config.ts',
          'source-timing-test.mjs','live-chart-test.mjs','final-stabilization-presentation-test.mjs','release-smoke.mjs'}
ROOT_FILES={'README.md','LICENSE','.gitignore','.env.example','.nvmrc','.python-version','requirements.txt'}
DENIED={'AGENTS.md','.env','.git','.venv','node_modules','__pycache__','dist','build','.codex','.agents','.aws'}
SECRET_PATTERNS=[re.compile(r'AIza[0-9A-Za-z_-]{30,}'),re.compile(r'sk-(?:proj-)?[0-9A-Za-z_-]{24,}'),
                 re.compile(r'gh[pousr]_[0-9A-Za-z]{30,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]


def selected(relative):
    p=Path(relative);parts=p.parts
    if p.is_absolute() or '..' in parts or not parts or any(part in DENIED for part in parts)or p.suffix in {'.pyc','.pkl','.npz','.npy','.pdf','.sqlite3','.log','.zip'}:return False
    if len(parts)==1:return p.name in ROOT_FILES
    top=parts[0]
    if top in SOURCE_DIRS:return p.suffix=='.py'
    if top in UI_DIRS:return p.suffix in {'.js','.css','.html'}
    if top=='frontend':
        if len(parts)==2:return p.name in FRONTEND
        if parts[1]=='public':return p.as_posix() in {'frontend/public/favicon.svg','frontend/public/assets/anatomical-heart.png','frontend/public/assets/metabolic-flow.png'}
        return parts[1]=='src' and p.suffix in {'.ts','.tsx','.css','.js','.json','.svg','.woff2'}
    if top=='configs':return len(parts)==2 and p.suffix in {'.json','.yaml'}
    if top=='tools':return len(parts)==2 and p.name in TOOLS
    if top=='release_tests':return len(parts)==2 and p.suffix=='.py'
    if top=='docs':return len(parts)==2 and p.name in DOCS
    return p.as_posix()in {'data/README.md','.github/workflows/ci.yml'}


def inventory(root=ROOT):
    result=[]
    # Prune ignored/private trees before walking: never read raw inputs or .env.
    def walk(folder):
        for p in sorted(folder.iterdir()):
            rel=p.relative_to(root)
            if p.is_symlink():continue
            if p.is_dir():
                if len(rel.parts)==1 and p.name not in SOURCE_DIRS|UI_DIRS|{'frontend','configs','tools','release_tests','docs','data','.github'}:continue
                if p.name in DENIED:continue
                if rel.parts[0]=='data':continue
                if rel.parts[0]=='docs'and len(rel.parts)>1:continue
                walk(p)
            elif selected(rel):result.append(p)
    walk(root)
    if(root/'data/README.md').is_file():result.append(root/'data/README.md')
    return sorted(set(result))


def audit(root):
    errors=[];count=0
    for f in inventory(root):
        count+=1
        if f.stat().st_size>10*1024*1024:errors.append('oversized public source file')
        if f.suffix in {'.png','.webp','.woff2'}:continue
        text=f.read_text()
        if any(pattern.search(text)for pattern in SECRET_PATTERNS):errors.append('credential pattern detected')
    return {'passed':not errors,'source_files':count,'errors':sorted(set(errors))}


def build(destination,archive=None):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Release destination must be new')
    source=inventory();destination.mkdir(parents=True)
    for f in source:
        target=destination/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    # Only retain QA commands whose scripts are included in the lean release.
    p=destination/'frontend/package.json';package=json.loads(p.read_text());package['name']='twincare';package['engines']={'node':'>=22 <23'}
    package['scripts']={'dev':'vite --host 127.0.0.1','build':'tsc --noEmit && vite build',
        'test:sources':'node source-timing-test.mjs','test:chart':'node live-chart-test.mjs',
        'test:presentation':'node final-stabilization-presentation-test.mjs','qa':'node release-smoke.mjs'}
    p.write_text(json.dumps(package,indent=2)+'\n')
    p=destination/'frontend/package-lock.json';lock=json.loads(p.read_text());lock['name']='twincare';lock['packages']['']['name']='twincare';lock['packages']['']['engines']={'node':'>=22 <23'};p.write_text(json.dumps(lock,indent=2)+'\n')
    result=audit(destination)
    if not result['passed']:raise RuntimeError('Public source audit failed; no archive created')
    files=inventory(destination)
    # Independently compare every copied file to the allowlist, not just inventory().
    if any(not selected(p.relative_to(destination))for p in destination.rglob('*')if p.is_file()):raise RuntimeError('Unexpected file in release')
    manifest={'schema_version':'twincare-source-release/1','source_only':True,'files':{
        p.relative_to(destination).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files}}
    if archive:
        archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED)as z:
            for p in files:z.write(p,'TwinCare/'+p.relative_to(destination).as_posix())
    private=ROOT/'data/release';private.mkdir(parents=True,exist_ok=True)
    (private/'source_release_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'source_files':len(files),'source_bytes':sum(p.stat().st_size for p in files),
                      'privacy_audit':result['passed'],'checkout':str(destination),'archive':str(archive)if archive else None}))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',default='release/TwinCare');p.add_argument('--archive',default='release/TwinCare-GitHub.zip');args=p.parse_args();build(args.destination,args.archive)
