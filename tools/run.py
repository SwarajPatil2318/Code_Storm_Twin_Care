"""Start the local patient workspace and both Twin services; Ctrl+C stops our children."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.doctor import check
ENV_KEYS={'MEAL_ANALYSIS_PROVIDER','GEMINI_API_KEY','TWINCARE_MEAL_VISION_MODEL','OPENAI_API_KEY',
          'TWINCARE_MEAL_ANALYSIS_ENDPOINT','TWINCARE_MEAL_ANALYSIS_TOKEN'}


def load_environment(path=ROOT/'.env'):
    """Read simple dotenv values as data, never shell code; process env takes precedence."""
    if not path.exists():return
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#'):continue
        if line.startswith('export '):line=line[7:].lstrip()
        key,sep,value=line.partition('=');key=key.strip();value=value.strip()
        if not sep or key not in ENV_KEYS:continue
        if len(value)>=2 and value[0]==value[-1] and value[0]in ('"',"'"):value=value[1:-1]
        os.environ.setdefault(key,value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8113);p.add_argument('--product-port',type=int,default=8097)
    p.add_argument('--metabolic-port',type=int,default=8098);p.add_argument('--judge-port',type=int,default=8116)
    p.add_argument('--with-judge',action='store_true');p.add_argument('--state-dir',default='data/patient_domain')
    args=p.parse_args();os.chdir(ROOT);load_environment()
    result=check()
    if not result['ready']:
        print('Preflight failed. Run: python tools/doctor.py',file=sys.stderr);return 1
    ports=[args.port,args.product_port,args.metabolic_port]+([args.judge_port]if args.with_judge else [])
    if len(set(ports))!=len(ports) or any(not 1<=port<=65535 for port in ports):
        print('Use distinct valid loopback ports.',file=sys.stderr);return 1
    for port in ports:
        with socket.socket()as s:
            try:s.bind(('127.0.0.1',port))
            except OSError:
                print(f'Port {port} is occupied; choose another port or stop that service yourself. No existing process was stopped.',file=sys.stderr);return 1
    if args.with_judge and not(ROOT/'data/productization_p9/selected_inputs.json').is_file():
        print('Judge playback requires a private bundle created with --with-judge.',file=sys.stderr);return 1
    state=Path(args.state_dir).resolve();state.mkdir(parents=True,exist_ok=True,mode=0o700)
    env={**os.environ,'PYTHONPATH':str(ROOT),'TWINCARE_RUNTIME_MANIFEST':str(ROOT/'data/runtime/manifest.json'),
         'TWINCARE_PRODUCT_UPSTREAM':f'http://127.0.0.1:{args.product_port}',
         'TWINCARE_METABOLIC_UPSTREAM':f'http://127.0.0.1:{args.metabolic_port}'}
    commands=[('cardiovascular',['src.product.server','--port',str(args.product_port)],'/api/health',args.product_port),
              ('metabolic',['src.metabolic_d41.server_fusion','--port',str(args.metabolic_port)],'/api/health',args.metabolic_port),
              ('workspace',['patient_experience.server','--port',str(args.port),'--db',str(state/'patients.sqlite3'),
                            '--storage',str(state/'uploads'),'--photos',str(state/'meal_photos')],'/api/product/health',args.port)]
    if args.with_judge:
        commands.append(('judge',['judge_scenarios.server','--port',str(args.judge_port),'--root',str(state/'judge')],'/api/judge/scenarios',args.judge_port))
    children=[]
    try:
        for name,argv,health,port in commands:
            child=subprocess.Popen([sys.executable,'-m',*argv],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children.append(child);deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if child.poll()is not None:raise RuntimeError(name+' failed during startup')
                try:
                    with urlopen(f'http://127.0.0.1:{port}'+health,timeout=1)as response:
                        if response.status==200:break
                except OSError:pass
                time.sleep(.1)
            else:raise RuntimeError(name+' startup timed out')
            print(name+' ready',flush=True)
        print(f'TwinCare: http://127.0.0.1:{args.port}',flush=True)
        print('Ctrl+C stops only services started by this launcher.',flush=True)
        while all(c.poll()is None for c in children):time.sleep(.25)
        raise RuntimeError('A TwinCare service exited')
    except KeyboardInterrupt:return 0
    except RuntimeError as e:print(str(e)+'. Check tools/doctor.py and README troubleshooting.',file=sys.stderr);return 1
    finally:
        for child in reversed(children):
            if child.poll()is None:child.terminate()
        for child in reversed(children):
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':raise SystemExit(main())
