"""Read-only loopback API for precomputed, entirely synthetic D3 demo states."""
import argparse,json,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from .reference import ROOT
from .state import validate
from .live import cached_replay

class DemoStore:
    def __init__(self,path=None):
        self.demo=json.loads((path or ROOT/'demo.json').read_text())
        self.cases={case['case']:case for case in self.demo['cases']}
        for case in self.cases.values():
            for frame in case['timeline']:
                validate(frame)
                if not frame['patient']['synthetic']:raise ValueError('Public demo API accepts only synthetic states')
    def route(self,path):
        if path=='/api/evidence/metabolic':
            # Reviewed aggregates only; no experimental models, data or CV at runtime.
            return json.loads(Path('artifacts/metabolic_d36/evidence.json').read_text())
        if path=='/api/health':return {'status':'ready','mode':'PRECOMPUTED SYNTHETIC DEMO','model_inference_on_request':False}
        if path=='/api/evidence':return self.demo['reference']
        if path=='/api/performance':return self.demo['latency_ms']
        if path=='/api/cases':return [{'case':k,'patient':v['patient'],'frames':len(v['timeline'])} for k,v in self.cases.items()]
        parts=path.strip('/').split('/')
        if len(parts)==4 and parts[:2]==['api','cases'] and parts[2] in self.cases:
            case=self.cases[parts[2]]
            if parts[3]=='state':return case['timeline'][-1]
            if parts[3]=='timeline':return case['timeline']
            if parts[3]=='events':return case['events']
            if parts[3]=='replay':return list(cached_replay(case))
        raise KeyError('Unknown demo resource')

def handler(store,runtime=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if urlparse(self.path).path=='/api/metabolic/live/performance' and runtime is not None:
                body=json.dumps(runtime.performance(),allow_nan=False).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                return
            try:body=json.dumps(store.route(urlparse(self.path).path),allow_nan=False).encode();status=200
            except KeyError:body=b'{"error":"Unknown demo resource"}';status=404
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            from src.metabolic_d41.runtime import RuntimeUnavailable,RevisionConflict
            started=time.perf_counter()
            path=urlparse(self.path).path
            try:
                if path not in ['/api/metabolic/live/'+name for name in ['session','advance','event','close']]:
                    raise KeyError('Unknown live resource')
                if runtime is None:raise RuntimeUnavailable('Runtime not enabled')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Invalid request size')
                origin=self.headers.get('Origin')
                if origin and origin not in ['http://'+self.headers.get('Host',''), 'http://127.0.0.1:8097', 'http://127.0.0.1:5175', 'http://127.0.0.1:5173']:
                    raise ValueError('Cross-origin runtime request rejected')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError('Object payload required')
                response=runtime.route(path,payload)
                runtime.record_latency(response,started)
                body=json.dumps(response,allow_nan=False).encode();status=200
            except KeyError:body=b'{"error":"Unknown live resource"}';status=404
            except RevisionConflict:body=b'{"error":"Session revision changed; reset the demo"}';status=409
            except (ValueError,TypeError,OverflowError):body=b'{"error":"Invalid live event or session request"}';status=400
            except Exception:
                # No request contents, identifiers, model paths or exception
                # payloads are logged. Cached endpoints remain available.
                body=b'{"error":"Live inference unavailable; verified replay remains available"}';status=503
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    return Handler

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8098);args=parser.parse_args()
    from src.metabolic_d41.runtime import RuntimeStore
    store=DemoStore();runtime=RuntimeStore(store.demo)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(store,runtime))
    print(f'Precomputed synthetic Metabolic Twin API: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
