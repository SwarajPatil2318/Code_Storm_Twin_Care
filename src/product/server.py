"""Loopback-only read-only cached product API and compiled frontend server.

No model loading, inference, simulator access or patient request logging at runtime.
"""
import argparse
import hashlib
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from src.product.contracts import validate_cache

class ProductStore:
    def __init__(self, path=Path('data/product/demo.json')):
        path = Path(path)
        audit = json.loads(path.with_name('build_audit.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit['api_cache_sha256']:
            raise ValueError('Product cache integrity failure; rebuild approved demo')
        self.cache = validate_cache(json.loads(path.read_text()))
        self.patients = {p['patient']['id']: p for p in self.cache['patients']}

    def get(self, path, query=''):
        parts = [unquote(p) for p in path.strip('/').split('/')]
        if parts and parts[0]=='api': parts=parts[1:]
        if parts==['health']: return {'status':'ok','version':self.cache['metadata']['version']}
        if parts==['evidence']: return self.cache['metadata']
        if parts==['patients']: return [p['patient'] for p in self.patients.values()]
        if len(parts) < 2 or parts[0]!='patients' or parts[1] not in self.patients: raise KeyError
        p = self.patients[parts[1]]
        if len(parts)==2: return p['patient']
        if len(parts)!=3: raise KeyError
        section = parts[2]
        if section=='twin':
            day = int(parse_qs(query).get('day',['0'])[0])
            if day < -7 or day > 0: raise ValueError('Replay day must be -7 through 0')
            return p['timeline'][day+7]
        if section=='timeline': return p['timeline']
        if section=='ehr': return p['ehr']
        if section=='evidence': return self.cache['metadata']
        if section=='observations': return p['observations']
        raise KeyError


def handler(store, static=Path('frontend/dist')):
    static = static.resolve()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # No record or identifier logging.
        def send(self,status,body,kind='application/json'):
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store' if kind=='application/json' else 'no-cache')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path.startswith('/metabolic-api/'):
                try:
                    with urlopen(os.environ.get('TWINCARE_METABOLIC_UPSTREAM', 'http://127.0.0.1:8098')+'/api/'+url.path.removeprefix('/metabolic-api/'),timeout=3) as response:
                        self.send(response.status,response.read())
                except URLError:self.send(503,b'{"error":"Cached D3 API unavailable"}')
                return
            if url.path.startswith('/api/') or url.path.startswith('/patients') or url.path in ('/health','/evidence'):
                try: payload=store.get(url.path,url.query);status=200
                except KeyError: payload={'error':'Resource not found'};status=404
                except (ValueError,TypeError): payload={'error':'Invalid replay request'};status=400
                self.send(status,json.dumps(payload,allow_nan=False).encode());return
            candidate = (static/unquote(url.path).lstrip('/')).resolve()
            if not candidate.is_relative_to(static): self.send(404,b'{}');return
            if not candidate.is_file():
                if candidate.suffix: self.send(404,b'{}');return
                candidate=static/'index.html'
            if not candidate.is_file(): self.send(503,b'{"error":"Build the frontend first"}');return
            self.send(200,candidate.read_bytes(),mimetypes.guess_type(candidate)[0] or 'application/octet-stream')
        def do_POST(self):
            # Only Metabolic live-demo plumbing; cardiovascular routes and
            # cached GET contracts retain their existing behavior.
            path=urlsplit(self.path).path
            if path not in ['/metabolic-api/metabolic/live/'+name for name in ['session','advance','event','close']]:
                self.send(404,b'{"error":"Resource not found"}');return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError
                origin=self.headers.get('Origin')
                if origin and origin!='http://'+self.headers.get('Host',''):
                    self.send(403,b'{"error":"Cross-origin request rejected"}');return
                request=Request(os.environ.get('TWINCARE_METABOLIC_UPSTREAM', 'http://127.0.0.1:8098')+'/api/'+path.removeprefix('/metabolic-api/'),
                    data=self.rfile.read(length),headers={'Content-Type':'application/json'},method='POST')
                with urlopen(request,timeout=15) as response:self.send(response.status,response.read())
            except HTTPError as error:self.send(error.code,error.read())
            except (ValueError,TypeError):self.send(400,b'{"error":"Invalid runtime request"}')
            except (URLError,TimeoutError):self.send(503,b'{"error":"Live inference unavailable"}')
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8080)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(ProductStore()))
    print(f'Synthetic research product available at http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
