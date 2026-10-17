"""P7 local origin: preserved dashboard plus additive notification surfaces."""
import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from patient_inference.server import handler as inference_handler
from patient_inference.connections import InferenceConnections
from patient_inference.readiness import InferenceReadiness
from patient_ingestion.service import IngestionService
from patient_meals.service import Meals
from src.patient_domain.repository import Repository,DEFAULT_DB
from .runtime import AlertInference

ROOT=Path(__file__).resolve().parents[1];UI=ROOT/'alerts_ui'


def handler(inference,connections,records):
    alerts=inference.alerts
    class Handler(inference_handler(inference,connections,records)):
        def dispatch(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
            parsed=urlsplit(self.path)
            files={'/alerts':'index.html','/wearable':'index.html','/alerts.js':'alerts.js','/alerts.css':'alerts.css','/alert-banner.js':'alert-banner.js'}
            if self.command=='GET' and parsed.path in files:
                name=files[parsed.path]
                return self.static((UI/name).read_bytes(),'text/html' if name.endswith('.html') else 'text/css' if name.endswith('.css') else 'text/javascript')
            if self.command=='GET' and parsed.path=='/twin-inference':
                raw=(ROOT/'inference_ui/index.html').read_bytes().replace(b'</head>',b'<link rel="stylesheet" href="/alerts.css"><script type="module" src="/alert-banner.js"></script></head>')
                return self.static(raw,'text/html')
            if self.command=='GET' and parsed.path=='/api/product/health':return self.reply(200,{'status':'ok','schema_version':7,'inference_connected':True,'alerts_connected':True,'native_wearable_push':False})
            parts=[unquote(v) for v in parsed.path.strip('/').split('/')]
            if parts[:2]==['api','product']:parts=parts[2:]
            if len(parts)>=3 and parts[0]=='patients':
                pid=parts[1]
                if parts[2:]==['notifications'] and self.command=='GET':return self.reply(200,alerts.inbox(pid))
                if parts[2]=='alerts':
                    if len(parts)==3 and self.command=='GET':return self.reply(200,alerts.list(pid))
                    if len(parts)==4 and self.command=='GET':return self.reply(200,alerts.active(pid) if parts[3]=='active' else alerts.get(pid,parts[3]))
                    if len(parts)==5:
                        aid,action=parts[3:]
                        if action=='deliveries' and self.command=='GET':return self.reply(200,alerts.deliveries(pid,aid))
                        if action=='evidence' and self.command=='GET':return self.reply(200,alerts.evidence(pid,aid))
                        if action=='acknowledge' and self.command=='POST':
                            value=self.body()
                            if set(value)-{'client'}:raise ValueError('Acknowledgement schema required')
                            return self.reply(200,alerts.acknowledge(pid,aid,value.get('client','dashboard')))
                        if action=='retry-delivery' and self.command=='POST':
                            if self.body():raise ValueError('Empty retry request required')
                            return self.reply(200,alerts.dispatch(pid,aid,retry=True))
            # Serve the existing compiled dashboard and its existing read-only
            # API from the preserved origin. No patient identity is mapped from
            # that historical demo cohort to a new product patient.
            if self.command in ('GET','POST') and (parsed.path.startswith(('/twin/','/assets/','/api/','/metabolic-api/')) and not parsed.path.startswith('/api/product/')):
                data=json.dumps(self.body()).encode() if self.command=='POST' else None
                try:
                    with urlopen(Request(os.environ.get('TWINCARE_PRODUCT_UPSTREAM', 'http://127.0.0.1:8097')+self.path,data=data,headers={'Content-Type':'application/json'},method=self.command),timeout=5) as response:
                        status=response.status;raw=response.read();mime=response.headers.get_content_type();csp=response.headers.get('Content-Security-Policy')
                except HTTPError as error:
                    status=error.code;raw=error.read();mime=error.headers.get_content_type();csp=error.headers.get('Content-Security-Policy')
                if mime=='text/html':raw=raw.replace(b'</head>',b'<link rel="stylesheet" href="/alerts.css"><script type="module" src="/alert-banner.js"></script></head>')
                self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)))
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
                if csp:self.send_header('Content-Security-Policy',csp)
                self.end_headers();self.wfile.write(raw);return
            return super().dispatch()
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8111);parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--storage',default='data/patient_domain/uploads');parser.add_argument('--photos',default='data/patient_domain/meal_photos');args=parser.parse_args()
    repo=Repository(args.db);records=IngestionService(repo,Path(args.storage));meals=Meals(repo,Path(args.photos))
    inference=AlertInference(InferenceReadiness(meals));connections=InferenceConnections(inference)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(inference,connections,records)).serve_forever()


if __name__=='__main__':main()
