"""Additive local P6 API and inspection surface; P1–P5 routes preserved."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs,unquote
from src.patient_domain.repository import Repository, DEFAULT_DB
from patient_ingestion.service import IngestionService
from patient_meals.service import Meals
from .readiness import InferenceReadiness
from patient_readiness.server import handler as readiness_handler
from .service import Inference
from .connections import InferenceConnections

ROOT=Path(__file__).resolve().parents[1]
UI=ROOT/'inference_ui'


def handler(inference,connections,records):
    class Handler(readiness_handler(inference.readiness,connections,records)):
        def dispatch(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
            parsed=urlsplit(self.path)
            files={'/twin-inference':'index.html','/inference.js':'inference.js','/inference.css':'inference.css'}
            if self.command=='GET' and parsed.path in files:
                name=files[parsed.path]
                return self.static((UI/name).read_bytes(),'text/html' if name.endswith('.html') else 'text/css' if name.endswith('.css') else 'text/javascript')
            if self.command=='GET' and parsed.path=='/api/product/health':
                return self.reply(200,{'status':'ok','schema_version':6,'inference_connected':True,'clinical_status':'NOT CLINICALLY VALIDATED'})
            parts=[unquote(v) for v in parsed.path.strip('/').split('/')]
            if parts[:2]==['api','product']:parts=parts[2:]
            if len(parts)>=3 and parts[0]=='patients':
                pid=parts[1]
                if parts[2:]==['twin','infer'] and self.command=='POST':
                    value=self.body()
                    if set(value)!={'reference_time'}:raise ValueError('Explicit reference_time required')
                    return self.reply(200,inference.run_twin_inference(pid,value['reference_time']))
                if parts[2:]==['twin'] and self.command=='GET':
                    query=parse_qs(parsed.query,keep_blank_values=True)
                    if set(query)-{'at'} or any(len(v)!=1 for v in query.values()):raise ValueError('Invalid Twin query')
                    return self.reply(200,inference.latest(pid,query.get('at',[None])[0]))
                if len(parts)==5 and parts[2]=='predictions' and parts[4]=='trace' and self.command=='GET':
                    return self.reply(200,inference.trace(pid,parts[3]))
            return super().dispatch()
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8109);parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--storage',default='data/patient_domain/uploads');parser.add_argument('--photos',default='data/patient_domain/meal_photos')
    args=parser.parse_args()
    repo=Repository(args.db);records=IngestionService(repo,Path(args.storage));meals=Meals(repo,Path(args.photos))
    readiness=InferenceReadiness(meals);inference=Inference(readiness);connections=InferenceConnections(inference)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(inference,connections,records)).serve_forever()


if __name__=='__main__':main()
