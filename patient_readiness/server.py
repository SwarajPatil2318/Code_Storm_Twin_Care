"""P5 onboarding origin retaining P1–P4 routes; no generalized inference."""
import argparse,json
from datetime import datetime,timezone,timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs,unquote
from src.patient_domain.repository import Repository,DEFAULT_DB
from patient_ingestion.service import IngestionService
from patient_meals.service import Meals
from patient_meals.server import handler as meal_handler
from .service import Readiness
from .connections import ReadinessConnections
from .demo import accelerate
ROOT=Path(__file__).resolve().parents[1];UI=ROOT/'readiness_ui';PRIVATE=ROOT/'data/productization_p5'


class EngineeringClock:
    """Controlled clock only for an explicitly isolated engineering service."""
    def __init__(self):self.offset=timedelta()
    def __call__(self):return datetime.now(timezone.utc)+self.offset
    def advance(self,minutes):
        if type(minutes) is not int or not 0<=minutes<=120:raise ValueError('Engineering clock advance must be 0–120 minutes')
        self.offset+=timedelta(minutes=minutes)


def handler(readiness,connections,records,engineering_demo=False):
    parent=meal_handler(readiness.meals,connections,records)
    class Handler(parent):
        def reply(self,status,value):
            # Service mutations have committed before reply. Reconcile current
            # readiness without changing existing response contracts.
            if self.command in ['POST','PATCH','DELETE'] and (200<=status<300 or status==503):
                parts=[unquote(p) for p in urlsplit(self.path).path.strip('/').split('/')]
                if parts[:2]==['api','product']:parts=parts[2:]
                pid=parts[1] if len(parts)>=2 and parts[0]=='patients' else value.get('patient_id') if isinstance(value,dict) and parts==['patients'] else None
                if pid:
                    from src.patient_domain.repository import NotFound
                    try:readiness.evaluate(pid)
                    except NotFound:pass
            return super().reply(status,value)
        def static(self,raw,mime):
            self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(raw)
        def dispatch(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
            parsed=urlsplit(self.path);files={'/':'index.html','/twin-setup':'index.html','/readiness.js':'readiness.js','/readiness.css':'readiness.css','/readiness-hub.js':'readiness-hub.js','/readiness-records.js':'readiness-records.js'}
            if self.command=='GET' and parsed.path in files:
                name=files[parsed.path];return self.static((UI/name).read_bytes(),'text/html' if name.endswith('.html') else 'text/css' if name.endswith('.css') else 'text/javascript')
            if self.command=='GET' and parsed.path=='/data-connections':
                raw=(ROOT/'connections_ui/index.html').read_text().replace('</head>','<script type="module" src="/meal-hub.js"></script><script type="module" src="/readiness-hub.js"></script></head>')
                return self.static(raw.encode(),'text/html')
            if self.command=='GET' and parsed.path=='/medical-records':
                raw=(ROOT/'patient_product_ui/index.html').read_text().replace('</head>','<script type="module" src="/readiness-records.js"></script></head>')
                return self.static(raw.encode(),'text/html')
            if self.command=='GET' and parsed.path=='/api/product/readiness/config':return self.reply(200,{'engineering_demo':engineering_demo,'inference_enabled':False})
            if engineering_demo and self.command=='POST' and parsed.path=='/api/product/readiness/engineering-clock':
                value=self.body()
                if set(value)!={'minutes','acknowledge_synthetic'} or value['acknowledge_synthetic'] is not True:raise ValueError('Engineering clock acknowledgement required')
                readiness.time.advance(value['minutes']);return self.reply(200,{'evaluated_at':readiness._stamp(),'engineering_only':True})
            parts=[unquote(v) for v in parsed.path.strip('/').split('/')]
            if parts[:2]==['api','product']:parts=parts[2:]
            if len(parts)==3 and parts[0]=='patients' and parts[2]=='twin-readiness' and self.command=='GET':
                query=parse_qs(parsed.query,keep_blank_values=True)
                if set(query)-{'at','inspect'} or any(len(v)!=1 for v in query.values()) or query.get('inspect',['false'])[0] not in ['true','false']:raise ValueError('Invalid readiness query')
                return self.reply(200,readiness.evaluate(parts[1],query.get('at',[None])[0],query.get('inspect',['false'])[0]=='true'))
            if len(parts)==3 and parts[0]=='patients' and parts[2]=='readiness-demo' and self.command=='POST':
                if not engineering_demo:return self.reply(404,{'error':'Engineering tooling is disabled'})
                value=self.body()
                if set(value)!={'minutes','acknowledge_synthetic'}:raise ValueError('Engineering history schema required')
                return self.reply(200,accelerate(readiness,connections,parts[1],value['minutes'],value['acknowledge_synthetic']))
            if len(parts)==4 and parts[0]=='patients' and parts[2:]==['readiness-demo','fail-source'] and self.command=='POST' and engineering_demo:
                value=self.body()
                if set(value)!={'acknowledge_synthetic'} or value['acknowledge_synthetic'] is not True or not readiness.repo.get_patient(parts[1])['synthetic']:raise ValueError('Explicit synthetic source-failure test required')
                class FailedAdapter:
                    def sync(self,*args):raise RuntimeError('Engineering adapter failure')
                original=connections.adapters['CGM'];connections.adapters['CGM']=FailedAdapter()
                try:
                    from patient_connections.service import ConnectionError
                    try:connections.sync(parts[1],'CGM')
                    except ConnectionError:pass
                finally:connections.adapters['CGM']=original
                return self.reply(200,readiness.evaluate(parts[1]))
            return super().dispatch()
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8107);parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--storage',default='data/patient_domain/uploads');parser.add_argument('--photos',default='data/patient_domain/meal_photos')
    parser.add_argument('--engineering-demo',action='store_true');args=parser.parse_args()
    if args.engineering_demo and (not Path(args.db).resolve().is_relative_to(PRIVATE.resolve()) or Path(args.db).resolve()==DEFAULT_DB.resolve()):
        raise ValueError('Engineering demo requires an isolated P5 data directory')
    time=EngineeringClock() if args.engineering_demo else None
    repo=Repository(args.db);records=IngestionService(repo,Path(args.storage));meals=Meals(repo,Path(args.photos),time=time);readiness=Readiness(meals,time=time)
    connections=ReadinessConnections(meals,time=time)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(readiness,connections,records,args.engineering_demo)).serve_forever()


if __name__=='__main__':main()
