"""Meal Logger product origin with intact P2/P3 surfaces and APIs."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote,parse_qs
from patient_connections.server import handler as connections_handler
from patient_ingestion.service import IngestionService
from src.patient_domain.repository import Repository,DEFAULT_DB
from .service import Meals
from .schema import MealError
from .providers import FixtureProvider,configured_provider
from .connections import MealConnections
from .images import MAX_BYTES

UI=Path(__file__).resolve().parents[1]/'meal_logger_ui'
QA=Path(__file__).resolve().parents[1]/'data/productization_p4'


def handler(meals,connections,records):
    parent=connections_handler(connections,records)
    class Handler(parent):
        def dispatch(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
            parsed=urlsplit(self.path)
            static={'/':'index.html','/meal-logger':'index.html','/meals.js':'meals.js','/meals.css':'meals.css','/meal-hub.js':'meal-hub.js'}
            if self.command=='GET' and parsed.path in static:
                name=static[parsed.path];raw=(UI/name).read_bytes();mime='text/html' if name.endswith('html') else 'text/css' if name.endswith('css') else 'text/javascript'
                self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers();self.wfile.write(raw);return
            if self.command=='GET' and parsed.path=='/data-connections':
                from patient_connections.server import UI as HUB
                raw=(HUB/'index.html').read_text().replace('</head>','<script type="module" src="/meal-hub.js"></script></head>').encode()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers();self.wfile.write(raw);return
            parts=[unquote(v) for v in parsed.path.strip('/').split('/')]
            if parts[:2]==['api','product']:parts=parts[2:]
            try:
                if self.command=='GET' and parts==['meal-analysis','status']:
                    return self.reply(200,{'available':meals.provider.available,'provider':meals.provider.name,'engineering_only':meals.provider.engineering_only,'manual_entry_available':True})
                if len(parts)>=3 and parts[0]=='patients' and parts[2]=='meals':
                    pid=parts[1]
                    if len(parts)==3 and self.command=='GET':return self.reply(200,meals.list(pid))
                    if parts[3:]==['photo'] and self.command=='POST':
                        if self.headers.get('Transfer-Encoding'):raise ValueError('Known-size image required')
                        size=int(self.headers.get('Content-Length','0'))
                        if not 0<size<=MAX_BYTES:raise MealError('Choose a photo smaller than 8 MB.',413)
                        self.connection.settimeout(15);raw=self.rfile.read(size)
                        if len(raw)!=size:raise ValueError('Incomplete image')
                        return self.reply(201,meals.upload(pid,unquote(self.headers.get('X-Filename','')),self.headers.get_content_type(),raw))
                    if parts[3:]==['drafts'] and self.command=='POST':
                        v=self.body()
                        if set(v)-{'source_draft_id'}:raise ValueError('Manual draft schema required')
                        return self.reply(201,meals.manual_draft(pid,v.get('source_draft_id')))
                    if len(parts)>=5 and parts[3]=='drafts':
                        did=parts[4]
                        if len(parts)==5:
                            if self.command=='GET':return self.reply(200,meals.draft(pid,did))
                            if self.command=='PATCH':
                                v=self.body()
                                if set(v)!={'revision','meal'}:raise ValueError('Review schema required')
                                return self.reply(200,meals.review(pid,did,v['revision'],v['meal']))
                        if len(parts)==6 and self.command=='POST':
                            v=self.body()
                            if parts[5]=='analyze':
                                if set(v)!={'revision'}:raise ValueError('Analysis schema required')
                                return self.reply(202,meals.start_analysis(pid,did,v['revision']))
                            if parts[5]=='run-analysis':
                                if set(v)!={'revision'}:raise ValueError('Analysis schema required')
                                return self.reply(200,meals.analyze(pid,did,v['revision']))
                            if parts[5]=='confirm':
                                if set(v)!={'revision','confirm'}:raise ValueError('Confirmation schema required')
                                return self.reply(201,meals.confirm(pid,did,v['revision'],v['confirm']))
                    if len(parts)==5 and parts[3]=='photos' and self.command=='GET':
                        raw,mime=meals.photo(pid,parts[4]);self.send_response(200);self.send_header('Content-Type',mime)
                        self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
                        self.send_header('Content-Security-Policy',"sandbox; default-src 'none'");self.end_headers();self.wfile.write(raw);return
                    if parts[3:]==['context'] and self.command=='GET':
                        q=parse_qs(parsed.query)
                        if set(q)-{'reference_time','lookback_minutes'} or 'reference_time' not in q or any(len(v)!=1 for v in q.values()):raise ValueError('Context query required')
                        return self.reply(200,meals.recent_context(pid,q['reference_time'][0],int(q.get('lookback_minutes',['120'])[0])))
                    if len(parts)==4:
                        mid=parts[3]
                        if self.command=='GET':return self.reply(200,meals.get(pid,mid))
                        v=self.body()
                        if self.command=='PATCH':
                            if set(v)!={'revision','meal','confirm'}:raise ValueError('Meal edit schema required')
                            return self.reply(200,meals.edit(pid,mid,v['revision'],v['meal'],v['confirm']))
                        if self.command=='DELETE':
                            if set(v)!={'revision'}:raise ValueError('Meal removal schema required')
                            return self.reply(200,meals.delete(pid,mid,v['revision']))
                    if len(parts)==5 and parts[4]=='history' and self.command=='GET':return self.reply(200,meals.history(pid,parts[3]))
            except MealError as e:return self.reply(e.status,{'error':str(e)})
            return super().dispatch()
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8104);p.add_argument('--db',default=str(DEFAULT_DB));p.add_argument('--storage',default='data/patient_domain/meal_photos')
    p.add_argument('--records-storage',default='data/patient_domain/uploads');p.add_argument('--engineering-fixtures',action='store_true');a=p.parse_args()
    provider=configured_provider()
    if a.engineering_fixtures:
        db_path=Path(a.db).resolve()
        if not db_path.is_relative_to(QA.resolve()) or db_path==DEFAULT_DB.resolve():raise ValueError('Engineering provider requires an isolated P4 QA database')
        provider=FixtureProvider(QA/'test_image_manifest.json')
    repo=Repository(a.db);records=IngestionService(repo,Path(a.records_storage));meals=Meals(repo,Path(a.storage),provider,engineering_test=a.engineering_fixtures)
    connections=MealConnections(meals)
    ThreadingHTTPServer(('127.0.0.1',a.port),handler(meals,connections,records)).serve_forever()


if __name__=='__main__':main()
