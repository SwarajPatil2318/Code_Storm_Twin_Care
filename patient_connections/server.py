"""P3 local hub plus unchanged P2 Medical Records routes on the same origin."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs
from patient_ingestion.server import handler as records_handler
from patient_ingestion.service import IngestionService
from src.patient_domain.repository import Repository, DEFAULT_DB
from .service import Connections, ConnectionError

UI=Path(__file__).resolve().parents[1]/'connections_ui'


def handler(connections, ingestion):
    class Handler(records_handler(ingestion)):
        def dispatch(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
            parsed=urlsplit(self.path)
            static={'/':'index.html','/data-connections':'index.html','/connections.js':'connections.js','/connections.css':'connections.css'}
            if self.command=='GET' and parsed.path in static:
                name=static[parsed.path];data=(UI/name).read_bytes()
                mime='text/html' if name.endswith('html') else 'text/css' if name.endswith('css') else 'text/javascript'
                self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers();self.wfile.write(data);return
            parts=[unquote(p) for p in parsed.path.strip('/').split('/')]
            if parts[:2]==['api','product']:parts=parts[2:]
            try:
                if len(parts)>=3 and parts[0]=='patients':
                    pid=parts[1]
                    if parts[2]=='connections':
                        if len(parts)==3:
                            if self.command=='GET':return self.reply(200,connections.list(pid))
                            if self.command=='POST':
                                value=self.body()
                                if set(value)!={'source_type','scenario','acknowledge_synthetic'}:raise ValueError('Connection schema required')
                                source=value.pop('source_type');return self.reply(202,connections.connect(pid,source,value))
                        if len(parts)==4:
                            source=parts[3]
                            if self.command=='GET':return self.reply(200,connections.status(pid,source))
                            if self.command=='DELETE':return self.reply(200,connections.disconnect(pid,source))
                        if len(parts)==5 and self.command=='POST':
                            source,action=parts[3:]
                            if action=='activate':return self.reply(200,connections.activate(pid,source))
                            if action=='sync':return self.reply(200,connections.sync(pid,source))
                            if action=='events':
                                value=self.body()
                                if set(value)!={'events'}:raise ValueError('Events schema required')
                                return self.reply(200,connections.ingest(pid,source,value['events']))
                    if parts[2]=='observations' and self.command=='GET' and (len(parts)==3 or parts[3:]==['recent']):
                        q=parse_qs(parsed.query)
                        if set(q)-{'source','event_type','start','end','limit'} or any(len(v)!=1 for v in q.values()):raise ValueError('Invalid query')
                        return self.reply(200,connections.observations(pid,source=q.get('source',[None])[0],event_type=q.get('event_type',[None])[0],
                            start=q.get('start',[None])[0],end=q.get('end',[None])[0],limit=int(q.get('limit',['100'])[0]),recent=len(parts)==4))
            except ConnectionError as error:return self.reply(error.status,{'error':str(error)})
            return super().dispatch()
        do_DELETE=records_handler(ingestion).run_request
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8102)
    parser.add_argument('--db',default=str(DEFAULT_DB));parser.add_argument('--storage',default='data/patient_domain/uploads')
    args=parser.parse_args();repo=Repository(args.db);ingestion=IngestionService(repo,Path(args.storage));connections=Connections(repo)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(connections,ingestion)).serve_forever()


if __name__=='__main__':main()
