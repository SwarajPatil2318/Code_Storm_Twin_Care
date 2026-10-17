"""Local Medical Records product surface and patient-scoped ingestion API."""
import argparse
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote,urlsplit,quote
from src.patient_domain.repository import Repository,DEFAULT_DB,NotFound
from src.patient_domain.server import handler as patient_handler
from .service import IngestionService,MAX_UPLOAD
from .schema import IngestionError

UI=Path(__file__).resolve().parents[1]/'patient_product_ui'


def handler(service):
    class Handler(patient_handler(service.repo)):
        def dispatch(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):
                return self.reply(403,{'error':'Origin denied'})
            path=urlsplit(self.path).path
            static={'/':'index.html','/medical-records':'index.html','/records.css':'records.css','/records.js':'records.js'}
            if self.command=='GET' and path in static:
                filename=static[path];raw=(UI/filename).read_bytes()
                mime='text/html; charset=utf-8' if filename.endswith('html') else 'text/css; charset=utf-8' if filename.endswith('css') else 'text/javascript; charset=utf-8'
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)))
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers();self.wfile.write(raw);return
            parts=[unquote(p) for p in path.strip('/').split('/')]
            if parts[:2]==['api','product']:parts=parts[2:]
            if len(parts)>=2 and parts[0]=='patients':
                pid=parts[1]
                if len(parts)==3 and parts[2]=='manual-extraction' and self.command=='POST':
                    return self.reply(201,service.manual(pid))
                if len(parts)>=3 and parts[2]=='documents':
                    if len(parts)==3:
                        if self.command=='GET':return self.reply(200,{'patient_id':pid,'items':service.list_documents(pid)})
                        if self.command=='POST':
                            if self.headers.get('Transfer-Encoding'):raise IngestionError('INVALID_UPLOAD','Upload a file with a known size.')
                            size=int(self.headers.get('Content-Length','0'))
                            if not 0<size<=MAX_UPLOAD:raise IngestionError('FILE_SIZE','Choose a nonempty document smaller than 4 MB.',413)
                            self.connection.settimeout(15)
                            raw=self.rfile.read(size)
                            if len(raw)!=size:raise IngestionError('INVALID_UPLOAD','The upload was incomplete. Please try again.')
                            doc=service.upload(pid,unquote(self.headers.get('X-Filename','')),self.headers.get('Content-Type',''),raw)
                            return self.reply(201,doc)
                    if len(parts)>=4:
                        did=parts[3]
                        if len(parts)==4 and self.command=='GET':return self.reply(200,service.get_document(pid,did))
                        if len(parts)==5:
                            action=parts[4]
                            if action=='source' and self.command=='GET':
                                doc=service.get_document(pid,did);file=Path(doc['storage_path']).resolve()
                                if file.parent!=service.storage or file.stem!=did.split(':')[1]:raise NotFound('Source not found')
                                raw=file.read_bytes()
                                if hashlib.sha256(raw).hexdigest()!=doc['content_hash']:raise IngestionError('SOURCE_CHANGED','The stored source did not pass its integrity check.',409)
                                self.send_response(200);self.send_header('Content-Type',doc['mime_type']);self.send_header('Content-Length',str(len(raw)))
                                self.send_header('Content-Disposition',"inline; filename*=UTF-8''"+quote(doc['filename'],safe=''))
                                self.send_header('Content-Security-Policy',"sandbox; default-src 'none'");self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
                                self.end_headers();self.wfile.write(raw);return
                            if action=='extraction':
                                if self.command=='GET':return self.reply(200,service.get_extraction(pid,did))
                                if self.command=='POST':return self.reply(200,service.extract(pid,did))
                                if self.command=='PATCH':
                                    value=self.body()
                                    if set(value)!={'revision','edits'}:raise ValueError('Review schema required')
                                    return self.reply(200,service.edit(pid,did,value['revision'],value['edits']))
                            if action=='confirm' and self.command=='POST':
                                value=self.body()
                                if set(value)!={'revision','confirm'}:raise ValueError('Confirmation schema required')
                                return self.reply(200,service.confirm(pid,did,value['revision'],value['confirm']))
                            if action=='refresh' and self.command=='POST':return self.reply(200,service.refresh(pid,did))
                            if action=='history' and self.command=='GET':return self.reply(200,{'patient_id':pid,'revisions':service.revisions(pid,did)})
            return super().dispatch()

        def run_request(self):
            try:self.dispatch()
            except IngestionError as error:self.reply(error.status,{'error':error.message,'code':error.code})
            except NotFound:self.reply(404,{'error':'Record not found'})
            except (ValueError,TypeError,UnicodeError):self.reply(400,{'error':'Check the information and try again.'})
            except Exception:self.reply(500,{'error':'The record could not be processed. Try again or enter information manually.'})
        do_GET=run_request
        do_POST=run_request
        do_PATCH=run_request
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8100)
    parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--storage',default='data/patient_domain/uploads')
    args=parser.parse_args()
    service=IngestionService(Repository(args.db),Path(args.storage))
    print(f'Medical Records is available at http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(service)).serve_forever()


if __name__=='__main__':main()
