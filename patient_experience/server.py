"""One local TwinCare shell; inherited P1–P7 API and legacy links preserved."""
import argparse,json,re
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote,parse_qs,quote
from patient_alerts.server import handler as alert_handler
from patient_alerts.runtime import AlertInference
from patient_inference.connections import InferenceConnections
from patient_inference.readiness import InferenceReadiness
from patient_ingestion.service import IngestionService
from patient_connections.service import ConnectionError
from patient_meals.service import Meals
from src.patient_domain.repository import Repository,DEFAULT_DB
from .views import ProductViews
ROOT=Path(__file__).resolve().parents[1];UI=ROOT/'experience_ui'

def handler(inference,connections,records,simulation=None):
    views=ProductViews(inference,connections,records)
    class Handler(alert_handler(inference,connections,records)):
        def static(self,raw,mime):
            self.send_response(200);self.send_header("Content-Type",mime+"; charset=utf-8" if not mime.startswith("image/") else mime);self.send_header("Content-Length",str(len(raw)))
            self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(raw)
        def dispatch(self):
            parts=[unquote(p) for p in urlsplit(self.path).path.strip('/').split('/')]
            try:
                if simulation and len(parts)>=4 and parts[:3]==['api','product','patients']:
                    with simulation.context(parts[3]):
                        return self._dispatch_product()
                return self._dispatch_product()
            except ConnectionError as error:
                return self.reply(error.status,{'error':str(error)})
        def _dispatch_product(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
            parsed=urlsplit(self.path);parts=[unquote(p) for p in parsed.path.strip('/').split('/')]
            if simulation and len(parts)>=5 and parts[:3]==['api','product','patients']:
                pid=parts[3]
                if parts[4]=='simulation':
                    if len(parts)==5:
                        if self.command=='GET':return self.reply(200,simulation.status(pid))
                        if self.command=='POST':return self.reply(201,simulation.create(pid,self.body()))
                    if len(parts)==6 and self.command in ('POST','PATCH'):
                        action=parts[5];value=self.body()
                        if action=='load-prior-history':
                            if value:raise ValueError('No extra history fields required')
                            return self.reply(200,simulation.load_history(pid))
                        if action in ('start','pause','resume','stop'):
                            if value:raise ValueError('No extra control fields required')
                            return self.reply(200,simulation.control(pid,action))
                        if action in ('speed','activity'):
                            if set(value)!={action}:raise ValueError('Explicit control value required')
                            return self.reply(200,simulation.control(pid,action,value[action]))
                if parts[4:]==['connections'] and self.command=='POST':
                    value=self.body()
                    if value.get('mode')=='simulated_live':
                        source=value.pop('source_type',None);value.pop('mode')
                        return self.reply(201,simulation.connect(pid,source,value))
                    # Preserve the existing historical connection contract.
                    if set(value)!={'source_type','scenario','acknowledge_synthetic'}:raise ValueError('Connection schema required')
                    source=value.pop('source_type');return self.reply(202,connections.connect(pid,source,value))
            if self.command=='GET':
                if parsed.path=='/favicon.svg':return self.static((ROOT/'frontend/public/favicon.svg').read_bytes(),'image/svg+xml')
                if parsed.path=='/api/product/health':return self.reply(200,{'status':'ok','schema_version':8,'inference_connected':True,'alerts_connected':True,'native_wearable_push':False,'live_simulation_available':simulation is not None})
                if parsed.path=='/api/product/experience/evidence':
                    evidence=json.loads((ROOT/'artifacts/productization_p8/model_evidence.json').read_text())
                    config=json.loads((ROOT/'configs/metabolic_d3.json').read_text())
                    evidence['presentation']={'trend_threshold':config['rising_display_slope_mg_dl_per_minute']}
                    return self.reply(200,evidence)
                if len(parts)==5 and parts[:3]==['api','product','patients'] and parts[4]=='experience':
                    result=views.snapshot(parts[3])
                    if simulation:
                        result['simulation']=simulation.status(parts[3]);result['live_simulation_available']=True
                    return self.reply(200,result)
                if len(parts)==5 and parts[:3]==['api','product','patients'] and parts[4]=='experience-summary':return self.reply(200,views.summary(parts[3]))
                if parsed.path.startswith('/experience-assets/'):
                    name=parsed.path.removeprefix('/experience-assets/')
                    if name=='wearable.css':return self.static((ROOT/'frontend/src/platform/wearable.css').read_bytes(),'text/css')
                    if name not in ('app.js','workflows.js','chart.js','styles.css','wearable.js','simulation.js','presentation.js'):return self.reply(404,{'error':'Asset not found'})
                    return self.static((UI/name).read_bytes(),'text/css' if name.endswith('.css') else 'text/javascript')
                if parsed.path=='/experience-flow.png':return self.static((ROOT/'frontend/public/assets/metabolic-flow.png').read_bytes(),'image/png')
                if parsed.path=='/' or parsed.path=='/patients' or len(parts)==3 and parts[0]=='patients' and parts[2] in ('overview','data','timeline','evidence','setup','wearable'):
                    return self.static((UI/'index.html').read_bytes(),'text/html')
                migrations={'/medical-records':('data','records'),'/data-connections':('data','connections'),'/meal-logger':('data','meals'),'/twin-setup':('setup',''),'/twin-inference':('overview',''),'/alerts':('timeline','alerts'),'/wearable':('wearable','')}
                if parsed.path in migrations:
                    query=parse_qs(parsed.query);pid=query.get('patient',query.get('patient_id',['']))[0]
                    route,section=migrations[parsed.path];location='/patients/'+quote(pid,safe='')+'/'+route if pid else '/patients'
                    if section and pid:location+='?section='+section
                    self.send_response(302);self.send_header('Location',location);self.send_header('Content-Length','0');self.end_headers();return
            return super().dispatch()
    return Handler

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8113);p.add_argument('--db',default=str(DEFAULT_DB));p.add_argument('--storage',default='data/patient_domain/uploads');p.add_argument('--photos',default='data/patient_domain/meal_photos');a=p.parse_args()
    from live_simulation.service import Simulation,LiveConnections,ReplayClock
    live_clock=ReplayClock()
    repo=Repository(a.db);records=IngestionService(repo,Path(a.storage),time=live_clock.stamp);meals=Meals(repo,Path(a.photos),time=live_clock.now)
    inference=AlertInference(InferenceReadiness(meals,time=live_clock.now));connections=LiveConnections(inference)
    simulation=Simulation(inference,connections,live_clock);simulation.start_worker()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(inference,connections,records,simulation))
    try:server.serve_forever()
    finally:simulation.close();server.server_close()
if __name__=='__main__':main()
