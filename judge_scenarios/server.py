"""Loopback-only judge tooling, isolated database, inherited P1–P8 product API."""
import argparse,json
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit
from patient_experience.server import handler as product_handler
from .catalog import ROOT,PROFILES
from .service import Scenarios
UI=ROOT/'judge_ui'
NAMES={'A':'Post-meal spike','B':'Stable response','C':'Personal deviation'}

def handler(service):
 class Handler(product_handler(service.runtime,service.connections,service.records)):
  def dispatch(self):
   origin=self.headers.get('Origin')
   if origin and origin!='http://'+self.headers.get('Host',''):return self.reply(403,{'error':'Origin denied'})
   parsed=urlsplit(self.path);parts=[unquote(x) for x in parsed.path.strip('/').split('/')]
   with service.lock:
    # The real P8 shell's shared React component must remain available even
    # when the optional historical-module proxy is not running.
    if self.command=='GET' and parsed.path.startswith('/assets/'):
     assets=(ROOT/'frontend/dist/assets').resolve();target=(assets/parsed.path.removeprefix('/assets/')).resolve()
     if not target.is_relative_to(assets) or not target.is_file() or target.suffix not in ('.js','.css','.png','.svg','.webp','.woff2'):return self.reply(404,{'error':'Asset not found'})
     mime={'.js':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml','.webp':'image/webp','.woff2':'font/woff2'}[target.suffix]
     return self.static(target.read_bytes(),mime)
    if self.command=='GET' and parsed.path in ('/','/demo','/dev/judge-scenarios'):
     return self.static((UI/'index.html').read_bytes(),'text/html')
    if self.command=='GET' and parsed.path.startswith('/judge-assets/'):
     name=parsed.path.removeprefix('/judge-assets/')
     if name not in ('launcher.js','playback.js','styles.css'):return self.reply(404,{'error':'Asset not found'})
     return self.static((UI/name).read_bytes(),'text/css' if name.endswith('.css') else 'text/javascript')
    if parts[:3]==['api','judge','scenarios']:
     if self.command=='GET' and len(parts)==3:
      return self.reply(200,{'scenarios':[{'key':k,'name':NAMES[k],'patient_name':PROFILES[k]['display_name'],'state':service.state(k) if k in service.sessions else {'status':'UNINITIALIZED'}} for k in service.configs], 'isolated_judge_service':True,'product_pipeline':'P1–P8'})
     if len(parts)>=4:
      key=parts[3]
      if key not in service.configs:return self.reply(404,{'error':'Unknown scenario'})
      if self.command=='GET' and len(parts)==4:
       return self.reply(200,service.state(key) if key in service.sessions else {'scenario':key,'status':'UNINITIALIZED'})
      if self.command=='GET' and parts[4:]==['milestones']:
       return self.reply(200,{'scenario':key,'milestones':service.state(key)['milestones'] if key in service.sessions else []})
      if self.command=='POST' and len(parts)==5:
       value=self.body();action=parts[4]
       if set(value)-({'speed'} if action in ('start','resume','speed') else set()):raise ValueError('Unsupported judge field')
       if action in ('initialize','reset'):return self.reply(200,service.reset(key))
       if action=='teardown':return self.reply(200,service.teardown(key))
       if key not in service.sessions:return self.reply(409,{'error':'Reset this scenario first'})
       if action in ('start','resume'):return self.reply(200,service.start(key,value.get('speed')))
       if action=='pause':return self.reply(200,service.pause(key))
       if action=='advance':
        service.pause(key);return self.reply(200,service.advance(key))
       if action=='speed':
        speed=value.get('speed')
        if speed not in ('1×','4×','8×','Jury Demo'):raise ValueError('Invalid speed')
        service.sessions[key]['speed']=speed;service.sessions[key]['next_due']=None
        # Rescheduling changes wall clock only; no physiological event skipped.
        if service.sessions[key]['status']=='RUNNING':service.sessions[key]['next_due']=__import__('time').monotonic()
        service._persist(key);return self.reply(200,service.state(key))
     return self.reply(405,{'error':'Judge action not supported'})
    pid=next((parts[i+1] for i,x in enumerate(parts[:-1]) if x=='patients'),None)
    key=service.for_patient(pid) if pid else None
    if self.command=='GET' and len(parts)==3 and parts[0]=='patients' and parts[2] in ('overview','data','timeline','evidence','setup','wearable'):
     raw=(ROOT/'experience_ui/index.html').read_bytes().replace(b'</head>',b'<link rel="stylesheet" href="/judge-assets/styles.css"><script type="module" src="/judge-assets/playback.js"></script></head>')
     return self.static(raw,'text/html')
    if key:
     with service.clock.at(service.reference(key)):
      response=super().dispatch()
      if self.command=='POST' and parts[-1]=='acknowledge' and 'alerts' in parts:
       notifications=service.runtime.alerts.inbox(pid)['items']
       if any(n['delivery']['status']=='ACKNOWLEDGED' for n in notifications) and not any(m['event_type']=='watch_acknowledged' for m in service.sessions[key]['milestones']):
        service._record(key,'watch_acknowledged');service._persist(key)
      return response
    return super().dispatch()
 return Handler

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8116);p.add_argument('--root',default=str(ROOT/'data/productization_p9/live'));p.add_argument('--inputs',default=str(ROOT/'data/productization_p9/selected_inputs.json'));a=p.parse_args()
 service=Scenarios(a.root)
 for key,config in json.loads(Path(a.inputs).read_text()).items():service.install(key,config)
 service.launch_worker();server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(service))
 try:server.serve_forever()
 finally:server.server_close();service.close()
if __name__=='__main__':main()
