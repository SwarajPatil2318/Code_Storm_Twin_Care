"""Same local API routes with frozen D2 trajectories and a separate frozen risk head."""
import argparse,copy,json
from http.server import ThreadingHTTPServer
from src.metabolic_d3.server import DemoStore,handler
from src.metabolic_d3.buffer import Buffer
from src.metabolic_d3.events import Event,timestamp
from .fusion import FusionAdapter,FusionRuntimeStore


class FusionDemoStore(DemoStore):
    def __init__(self,path=None,adapter=None):
        super().__init__(path);self.adapter=adapter or FusionAdapter()
        self.demo=copy.deepcopy(self.demo);self.cases={c['case']:c for c in self.demo['cases']}
        for case in self.cases.values():
            for i,frame in enumerate(case['timeline']):
                buffer=Buffer(case['patient']['id']);cutoff=timestamp(frame['timestamp'])[0]
                for raw in case['events']:
                    event=Event(**raw)
                    if event.available_ns<=cutoff and event.observed_ns<=cutoff:buffer.ingest(event)
                buffer.as_of=cutoff;case['timeline'][i]=self.adapter.enrich(frame,buffer)[0]
        self.demo['reference']['multimodal_risk']=self.adapter.manifest
    def route(self,path):
        if path=='/api/evidence/metabolic':
            payload=super().route(path);return {**payload,'multimodal_risk':self.adapter.manifest}
        return super().route(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8098);args=parser.parse_args()
    adapter=FusionAdapter();store=FusionDemoStore(adapter=adapter);runtime=FusionRuntimeStore(store.demo,adapter=adapter)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(store,runtime))
    print(f'TwinCare multimodal synthetic-demo API ready on loopback port {args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
