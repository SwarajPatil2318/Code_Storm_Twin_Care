"""Commit observations, reconcile readiness, then invoke backend inference."""
from patient_readiness.connections import ReadinessConnections


class InferenceConnections(ReadinessConnections):
    def __init__(self,inference,**kwargs):
        self.inference=inference
        super().__init__(inference.meals,time=inference.readiness.time,**kwargs)

    def _reconcile(self,pid,source,result):
        self.inference.readiness.evaluate(pid)
        if source=='CGM' and result.get('events_received',0)>0:
            try:
                generated=self.inference.run_twin_inference(pid,self._stamp())
                result['inference']={key:generated.get(key) for key in ('status','prediction_id','reference_time')}
            except Exception:
                # The observation is already committed. A failed forecast must
                # never make callers retry or mistake ingestion for a failure.
                result['inference']={'status':'INFERENCE_UNAVAILABLE','prediction_id':None}
        return result

    def ingest(self,pid,source,events):
        return self._reconcile(pid,source,super().ingest(pid,source,events))

    def sync(self,pid,source):
        return self._reconcile(pid,source,super().sync(pid,source))
