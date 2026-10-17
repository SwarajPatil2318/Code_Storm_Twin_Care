"""P6 subclass preserves its outputs and evaluates persisted states server-side."""
from patient_inference.service import Inference
from .service import Alerts


class AlertInference(Inference):
    def __init__(self,readiness,provider=None):
        super().__init__(readiness);self.alerts=Alerts(self,provider)

    def run_twin_inference(self,patient_id,reference_time):
        with self._lock:
            result=super().run_twin_inference(patient_id,reference_time)
            if result.get('twin_state') is not None:
                try:result['alert_evaluation']=self.alerts.evaluate_alerts(patient_id,result['twin_state']['twin_state_id'])
                except Exception:
                    # A persisted forecast stays valid if alert processing fails.
                    # Re-evaluating this state retries the independently stored
                    # alert transaction without executing a new forecast.
                    result['alert_evaluation']={'status':'ALERT_EVALUATION_UNAVAILABLE','patient_id':patient_id}
            return result
