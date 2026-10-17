"""Optional real-model release checks. Records exist only in temporary local storage."""
import contextlib
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT/'data/runtime/manifest.json').is_file(),'Trusted private runtime bundle required')
class RuntimeAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'TWINCARE_RUNTIME_MANIFEST':str(ROOT/'data/runtime/manifest.json'),
                                       'MEAL_ANALYSIS_PROVIDER':'disabled'});self.env.start()
        from src.patient_domain.repository import Repository
        from patient_ingestion.service import IngestionService
        from patient_meals.service import Meals
        from patient_inference.readiness import InferenceReadiness
        from patient_alerts.runtime import AlertInference
        from live_simulation.service import ReplayClock,Simulation,LiveConnections
        from tools.create_sample_records import create
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);self.clock=ReplayClock()
        self.repo=Repository(root/'patients.sqlite3');self.records=IngestionService(self.repo,root/'records',time=self.clock.stamp)
        self.meals=Meals(self.repo,root/'photos',time=self.clock.now);self.runtime=AlertInference(InferenceReadiness(self.meals,time=self.clock.now))
        self.connections=LiveConnections(self.runtime);self.sim=Simulation(self.runtime,self.connections,self.clock)
        with contextlib.redirect_stdout(StringIO()):sample=create(root/'examples')
        self.input=sample.read_bytes();self.pid=self.new_patient()

    def tearDown(self):
        self.sim.close();self.tmp.cleanup();self.env.stop()

    def new_patient(self):
        pid=self.repo.create_patient(display_name='Release Integration Fixture',synthetic=True)['patient_id']
        doc=self.records.upload(pid,'clinical_history.json','application/json',self.input)
        draft=self.records.extract(pid,doc['document_id'])
        draft=self.records.edit(pid,doc['document_id'],draft['revision'],[{'fact_id':f['fact_id'],'decision':'accept'}for f in draft['facts']])
        self.records.confirm(pid,doc['document_id'],draft['revision'],True)
        return pid

    def ready(self,history):
        self.sim.connect(self.pid,'CGM',{'acknowledge_synthetic':True,'load_prior_history':history,'seed':1234})
        self.sim.connect(self.pid,'WEARABLE',{'acknowledge_synthetic':True});self.sim.control(self.pid,'start')

    def test_new_patient_real_warm_forecast_baseline_risk_and_ownership(self):
        self.ready(True);self.sim.step(self.pid);result=self.runtime.latest(self.pid);state=result.get('twin_state')
        self.assertTrue(result['status']=='INFERRED','Real model inference did not complete')
        self.assertTrue(state['forecast_mode']=='warm','Prior history should enable warm model')
        self.assertTrue(state['shadow']['available'],'Historical expected trajectory unavailable')
        self.assertTrue(state['spike']['available'],'Complete confirmed context should enable risk')
        self.assertTrue(len(state['forecast'])==4,'Four horizons required')
        trace=self.runtime.trace(self.pid,state['prediction_id'])
        self.assertTrue(trace['artifact_execution']['forecast']['executed'])
        self.assertTrue(trace['artifact_execution']['risk']['executed'])
        from patient_connections.contracts import clock
        self.assertTrue(all(clock(e['observed_at'])<=clock(state['reference_time'])and clock(e['received_at'])<=clock(state['reference_time'])for e in trace['inputs']['CGM']))
        other=self.new_patient();self.assertTrue(self.runtime.latest(other)['status']=='NO_PREDICTION','Owner state crossed patients')
        with self.sim.context(self.pid):
            draft=self.meals.manual_draft(self.pid)
            value={'title':'Fictional test meal','foods':[{'name':'Rice','confidence':None}],
                   'meal_occurred_at':self.clock.stamp(),'carbohydrate_category':'HIGH','glycemic_impact_category':'HIGH',
                   'estimated_carb_range_g':None,'user_notes':''}
            draft=self.meals.review(self.pid,draft['draft_id'],draft['revision'],value)
            self.meals.confirm(self.pid,draft['draft_id'],draft['revision'],True)
        self.assertTrue(self.runtime.latest(self.pid)['prediction_id']==result['prediction_id'],'Meal confirmation must not rewrite a frozen prediction')
        self.sim.step(self.pid)
        self.assertTrue(self.runtime.latest(self.pid)['prediction_id']!=result['prediction_id'],'New observation should generate a new prediction')

    def test_start_fresh_requires_history_and_then_uses_cold_model(self):
        self.ready(False)
        for _ in range(7):self.sim.step(self.pid)
        self.assertTrue(self.runtime.latest(self.pid)['status']=='NO_PREDICTION')
        self.sim.step(self.pid);state=self.runtime.latest(self.pid).get('twin_state')
        self.assertTrue(state is not None,'Complete native history should enable inference')
        self.assertTrue(state['forecast_mode']=='cold');self.assertFalse(state['shadow']['available'])

    def test_modified_scientific_fingerprint_rejected_in_deployment(self):
        from src.metabolic_d3.reference import register
        with patch('src.metabolic_d2.protocol.code_hash',return_value='modified'):
            with self.assertRaises(RuntimeError):register()

if __name__=='__main__':unittest.main()
