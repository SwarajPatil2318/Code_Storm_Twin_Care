"""Calibrate a scenario hazard to configured synthetic patient event prevalence.

This fits no predictor and never uses real outcomes or pilot diagnostic scores.
A separate virtual calibration population supplies common-random-number draws.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from src.synthetic.config import load
from src.synthetic.patient_generator import patient,vulnerability
from src.synthetic.baseline_generator import baseline
from src.synthetic.state_model import simulate
from src.vitaldb_feasibility.source import private_json


def calibrate(path='configs/synthetic_v1.yaml',patients=512,seed=2031):
    config=load(path);grounding=json.loads(Path(config['grounding_path']).read_text());population=[]
    for index in range(20000,20000+patients):
        p=patient(index,config,seed);b=baseline(p,config,grounding,seed);population.append((p,vulnerability(p,b['mean'],config)))
    target=config['states']['target_patient_event_prevalence'];low,high=.00001,.01;curve=[];best=None
    for iteration in range(12):
        trial=(low+high)/2;config['states']['event_hazard_per_hour']=trial
        count=sum(simulate(p,v,config,seed)['event_time'] is not None for p,v in population);rate=count/patients
        curve.append({'base_hazard_per_hour':trial,'event_patient_prevalence':rate,'event_patients':count})
        if best is None or abs(rate-target)<abs(best[1]-target):best=(trial,rate)
        if rate<target:low=trial
        else:high=trial
    data=json.loads(Path(path).read_text());data['states']['event_hazard_per_hour']=best[0];data['states']['hazard_calibration']='separate512virtual patients,seed2031,indices20000..20511; common-random-number bisection; not real epidemiology'
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    result={'calibration_patients':patients,'seed':seed,'patient_indices':[20000,20000+patients-1],'target':target,'selected_hazard_per_hour':best[0],'calibration_event_prevalence':best[1],
      'curve':curve,'not_clinical_validation':True,'no_predictor_fitted':True,'evidence':'SYNTHETIC POC','evidence_label':'NOT VALIDATED'}
    private_json('data/synthetic_grounding/hazard_calibration.json',result);print({'hazard_calibrated':best[0],'scenario_target':target,'calibration_rate':best[1]},flush=True)
    return result

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):calibrate()
