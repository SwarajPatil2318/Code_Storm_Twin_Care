"""Read-only reference registration; never opens spent holdout trajectories."""
import json
from pathlib import Path
from src.metabolic_d2.protocol import ROOT as D2_ROOT,digest,verify_frozen,write_json
from src.metabolic_d2.models import ForecastModel
from src.metabolic_d2.risk import RiskBundle

ROOT=Path('data/metabolic_d3')
NAME='METABOLIC_D2_REFERENCE'

def register():
    from src.runtime_assets import portable_reference
    portable = portable_reference()
    if portable is not None:
        return portable
    frozen=verify_frozen()  # Source/model/config hashes only; no holdout trajectory reads.
    paths={role:D2_ROOT/'models'/(name+'.pkl') for role,name in
        [('cold',frozen['forecast_selection']['population']),('warm',frozen['forecast_selection']['warm_model']),('risk','risk')]}
    record={'name':NAME,'D2_manifest_sha256':digest(D2_ROOT/'FROZEN_MODEL_MANIFEST.json'),
        'D2_scientific_code_sha256':frozen['code_sha256'],
        'model_hashes':{k:digest(v) for k,v in paths.items()},'paths':{k:str(v) for k,v in paths.items()},
        'threshold':frozen['spike_threshold'],'risk_method':frozen['risk_method'],
        'protected_event_hashes':{p.name:digest(p) for p in [D2_ROOT/'HOLDOUT_EVALUATION_STARTED.json',D2_ROOT/'HOLDOUT_EVALUATION_COMPLETED.json',D2_ROOT/'access_ledger.json']},
        'historical_metrics_scope':'D2 aggregate results only; D3 does not access spent holdout records',
        'D3_config_sha256':digest('configs/metabolic_d3.json')}
    path=ROOT/'reference_registry.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise RuntimeError('D2 reference or D3 protocol changed')
    else:write_json(path,record,exclusive=True)
    return record

class Reference:
    def __init__(self):
        self.registration=register()
        self.cold=ForecastModel.load(self.registration['paths']['cold'])
        self.warm=ForecastModel.load(self.registration['paths']['warm'])
        self.risk=RiskBundle.load(Path(self.registration['paths']['risk']))
    def predict(self,data):
        model=self.warm if data['warm'][0] else self.cold
        p=model.predict(data)
        probability=self.risk.probabilities(data,p)[self.risk.method]
        return p[0],float(probability[0]),'warm' if data['warm'][0] else 'cold'

def assert_protected():
    register()
