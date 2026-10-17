"""Final completeness/preservation audit; no new clinical TEST decoding."""
import json,stat
from pathlib import Path
from src.twin.protocol import load,EVIDENCE
from src.twin.sealing import verify,REGISTRY
from src.twin.report import DOCS
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def run(config=None):
    c=config or load();out=Path(c['output']);seal=verify(c);results=json.loads((out/'test_results.json').read_text());completion=json.loads((out/'evaluation_complete.json').read_text());registry=json.loads((REGISTRY/(seal['cohort_manifest_sha256']+'.json')).read_text());registration=json.loads((out/'registration.json').read_text());gates={}
    gates['exactly_one_frozen_joint_test_evaluation']=completion['status']=='COMPLETE' and completion['logical_test_evaluations']==1 and registry==completion and results['joint_test_evaluations']==1 and results['transaction_event_id']==completion['event_id'] and completion['results_sha256']==file_hash(out/'test_results.json')
    gates['all_required_models']=set(results['models'])==set(c['model_variants'])
    gates['all_horizons_metrics_bootstraps']=True
    for model in results['models'].values():
        gates['all_horizons_metrics_bootstraps'] &= set(model['horizons'])=={'24','48','72'} and model['patients']==1500 and model['rows']==76500
        for horizon in ['24','48','72']:
            expected={'auprc','auroc','brier','sensitivity','specificity','ppv','npv','f1','prevalence'};gates['all_horizons_metrics_bootstraps'] &= expected<=set(model['horizons'][horizon]['metrics']) and expected|{'calibration_slope','calibration_intercept'}<=set(model['patient_bootstrap95'][horizon]) and all(v['requested']==1000 for v in model['patient_bootstrap95'][horizon].values())
    gates['test_metadata_fixed_schedule']=all(json.loads((out/'features'/role/'manifest.json').read_text())['rows']==76500 and json.loads((out/'features'/role/'manifest.json').read_text())['patients']==1500 for role in ['test','consumer_test'])
    gates['all_documentation']=all((Path('docs')/(name+'.md')).exists() and all(label in (Path('docs')/(name+'.md')).read_text() for label in EVIDENCE.values()) for name in DOCS)
    figures=json.loads((out/'figures/manifest.json').read_text());gates['all_plot_categories']=len(figures['types'])>=12 and all((out/'figures'/name).exists() for name in figures['figure_files'])
    gates['causal_sensor_ablations_and_stress']=set(results['sensor_ablations']['experiments'])==set(c['sensor_ablations']) and all(v['removed_before_baseline'] for v in results['sensor_ablations']['experiments'].values()) and set(results['stress']['variants'])=={'normal_noise','high_sensor_noise','high_missingness','reduced_sensor_set','baseline_shift'}
    gates['separate_uncertainty_sources']=set(results['uncertainty'])=={'full_twin',results['primary_model']} and all(v['ensemble_members']==3 and 'model_sd72' in v and 'data_quality' in v for v in results['uncertainty'].values())
    gates['challenge_frozen_observable']=not results['challenge']['outcome_used_to_define_subset'] and results['challenge']['definition']==seal['challenge_definition']
    reasons=json.loads((out/'why_now_manifest.json').read_text());gates['structured_reasons_every_full_twin_prediction']=reasons['rows']==76500 and reasons['deterministic_structured_evidence'] and not reasons['LLM_generated'] and reasons['sha256']==file_hash(out/'why_now.jsonl.gz')
    gates['previous_public_and_phase5_preserved']=file_hash('data/synthetic/phase5_complete.json')==registration['phase5_completion_sha256'] and all(file_hash(p)==h for p,h in registration['previous_public_sha256'].items())
    prior=json.loads(Path('data/synthetic_grounding/prior_preservation.json').read_text());gates['prior_private_metadata_preserved']=all(Path(p).stat().st_size==v['bytes'] and Path(p).stat().st_mtime_ns==v['mtime_ns'] for p,v in prior['private_size_mtime'].items())
    gates['private_permissions']=stat.S_IMODE(out.stat().st_mode)==0o700 and all(stat.S_IMODE(p.stat().st_mode)==(0o700 if p.is_dir() else 0o600) for p in out.rglob('*'))
    record={'gates':gates,'all_passed':bool(all(gates.values())),'test_clinical_decodes_during_this_audit':0,'scope':'Saved frozen outputs and file metadata/checksums only; no new fitting or evaluation','test_transaction_event_id':completion['event_id'],**EVIDENCE};private_json(out/'final_audit.json',record)
    print({'phase6_final_audit_passed':record['all_passed'],'failed':[name for name,ok in gates.items() if not ok]},flush=True)
    if not record['all_passed']:raise ValueError('Phase6 final completeness audit failed')
    return record

if __name__=='__main__':run()
