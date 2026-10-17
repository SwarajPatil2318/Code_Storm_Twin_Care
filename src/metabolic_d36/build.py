"""Build reviewed aggregate-only product metadata from completed evidence."""
import json
from pathlib import Path
from src.metabolic_d2.protocol import digest, write_json

OUTPUT = Path('artifacts/metabolic_d36/evidence.json')
CLAIM = 'The Twin combines clinical context and live sensor state while keeping the validated glucose forecaster unchanged.'


def build():
    demo = json.loads(Path('data/metabolic_d3/demo.json').read_text())
    manifest = json.loads(Path('artifacts/metabolic_d35/fusion_manifest.json').read_text())
    research = json.loads(Path('data/metabolic_d35/results.json').read_text())
    if manifest['decision'] != 'NO MODEL REPLACEMENT' or manifest['fusion_supported']:
        raise ValueError('D3.6 requires the reviewed no-replacement interpretation')
    result = {
        'schema_version': 'metabolic-product-evidence/1',
        'model_version': 'METABOLIC_D2_REFERENCE', 'production_model_changed': False,
        'claim': CLAIM,
        'd2': {**demo['reference']['historical_validation'], 'participants':100,
               'holdout_windows':21820, 'holdout_status':'Spent after one frozen evaluation; not reused for D3.5',
               'persistence_MAE_mg_dl':[12.212,20.626,26.951,31.475],
               'horizons_minutes':[30,60,90,120]},
        'fusion': {
            'title':'Static + dynamic fusion experiment', 'participants':research['participants'],
            'folds':research['folds'], 'windows':research['windows'], 'scope':'Development-only participant-grouped cross-validation; not D2 holdout validation',
            'static_features_used':manifest['static_features_used'], 'excluded_features':manifest['excluded_features'],
            'dynamic_MAE_mg_dl':research['metrics']['dynamic_only']['subject_macro']['MAE'],
            'fused_MAE_mg_dl':research['metrics']['fused']['subject_macro']['MAE'],
            'delta_MAE_mg_dl':research['fused_minus_dynamic']['delta_mae'],
            'bootstrap_ci95':research['fused_minus_dynamic']['ci95'],
            'bootstrap_replicates':research['bootstrap_replicates'], 'bootstrap_unit':'participant',
            'shuffled_static_120m_MAE':research['metrics']['shuffled_static']['subject_macro']['MAE'][3],
            'fused_minus_shuffled_120m_ci95':research['fused_minus_shuffled']['ci95'][3],
            'static_permutation_120m_MAE_increase':research['modality_importance']['static']['mae_degradation'][3],
            'dynamic_permutation_120m_MAE_increase':research['modality_importance']['dynamic']['mae_degradation'][3],
            'fusion_supported':False, 'decision':manifest['decision'],
            'conclusion':'No statistically supported incremental forecasting gain from static clinical context.',
            'negative_control_explanation':'Correct patient linkage performed better than shuffled clinical profiles, but improvement beyond dynamic-only forecasting remained inconclusive.',
            'permutation_scope':'Predictive reliance under validation permutation; not a causal effect.'},
        'provenance_categories': {
            'REAL OPEN-DATA MODEL EVIDENCE':'ShanghaiT2DM development and historical held-out model validation; distinct from demo inputs.',
            'SYNTHETIC DEMO PATIENT':'Demo identity, displayed EHR and wearable/meal context; not records belonging to a real Shanghai participant.',
            'MODEL PREDICTED':'Frozen model glucose forecasts and spike probability; derived peak/status/rendering are estimates.',
            'PERSONAL EXPECTED':'Causal Shadow Twin response and personal-baseline comparisons; exploratory expectation.',
            'OBSERVED / REPLAYED':'Received CGM observations and timestamps; current public replay inputs are synthetic.'},
        'limitations':['Research PoC; not clinically validated.', 'No prospective trial.',
                       'Short-horizon forecast primarily CGM-driven, with eligible historical CGM features.',
                       'EHR incremental forecast gain inconclusive.', 'Smartwatch context simulated for demo.',
                       'Displayed EHR is synthetic demo context.', 'CGMacros external validation remains unspent.',
                       'Historical clinical onboarding availability is an assumption; undated labs and anthropometry were excluded.',
                       'D3.5 participant bootstrap describes this fixed development CV run, not independent clinical validation.'],
        'lineage': {'d35_manifest_sha256':digest('artifacts/metabolic_d35/fusion_manifest.json'),
                    'd35_predictions_sha256':research['prediction_sha256'],
                    'd2_reference_sha256':digest('data/metabolic_d3/reference_registry.json'),
                    'source_reports':['docs/GLUCOSE_FORECAST_RESULTS.md','docs/SPIKE_TARGET_ANALYSIS.md',
                                      'docs/METABOLIC_D35_STATIC_FEATURE_AUDIT.md','docs/METABOLIC_D35_FUSION_RESULTS.md']}
    }
    write_json(OUTPUT,result)
    print(json.dumps({'cached_evidence_built':True,'contains_participant_records':False,'production_model_changed':False}))
    return result


if __name__=='__main__':build()
