"""Reviewed aggregate-only documentation and integration evidence."""
import json
from pathlib import Path
from src.metabolic_d2.protocol import write_json, digest
from .data import FIELD_MAP, SAFE
from .protocol import ROOT, MANIFEST, validate_manifest

PUBLICATION = 'https://acris.aalto.fi/ws/portalfiles/portal/100206019/Chinese_diabetes_datasets_for_data_driven_machine_learning.pdf'


def disposition(field):
    if field in FIELD_MAP:
        reason = 'Historical medical-record context; explicit before-CGM medication statement; coarse source-text category, not dose/adherence.' if 'Agents' in field else 'Stable demographic or explicitly historical attribute; earliest-recording onboarding assumption, actual receipt unknown.'
        return 'SAFE', 'Historical/onboarding; actual availability unknown', reason
    if field == 'Patient Number': return 'EXCLUDE', 'Identity metadata', 'Used only to link source recording to namespaced participant, never a predictor.'
    if field == 'Hypoglycemia (yes/no)': return 'EXCLUDE', 'Recording outcome possible', 'Could summarize future CGM events.'
    if field == 'Type of Diabetes': return 'EXCLUDE', 'Pre-existing cohort membership', 'Constant T2D selection attribute; no predictive variation.'
    if field in ['Height (m)', 'Weight (kg)', 'BMI (kg/m2)']:
        return 'AMBIGUOUS', 'Undated physical examination', 'Pre-forecast measurement timing not established; EXCLUDED from all models.'
    if 'Complications' in field or field == 'Comorbidities':
        return 'AMBIGUOUS', 'Medical-record summary without diagnosis dates', 'Cannot establish pre-existing status for every forecast; EXCLUDED from all models.'
    return 'EXCLUDE', 'Labs may precede or follow CGM', 'Publication permits labs within six months before OR after CGM; no per-lab date proves prior availability.'


def audit_report(audit):
    lines = ['# D3.5 static-feature audit', '',
             f'Development only: {audit["development_participants"]} participants and {audit["clinical_recordings"]} recording-linked clinical rows. No holdout clinical feature rows were decoded. The audit inspects the real XLSX header and authorized rows, not assumed EHR fields.', '',
             f'[Source publication]({PUBLICATION}) describes demographic/history collection, undated physical examinations, before-CGM medication summaries and labs that may occur before or after CGM. The interpretation below separates those facts from the historical-onboarding assumption.', '',
             '| Field | Source | Same-person linkage | Timing status | Missing rows | Decision | Reason |',
             '|---|---|---|---|---:|---|---|']
    for field in audit['header_fields']:
        status, timing, reason = disposition(field)
        missing = audit['recording_missingness'].get(field, 0)
        lines.append(f'| {field} | Shanghai T2DM summary | Exact recording key → namespaced participant | {timing} | {missing}/{audit["clinical_recordings"]} | {status} | {reason} |')
    lines += ['', 'AMBIGUOUS fields are excluded, not imputed or modeled. No HbA1c, fasting glucose/insulin, C-peptide, lipids, renal or other labs enter any model. No Synthea record is paired with Shanghai.', '',
              'Safe model features: '+', '.join(SAFE)+'.', '',
              '| Safe feature | Missing participant profiles |', '|---|---:|']
    for key in SAFE: lines.append(f'| {key} | {audit["profile_missingness"][key]}/80 |')
    lines += ['', 'Every model profile is selected from the earliest CGM recording belonging to that participant. Later visits never supply retrospective updates. Only one profile is used per participant. Original recording keys, raw values, source-unit headers, missingness and provenance remain private in `data/metabolic_d35/static_provenance.json`; model-ready profiles are in `participant_static.parquet`.', '',
              'Numeric fields retain years/pack-years. Invalid numeric values are quarantined to missing, with original values retained. Sex uses the source female=1/male=2 code. Alcohol retains the source category; no physiological inference is made. Medication categories only distinguish explicit insulin terms from other recorded text. Absent medication text is missing, not asserted to mean no treatment. Missingness indicators are retained, and categorical vocabularies/numeric medians are training-fold-only.', '',
              'Historical availability is an explicit onboarding assumption for stable demographics and history, not a measured receipt. Actual observation and receipt timestamps remain null. Before-CGM medication context has explicit publication support. BMI/weight, diagnosis/complication dates and all laboratory timing do not pass the gate. No safe dated comorbidity flag was found.', '',
              'The shared XLSX is physically scanned for identities; clinical cells and shared-string values are resolved only for authorized development rows. No whole-cohort clinical dataframe is created. The spent holdout identities are used solely as a denylist, never as features or for selection.']
    Path('docs/METABOLIC_D35_STATIC_FEATURE_AUDIT.md').write_text('\n'.join(lines)+'\n')


def report(results, audit):
    audit_report(audit)
    supported = results['fusion_supported']; outcome = results['outcome']
    decision = 'GO FOR MODEL-LEVEL FUSION' if supported else 'NO MODEL REPLACEMENT'
    if supported:
        action = 'Research fusion promotion only after separate preservation and new independent validation; do not change current D2/D3 or UI.'
        claim = 'Development-only same-patient testing supports fusing historical clinical context with CGM for personalized forecasting; the production forecaster remains unchanged pending independent validation.'
    else:
        action = 'Retain METABOLIC_D2_REFERENCE and existing D3/UI; clinical history remains contextual, with no model replacement.'
        claim = 'The Twin combines clinical context and live sensor state while keeping the validated glucose forecaster unchanged.'
    primary = results['fused_minus_dynamic']
    excluded = [f for f in audit['header_fields'] if f not in FIELD_MAP]
    manifest = {'decision': decision, 'static_features_used': SAFE, 'excluded_features': excluded,
                'dynamic_model_120m_mae': results['metrics']['dynamic_only']['subject_macro']['MAE'][3],
                'fused_model_120m_mae': results['metrics']['fused']['subject_macro']['MAE'][3],
                'delta_mae_120m': primary['delta_mae'][3], 'bootstrap_ci_120m': primary['ci95'][3],
                'fusion_supported': supported, 'recommended_production_action': action, 'safe_ui_claim': claim,
                'claim_scope': 'Offline development-only fusion evidence; live UI EHR remains synthetic demo context. Not clinical validation.',
                'participants': 80, 'folds': 5, 'horizons_minutes': [30,60,90,120], 'outcome': outcome,
                'runtime_seconds': results['runtime_seconds'], 'bootstrap_unit': 'participant',
                'bootstrap_replicates': results['bootstrap_replicates'], 'production_modified': False,
                'prediction_sha256': results['prediction_sha256'], 'protocol_sha256': digest('configs/metabolic_d35.json')}
    validate_manifest(manifest); write_json(MANIFEST, manifest)
    lines = ['# D3.5 same-patient static/CGM fusion results', '', decision+'. '+action, '',
             '## Scope and linkage', '',
             f'ShanghaiT2DM only: {results["participants"]} existing development participants, {results["windows"]} unchanged D2 forecast windows. No spent-holdout trajectories, predictions or metrics were loaded for this experiment. The prior 16 calibration participants are used only as part of the allowed development pool; production calibration is unchanged.', '',
             'Static fields: '+', '.join(SAFE)+'. No safe dated comorbidity flags were found. Excluded: all labs, undated BMI/weight/height, undated diagnoses/complications, recording hypoglycemia outcomes, identifiers, constant cohort type and synthetic EHR. See [field audit](METABOLIC_D35_STATIC_FEATURE_AUDIT.md).', '',
             'Exact recording keys establish same-person linkage; one earliest-recording historical profile is retained per participant. True receipt times are absent and historical onboarding remains an explicit assumption. Source windows remain native observed-only CGM, nine observations over 120 minutes, all four unchanged horizons and ±7.5-minute tolerance.', '',
             '## Comparison design', '',
             'Five participant-grouped folds, 64 train/16 validation each, seed 3502026. Identical participants/windows/targets for all variants. Fixed D2 XGBoost histogram multi-output trees, 180 rounds/depth 4, participant-balanced pseudo-Huber loss. Every preprocessor is fitted within the training fold. Dynamic preprocessing reuses the unchanged D2 class; static numeric medians use unique training profiles, categories use training-only one-hot encoding with missingness flags.', '',
             'The dynamic baseline is an experimental fold clone, not the frozen production engine: it uses established CGM features plus eligible historical-prefix features in a common research model. Dynamic variants predict residuals from current glucose; static-only predicts residuals from a training-fold participant-macro target mean and never receives current CGM. Forecast targets are identical. This offset difference is explicit and necessary to make static-only truly static.', '',
             '## Forecast metrics', '',
             'All values are mg/dL, equal-participant macro MAE/RMSE. RMSE is the mean of participant RMSEs, not the square root of pooled participant errors. Pooled values remain separately reported in private results.', '',
             '| Variant | +30 MAE / RMSE | +60 MAE / RMSE | +90 MAE / RMSE | +120 MAE / RMSE |',
             '|---|---:|---:|---:|---:|']
    for variant in ['static_only','dynamic_only','age_sex','fused','shuffled_static']:
        m = results['metrics'][variant]['subject_macro']
        lines.append('| '+variant+' | '+' | '.join(f'{a:.3f} / {r:.3f}' for a,r in zip(m['MAE'],m['RMSE']))+' |')
    lines += ['', '## Paired participant uncertainty', '',
              'ΔMAE = FUSED − DYNAMIC; negative favors fusion. Paired percentile 95% CIs resample 80 participants 2,000 times with their complete within-person predictions, not independent windows. Out-of-fold estimates share training folds; these CIs describe participant heterogeneity conditional on this fixed CV run, not retraining uncertainty. They are not independent external validation or multiplicity-adjusted tests.', '',
              '| Horizon | ΔMAE | 95% CI | Participants improved |', '|---|---:|---|---:|']
    for h,d,ci,n in zip(results['horizons'],primary['delta_mae'],primary['ci95'],primary['participants_improved']):
        lines.append(f'| +{h} min | {d:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {n}/80 |')
    lines += ['', '## Negative control and demographic ablation', '',
              'Complete static profiles were deranged between training participants once per fold, never window-by-window. Dynamic streams retain their owners; validation static profiles remain correctly linked. One fixed seeded permutation per fold limits negative-control precision. The control and fusion share model capacity, training rows and hyperparameters.', '',
              '| Paired comparison | +30 ΔMAE [CI] | +60 ΔMAE [CI] | +90 ΔMAE [CI] | +120 ΔMAE [CI] |', '|---|---|---|---|---|']
    for key in ['fused_minus_shuffled','shuffled_minus_dynamic','age_sex_minus_dynamic','fused_minus_age_sex']:
        comparison = results[key]
        lines.append('| '+key+' | '+' | '.join(f'{d:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]' for d,ci in zip(comparison['delta_mae'],comparison['ci95']))+' |')
    lines += ['', '## Static and dynamic utilization', '',
              'Validation-fold permutation diagnostic: static profiles are deranged among validation participants; complete dynamic rows are permuted within each participant, including current-glucose prediction offsets. Each is one fixed perturbation per fold. MAE increases measure predictive reliance; negative values indicate that perturbation improved error. Values are not causal effects, calibrated percentages or XGBoost gain.', '',
              '| Perturbed modality | +30 MAE increase | +60 | +90 | +120 |', '|---|---:|---:|---:|---:|']
    for group, m in results['modality_importance'].items(): lines.append('| '+group+' | '+' | '.join(f'{d:.3f}' for d in m['mae_degradation'])+' |')
    lines += ['', '## Optional spike ranking', '',
              'No spike model/calibrator was fitted. The unchanged D2 incident target and forecast-derived trajectory score were reused on current-glucose ≤180 mg/dL windows. These are uncalibrated trajectory ranking results, not performance of the released calibrated risk bundle. Production threshold remains unchanged.', '',
              '| Variant | AUPRC | AUROC | Eligible windows | Positive windows |', '|---|---:|---:|---:|---:|']
    for variant,m in results['spike'].items(): lines.append(f'| {variant} | {m["AUPRC"]:.4f} | {m["AUROC"]:.4f} | {m["eligible_windows"]} | {m["positive_windows"]} |')
    interpretation = 'Fusion consistently improves all horizons with supported participant intervals and beats the shuffled control at 120 minutes.' if supported else 'Static context was integrated and evaluated, but a model replacement is not supported by the prespecified horizon and shuffled-control gates.'
    if outcome == 'hurts': interpretation += ' At 120 minutes fusion is reliably worse within this development experiment.'
    lines += ['', '## Conclusion and limitations', '', interpretation, '',
              f'Outcome: **{outcome}**. Fusion supported: **{supported}**. '+action, '',
              'Limitations: one small previously-used development cohort; no external/new held-out validation; few coarse safe historical attributes; unavailable dated labs/BMI/comorbidities; unknown observation-to-availability delays; single negative-control permutation and single diagnostic permutation per modality/fold. Static medication missingness is not treatment absence. Causal feature construction does not establish causal clinical effects. No clinical utility claim is warranted.', '',
              'Safe wording recommendation: “'+claim+'” This does not authorize a UI change. Any live demo must retain its synthetic-EHR/source qualifications.', '',
              f'Experiment wall runtime: {results["runtime_seconds"]:.2f} seconds, excluding subsequent preservation/API/tests. No optimization continues after this fixed comparison.', '',
              '## Artifacts and preservation', '',
              '`artifacts/metabolic_d35/fusion_manifest.json` contains reviewed aggregates and the production decision. Private `data/metabolic_d35/` contains participant profiles/provenance, per-participant MAE differences, OOF predictions, fold-specific experimental models/preprocessors, fold assignments/donors and preservation evidence. These are not added to Git or logs.', '',
              'Production D2 code/preprocessing/model/calibration/threshold, D3 Shadow/WhyNow/state/stream/API/demo and all frontend files are protected by before/after hashes and artifact metadata. The cases HTTP response is also compared exactly. The completion audit records these checks separately.']
    Path('docs/METABOLIC_D35_FUSION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    architecture = '''# D3.5 integration architecture (research only)

```mermaid
flowchart TD
  A[Shanghai historical clinical summary] --> B[Same-participant allowlist and timing audit]
  B --> C[Earliest-recording static profile]
  C --> D[Training-fold numeric and categorical encoder]
  E[Shanghai native dynamic CGM] --> F[Unchanged D2 causal history and prefix features]
  D --> G[Offline fusion research model]
  F --> G
  G --> H[OOF +30 / +60 / +90 / +120 forecasts]
  H --> I[Participant metrics, bootstrap and shuffled control]
  I --> J[Integration manifest and production decision]
  J -. Separate future validation and preservation gate .-> K[Potential future Personal TwinState integration]
  L[Frozen current METABOLIC_D2_REFERENCE] --> M[Unchanged current D3 TwinState]
  M --> N[Expected personal response / forecast / spike risk / WhyNow]
```

The solid fusion path exists only in the offline experiment. The dashed future path is not implemented. Current D3 Shadow, WhyNow, spike calibration, streaming, APIs, demo cases and UI continue to use the frozen engine. No D3.5 import/runtime dependency is introduced into production. Forecast-derived spike ranking in the experiment is not a new calibrated spike head.

Static and dynamic branches are joined by namespaced same-participant identities only. Numeric/categorical encoders are fold-specific. No synthetic EHR or unrelated cohort is attached to a real CGM stream. Future integration requires a validated versioned model, explicit availability contract, preservation gate and independent evaluation; it is not authorized by this evidence package alone.
'''
    Path('docs/METABOLIC_D35_FUSION_ARCHITECTURE.md').write_text(architecture)
    write_json(ROOT/'architecture.json', {'version':'research-offline-d35', 'static_branch':SAFE,
        'dynamic_branch':audit['dynamic_features'], 'join':'same participant, earliest recording profile',
        'forecast_horizons':[30,60,90,120], 'current_production':'METABOLIC_D2_REFERENCE unchanged',
        'future_outputs':['expected personal response','forecast','spike risk','WhyNow'],
        'runtime_integration':False,'requires_independent_validation':True})
