"""Aggregate-only Phase 5 documentation and final evidence inventory.

Run only after the pilot, large-cohort audit, variants and figures finish. Does
not rewrite generation artifacts or the immutable pilot authorization.
"""
import json,re,sys
from importlib.metadata import version
from pathlib import Path
from src.synthetic.config import load,SIGNALS,UNITS
from src.synthetic.generate import code_hash
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def read(path):return json.loads(Path(path).read_text())
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def write(name,text):
    Path('docs').mkdir(exist_ok=True);Path('docs',name+'.md').write_text(text.strip()+'\n')

def report():
    pilot=Path('data/synthetic/pilot');large=Path('data/synthetic/cohort10000');p=read(pilot/'validation.json');m=read(large/'manifest.json');a=read(large/'large_validation.json');v=read('data/synthetic/variants/variant_summary.json');g=read('data/synthetic_grounding/grounding.json');d=read(pilot/'diversity_metrics.json');r=read(pilot/'realism_metrics.json');diag=read(pilot/'difficulty_metrics.json');c=load()
    if not (p['all_gates_passed'] and a['all_gates_passed'] and v['event_histories_identical']):raise ValueError('Do not publish a success decision before all cohort gates pass')
    if m['simulator_code_sha256']!=code_hash():raise ValueError('Core generator changed')
    figures=read(pilot/'figures/manifest.json')['figures']
    if len(figures)!=8 or not all((pilot/'figures'/f).exists() for f in figures):raise ValueError('Eight figures required')
    tests=(pilot/'regression_tests.log').read_text();match=re.search(r'Ran (\d+) tests',tests)
    if not match or not re.search(r'\nOK\s*$',tests):raise ValueError('Verified regression results required')
    test_count=int(match.group(1));run_p=read(pilot/'runtime.json');run_l=read(large/'runtime.json');max_q=max(x['maximum_quantile_distance_real_sd'] for x in r['comparison'].values());min_q=min(x['maximum_quantile_distance_real_sd'] for x in r['comparison'].values());min_w=min(x['wasserstein_real_sd'] for x in r['comparison'].values());max_w=max(x['wasserstein_real_sd'] for x in r['comparison'].values());source_hashes=table(['Read-only grounding input','SHA-256'],[(path,'`'+h+'`') for path,h in sorted(g['sources_sha256'].items())])
    evidence='Evidence: **SYNTHETIC POC VALIDATED** for the stated simulator checks. The cohort is **SYNTHETIC POC**; these checks are not clinical validation.\n'
    design='''# Synthetic cohort design

'''+evidence+'''
The simulator produces paired historical EHR and 60 days of home/wearable physiology for each new virtual patient. It combines heterogeneous personal baselines, stochastic semi-Markov deterioration, independent decoys, sparse devices, measurement noise and receipt delays. Events arise from private latent burden and vulnerability. The final Digital Twin predictor has not been trained.

## Population and longitudinal model

Seven overlapping HF-like profiles affect age, sex, BMI, EF, severity, comorbidities, physiological baseline and activity. They are scenario priors, not validated clinical phenotype classifiers. Profile probabilities are '''+str(c['phenotype_proportions'])+''' in the explicit `phenotype_order` in the configuration. Sampling is probabilistic, so actual proportions vary. Every patient gets an independent continuous baseline draw and personal variance multiplier.

The six measured home channels use a Gaussian copula over aggregate PerHeart patient-median quantiles. Pairwise covariance is shrunk and projected to positive definiteness. This generates new values; it does not copy a real patient's record. Healthy baseline drift, daily/hourly autoregression, circadian structure, shared factors and individual response heterogeneity produce correlated histories. No baseline is estimated from a patient's future trajectory, and deterioration never updates the private healthy baseline. There is no weekday/weekend effect because the supplied sources do not support its calibration.

HRV (RMSSD proxy), sleep duration, activity and home respiration rely on explicit assumptions. `heart_rate_sd` from CHIRON is never called HRV; lying duration is never called sleep. `steps` is an hourly count, with a configured daily total expectation; it is not a continuous accelerometer trace.

## Paired EHR and device acquisition

Historical context includes age, female indicator, BMI, EF, HF severity, hypertension, diabetes, CKD, beta-blocker, ACE/ARB, diuretic, MRA, SGLT2 context, creatinine, sodium, hemoglobin and BNP. Labs can be missing. HF context, diagnoses, medication requests, labs, demographics and encounters use the same new virtual patient ID as every wearable observation. Context is available before monitoring begins. Severity, age, CKD, diabetes, hypertension, lower baseline BP and medication context affect probabilistic vulnerability, with unobserved heterogeneity. These are scenario associations, not estimated treatment effects.

The audited Synthea sample supplies resource shapes and available codes; this phase does not run a fresh Synthea Java engine. Unsupported codes use an explicit local synthetic coding system. FHIR resources include Patient, Condition, MedicationRequest, Observation and Encounter, with absolute fullUrls and consistent patient references. Quantity units use UCUM. Checks establish structural linkage and serialization; they do not certify full FHIR R4 profiles, terminology or clinical correctness. See the [Synthea project](https://synthetichealth.github.io/synthea/) and [FHIR R4 Observation specification](https://hl7.org/fhir/R4/observation.html).

'''+table(['Signal','Unit','Scheduled acquisition'],[(s,u,c['sensors']['opportunities'][s]) for s,u in zip(SIGNALS,UNITS)])+'''

HR/activity have hourly summary opportunities; HRV is a nightly proxy, BP/MAP/SpO2/RR use morning/evening home opportunities, and weight/temperature/sleep are morning measurements. MAP derives from the same BP measurement as `(SBP + 2*DBP)/3`; it is not an independent continuous consumer wearable channel.

## Labels and model inputs

Every 60-day history has 51 fixed daily landmarks, from hour 168 through hour 1368. Features use `(landmark - 168h, landmark]`, subject to usable quality and receipt time at or before the landmark. Exclusive future intervals are `0 < t <= 24`, `24 < t <= 48`, `48 < t <= 72`; `event_within_72h` is their sum. No-event includes a future event beyond 72 hours or no additional generated event. `time_to_event_hours` belongs only to the outcome table; it can exceed 72 hours.

This version generates at most one adverse event per patient. Fixed landmarks continue after that event, where labels denote no further generated event. That avoids event-dependent landmark-count leakage but does not establish recurrent-risk validity. Phase 6 must explicitly retain this task interpretation and report pre-event-only results as an additional analysis, without changing the primary eligibility silently.

The 83-feature allowlist contains 17 EHR values and six statistics for each of 11 signals: mean, SD, last, slope/day, recent-48h versus earlier-history change, and coverage. IDs and landmark times are linkage metadata, never prediction features. Private state, pathway, hazard, event time, true physiology and healthy baseline are kept separate.

## Outputs, scaling and privacy

Each cohort contains `patients`, `ehr`, `observations`, `outcomes`, `ground_truth`, `quality`, `model_features`, `splits.json`, `schemas.json`, a manifest, runtime statistics, selected FHIR demos and canonical demos. Compressed NPZ stores typed patient/time/signal arrays in batches of 64; gzip JSONL streams nested records. Parquet dependencies are absent, so no additional dataframe stack is required. Raw value, usable value, quality, missingness, scheduled opportunity, source, units, receipt hour and missing reason remain explicit. The lazy canonical readers map observations and historical context to existing shared dataclasses without opening labels or hidden truth.

Histories are generated before a patient-level 70/15/15 hash split, using identical rules in every split. The 1,000-person pilot uses simulator indices 0–999; the fresh 10,000-person cohort uses 1000–10999, so Phase 6 patients do not overlap pilot diagnostic patients. No real patients are paired across datasets. All records, arrays, weights, plots, codes selected from raw inputs and hidden ground truth remain private under Git-ignored `data/`, with directories 0700 and files 0600. Public documentation contains reviewed aggregates only.

Individual exports retain `NOT VALIDATED`: a generator gate cannot establish clinical truth for an individual observation. Acceptance certificates state `SYNTHETIC POC VALIDATED` only for the enumerated simulation checks, with `not_clinical_validation=true`. Real descriptive summaries are `REAL DATA EXPLORATORY`; Synthea context fidelity remains `NOT VALIDATED`.

## Reproduction and workflow

Configuration `configs/synthetic_v1.yaml` is JSON-subset YAML, parsed with the standard library. It stores population priors, sensor schedules/noise, bounds, latent transition controls, effect strengths and predeclared engineering acceptance gates. Version: '''+c['version']+'''; design revision: '''+c['design_revision']+'''. Component-specific random streams isolate baseline, state, physiology, decoys and devices.

```bash
.venv/bin/python -m src.synthetic.grounding
# Build grounding only in a new setup; do not rebuild during a cohort run.
.venv/bin/python -m src.synthetic.generate --patients 1000 --days 60 --seed 2026
.venv/bin/python -m src.synthetic.validation --root data/synthetic/pilot
.venv/bin/python -m src.synthetic.verify --root data/synthetic/pilot
.venv/bin/python -m src.synthetic.generate --patients 10000 --days 60 --seed 2026
.venv/bin/python -m src.synthetic.large_audit
.venv/bin/python -m src.synthetic.variants
.venv/bin/python -m src.synthetic.plots
.venv/bin/python -m src.synthetic.reference_dynamics
.venv/bin/python -m src.synthetic.report
```

Outputs are immutable: use a fresh output directory for another run. Existing published cohorts must not be overwritten. Scaling above 1,000 is denied unless the full pilot certificate, seed, duration, config, generation code, grounding and original pilot artifact hashes match. Arbitrary positive sizes work; a new seed/duration/design above 1,000 requires its own matching pilot validation. Patient offsets guarantee pilot/large isolation, not automatic disjointness between two independently requested large runs with the same seed.

Generation manifests hash all original data artifacts. Reproduction compares 24 measured-grounding patients twice byte for byte, including compressed arrays, records, FHIR and split files, and verifies agreement with the pilot prefix. Runtime metadata is intentionally excluded from byte equivalence. Private final evidence inventory records current source/config/input/artifact hashes; it complements the frozen core-generation hash.
'''
    write('SYNTHETIC_COHORT_DESIGN',design)
    grounding='''# Physiological grounding

Evidence: **REAL DATA EXPLORATORY** for measured descriptive summaries. Numerical latent mechanics, home HRV/activity/sleep and EHR epidemiology are **NOT VALIDATED** clinical assumptions. Cohort checks are **SYNTHETIC POC VALIDATED** within the stated scope.

All sources were already locally audited. The generator uses aggregate distributions and relationships, never links real patients across cohorts, never opens the spent eICU clinical test partition, and does not reinterpret earlier real-data results. No genuine long-lead paired HF event cohort was available to calibrate the synthetic endpoint.

'''+table(['Source','Measured support','Use and limitation'],[
    ['PerHeart','27 home-monitoring patients; descriptor-supported HR/BP/SpO2/weight/temperature; daily values, patient medians, variance, quantiles, paired correlations, frequency and coverage','Primary home marginal/copula anchor; no supplied wrist/activity/sleep records; observed medical span is not verified enrollment or nonwear'],
    ['CHIRON','25 subjects; daily HR/BP/SpO2/weight and related summaries, day changes, lag correlation','Separate longitudinal reference; provisional units are never pooled with PerHeart; personalized _pers fields quarantined; HR_SD is not HRV, skin temperature is not core temperature'],
    ['eICU development','Already-isolated Phase 4.5B development-only 73-bin multivariate windows','Acute BP/MAP/HR/SpO2/RR distributions, correlation and descriptive dynamics; no source outcomes or spent-test labels opened in this phase'],
    ['VitalDB','Selected 12-bin surgical windows','Short cardiovascular dynamics and multivariate correlation reference; setting/cadence cannot validate home HF physiology or 72h events'],
    ['Synthea','Audited synthetic Patient/Condition/MedicationRequest/Observation/Encounter shapes and codes','FHIR/EHR foundation only; no claim that source sample diagnoses match HF profile prevalence']])+'''

Grounding uses usable readings and retains original QC/provenance in source tables. PerHeart daily aggregation uses UTC calendar days from normalized timestamps; source local dates are preserved upstream but are not used in this statistical view. This day-boundary choice is an acquisition limitation. HR comparisons sample synthetic morning/evening values to match sparse home acquisition instead of comparing all hourly wrist summaries with cuff-associated HR.

## Parameter support map

'''+table(['Configuration family / parameter','Grounding or assumption'],[
 ['baseline marginal_shape, population quantiles/correlation/location_offset','PerHeart aggregate patient medians; continuous interpolated quantile copula, correlation shrinkage 0.4 is an engineering regularizer; common temperature offset −0.05 °C corrects observed simulated daily-tail bias'],
 ['baseline daily_variability_sd, factor structure','Adjusted to descriptive PerHeart within-person/day-change variability; BP coupling cross-checked against CHIRON, eICU and VitalDB; factor loadings are assumptions'],
 ['baseline hourly_ar/daily_ar/circadian_amplitudes/drift','Positive temporal structure checked against home daily summaries; exact autoregression, phase and healthy drift are scenario assumptions'],
 ['baseline HRV/sleep/steps/home RR priors','Unsupported home modalities: configured assumptions; not substitutions for CHIRON HR_SD or lying time'],
 ['phenotypes/proportions/offsets; ehr demographics, EF, lab and medication priors','Scenario priors with overlap and bounds; Synthea shapes/codes support serialization, not their distributions'],
 ['vulnerability and medication_protection','Assumed stochastic context association; no estimated medication efficacy or causal intervention effect'],
 ['states / paths / event hazard / duration / recovery','Interpretable scenario assumptions; 512 separate virtual patients calibrated to a scenario prevalence target, not real epidemiology'],
 ['physiology deterioration gains','Qualitative symptom/pathway motivation; all numerical causal strengths and delays are assumptions'],
 ['decoys strengths/frequency/duration','Independent negative-control scenarios, not adjudicated real negative events'],
 ['sensors sparse home opportunities','Anchored to PerHeart frequency/coverage; exact morning/evening schedules, adherence distribution and missing mechanisms are assumptions'],
 ['sensors wrist/HRV/sleep/noise/artifact/delay','Device stress assumptions; source files do not establish true nonwear/charging or device error mechanisms'],
 ['signal_bounds / quality_bounds','Simulation guards and QC bounds; not diagnostic or treatment thresholds'],
 ['variants','Controlled synthetic acquisition/baseline stress, not real-world model robustness evidence'],
 ['landmarks / history / horizon / splits / validation','User task and frozen engineering acceptance contracts, not clinical validity thresholds']])+'''

The [American Heart Association HF symptom description](https://www.heart.org/en/health-topics/heart-failure/warning-signs-of-heart-failure) supports qualitative links among fluid retention/weight, dyspnea, fatigue and heart-rate change. It does not establish this simulator's coefficients, pathways, durations, event definition or prevalence. Directions involving HRV, sleep and home respiration remain assumptions even when a generated relationship passes a sign check.

## Numerical reference relationships

Development eICU MAP–DBP correlation is '''+f"{g['eicu_development']['correlations']['map__dbp']:.3f}"+''', MAP–HR '''+f"{g['eicu_development']['correlations']['map__hr']:.3f}"+''', and SpO2–RR '''+f"{g['eicu_development']['correlations']['spo2__rr']:.3f}"+'''. VitalDB SBP–DBP is '''+f"{g['vitaldb']['correlations']['sbp__dbp']:.3f}"+''', MAP–DBP '''+f"{g['vitaldb']['correlations']['map__dbp']:.3f}"+''', and SpO2–RR '''+f"{g['vitaldb']['correlations']['spo2__rr']:.3f}"+'''. These demonstrate dependent but noisy physiology; they do not justify forcing identical correlations across ICU, surgery and home monitoring. The home simulator derives cuff MAP algebraically and uses modest shared factors rather than independently sampling BP channels.

`data/synthetic_grounding/reference_dynamics.json` adds within-window first/last-bin changes, adjacent-bin distributions and lag correlations using only the same unlabelled allowlisted NPZ files. Its descriptive MAP-decline subset is a physiological comparison, not an event label. Overlapping windows are dependent. This supplementary reference is separate from frozen generation inputs and was not used to fit the HF hazard.

Complete measured mean/variance/quantile, within/between-person, slope, day-change, lag and acquisition summaries are in private `grounding.json` and `realism_metrics.json`. Grounding cache files contain aggregate distribution samples without patient identities; they remain private. All numeric configuration leaves are covered by the family support map above, with exact values retained in the versioned configuration. `physiology.weight_hourly_sd` is a reserved setting in v1; actual weight variability comes from the shared home factors and measurement-noise parameter. Cohort size/duration/seed are explicitly selected through the CLI; the versioned default scenario records 1,000/60/2026.

## Provenance

Configuration SHA-256: `'''+p['config_sha256']+'''`.
Core generator SHA-256: `'''+p['simulator_code_sha256']+'''`.
Frozen aggregate grounding SHA-256: `'''+p['grounding_sha256']+'''`.

'''+source_hashes+'''
No input checksums are interpreted as clinical validity. Prior private-phase preservation uses size/mtime metadata, avoiding clinical test-file decoding or hashing.
'''
    write('PHYSIOLOGICAL_GROUNDING',grounding)
    state='''# Deterioration state model

'''+evidence+'''
Each patient has a stochastic sequence of stress episodes after an initial seven-day stable period. A semi-Markov process advances through STABLE → EARLY_STRESS → COMPENSATING → EARLY_DETERIORATION → WORSENING. Episodes may recover without an event; an adverse event is sampled probabilistically during EARLY_DETERIORATION/WORSENING, followed by recovery. Four private factors represent autonomic stress, congestion, pressure burden and illness. Observable signals are noisy, individually scaled manifestations of those factors.

Gradual planned episodes last 96–264 hours; 18% of episodes use a rapid 24–72-hour scenario. Stable periods and episode starts have random timing, with 12% configured never-episode scenarios. Thirty-five percent are scheduled for early recovery, while other event-free episodes also eventually recover. The reported recovered-without-event count includes all episode recoveries without a sampled event, not just scheduled early recoveries. Quiet gaps, stochastic peaks and response multipliers prevent a universal timing/signature.

'''+table(['Path','Probability','Autonomic / congestion / pressure / illness loading'],[(k,str(x['probability']),str(x['loadings'])) for k,x in c['paths'].items()])+'''

Autonomic-first emphasizes increased HR, decreased HRV/sleep and less activity. Congestion-like emphasizes rising weight, reduced SpO2/activity and rising respiration. Pressure/hemodynamic emphasizes falling BP/MAP with compensatory HR. Mixed combines smaller changes; noisy/atypical uses weak loadings that ordinary variation can obscure. Lognormal response heterogeneity and independent sensor noise create overlap.

For private latent level `b`, patient vulnerability `v` and episode duration so far `d`, the hourly probability is:

```text
p = 1 - exp(-h0 * exp(1.2*(v-0.5) + 1.4*b + 0.3*min(d/168, 1)))
h0 = 0.0015709375 per hour
```

The private scalar level is the common episode driver; pathway loadings determine its multivariate manifestations. The hazard never reads observed MAP, HR, device values, future feature windows, decoys or sensor noise. The random event time lies inside the selected hour. A separate 512-person virtual calibration population (seed 2031, indices 20000–20511) set the hazard for a 30% scenario target; observed calibration prevalence was 30.078%. The accepted engineering band is 20–40%, not a clinical HF incidence range.

Baseline context affects vulnerability through severity, age, comorbidities, lower baseline BP, medication context and unobserved random variation. No EHR attribute deterministically determines the outcome. Controlled CKD-context and hazard tests confirm the code path; pilot EHR-only AUROC does not establish a predictive fusion benefit.

Private episode records preserve onset, planned duration, pathway, transitions, exact event time, recovery start/end and whether recovery occurred without an event. Hourly truth arrays preserve state, latent factors, event hazard, true physiology and healthy baseline. Some episodes end at the monitoring boundary; recorded recovery/end times describe that truncated simulation rather than verified later recovery.

Eight independent decoys include poor sleep, tachycardia, exercise, BP shifts, isolated SpO2 changes, weight fluctuation, sensor artifact and mild illness. They may coexist with latent stress by chance, but never cause an event. The simulator separately preserves exogenous healthy drift; neither deterioration nor decoys redefine the true healthy baseline.

## Observed pilot diversity

'''+table(['Quantity','Pilot count'],[
 ['Unique baseline vectors',d['baseline_unique_count']],['Rapid episodes',d['rapid_episodes']],['Recovered episodes without event',d['recovered_without_event']],['High-vulnerability stable non-event patients (≥14 stable days)',d['stable_high_risk_non_event_patients']],['Low-vulnerability abnormal decoy non-event patients',d['low_risk_abnormal_non_event_patients']],['Distinct hypertension/diabetes/CKD profiles',d['comorbidity_profiles']]])+'''

Pilot event times have SD '''+f"{d['event_time_hours']['sd']:.1f}"+''' hours, and patient adherence SD '''+f"{d['adherence']['sd']:.3f}"+'''. Counts by path are episode counts, not distinct event patients:

'''+table(['Path','Pilot episodes'],sorted(d['paths'].items()))+'''
The event is an internal synthetic cardiovascular deterioration marker, not adjudicated hospitalization, death, clinical decompensation or a measured HF endpoint. This scope is mandatory for Phase 6.
'''
    write('DETERIORATION_STATE_MODEL',state)
    comparison=table(['Signal','Wasserstein / real SD','KS','Max quantile gap / real SD','Within-person SD ratio','Day-change SD ratio','Synthetic lag-1'],[(s,*[f'{x[k]:.3f}' for k in ['wasserstein_real_sd','ks_statistic','maximum_quantile_distance_real_sd','within_sd_ratio','day_change_sd_ratio']],f"{x['synthetic']['lag1_daily_correlation']:.3f}") for s,x in sorted(r['comparison'].items())])
    diagnostic_table=table(['Pilot validation diagnostic','AUROC','AUPRC'],[(name,f"{x['auroc']:.3f}",f"{x['auprc']:.3f}") for name,x in diag['models'].items() if isinstance(x,dict)])
    variant_table=table(['Variant (128 matched patients each)','HR missing opportunity fraction','HR measurement-error SD (bpm)','SBP measurement-error SD (mmHg)'],[(name,f"{x['missing_fraction_at_opportunities']['heart_rate']:.3f}",f"{x['measurement_error_sd']['heart_rate']:.3f}",f"{x['measurement_error_sd']['systolic_bp']:.3f}") for name,x in v['variants'].items()])
    realism='''# Synthetic realism validation

'''+evidence+'''
The 1,000-person pilot passed all '''+str(len(p['gates']))+''' declared gates before the 10,000-person run was authorized. The replacement cohort was then audited with the same home-signal realism limits and simulation contracts. The temperature correction used the first large run's aggregate distributions, so this is an engineering recheck rather than an independent external validation population. Phase 6 test patients remain unused for model-performance evaluation. This is bounded engineering acceptance, not proof of a clinically faithful HF generator.

## Home distribution and temporal comparisons

'''+comparison+'''

Gates require normalized Wasserstein ≤0.60, maximum selected-quantile discrepancy ≤0.85 real SD, within-person and day-change SD ratios within 0.35–2.20, and lag-1 ≥0.15 for supported home channels. Means, variances, seven quantiles, slope distributions and between-patient means/variances are retained in the private metrics. These loose predeclared scenario limits permit substantial mismatch; they do not test clinical equivalence. The first large run missed the unchanged temperature quantile gate by 0.0026 real SD. A common −0.05 °C location correction was then applied, the pilot and all gates regenerated, and a fresh large run audited. This correction calibrates descriptive source distributions, not clinical outcomes. Weight's near-unity pooled lag correlation reflects persistent between-patient weight and is not evidence of accurate within-person physiology.

Observed daily PerHeart references comprise only 27 patients, with different observation counts and follow-up. Distribution metrics give more weight to patients contributing more observed days, and no claim of independent observations is made. CHIRON remains a separate provisional-unit contextual reference. Activity, HRV and sleep have no matching supplied home measurements, so real-vs-synthetic similarity for those channels is NOT VALIDATED.

The initial candidate failed HR/temperature tail checks. The simulator was revised to aggregate quantile-based marginal shapes and adjusted ordinary daily variance, then a separate virtual calibration lowered event incidence. After the first large cohort slightly missed the same temperature gate, a documented common −0.05 °C baseline location correction addressed that bias without changing the limit. The failed cohort and prior pilots are preserved, and pilot/large outputs were regenerated under the revised configuration. Subsequent causality testing removed normalization by an entire future activity trajectory. The final pilot was regenerated and every gate rerun without relaxing acceptance thresholds. Archived experimental pilots cannot authorize scaling because their config/core hashes no longer match.

## Cross-signal, diversity and quality checks

'''+table(['Within-patient relationship','Observed synthetic daily correlation'],[(k,f'{x:.3f}') for k,x in r['cross_signal_within_patient'].items()])+'''

The weight/congestion and activity/deterioration relationships use private latent burden as an evaluation reference; it is never a classifier feature. These relationships have overlap and weak effects, particularly activity/deterioration. True-physiology correlations are separately retained rather than presented as observed sensor evidence. All seven HF profiles, five paths, eight decoys, unique baselines, variable event timing and variable adherence are present. Stable high-risk and abnormal low-risk non-event controls prevent unusual physiology from equaling an outcome.

Missingness is measured against scheduled opportunities, not all hourly grid slots. Pilot missing/invalid fractions range from '''+f"{min(p['physiology_quality']['missing_fraction_by_signal'].values()):.1%}"+' to '+f"{max(p['physiology_quality']['missing_fraction_by_signal'].values()):.1%}"+'''. Missing nights, nonwear, charging, patient adherence, sparse devices, isolated invalid raw readings and receipt delays are represented. Invalid readings preserve raw value but have no usable value. Suspect readings remain flagged. Small current-state adherence penalties are checked by missingness-only attacks; no explicit outcome determines availability.

## Diagnostic difficulty, not final training

Only pilot TRAIN patients (700) fit the imputer, scaler and simple classifiers. Pilot VALIDATION patients (150; 7,650 windows) supplied these engineering checks; their 72-hour window prevalence is '''+f"{diag['models']['logistic_regression']['prevalence']:.3%}"+'''. Neither pilot TEST nor the fresh large TEST receives model-performance evaluation. Repeated windows are correlated, metrics are descriptive, and selection of the best single threshold on validation is an intentionally conservative shortcut search.

'''+diagnostic_table+'''

Best single-variable threshold AUROC: '''+f"{p['maximum_single_variable_auroc']:.3f}"+'''. Maximum timestamp/missingness/profile-ID/patient-hash nuisance AUROC: '''+f"{p['maximum_nuisance_auroc']:.3f}"+'''. Joint diagnostic AP is about '''+f"{diag['models']['logistic_regression']['auprc']/diag['models']['logistic_regression']['prevalence']:.1f}"+''' times its validation prevalence. The task is learnable and does not approach perfect single-channel separation. Physiology-only exceeds joint LR here, so this phase demonstrates no EHR fusion gain, personalization gain or Digital Twin benefit. Hazard-context influence is an implementation property, not a validated performance claim.

## Matched stress cohorts

'''+variant_table+'''

High noise multiplies measurement error by 2.5 except channel-specific physical/Poisson constraints. High missingness multiplies adherence by 0.65. Reduced sensors removes HRV, SpO2, respiration and temperature. Baseline shift adds HR +5 bpm, SBP +8, DBP +3, SpO2 −0.6%, RR +0.5/min and weight +1 kg from day 30; MAP is rederived. Each variant preserves the same latent event times. These 128 virtual patients are a subset of the pilot and are disjoint from the 10,000-patient Phase 6 population; use them as secondary stress checks, not as a replacement for the locked large-cohort test. That isolates observation/baseline stress but is not a causal treatment experiment or model robustness result.

## Figures and large-cohort audit

Eight private PNG artifacts are saved in `data/synthetic/pilot/figures/`: '''+', '.join('`'+x+'`' for x in figures)+'''. They show selected stable/deteriorating examples, each path, quantile comparisons, correlation heatmaps, missingness, event timing and pre-event median/P10–P90 trajectories. Pathway examples divide change by the maximum of first-week SD and one signal unit to stabilize small references; this is an illustrative scale, not a clinical score. Example trajectories are illustrative selections. Pre-event plots omit patients without any first-week reading for the channel, and do not substitute hidden baseline values. Correlation comparisons distinguish case mix from centered within-patient checks.

The large audit verified '''+f"{a['landmarks_checked']:,}"+''' feature/outcome links and future labels, all original artifact hashes, patient splits, freshness relative to the pilot, FHIR links, canonical context serialization, physiological bounds, derived MAP, receipt/masks, diversity, missingness/clipping, source integrity and preservation of earlier phases. No diagnostic model was rerun on large-cohort test data. Full checks and exact large distribution metrics remain in private `large_validation.json` and `realism_metrics.json`.
'''
    write('SYNTHETIC_REALISM_VALIDATION',realism)
    leakage='''# Synthetic leakage audit

'''+evidence+'''
Prediction interfaces contain historical EHR and received usable observations only. Simulator truth and future labels are physically separate tables. The canonical reader opens patients/EHR/observations and the manifest; it has no outcome or latent-state reader interface.

'''+table(['Boundary','Verified contract'],[
 ['Public features','Exact 83-field allowlist: 17 EHR values plus 6 statistics × 11 channels; no dynamic field discovery'],
 ['Linkage metadata','patient_id/window_id/landmark_hour/evidence labels stay outside all diagnostic input matrices'],
 ['Private simulator truth','Profile/path/state/latent/hazard/vulnerability/exact event time/seed/true physiology/healthy baseline excluded from feature tables and observation arrays'],
 ['Future labels','Exclusive 24h buckets and within72h flag recomputed from exact private event times; time-to-event remains outcome-only'],
 ['Observation clock','Only preceding 168h measurements whose receipt is ≤prediction cutoff contribute'],
 ['Future-value poisoning','Changing future values and their receipt times cannot change past features'],
 ['Late-receipt poisoning','Changing observations that had not yet arrived cannot change features'],
 ['Future process innovations','Replacing future autoregressive innovations does not alter past physiology; full-trajectory activity normalization removed'],
 ['Hidden-context poisoning','Extra state/event/seed/profile fields do not enter feature extraction'],
 ['Eligibility','Every patient has the same fixed 51 landmarks; no event-dependent row-count censor sidechannel'],
 ['Patient split','All histories generated first; non-overlapping patient roles; all imputation/scaling fits on pilot training patients'],
 ['Fresh Phase 6 population','10,000 virtual patient IDs disjoint from the 1,000 pilot IDs; same generator rules'],
 ['Nuisance attacks','Timestamp, missingness-only, clinical profile-ID and numeric patient-ID hash tried with LR/tree; below declared 0.75 AUROC limit'],
 ['Filesystem/Git','Private ground truth, records, arrays, weights and figures use 0700/0600 permissions and Git-ignore exclusion']])+'''

The pilot's full leakage contracts passed; large audit recomputed every exported future label and checked every feature schema/link. The tests exercise inclusive/exclusive horizon edges, insufficient follow-up, receipt delays, post-event semantics, no hidden-field export and future-process invariance. FHIR patient references and canonical dataclasses preserve namespaces and availability.

This is an aggressive bounded audit, not a proof against every classifier or encoding attack. The patient-hash nuisance check is one numeric SHA projection; it does not exhaust all possible identifier transformations. Profile-ID attacks use private truth only inside the audit, never normal model features. Clinical EHR remains correlated with risk by design but is not deterministic. Missingness can depend weakly on current latent state and therefore can carry some physiology information; its nuisance attack is measured, not assumed absent.

A fixed post-event schedule avoids outcome-dependent window counts. In the one-event simulator, however, post-event histories and no-additional-event labels remain a task-design limitation; Phase 6 should separately report performance for pre-event histories. Synthetic test patients remain fresh for model assessment, although their simulator-level integrity/distribution checks have necessarily been run here.

Grounding uses only the already-isolated eICU development export and selected VitalDB windows. It opens no real event labels in this phase, fits no real-data predictor/encoder, and never evaluates the spent real test cohort. Earlier phase results remain unchanged. Preservation proof covers '''+str(p['verification']['previous_public_files_unchanged'])+''' prior public source/docs/tests checksums and '''+f"{p['verification']['previous_private_metadata_unchanged']:,}"+''' private size/mtime records without decoding spent-test clinical files.

Scope certificates and an explicit artifact inventory distinguish model-accessible tables, linkage metadata, evaluation outcomes and private truth. Clinical validity, complete adversarial leakage resistance, counterfactual validity and real-world transportability are NOT VALIDATED.
'''
    write('SYNTHETIC_LEAKAGE_AUDIT',leakage)
    remaining='Small and selectively observed real home cohorts; unsupported home HRV/activity/sleep; assumed latent mechanics and HF incidence; no adjudicated 72h clinical endpoint; synthetic EHR epidemiology and treatment effects; incomplete FHIR profile certification; correlated overlapping windows; one-event/post-event label semantics; no demonstrated EHR fusion or personalization advantage.'
    items=[
      ('Simulator design','Aggregate-quantile copula baselines, autoregressive/circadian healthy variability, semi-Markov latent episodes, decoys and device observation layer.'),
      ('Real grounding sources','CHIRON daily structure; PerHeart verified home distributions/cadence; development-only eICU acute multivariate references; VitalDB surgical dynamics; Synthea FHIR shapes/codes. No unrelated real patient pairing.'),
      ('Physiological signals',', '.join(SIGNALS)+'. MAP derives from SBP/DBP; steps are hourly counts; unsupported home channels are labelled assumptions.'),
      ('EHR/context','Age/sex/BMI/EF/HF severity, hypertension/diabetes/CKD, five medication classes, creatinine/sodium/hemoglobin/BNP, diagnoses and historical encounters; all link to the same virtual patient.'),
      ('Deterioration paths','Autonomic-first, congestion-like, pressure/hemodynamic, mixed and noisy/atypical; rapid/gradual progression and recovery.'),
      ('Hidden-state model','Stable, early stress, compensating, early deterioration, worsening, adverse event and recovery; private factors/transitions/baselines.'),
      ('Event definition','Single synthetic latent-hazard adverse marker; stochastic burden × vulnerability × duration, never an observed-variable threshold. No clinical endpoint claim.'),
      ('Generated event prevalence',f"Pilot: 345/1000 (34.5%). Fresh large cohort: {m['event_patients']:,}/{m['patients']:,} ({m['event_patient_prevalence']:.2%}). Validation window prevalence {diag['models']['logistic_regression']['prevalence']:.3%}; patient and window incidence differ."),
      ('Cohort diversity',f"Pilot: 1,000 unique baselines, seven profiles, five paths, eight decoys, {d['rapid_episodes']} rapid episodes, {d['recovered_without_event']} event-free episode recoveries, {d['stable_high_risk_non_event_patients']} high-risk stable controls and {d['low_risk_abnormal_non_event_patients']} low-risk abnormal non-event patients."),
      ('Real-vs-synthetic similarity',f'All six supported home-channel gates pass in pilot and large audit; pilot normalized Wasserstein {min_w:.3f}–{max_w:.3f}, max selected-quantile discrepancy {min_q:.3f}–{max_q:.3f} real SD. See full ratios, slopes, quantiles, variance and correlations in realism report; no clinical equivalence claim.'),
      ('Simple-model difficulty',f"Pilot validation joint LR AUROC {diag['models']['logistic_regression']['auroc']:.3f}, AP {diag['models']['logistic_regression']['auprc']:.3f}; best single variable {p['maximum_single_variable_auroc']:.3f}; MAP-only {diag['models']['map_only']['auroc']:.3f}, HR-only {diag['models']['hr_only']['auroc']:.3f}, simple tree {diag['models']['simple_tree']['auroc']:.3f}. Diagnostics only."),
      ('Leakage audit',f"All explicit allowlist/hidden-state/future/late-receipt/process causality checks pass; nuisance maximum AUROC {p['maximum_nuisance_auroc']:.3f}; large audit verifies every label/link. Metadata excluded; patient splits isolated."),
      ('Missingness/noise','Pilot 26.8–30.6% missing/invalid at scheduled opportunities with individual adherence, nonwear, charging, missing nights, delayed/invalid/suspect readings. All five matched variants complete; latent event histories unchanged.'),
      ('Reproducibility',f"Seed 2026; 24 real-grounded virtual patients generated twice with all {p['verification']['artifact_files_byte_identical']} artifact files byte-identical and agreement with pilot prefix. Input/config/core/output checksums recorded; elapsed runtime metadata excluded."),
      ('Tests',f"{test_count} regression tests passed: 31 new synthetic tests plus 154 prior tests. Pilot/large artifact/schema/FHIR/privacy audits also pass."),
      ('Storage/runtime',f"Pilot generation {run_p['seconds']:.1f}s, original artifacts {run_p['bytes']/2**20:.1f} MiB. Large generation {run_l['seconds']:.1f}s, original artifacts {run_l['bytes']/2**30:.3f} GiB. Large audit {a['audit_seconds']:.1f}s. These are measured local runtime/storage, not arbitrary-scale guarantees."),
      ('1,000-patient pilot',f"PASS: all {len(p['gates'])} gates, before authorization to scale."),
      ('10,000-patient cohort',f"Generated and audited: 60 days, {m['landmarks']:,} fixed landmarks, train/validation/test {a['split_counts']['train']:,}/{a['split_counts']['validation']:,}/{a['split_counts']['test']:,}; patients disjoint from pilot; always SYNTHETIC POC."),
      ('Remaining limitations',remaining),
      ('Decision','GO for the bounded SYNTHETIC POC generator and synthetic Phase 6 experiments. Clinical validity and real-data end-to-end HF prediction remain NOT VALIDATED; prior real-data NO-GO findings are unchanged.'),
      ('Exact Phase 6 recommendation','Begin a separately scoped synthetic experiment on the fresh 10k split: use only EHR + received seven-day physiology; fit transformations/baselines on training patients; compare static, population and personal-history approaches with diagnostic baselines; reserve test for a locked final assessment; report 24/48/72h metrics and uncertainty by patient, pathway and sensor variant. Evaluate healthy-baseline estimation against private truth only after training, report pre-event-only sensitivity analysis, and claim no clinical, fusion or personalization advantage until measured. Do not train final Twin/Shadow/TDI/TTE/neighbours/sensing/UI during Phase 5.')]
    decision='''# Phase 5 decision

'''+evidence+'''
**GO for the synthetic generator and a separately scoped Phase 6 synthetic experiment.** All pilot gates passed before scaling; the fresh 10,000-patient cohort and its generation contracts passed the large audit. The final Digital Twin, Shadow Twin, TDI, time-to-event network, Twin Neighbours, Adaptive Sensing and clinician UI remain untrained/unbuilt in this phase. Earlier real-data results are preserved and are not reinterpreted.

## Required completion report

'''+''.join(f"### {i}. {title}\n\n{text}\n\n" for i,(title,text) in enumerate(items,1))+'''
## Reviewable artifacts

- [Design](SYNTHETIC_COHORT_DESIGN.md), [grounding and parameter support](PHYSIOLOGICAL_GROUNDING.md), [latent/event model](DETERIORATION_STATE_MODEL.md), [realism/difficulty/variants](SYNTHETIC_REALISM_VALIDATION.md), [leakage boundaries](SYNTHETIC_LEAKAGE_AUDIT.md).
- Implementation: `src/synthetic/`; versioned configuration: `configs/synthetic_v1.yaml`; regression tests: `tests/test_synthetic.py`.
- Private pilot certificate: `data/synthetic/pilot/validation.json`; large integrity certificate: `data/synthetic/cohort10000/large_validation.json`.
- Private stress comparison: `data/synthetic/variants/variant_summary.json`; eight scientific PNGs: `data/synthetic/pilot/figures/`.
- Private final source/output/evidence inventory: `data/synthetic/phase5_complete.json`.

Individual observations and assumed clinical context retain NOT VALIDATED. SYNTHETIC POC VALIDATED applies to reproducibility, schemas and enumerated simulation acceptance gates; it does not validate clinical physiology, risk predictions or treatment effects.
'''
    write('PHASE5_DECISION',decision)
    # Give every derived result an explicit evidence scope without changing the
    # immutable original exports or the pilot certificate used by the large run.
    for root in [pilot,large]:
        for name in ['realism_metrics.json','difficulty_metrics.json','diversity_metrics.json','leakage_audit.json']:
            path=root/name
            if path.exists():value=read(path);value.update(evidence='SYNTHETIC POC',evidence_label='SYNTHETIC POC VALIDATED',validation_scope='Enumerated simulator/diagnostic contracts only; not clinical validation');private_json(path,value)
    inventory=[]
    for base in [Path('data/synthetic'),Path('data/synthetic_grounding')]:
        for path in base.rglob('*'):
            path.chmod(0o700 if path.is_dir() else 0o600)
        base.chmod(0o700)
    for base in [pilot,large,Path('data/synthetic/variants'),Path('data/synthetic_grounding')]:
        for path in sorted(base.rglob('*')):
            if path.is_file():
                classification='private simulator ground truth' if 'ground_truth' in path.parts else 'model features' if 'model_features' in path.parts else 'evaluation outcomes' if 'outcomes' in path.parts else 'private record/reference/diagnostic artifact'
                label='NOT VALIDATED'
                if path.suffix=='.json':
                    try:label=read(path).get('evidence_label','NOT VALIDATED')
                    except (json.JSONDecodeError,AttributeError):pass
                inventory.append({'path':str(path),'sha256':file_hash(path),'bytes':path.stat().st_size,'classification':classification,'evidence_label':label})
    complete={'decision':'GO for synthetic POC only','pilot_all_gates_passed':True,'large_all_gates_passed':True,'patients':m['patients'],'event_patients':m['event_patients'],'tests_passed':test_count,'final_predictor_trained':False,'evidence':'SYNTHETIC POC','evidence_label':'SYNTHETIC POC VALIDATED','not_clinical_validation':True,'pilot_certificate_sha256':file_hash(pilot/'validation.json'),'large_manifest_sha256':file_hash(large/'manifest.json'),'large_certificate_sha256':file_hash(large/'large_validation.json'),'config_sha256':c['_config_sha256'],'simulator_code_sha256':code_hash(),'source_sha256':{str(x):file_hash(x) for x in sorted(Path('src/synthetic').glob('*.py'))},'environment_versions':{'python':sys.version.split()[0],**{name:version(name) for name in ['numpy','scipy','scikit-learn','joblib','matplotlib']}},'test_source_sha256':file_hash('tests/test_synthetic.py'),'documentation_sha256':{str(x):file_hash(x) for x in sorted(Path('docs').glob('*.md')) if x.stem in ['SYNTHETIC_COHORT_DESIGN','PHYSIOLOGICAL_GROUNDING','DETERIORATION_STATE_MODEL','SYNTHETIC_REALISM_VALIDATION','SYNTHETIC_LEAKAGE_AUDIT','PHASE5_DECISION']},'original_large_artifact_bytes':run_l['bytes'],'current_accepted_and_grounding_artifact_bytes':sum(x['bytes'] for x in inventory),'artifact_inventory':inventory}
    private_json('data/synthetic/phase5_complete.json',complete);print({'phase5_documentation_complete':True,'required_docs':6,'figures':8,'decision':'GO for SYNTHETIC POC only','tests_passed':test_count},flush=True)
    return complete

if __name__=='__main__':report()
