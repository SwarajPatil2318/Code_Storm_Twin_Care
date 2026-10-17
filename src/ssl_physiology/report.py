"""Publish aggregate-only Phase45B documentation from private experiment results."""
import json,re
from pathlib import Path
import numpy as np
from src.ssl_physiology.protocol import ROOT


def read(name):return json.loads((ROOT/name).read_text())
def f(value,d=3):return 'undefined' if value is None else f'{value:.{d}f}'
def ci(value):return 'undefined' if value is None else '['+', '.join(f(x) for x in value)+']'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def write(name,text):
    # Format prose quantities without changing paths, code tokens or source URLs.
    units=r'(minutes|minute|epochs|epoch|patients|windows|parameters|channels|iterations|folds|fold|tests|subjects|cases|positivepatients|positivewindows|saved|public|private|years|seconds|mmHg|bpm|hours|grid|observations|percentagepoint|breaths)'
    text=re.sub(r'(\d)(?='+units+r'\b)',r'\1 ',text)
    for before,after in [('Phase4.5A','Phase 4.5A'),('Phase4','Phase 4'),('Phase5','Phase 5'),('MIMIC-IV3.1','MIMIC-IV 3.1'),('eICU2.0','eICU 2.0'),('kernel3','kernel 3'),('batch128','batch 128'),('alpha10','alpha 10'),('seeds2026/2031','seeds 2026/2031'),('learning rate0.003','learning rate 0.003'),('learning rate0.0005','learning rate 0.0005'),('penalties0.1','penalties 0.1'),('fixed20','fixed 20'),('declared fixed20','declared fixed 20'),('byT0','by T0'),('MAP medians81vs82','MAP medians 81 vs 82'),('HR72vs86','HR 72 vs 86'),('ages61vs67','ages 61 vs 67'),('RR14vs19/min','RR 14 vs 19/min'),('available1,063','available 1,063'),('error0','error 0'),('all137','all 137'),('all180','all 180'),('All180','All 180'),('Previous112','Previous 112'),('Prevalence0.0382','Prevalence 0.0382'),('prevalence0.0382','prevalence 0.0382'),('arterial220','arterial 220'),('noninvasive220','noninvasive 220'),('total224','total 224'),('only2/9','only 2/9'),('only5/10','only 5/10'),('in5/10','in 5/10'),('required8/10','required 8/10'),('≥8/10folds','≥8/10 folds'),('of roughly18–21','of roughly 18–21')]:
        text=text.replace(before,after)
    text=re.sub(r',(?=[A-Za-z])',', ',text)
    text=re.sub(r'(?<![0-9]),(?=[0-9])',', ',text)
    text=re.sub(r';(?=\S)','; ',text)
    text=re.sub(r':(?=[0-9])',': ',text)
    text=re.sub(r',(?=[0-9]+%)',', ',text)
    for token in ['primary','dilations','and','only','top','files','directories']:
        text=re.sub(r'\b'+token+r'(?=[0-9])',token+' ',text)
    text=text.replace('positivepatients','positive patients').replace('positivewindows','positive windows').replace('17new','17 new').replace('verified.30','verified. 30')
    Path('docs',name).write_text(text.strip()+'\n')


def run():
    cohort=read('vitaldb_cohort.json');ssl=read('ssl_results.json');summary=read('transfer_summary.json');domain=read('domain_shift.json');robust=read('robustness_results.json');verification=read('verification.json')
    models=summary['models'];selected=ssl['selected_task'];quality=ssl['models'][selected+'__all']['heldout'];mask=ssl['models']['causal_masked_reconstruction__all']['heldout']
    cohorts=table(['History','Training sequences / cases / subjects','Validation sequences / cases / subjects'],[
      [minutes+' min',' / '.join(str(cohort['sequence_durations'][minutes]['train'][k]) for k in ['sequences','cases','subjects']),
       ' / '.join(str(cohort['sequence_durations'][minutes]['validation'][k]) for k in ['sequences','cases','subjects'])] for minutes in ['30','60','120']])
    quality_table=table(['Signal / units','Future latent probe MAE','Persistence MAE','Handcrafted linear MAE','Current-property probe R²'],[
       [signal,f(quality['future15_linear_probe'][signal]['mae_original_units']),f(quality['future15_persistence'][signal]['mae_original_units']),
        f(quality['future15_handcrafted_linear'][signal]['mae_original_units']),f(quality['current_property_linear_probe'][signal]['r2'])] for signal in ['map','sbp','dbp','hr','spo2','rr']])
    write('SSL_PHYSIOLOGY_ENCODER.md',f'''# Phase 4.5B physiological encoder

The encoder learns surgical physiological variation, but its development transfer does not establish useful improvement over MAP. This is a research artifact. Independent validation still required.

## Cohort and causal tensors

The label-free Phase 4.5A numeric sample contains 1,227 selected MAP cases, including a hash sample and long-record enrichment. Adult sequences require MAP/SBP/DBP/HR/SpO2 overlap; respiration is optional. Source patient identities are grouped across repeat cases. No cross-dataset patient linking occurs. Subject assignments are outcome independent and shared across durations.

{cohorts}

Choose **60 minutes**: it retains 742 training and 190 validation subjects with sufficient multivariate context. The 120-minute option loses many subjects; 30 minutes retains more short records but provides less context. This choice preceded eICU results. The primary 2,729 sequences cover 954 cases and 932 subjects. This local-window QC count can exceed Phase 4.5A's 827 whole-record usable cases: the criteria and denominator differ. Eligible windows are not evidence that an entire record is continuously usable. The candidate split contains 817 training/206 validation subjects; primary-duration availability reduces those counts without reallocating subjects.

Each closed five-minute bin uses at least four QC-valid closed one-minute medians from (t−5,t]. Native minute QC enforces impossible-value screens, duplicate handling, sample support and maximum gaps inherited from Phase 4.5A; source units, original QC flags and provenance remain available. Core channels must each have ≥80% observed bins with at most two consecutive missing bins. Missing values remain NaN in stored tensors. No gap interpolation or forward fill occurs. Jump flags are diagnostic, not a validated artifact classifier; residual artifacts remain possible. Cross-channel timestamp agreement is not independently certified hardware synchronization.

The canonical channel order is MAP, SBP, DBP, HR, SpO2, RR: six values, six observation masks, six causal ages since last available observation (capped at 360 minutes). Ages and masks expose missingness; absent standardized values are zero only alongside explicit absent masks. Training-subject observed median/IQR scales are frozen; degenerate IQR falls back to observed standard deviation with a minimum scale of one. Encoder inputs are clipped at ±8 normalized units; original units and values remain stored. Validation never fits normalization. VitalDB medians summarize dense bins whereas eICU bins typically contain a single periodic observation; cadence differences remain.

## Tasks, architecture and selection

Two tasks received ten-epoch pilots and fixed sixty-epoch runs: causal masked reconstruction and prediction fifteen minutes ahead. Reconstruction hides 25% of observed cells plus short contiguous masks, removes their observation indicators and recomputes age from surviving past observations. It predicts hidden current values from causal context only. Forecast targets are shifted three five-minute bins and include a separate future target array; these future values never enter encoder inputs. Missing targets contribute no loss. Both objectives use patient-window weights, batch128, Adam learning rate0.003, L2=0.001 and gradient clipping.

A compact causal TCN uses three tanh convolutions, kernel3, eight channels, dilations1/2/4, left zero padding and mean-plus-last pooling: **16-dimensional latent**, **894 parameters including the SSL head**. Each hidden state has a 15-grid-step receptive field; averaging pools the entire available input history. Padding makes the 60-minute training context shorter than the maximum receptive field. All kernel taps receive training support. A GRU/Transformer sweep was not justified by cohort size or the modest scientific question. The implementation uses NumPy with audited analytic gradients; finite differences test convolution and SSL-output derivatives. Larger models and 500Hz waveforms were not introduced.

Task selection used the all-signal held-out VitalDB future-state linear probe, before any eICU evaluation. Mean normalized MAE is {f(mask['selection_mean_future_normalized_mae'],4)} for masked reconstruction and **{f(quality['selection_mean_future_normalized_mae'],4)} for future prediction**. Selected task: **{selected}**. Encoders were separately pretrained for all channels, no MAP, and no BP. Excluded values, masks, ages and loss targets are removed. No-MAP still sees SBP/DBP, so it cannot establish independence from BP.

## Held-out physiological quality

The same fixed Ridge(alpha10) probe is fitted only on training subjects and tested on 541 sequences from 190 held-out subjects. Future targets are the state fifteen minutes after the sequence end; property targets are medians over its final fifteen minutes. These are observed physiology targets, not deterioration labels. MAE units are mmHg for BP, bpm for HR, percentage points for SpO2, and breaths/min for RR. Channel denominators vary with available future targets and are stored privately.

{quality_table}

The representation improves BP forecasting over persistence but does **not** beat the simple handcrafted linear probe for BP/HR/RR. Persistence is better for HR/SpO2/RR. A small SpO2 advantage over the handcrafted probe does not establish practical benefit. Current-property probes recover physiological variation, but this does not prove superior prognostic information or validate clinical latent states. No clinically interpreted clusters or trajectory retrieval claim is made. No independent trajectory-similarity or Shadow modelling utility was tested, so these results do not justify PARTIAL GO for a Twin feature. VitalDB remains unsuitable for the requested six-hour-history, three-hour deterioration outcome; no such model or clinical performance claim was fitted here.

## Transfer and artifacts

Use approved original eICU development patients only, with buffered six-hour tensors ending at T0−5 minutes (73 grid endpoints) and as-of-T0 EHR features. Frozen transfer fits a small logistic head. Limited fine-tuning updates only the final convolution and head, with early layers fixed, learning rate0.0005, L2/anchor penalties0.1 and fixed20epochs. Inner three-patient-fold predictions select head C=0.1/1 and fit Platt calibration. The planning protocol mentioned inner early selection; implementation uses the predeclared fixed20epochs and performs no early stopping or outer-label epoch selection. This execution difference is retained explicitly in private metadata. Scratch and random-encoder controls are included.

Architecture/config, source hashes, private weights, latent arrays and fold models reside under `data/results/phase45b/`. Medical records, patient identifiers, weights and predictions are excluded from Git; directories700/files600. Source implementation: `src/ssl_physiology/`. Reproduction commands and exact hashes are in private `reproducibility.json`; see [transfer results](REPRESENTATION_TRANSFER_RESULTS.md) and [decision](PHASE4_5B_DECISION.md).
''')
    metric_table=table(['Model','AUPRC','AUROC','Calibrated Brier','Mean ECE','Positive ΔAP folds /10'],[
      [name,f(v['mean_repeat_raw']['auprc'],4),f(v['mean_repeat_raw']['auroc'],4),f(v['mean_repeat_calibrated']['brier'],4),
       f(np.mean([x['ece_equal_width_10_bins'] for x in v['calibration_diagnostics']]),4),v['fold_ap_gain_over_map']['positive_folds']] for name,v in models.items()])
    variance_table=table(['Model','Fold AP mean ± SD','Fold AP min–max','Paired averaged-prediction ΔAP vs MAP [95% CI]'],[
      [name,f(v['fold_ap']['mean'])+' ± '+f(v['fold_ap']['std']),f(v['fold_ap']['min'])+'–'+f(v['fold_ap']['max']),
       f(v['paired_ap_vs_map']['auprc_difference'])+' '+ci(v['paired_ap_vs_map']['interval95'])] for name,v in models.items()])
    contrast_table=table(['Contrast','Paired ΔAP','95% patient-cluster interval'],[[name,f(v['auprc_difference'],4),ci(v['interval95'])] for name,v in summary['contrasts'].items()])
    write('REPRESENTATION_TRANSFER_RESULTS.md',f'''# Phase 4.5B development transfer results

**C. NO-GO for adopting pretraining.** The declared improvement rule is not met. All results below are development estimates. Independent validation still required.

## Population, model fitting and spent-test boundary

Original eICU TRAIN + VALIDATION: **210 patients, 59 patients with a positive window, 2,329 windows, 89 positive windows (3.82%)**. Five patient-stratified outer folds repeat at seeds2026/2031. All stays/windows from a patient remain together. Ten fold fits provide two held-out predictions per development window; repeats do not create 4,658 independent observations or a new test set. The original spent partition was never evaluated, selected upon, or used to normalize/fit/calibrate models.

The old whole-cohort loader was bypassed. Mixed JSONL/compressed CSV streams were routed by identity against original TRAIN/VALIDATION metadata **before clinical deserialization**. Rejected-row bytes pass through the routing stream; only identity is decoded, not test clinical fields or labels. No whole-cohort NumPy array or original test predictions were opened. This is the precise access boundary, not a claim that no bytes of mixed source files were read. A poison-invalid rejected-label test verifies rejection before JSON parsing. Private `development_access.json` records materialized patients, source paths, QC and routing counts.

Baseline MAP/BP/handcrafted/EHR features reproduce the approved Phase4 pure causal construction. Physiology inputs end at T0−5; EHR uses available information by T0. Cutoff physiology excludes T0 audit MAP. Handcrafted physiology retains its approved temperature feature, whereas the SSL tensor uses the six selected channels. Original target/washout eligibility remains unchanged. Baselines select logistic C=0.1/1 or depth-limited histogram boosting (three leaves,100iterations) with inner three patient folds. MAP-logistic controls classifier capacity. Latent heads select C=0.1/1 with fold-training-only scaling; combined MAP/EHR feature preprocessing is fitted separately inside every inner/outer training population. Platt calibration uses only inner out-of-fold predictions/labels from outer training. Diagnostic calibration slopes/intercepts are evaluation summaries and are not applied to predictions.

Fine-tuning uses fixed20epochs and updates only the final convolution/head; early layers remain byte identical to pretrained weights. Random and supervised-scratch TCN controls assess benefit attributable to pretraining. Matched recent-state heads control context/feature changes in personalized comparisons. This small declared family remains an exploratory development comparison with multiplicity; no post-result architecture/epoch/sensor tuning was performed.

## Pooled out-of-fold results

Each metric is calculated across all 2,329 held-out development windows separately for each repeat, then averaged. AUPRC uses average precision; AUROC uses raw risks. Brier/ECE use frozen inner-OOF Platt calibration. ECE is ten equal-width bins; rare-event prevalence can make ECE appear small despite poor ranking.

{metric_table}

MAP AUPRC {f(models['map_only']['mean_repeat_raw']['auprc'],4)} is a new development estimate, not an update to historical Phase4 test AUPRC. Differences in patient mix and prevalence prevent direct comparison with the spent-test score. The frozen encoder's modest AUROC does not establish an AUPRC advantage. EHR fusion worsens the latent point estimate. Calibration improves Brier but cannot create missing discrimination.

## Fold variability and paired uncertainty

{variance_table}

Fold-mean AP and pooled-repeat AP are different estimands: AP is nonlinear and cross-fold risks have different fitted scales. The frozen model's fold mean exceeds MAP, while its pooled-repeat estimate is lower. This cannot be resolved by choosing the favorable aggregation. Prespecified stability requires ≥8/10 positive folds; frozen and fine-tuned achieve only5/10. Neither has mean fold gain≥0.02. No latent model passes the numeric rule, which additionally requires a positive lower95% paired AP bound, acceptable Brier, and credible conditional/no-BP/hard-case evidence.

Paired intervals resample **patients**, retaining their windows, with1,000 fixed-seed replicates and frozen fitted models. The estimand is AP of each window's mean raw prediction across the two repeats; it differs from the mean of two separately calculated APs above. Intervals are conditional on this development cohort/model selection and do not incorporate refitting/model-selection uncertainty or correct multiplicity. They are exploratory, not independent validation.

{contrast_table}

Pretrained-vs-random and pretrained-vs-scratch intervals include zero. [Beyond-MAP analysis](BEYOND_MAP_ANALYSIS.md) details ablations and the small hard-case signal; [domain shift](VITALDB_EICU_DOMAIN_SHIFT.md) details sensor stress tests. Private fold-level raw/calibrated metrics, saved models, allowlists, OOF risks and latents remain available for reproducible inspection.
''')
    hard=summary['strata']['hard'];normal=summary['strata']['normal80']
    hard_table=table(['Model','Hard AP','Hard ROC','Positive windows in top77','Hard paired ΔAP vs MAP [95% CI]'],[
      [name,f(hard['models'][name]['mean_prediction_two_repeats']['auprc'],4),f(hard['models'][name]['mean_prediction_two_repeats']['auroc'],4),
       hard['models'][name]['positive_windows_in_top10percent'],f(hard['paired_ap_vs_map'][name]['auprc_difference'])+' '+ci(hard['paired_ap_vs_map'][name]['interval95']) if name!='map_only' else 'reference']
       for name in ['map_only','latent_frozen','latent_finetuned','latent_no_map','latent_no_bp','latent_plus_map']])
    normal_table=table(['Model','Normal-MAP AP','Positive windows in top151','Paired ΔAP vs MAP [95% CI]'],[
      [name,f(normal['models'][name]['mean_prediction_two_repeats']['auprc'],4),normal['models'][name]['positive_windows_in_top10percent'],
       f(normal['paired_ap_vs_map'][name]['auprc_difference'])+' '+ci(normal['paired_ap_vs_map'][name]['interval95']) if name!='map_only' else 'reference']
       for name in ['map_only','latent_frozen','latent_finetuned','latent_no_map','latent_no_bp','latent_plus_map']])
    write('BEYOND_MAP_ANALYSIS.md',f'''# Phase 4.5B beyond-MAP and personalized-state analysis

There is an exploratory hard-case ranking signal, but the overall, conditional-MAP and no-BP comparisons do not support adopting the representation. Independent validation still required.

## BP dependence and conditional comparisons

Frozen all-signal development AUPRC is {f(models['latent_frozen']['mean_repeat_raw']['auprc'],4)} versus MAP {f(models['map_only']['mean_repeat_raw']['auprc'],4)}. Separate no-MAP pretraining yields {f(models['latent_no_map']['mean_repeat_raw']['auprc'],4)}; SBP/DBP remain available and encode pressure proximity. Removing all MAP/SBP/DBP values, masks and ages yields **{f(models['latent_no_bp']['mean_repeat_raw']['auprc'],4)}**, near prevalence0.0382, AUROC {f(models['latent_no_bp']['mean_repeat_raw']['auroc'],4)}. Its paired averaged-prediction ΔAP vs MAP is {f(models['latent_no_bp']['paired_ap_vs_map']['auprc_difference'])}, CI {ci(models['latent_no_bp']['paired_ap_vs_map']['interval95'])}. This is evidence against useful BP-independent adverse-event discrimination in this population, not proof that non-BP physiology is never informative.

Representation + MAP yields {f(models['latent_plus_map']['mean_repeat_raw']['auprc'],4)}, versus matched MAP-logistic {f(models['map_logistic']['mean_repeat_raw']['auprc'],4)} and selected MAP {f(models['map_only']['mean_repeat_raw']['auprc'],4)}. Matched-head paired ΔAP is {f(summary['contrasts']['latent_plus_map__vs__map_logistic']['auprc_difference'],4)} with interval {ci(summary['contrasts']['latent_plus_map__vs__map_logistic']['interval95'])}. It does not establish added information beyond MAP under these feature/head comparisons. Limited fine-tuning and EHR fusion do not rescue this conclusion. No latent coordinate interpretation or causal claim is made.

## Hard cases

The preserved Phase4 audit rule requires MAP at T0 and buffered latest≥80mmHg, fresh T0 MAP≤10min old, buffered last-hour minimum≥75 and slope≥−1mmHg/min, supported history and no low MAP in the prior buffered120min. T0 audit values define the evaluation stratum only and never enter encoder/head inputs.

There are **{hard['windows']} hard windows from {hard['patients']} patients, only {hard['positive_windows']} positive windows from {hard['positive_patients']} positive patients** (1.17% prevalence). Metrics use each window's mean raw prediction across the two held-out repeats.

{hard_table}

Frozen/fine-tuned/no-MAP ranking improves here; the paired conditional intervals have positive lower bounds in this exploratory subgroup. This is a result worth preserving, not dismissing. However, nine positive patients, very wide intervals, multiple comparisons and only2/9 positives among the frozen/fine-tuned highest77 ranked windows do not support a robust clinical advantage. Top10% is a ranking budget, not a trained alarm threshold or clinical sensitivity estimate. No new threshold was chosen on outer labels. A no-MAP gain still allows BP surrogates; no-BP evidence remains unsupported. The full development gain/stability gate fails.

Broader clearly-normal-at-T0 MAP≥80 contains **{normal['windows']} windows, {normal['positive_windows']} positives from {normal['positive_patients']} positive patients**. It does not enforce the complete hard-case history rule.

{normal_table}

All reported normal-MAP paired gain intervals include zero. This limits generalization of the stricter hard-case result. These strata are development analyses, not replacements for an independent test or heart-failure validation.

## Causal personalized deviation prototype

Current state embeds the last60minutes. The reference embeds the first120minutes of the same buffered six-hour history, wholly before the recent state, requiring ≥80% coverage in the five core channels and observed MAP≥75. This is a quality/stability screen, not proof of a healthy state. **{summary['personal_reference']['available_windows']} / {summary['personal_reference']['total_windows']} windows** across {summary['personal_reference']['available_patients']} patients have an available reference. Missing references yield zero deviation plus an explicit availability flag.

The experimental head sees current z, Δz=current−reference and availability. A matched current-state/availability head controls the shorter recent context. Delta AUPRC {f(models['latent_delta']['mean_repeat_raw']['auprc'],4)} versus recent-only {f(models['latent_recent']['mean_repeat_raw']['auprc'],4)}; paired ΔAP {f(summary['contrasts']['latent_delta__vs__latent_recent']['auprc_difference'],4)}, CI {ci(summary['contrasts']['latent_delta__vs__latent_recent']['interval95'])}. With EHR, paired ΔAP {f(summary['contrasts']['latent_delta_ehr__vs__latent_recent_ehr']['auprc_difference'],4)}, CI {ci(summary['contrasts']['latent_delta_ehr__vs__latent_recent_ehr']['interval95'])}. Neither supports added personalization value. This is not TDI, a healthy Shadow Twin, an intervention counterfactual or a full Twin implementation.
''')
    domain_table=table(['Signal','VitalDB median [Q25,Q75]','eICU median [Q25,Q75]','Missing VitalDB / eICU','KS statistic','Wasserstein in source units'],[
      [name,f(v['vitaldb']['q05_q25_median_q75_q95'][2])+' ['+f(v['vitaldb']['q05_q25_median_q75_q95'][1])+','+f(v['vitaldb']['q05_q25_median_q75_q95'][3])+']',
       f(v['eicu_development']['q05_q25_median_q75_q95'][2])+' ['+f(v['eicu_development']['q05_q25_median_q75_q95'][1])+','+f(v['eicu_development']['q05_q25_median_q75_q95'][3])+']',
       f(100*v['vitaldb']['missing_fraction'],1)+'% / '+f(100*v['eicu_development']['missing_fraction'],1)+'%',f(v['ks_statistic']),f(v['wasserstein_original_units'])] for name,v in domain['signals'].items()])
    stress_table=table(['Perturbation','Median latent cosine','Median / P95 latent RMS','Frozen AP','Δ frozen AP','P95 absolute raw risk change','Fine-tuned AP'],[
      [name,f(v['embedding']['cosine_median']),f(v['embedding']['rms_median'])+' / '+f(v['embedding']['rms_p95']),
       f(v['models']['latent_frozen']['mean_ap'],4),f(v['models']['latent_frozen']['mean_ap_change'],4),
       f(v['models']['latent_frozen']['absolute_raw_risk_change_p95'],4),f(v['models']['latent_finetuned']['mean_ap'],4)] for name,v in robust['eicu_development'].items()])
    vital_stress=table(['Perturbation','VitalDB median latent cosine','VitalDB median / P95 latent RMS'],[
      [name,f(v['embedding']['cosine_median']),f(v['embedding']['rms_median'])+' / '+f(v['embedding']['rms_p95'])] for name,v in robust['vitaldb_heldout'].items()])
    write('VITALDB_EICU_DOMAIN_SHIFT.md',f'''# Phase 4.5B measured domain shift and robustness

Surgical pretraining does not establish ICU transfer or wearable robustness. Independent validation still required.

## Observed distribution shift

Statistics cover unique case/stay five-minute grid points represented in eligible windows; overlapping windows are deduplicated. They describe selected observed histories, not population prevalence. There are {domain['point_count']['vitaldb']:,} VitalDB and {domain['point_count']['eicu_development']:,} eICU development grid points. Selection requires well-observed pretraining windows and supported eICU MAP histories; missingness outside eligible histories is not measured here. Repeated grid points within a patient are dependent, so KS statistics describe separation without inferential p-values.

{domain_table}

MAP distributions are relatively close while HR, SBP, SpO2 and respiration shift substantially. SpO2 is heavily concentrated at100% in surgical monitoring, producing a zero selected-point IQR; its Wasserstein/IQR ratio is undefined rather than a huge finite effect. Subject-weighted age median [IQR] is **61 [51,69] years in VitalDB** versus **67 [56,77] in eICU**. eICU age is absent for3/210 patients and source age>89 is topcoded with an explicit feature flag. Repeat-case VitalDB ages are reduced to one observed metadata value per subject.

Native numeric cadence is typically about2seconds in VitalDB versus5minutes in eICU vitalPeriodic; both become5-minute canonical bins, but dense-bin medians and sparse observations remain different measurement processes. Typical cadence comes from Phase4.5A interval QC and eICU observation offsets, not an assumption of identical grids. VitalDB's [official dataset description](https://vitaldb.net/dataset/) describes single-center intraoperative surgery/anesthesia; [eICU](https://physionet.org/content/eicu-crd/2.0/) is multicenter critical care. Ventilation, anesthesia, interventions, case mix, duration and monitoring choices are plausible shift mechanisms; these descriptive comparisons do not identify their causal effects.

Frozen normalization remains fitted on VitalDB training subjects. eICU head/fine-tuning preprocessing uses outer/inner development training patients only. Selected cohorts already favor available BP, so they do not establish general ICU/home availability or deployment safety. Negative transfer is visible in the worse overall latent AUPRC, weak no-BP discrimination and unsupported pretrained-vs-scratch benefit.

## Controlled sensor perturbations

Whole-channel absence is tested for SpO2, RR and HR. Observed-only Gaussian noise has SD3/5/3mmHg MAP/SBP/DBP,5bpm HR,1percentagepoint SpO2 and2/min RR. Noise changes observed values without altering timing/masks; it is a controlled stress test, not a calibrated device error model or a new QC pass. Sparse BP keeps every fourth grid observation (20-minute cadence) with gaps left missing and ages recalculated from past surviving observations. Truncation retains only the final60minutes. EHR remains unchanged and calibration frozen. No future observation or outcome changes.

The selected original encoder's mean-plus-last latent is compared with perturbed inputs on every development window. Fitted frozen/fine-tuned physiology-only and EHR heads are evaluated only on their own held-out patients. The table shows two-repeat mean AP and raw-risk changes for the physiology-only heads; all four heads and calibrated metrics are stored privately.

{stress_table}

Noise has relatively modest embedding changes, but missing SpO2/RR and sparse BP change risks materially. Sparse BP lowers frozen AP from {f(models['latent_frozen']['mean_repeat_raw']['auprc'],4)} to {f(robust['eicu_development']['sparse_bp']['models']['latent_frozen']['mean_ap'],4)} and fine-tuned AP to {f(robust['eicu_development']['sparse_bp']['models']['latent_finetuned']['mean_ap'],4)}. P95 raw-risk changes of roughly18–21 percentage points under sparse BP/truncation are substantial. Removing RR or HR sometimes improves AP; this suggests domain-sensitive/redundant signals and instability, not a license to retune the model on these same labels. No sensor-removal model was selected from stress-test results.

On 541 held-out VitalDB sequences, the same perturbations also affect embeddings and the direct self-supervised fifteen-minute forecasting head. Detailed observed-target per-channel MAE, counts and R² are saved in private `robustness_results.json`.

{vital_stress}

Cosine similarity alone can conceal risk changes through a sensitive downstream head. These tests do not validate wearable sensing, transfer to unmeasured devices, cardiac morphology, a connected patient or an HF-specific physiology model.
''')
    write('PHASE4_5B_DECISION.md',f'''# Phase 4.5B decision: C. NO-GO

**Retain the simpler MAP/physiology approach. Do not adopt VitalDB SSL as the core temporal encoder or as a supported Shadow/Twin feature. Independent validation still required.**

The prespecified overall improvement and fold-consistency rule fails; conditional MAP addition is unsupported, and no-BP performance is near prevalence. Physiological probes show learned variation but no consistent advantage over handcrafted forecasting. The small hard-case ranking signal is retained as an exploratory hypothesis and does not establish practical value sufficient for GO/PARTIAL GO. This conclusion applies to the implemented tasks, cohort and transfer scheme; it does not rule out every future representation approach.

## Requested evidence record

1. **Pretraining cohort:** primary60minutes,2,188 training sequences/763 cases/742 subjects;541 validation sequences/191 cases/190 subjects. Subject grouping and train-only normalization verified.30/120-minute retention documented in [encoder report](SSL_PHYSIOLOGY_ENCODER.md).
2. **SSL task:** fifteen-minute future physiology prediction, selected against causal masked reconstruction on held-out VitalDB common linear-probe normalized MAE: {f(quality['selection_mean_future_normalized_mae'],4)} vs {f(mask['selection_mean_future_normalized_mae'],4)}. No deterioration outcome was fitted in VitalDB.
3. **Encoder:** compact causal three-layer TCN,8channels,kernel3,dilations1/2/4,16D latent,894parameters with SSL head. No waveform/Transformer sweep.
4. **Held-out representation quality:** future MAP MAE {f(quality['future15_linear_probe']['map']['mae_original_units'])}mmHg vs persistence {f(quality['future15_persistence']['map']['mae_original_units'])} and handcrafted {f(quality['future15_handcrafted_linear']['map']['mae_original_units'])}. BP forecasts beat persistence but no consistent multivariate advantage over handcrafted probes; HR/SpO2/RR persistence is better.
5. **eICU development MAP baseline:**210patients/2,329windows/89positivewindows; two repeats of five patient folds. Mean-repeat AUPRC {f(models['map_only']['mean_repeat_raw']['auprc'],4)}, AUROC {f(models['map_only']['mean_repeat_raw']['auroc'],4)}, calibrated Brier {f(models['map_only']['mean_repeat_calibrated']['brier'],4)}.
6. **Transferred physiology:** frozen AUPRC {f(models['latent_frozen']['mean_repeat_raw']['auprc'],4)}; limited fine-tuned {f(models['latent_finetuned']['mean_repeat_raw']['auprc'],4)}. Frozen paired averaged-prediction ΔAP vs MAP {f(models['latent_frozen']['paired_ap_vs_map']['auprc_difference'],4)},95% interval {ci(models['latent_frozen']['paired_ap_vs_map']['interval95'])}.
7. **Representation + EHR:** frozen AUPRC {f(models['latent_frozen_ehr']['mean_repeat_raw']['auprc'],4)}; fine-tuned {f(models['latent_finetuned_ehr']['mean_repeat_raw']['auprc'],4)}. EHR benefit unsupported.
8. **Without MAP:** no-MAP AUPRC {f(models['latent_no_map']['mean_repeat_raw']['auprc'],4)} (still sees SBP/DBP); no-BP {f(models['latent_no_bp']['mean_repeat_raw']['auprc'],4)}, AUROC {f(models['latent_no_bp']['mean_repeat_raw']['auroc'],4)}. Representation+MAP adds no supported matched-head advantage.
9. **Hard-case development:**768windows,9positivepatients; averaged-prediction hard AP MAP {f(hard['models']['map_only']['mean_prediction_two_repeats']['auprc'],4)}, frozen {f(hard['models']['latent_frozen']['mean_prediction_two_repeats']['auprc'],4)}, fine-tuned {f(hard['models']['latent_finetuned']['mean_prediction_two_repeats']['auprc'],4)}. Exploratory conditional positive bounds coexist with wide intervals and only2/9 positives in the frozen/fine-tuned top10% rank budget. Broader normal-MAP intervals include zero; not a clinical operating-point claim.
10. **Variability:** frozen fold AP {f(models['latent_frozen']['fold_ap']['mean'])}±{f(models['latent_frozen']['fold_ap']['std'])},range {f(models['latent_frozen']['fold_ap']['min'])}–{f(models['latent_frozen']['fold_ap']['max'])}; improves over MAP in5/10folds, below required8/10. No latent variant passes the declared numeric gate.
11. **Sensor robustness:** sparse-BP frozen AP {f(robust['eicu_development']['sparse_bp']['models']['latent_frozen']['mean_ap'],4)}; P95 raw-risk shift {f(robust['eicu_development']['sparse_bp']['models']['latent_frozen']['absolute_raw_risk_change_p95'],4)}. Missing sensors/truncation materially affect embeddings/risks. No wearable robustness claim.
12. **Latent personalization:** earlier screened reference available1,063/2,329windows. Delta vs matched recent-state paired ΔAP {f(summary['contrasts']['latent_delta__vs__latent_recent']['auprc_difference'],4)}, CI {ci(summary['contrasts']['latent_delta__vs__latent_recent']['interval95'])}; unsupported. No TDI/full Shadow Twin.
13. **Domain shift:** MAP medians81vs82mmHg; HR72vs86bpm; ages61vs67years; RR14vs19/min; dense surgical vs periodic ICU acquisition. Selected-window missingness differs; [measured shift and stress tests](VITALDB_EICU_DOMAIN_SHIFT.md).
14. **Tests/reproducibility:**154tests pass, including all137 earlier regressions and17new contracts. All180saved fold models reproduce raw/calibrated OOF predictions exactly; latent reload error0. Previous112public file hashes and28,622private file size/mtime pairs unchanged. Private artifacts Git-ignored, files600/directories700. Old private checks use metadata only; this does not prove absence of reads. The identity-routing boundary and poison-row tests establish the clinical decoding protection.
15. **Decision:** exactly **C. NO-GO** for adoption in this phase. Keep weights/results for audit, do not continue tuning against these repeatedly used development labels to manufacture a GO.
16. **Phase5 recommendation:** freeze the simple MAP-based comparator, establish availability/QC/uncertainty reporting and implement the independent validation data mapping below. Keep personalization/EHR fusion/SSL research outputs disabled in any clinical claim or alert workflow. Do not build a full Shadow Twin, TDI or physiological forecasting UI on the unsupported representation.

## Concrete independent validation plan

Use credentialed [MIMIC-IV3.1](https://physionet.org/content/mimiciv/3.1/) or credentialed [full eICU2.0](https://physionet.org/content/eicu-crd/2.0/). Access is a future dependency; no credentialed clinical data were fetched in this phase. The [official MIMIC vital-sign SQL](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/vitalsign.sql) identifies the candidate item mappings below. Freeze and locally validate dictionary labels/units before outcome construction. Preserve invasive/noninvasive source distinctions rather than pooling them blindly.

| Concept | MIMIC candidate item IDs | Canonical units / rule |
| --- | --- | --- |
| MAP | arterial220052/225312; noninvasive220181 separately | mmHg; arterial primary, separately prespecified fallback |
| SBP / DBP | arterial220050/225309 and220051/225310; noninvasive220179/220180 separately | mmHg; paired-source consistency QC |
| HR |220045| bpm |
| SpO2 |220277| percent |
| RR |220210; total224690 separately| breaths/min; preserve measurement type |

Before exposing outcomes: namespace/group subject_id across admissions/stays; exclude every subject previously inspected in the MIMIC demo if overlapping. For full eICU exclude **all original demo patients**, including development and spent-test patients, plus any subsequently inspected patients; verify within-dataset identity compatibility before split. Never infer cross-cohort patient identities from demographics. Select an untouched hospital/time/site cohort and lock its identifiers privately before training/analysis. If enough independent data exist, reserve a distinct calibration/training population separate from the locked validation population; do not fit calibration on validation outcomes. If identity exclusion or adequate continuity cannot be verified, this cannot serve as the independent cohort.

Map subject/hadm/stay identifiers, admission/discharge offsets, charttime and storetime; establish receipt/availability semantics. Predictor availability must satisfy the approved cutoff, using charttime plus a validated storetime/receipt rule; receipt must not be assumed from observation time. Canonical five-minute tensors retain masks and source cadence with no interpolation across large gaps; preserve original impossible-value, duplicate and cross-BP ordering QC. Create only adult records with supported six-hour histories; preserve the five-minute physiology buffer, approved120-minute low-MAP washout, three-hour horizon and sustained MAP<65 for30minutes with prespecified support/gap rules. Check local cadence before label generation: unobserved minutes cannot establish continuous hypotension. Report inadequate labels as a feasibility failure, not manufactured events. Freeze missingness handling, patient grouping, admission censoring, demographic rules, hard-case definition and EHR as-of mappings before examining outcomes. No discharge/billing diagnoses become pre-cutoff predictors unless availability is demonstrated.

Freeze simple-model coefficients/config, scaling/calibration/thresholds on the authorized development or separate calibration population before a **single** independent evaluation. Predetermine AUPRC/AUROC, Brier/reliability, patient-cluster intervals, alert burden, lead times and hard-case ranking. Treat the retained SSL encoder only as a predeclared secondary comparator, with the no-BP and conditional-MAP hypotheses unchanged. Do not optimize on new validation labels or call another repeated development split external validation. Record exclusions, endpoint ascertainment, missingness and patient/site counts. A successful validation still does not establish heart-failure-specific usefulness or treatment benefit.

## Reproduction and preserved scope

Run the new `src.ssl_physiology` cohort/pretrain/transfer/analysis/verify modules using `.venv/bin/python`, then `.venv/bin/python -m unittest discover -s tests -q`; `report` publishes aggregate docs. Private `protocol.json`, `execution_notes.json`, `architecture.json`, `fold_metrics.json`, `transfer_summary.json`, `robustness_results.json`, `domain_shift.json`, `development_access.json`, `verification.json` and `reproducibility.json` retain exact configuration/evidence. Never invoke legacy whole-cohort evaluation to reproduce this phase. Fixed seeds/one BLAS thread and saved-pipeline reload checks support deterministic evaluation; no fresh test set was created or spent. Historical clinical targets, previous model artifacts and prior source/docs/tests remain unchanged.
''')
    print({'phase45b_aggregate_docs_created':5,'decision':'C. NO-GO'},flush=True)

if __name__=='__main__':run()
