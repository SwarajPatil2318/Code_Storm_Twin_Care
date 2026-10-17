"""Development-only plan, frozen before representation/transfer outcomes."""
from pathlib import Path
ROOT=Path('data/results/phase45b')
SIGNALS=['map','sbp','dbp','hr','spo2','rr']
EICU_FIELDS=['systemicmean','systemicsystolic','systemicdiastolic','heartrate','sao2','respiration']
PROTOCOL={'version':'phase45b_v1','seed':2026,'vitaldb_sequences_minutes':[30,60,120],
 'primary_sequence_minutes':60,'grid_minutes':5,'vitaldb_split':'subject SHA256,80percent train/20percent validation, label-independent',
 'sequence_qc':'core five channels >=80percent observed bins each; max2 consecutive missing bins; RR optional; no filling',
 'source_artifacts':'Phase45A minute masks/counts/maxgap/provenance; native numeric only, no additional waveforms',
 'normalization':'train-subject observed median and IQR per signal; fixed across transfer; missing standardized values zero with explicit observed and age channels',
 'ssl_tasks':['causal_masked_reconstruction','future15minute_prediction'],
 'pilots':'10epochs both objectives on training subjects, validation diagnostics; primary60minute chosen for available context/retention, not outcome scores',
 'encoder':'NumPy compact causal TCN, kernel3,dilations1/2/4,8channels; mean+last16D latent; all taps train on60minute context; finite-difference gradients and causal prefix tests',
 'ssl_epochs':60,'ssl_batch_size':128,'ssl_learning_rate':.003,'ssl_l2':.001,
 'ssl_selection':'held-out VitalDB future-state linear-probe mean normalized MAE vs persistence; select objective on all-signal mode only, before eICU transfer; retain other objective results',
 'pretraining_modes':['all','no_map','no_bp'],'ablation':'separate encoders pretrained without excluded channels; excluded values/masks/age never enter encoder',
 'eicu_population':'originalTRAIN+VALIDATION only; original TEST denied before clinical row decoding; shared raw sources identity-routed, no test labels/predictions parsed',
 'eicu_cv':'5patient-stratified folds,2fixed seeds2026/2031; all repeated stays/windows grouped; development only, no new test',
 'comparators':['map_only','bp_only','handcrafted','handcrafted_ehr','latent_frozen','latent_frozen_ehr','latent_finetuned','latent_finetuned_ehr','latent_no_map','latent_no_bp','latent_plus_map','latent_delta','latent_delta_ehr','random_encoder','supervised_scratch'],
 'head':'small ridge logistic C=.1/1, inner3patient folds; fold-training only scaler; primary metrics raw AUPRC/AUROC and inner-OOF Platt calibrated Brier/ECE',
 'fine_tuning':'last convolution and head only, early layers frozen;20epochs,lr.0005,L2.1 and anchor penalty toward pretrained later weights; inner-patient early selection only',
 'personalization':'reference embedding from first120minute segment of same buffered6h history, before last60minute state; MAP>=75 when observed, >=80percent core coverage; missing reference has zero delta plus absence flag; no healthy/stable clinical claim',
 'robustness':['missing_spo2','missing_rr','missing_hr','noise','sparse_bp','truncate60'],
 'robustness_scope':'VitalDB held-out subjects and eICU development OOF; paired embedding cosine/RMS and risk/AP changes; not wearable validation',
 'success_gate':'paired patient-bootstrap lower95% AP gain>0 and mean gain>=.02 overMAP, positive gains>=8/10folds; Brier no worse>.005; credible noBP/conditional beyondMAP signal and hard-case ranking; no claims from losses/plots/onefold',
 'bootstrap':1000,'selection_multiplicity':'small declared family, paired uncertainty exploratory and development-selection conditional; independent validation still required',
 'domain_shift':'observed distribution/age/cadence/missingness comparisons with development eICU only; surgical/anesthesia vs ICU',
 'independent_validation':'credentialed MIMIC-IV or new nonoverlapping ICU cohort; mapping/QC/target/availability frozen before outcomes',
 'prior_files':'source/docs/test hashes preserved; prior private files metadata-only snapshots, no spent-test file reads',
 'excluded':'fullTwin/UI/waveformencoder/HFclinicalvalidation/externalperformanceclaims'}
