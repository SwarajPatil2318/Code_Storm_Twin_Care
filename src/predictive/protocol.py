"""Pre-test choices: declared independently of final-test features and outcomes."""
SEED = 2026
MODALITIES = ['ehr', 'map_only', 'bp_only', 'phys_no_map', 'phys_no_bp', 'phys', 'fused', 'personalized']
SIGNALS = ['heartrate', 'systemicmean', 'systemicsystolic', 'systemicdiastolic', 'sao2', 'respiration', 'temperature']
BP = ['systemicmean', 'systemicsystolic', 'systemicdiastolic']
SCALES = [30, 60, 180, 360]
RANGES = {'heartrate': [50,100], 'systemicmean': [65,100], 'systemicsystolic': [90,140],
          'systemicdiastolic': [50,90], 'sao2': [90,100], 'respiration': [8,24], 'temperature': [36,38]}
CANDIDATES = [
 {'kind':'logistic','penalty':None,'C':1.,'balanced':False},
 {'kind':'logistic','penalty':'l2','C':.1,'balanced':False},
 {'kind':'logistic','penalty':'l2','C':1.,'balanced':False},
 {'kind':'logistic','penalty':'l2','C':.1,'balanced':True},
 {'kind':'hist_gradient_boosting','leaves':3,'iterations':100,'balanced':False},
 {'kind':'hist_gradient_boosting','leaves':7,'iterations':120,'balanced':True},
]
PROTOCOL = {
 'version':'phase4_v1','seed':SEED,'target':'map65_30min','horizon_minutes':180,'lookback_minutes':360,
 'input_buffer_minutes':5,'minimum_lead_minutes':15,'washout_minutes':120,
 'split':'copy authoritative Phase35 seed2026 patient-level stratification unchanged',
 'training_weights':'inverse within-patient window count, normalized mean1; optional class weighting computed on TRAIN only',
 'modalities':MODALITIES,'scales_minutes':SCALES,'signal_feature_ranges':RANGES,
 'personal_reference':'first five hours of the same six-hour history, ending T0-65; >=30 points, >=3 connected hours; observed reference, not healthy counterfactual',
 'sequence':'73 five-minute causal bins through T0-5; observed medians, masks, elapsed time; no interpolation/backfill',
 'tabular_candidates':CANDIDATES,
 'temporal':'three small tanh causal convolution layers, 4 channels, kernel3, dilations1/4/16; mean+last pooling; Adam150 epochs, L2 .01/.1; optional four-unit EHR representation',
 'calibration':'none / positive-slope Platt / isotonic; 3 patient-stratified validation folds produce out-of-fold calibrated probabilities; lowest OOF Brier, simplicity tolerance1e-4; final calibrator fit all validation',
 'operating_points':{'high_sensitivity':'>=80% validation OOF sensitivity then maximum threshold',
                     'balanced':'maximum validation OOF F1',
                     'burden_aware':'maximum sensitivity subject to <=.02 false alerts per eligible patient-hour'},
 'map_strata':{'below_threshold':'MAP<65','near_threshold':'65<=MAP<70','moderate':'70<=MAP<80','clearly_normal':'MAP>=80'},
 'hard_subset':'audit MAP at T0 and buffered latest>=80; minimum buffered last60m>=75; last60m slope>=-1 mmHg/h; no MAP<65 in buffered previous120m; current audit measurement age<=10min',
 'selection':'validation raw AUPRC; add fusion/personal/temporal complexity only if paired patient-bootstrap AUPRC lower bound>0 against simpler reference; among remaining within .01 best require acceptable OOF Brier, inspect hard subset when >=3 positive patients, favor simpler model and lower balanced false-alert burden; no test feedback',
 'metric_bootstraps':2000,'uncertainty_refits':20,
 'uncertainty':'training-patient bootstrap refits with frozen hyperparameters plus validation-patient bootstrap recalibration; 5/95% prediction quantiles; conditional, not clinical confidence guarantees',
 'test':'global study registry plus single exclusive evaluation transaction for ALL frozen modality comparators and selected primary; write started marker before any test feature construction; subsequent calls return sealed results only',
 'risk_trajectories_minutes_before_onset':[360,240,180,120,60,30,0],
 'evidence':'retrospective ICU monitor proxy; HF exploratory; no guaranteed 3h lead, no 48-72h, no consumer-wearable/clinical-benefit claim',
}
