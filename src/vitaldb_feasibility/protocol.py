"""Pre-outcome feasibility choices; no predictive-model selection."""
PROTOCOL={
 'version':'phase4_5A_v1.2','seed':2026,'population':'adult cases age>=18; all-age signal-QC counts separately',
 'primary_history_minutes':360,'history_sensitivity_minutes':[30,60,180,360],
 'horizons_minutes':[30,60,120,180],'buffer_minutes':5,'minimum_lead_minutes':15,'washout_minutes':120,
 'event':'consecutive usable invasive MAP<65 for>=1800 seconds, maximum gap10seconds, at least181 observations; no interpolation; confirmation recorded when30minutes observed',
 'event_sensitivity':'native10second-gap primary; causal5minute as-of cadence emulation (age<=10seconds) separately, not proof of continuously low pressure','episode_merge_minutes':120,
 'qc_pressure_ranges_mmHg':{'map':[10,200],'sbp':[30,300],'dbp':[10,200]},
 'pressure_jump_flags':'absolute change>40mmHg over<=10seconds flagged, not automatically removed',
 'flatline_flags':'plausible constant values flagged, not automatically interpreted as artifact; waveform verification remains needed',
 'minute_cache':'raw values assigned to closed ceil(Time/60) bins; latest/median/min/max/count/latest raw timestamp/max gap; no interpolation, no fill',
 'history_coverage':.8,'future_coverage':.9,'history_max_missing_minutes':2,'future_max_missing_minutes':2,
 'followup':'through horizon+31minutes; extra closed minute accommodates up to10seconds native confirmation jitter after30minutes; recording must actually reach endpoint; incomplete future censored, not negative',
 'implementation_correction':'v1.1 corrects exact-horizon confirmation censoring and partial final-minute ascertainment; v1.2 enforces the full120minute washout ending at the buffered cutoff rather than115minutes; clinical thresholds, requested durations, horizons and stability unchanged; not a sample-count optimization',
 'landmarks':'max(history,120minute washout)+5min then fixed30minute stride; at most one positive landmark per episode/configuration/stability; overlapping controls counted as windows, not independent cases',
 'lead_bands':'cumulative (15,horizon] and terminal30minute bands (max(15,horizon-30),horizon]; 3h terminal band (150,180]min is genuine approximately2.5–3h warning',
 'stability':{
  'washed_out':'no usable MAP<65 in [T0-125,T0-5] full120minute buffered washout; no ongoing confirmed event atT0',
  'normal80':'washed_out + latest atT0 and buffered latest>=80; current measurement age<=60seconds',
  'hard':'normal80 + minimum buffered last60minutes>=75 and slope>=-1mmHg/hour',
  'normal90':'washed_out + latest atT0 and buffered latest>=90; minimum last60minutes>=80; current measurement age<=60seconds'},
 'candidate_scale_gates':'engineering feasibility gates, not power calculations: >=100 independent positive subjects and>=500 control subjects for supervised pretraining; >=1000 usable multivariate subjects for representation learning',
 'numeric_retrieval':'metadata first; named numeric tracks only; primary source priority Solar8000/ART_MBP then FEM_MBP then EV1000/ART_MBP; waveform snippets only for small verification sample',
 'network':'5 GET requests/sec,4 workers,4MiB per numeric body,512MiB maximum new body bytes per run; hashes and retrieval dates; only minute summaries retained for cohort streams',
 'identity':'vitaldb:subject:{subjectid}; group repeated cases by subject in all future splits; never link unrelated eICU identities',
 'clinical_features':'demographics/preop context only under declared preoperative availability assumptions; untimed dx/surgical details need review; postop outcomes, whole-case drug/fluid totals and future labs excluded; timed lab receipt unverified',
 'training':'none; no final predictive fit or spent eICU test access',
 'scope':'feasibility only, intraoperative physiology; not HF validation, no UI or full Digital Twin layers'}

if __name__=='__main__':
 from src.vitaldb_feasibility.source import private_json,ROOT
 private_json(ROOT/'protocol.json',PROTOCOL);print({'protocol':PROTOCOL['version'],'thresholds_prespecified':True,'documented_sampling_boundary_correction':True,'models_trained':0})
