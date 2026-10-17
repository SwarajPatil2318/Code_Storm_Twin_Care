"""Frozen, pre-model Phase 3.5 protocol; parameters do not use model errors."""
PERIODIC = {
 'heartrate':('heart_rate','bpm',20,250),
 'systemicsystolic':('invasive_systolic_bp','mmHg',30,300),
 'systemicdiastolic':('invasive_diastolic_bp','mmHg',10,200),
 'systemicmean':('invasive_map','mmHg',10,200),
 'sao2':('spo2','%',50,100), 'respiration':('respiratory_rate','breaths/min',1,80),
 'temperature':('temperature','degC',30,43), 'cvp':('cvp','mmHg',-10,50),
 'etco2':('etco2','mmHg',5,80),
 'pasystolic':('pulmonary_systolic_pressure','mmHg',1,150),
 'padiastolic':('pulmonary_diastolic_pressure','mmHg',-5,100),
 'pamean':('pulmonary_mean_pressure','mmHg',0,120),
 'icp':('intracranial_pressure','mmHg',-10,100),
 'st1':('st_segment_1','unknown',-10,10),'st2':('st_segment_2','unknown',-10,10),'st3':('st_segment_3','unknown',-10,10)}
APERIODIC = {
 'noninvasivesystolic':('noninvasive_systolic_bp','mmHg',30,300),
 'noninvasivediastolic':('noninvasive_diastolic_bp','mmHg',10,200),
 'noninvasivemean':('noninvasive_map','mmHg',10,200),
 'paop':('pulmonary_occlusion_pressure','mmHg',-10,60),
 'cardiacoutput':('cardiac_output','L/min',.1,30),
 'cardiacinput':('cardiac_index','L/min/m2',.1,15),
 'svr':('systemic_vascular_resistance','unknown',0,5000),
 'svri':('systemic_vascular_resistance_index','unknown',0,10000),
 'pvr':('pulmonary_vascular_resistance','unknown',0,3000),
 'pvri':('pulmonary_vascular_resistance_index','unknown',0,6000)}
PRIMARY_SIGNALS = ['heartrate','systemicsystolic','systemicdiastolic','systemicmean','sao2','respiration','temperature']
CANDIDATES = {
 'map60_30min':{'source':'vitalPeriodic','field':'systemicmean','threshold':60,'inclusive':True,'duration':30,'max_gap':5,'min_points':7},
 'map65_30min':{'source':'vitalPeriodic','field':'systemicmean','threshold':65,'inclusive':False,'duration':30,'max_gap':5,'min_points':7},
 'drop20_map65':{'source':'vitalPeriodic','field':'systemicmean','threshold':65,'inclusive':False,'duration':30,'max_gap':5,'min_points':7,'drop':20},
 'cuff65_30min':{'source':'vitalAperiodic','field':'noninvasivemean','threshold':65,'inclusive':False,'duration':30,'max_gap':15,'min_points':3}}
PROTOCOL = {'seed':2026,'horizons_minutes':[60,180,360],'lookbacks_minutes':[360,720,1440],
 'history_coverage':.8,'future_coverage':.9,'max_coverage_gap':10,'feature_lag_minutes':5,
 'min_lead_minutes':15,'washout_minutes':120,'episode_merge_minutes':120,
 'window_stride':'horizon length, anchored to ICU admission; positive windows max one per event/configuration',
 'primary_candidate':'map65_30min','primary_horizon':180,'primary_lookback':360,
 'go_gates':{'eligible_patients':200,'positive_patients':50,'negative_patients':100,'development_positive_patients':30,
             'validation_positive_patients':10,'test_positive_patients':10,'hf_positive_patients':20}}

LAB_SCREENS = {
 'creatinine':({'mg/dl'},.1,25), 'potassium':({'mmol/l','meq/l'},1,10),
 'sodium':({'mmol/l','meq/l'},80,200), 'lactate':({'mmol/l'},.1,30),
 'bun':({'mg/dl'},1,300), 'hgb':({'g/dl'},3,25), 'glucose':({'mg/dl'},10,1500)}
