"""Reviewed aggregate Phase 3.5 reports, contracts and one explicit recommendation."""
import argparse
import json
from pathlib import Path
from src.data.io import dumps
from src.data.adapters.eicu import SCHEMAS
from src.data.eicu_spec import PERIODIC,APERIODIC,PRIMARY_SIGNALS,PROTOCOL

SOURCES={
 'eicu':'https://physionet.org/content/eicu-crd-demo/2.0.1/',
 'periodic':'https://eicu.mit.edu/eicutables/vitalperiodic/',
 'aperiodic':'https://eicu.mit.edu/eicutables/vitalaperiodic/',
 'patient':'https://eicu.mit.edu/eicutables/patient/',
 'lab':'https://eicu.mit.edu/eicutables/lab/',
 'history':'https://eicu.mit.edu/eicutables/pasthistory/',
 'diagnosis':'https://eicu.mit.edu/eicutables/diagnosis/',
 'admissiondx':'https://eicu.mit.edu/eicutables/admissiondx/',
 'medication':'https://eicu.mit.edu/eicutables/medication/',
 'infusion':'https://eicu.mit.edu/eicutables/infusiondrug/',
 'ahe':'https://physionet.org/content/challenge-2009/1.0.0/',
 'map65':'https://sccm.org/clinical-resources/guidelines/guidelines/surviving-sepsis-guidelines-2021',
 'mimic':'https://mimic.mit.edu/docs/iv/modules/hosp/'}


def availability_contract():
    rows=[]
    for path,fields in sorted(SCHEMAS.items()):
        if not path.startswith('eicu_demo/'):continue
        table=Path(path).name.removesuffix('.csv.gz')
        for field in fields:
            include=False;available='unknown';time=None;reason='audit/integration only; not in initial predictor allowlist';transform=None
            if field in ['patientunitstayid','patienthealthsystemstayid','uniquepid','hospitalid','wardid'] or field==fields[0]:
                reason='identity/linkage/source locator only; never a predictor'
            elif any(x in field.lower() for x in ['discharge','mortality','actual','died','activeupondischarge']):
                available=False;reason='future outcome, final status, discharge information or outcome-derived field'
            elif table=='patient' and field in ['age','gender','unittype','unitadmitsource']:
                include=True;available=True;time='ICU admission (0), semantic index assumption';reason='immutable demographics or admission context; capture timestamp not independently recorded'
                transform='preserve age >89 as topcoded category, not invented numeric age'
            elif table in ['vitalPeriodic','vitalAperiodic'] and field in {**PERIODIC,**APERIODIC}:
                available=False;time='observationoffset (entered offset); extra 5-min buffer for model inputs'
                include=table=='vitalPeriodic' and field in PRIMARY_SIGNALS
                reason='causal numeric summaries only; QC and ICU-boundary checks required; no interpolation'
                if field in ['cvp','st1','st2','st3','svr','svri','pvr','pvri']:reason='audit only: unit/scaling ambiguity or unknown units; excluded from primary model'
                if table=='vitalAperiodic':reason='separate irregular cuff/invasive measurement source; audited, not pooled with periodic arterial MAP'
                transform='past-only count, missingness, last, mean, median, SD, extrema, slope and staleness'
            elif table in ['diagnosis','pastHistory','admissionDx'] and field in ['diagnosisstring','icd9code','pasthistorypath','pasthistoryvalue','pasthistoryvaluetext','admitdxpath','admitdxname','admitdxtext']:
                include=True;available=False
                time={'diagnosis':'diagnosisoffset','pastHistory':'max(pasthistoryoffset,pasthistoryenteredoffset)','admissionDx':'admitdxenteredoffset'}[table]
                reason='only positive HF/history/HTN/diabetes/CKD/CV flags from evidence documented by T0; absence remains unknown; not active-disease adjudication'
                transform='explicit code/text mapping; negatives/uncertainty rejected; no final status'
            elif table=='lab' and field in ['labname','labresult','labmeasurenamesystem','labmeasurenameinterface']:
                include=True;available=False;time='labresultrevisedoffset, with draw <= revised <= T0'
                reason='only screened creatinine, potassium, sodium, lactate, BUN, Hgb and glucose with explicit recognized units; inverted times/conflicting revisions blocked'
                transform='latest drawn eligible specimen then revision; never backfill a later result to draw time'
            elif table.startswith('apache') or field=='apacheadmissiondx':
                reason='APACHE capture time/first-day aggregation uncertain; admission name does not prove availability at ICU index; some fields encode future outcomes'
            elif table in ['medication','admissiondrug','infusiondrug','treatment']:
                reason='excluded from primary features; order is not administration, later stop/cancel status and post-event therapy leak; first positive infusion record is not proven initiation'
                time={'medication':'drugorderoffset','admissiondrug':'max(drugoffset,drugenteredoffset)','infusiondrug':'infusionoffset','treatment':'treatmentoffset'}[table]
            elif table=='intakeOutput':
                reason='audit only; repeated cumulative totals must not be summed across cell rows; units/aggregation need review'
                time='max(intakeoutputoffset,intakeoutputentryoffset)'
            rows.append({'feature':field,'source':table,'available_at_index_time':available,'availability_time':time,
                         'include_in_baseline':include,'reason':reason,'transformation':transform,
                         'cutoff_rule':'observed_time <= T0 and available_time <= T0; dynamic data <= T0-5 minutes' if include else None})
    return {'version':'phase3_5_v1','prediction_time':'ICU-relative minute T0','rows':rows,
            'approved_raw_sources':'exact allowlist; inclusion never means an entire source row can be flattened',
            'quality_meaning':'VALID means numeric/temporal screen passed, not bedside clinical validation',
            'future_outputs_separate':True,'wearable_claim':False}


def write_doc(docs,name,lines):
    (docs/name).write_text('\n'.join(lines)+'\n')


def render(root='data/processed/eicu_demo',docs='docs'):
    root=Path(root);docs=Path(docs);r=json.loads((root/'report.json').read_text());m=json.loads((root/'mimic_audit.json').read_text());c=r['counts'];decision=r['decision']
    archive=json.loads((root/'manifest.json').read_text())['inputs']['archive_sha256']
    contract=availability_contract();(docs/'EICU_FEATURE_AVAILABILITY.json').write_text(json.dumps(contract,sort_keys=True,indent=2)+'\n')
    audit=['# eICU data audit — Phase 3.5','',f'Archive: `Datasets/aibhs_datasets-main.zip`; SHA256 `{archive}`. Original ZIP remains unchanged. Only 16 eICU and 17 MIMIC compressed CSV tables plus licensing/checksum metadata are extracted into ignored `data/raw/phase3_5/`. The included SQLite database, unrelated cohorts and free-text notes are not extracted.',
           '',f"Actual eICU files: {c['patients']} patient IDs, {c['hospital_stays']} hospital stays and {c['icu_stays']} ICU stays. Adults: {c['adult_patients']} patients / {c['adult_stays']} ICU stays. Four missing-age and four pediatric stays are excluded from adult risk sets. {c['topcoded_age_stays']} stays record age >89; no exact age is invented.",
           '',f"{c['repeated_stay_patients']} patients have repeated ICU stays. All stays link through patientunitstayid → patienthealthsystemstayid and uniquepid. Patient membership uses namespaced uniquepid, never ICU ID alone. Hospital ordering across separate admissions in the same year is not reliably established; histories are not concatenated across stays. [Source timing/identity documentation]({SOURCES['patient']}).",
           '',f"The files contain {c['coded_hospital_ids']} distinct coded hospital IDs and {c['coded_ward_ids']} coded ward IDs. This contradicts the source landing-page description of 20 hospitals and removed site IDs. Extracted primary files match the packaged supplier checksums; patient, hospital, diagnosis and vitalPeriodic hashes also match the official 2.0.1 checksum listing. Report actual coded IDs without identifying hospitals or claiming that the landing-page site description is verified. [Source description]({SOURCES['eicu']}).",
           '',f"Age quantiles (min, 1%, 25%, median, 75%, 99%, max), excluding topcoding: {c['age_quantiles_not_topcoded']}. ICU length-of-stay hours: {c['unit_length_hours_quantiles']}.",
           '', '## Demographics and recorded outcomes','', '| Field | Aggregate distribution |','|---|---|']
    for k,v in c['demographics'].items():audit.append(f'| {k} | {v} |')
    audit+=['', 'Mortality/discharge distributions count ICU-stay rows; repeated hospital outcomes are not independent patients. They are descriptive outcomes only, never predictors. Negative hospital-admit offsets are expected; all offsets are centered on the individual ICU admission.',
            '', '## Clinical coverage','', '| Table | Rows | Patients | ICU stays | Missing times | After-discharge entries | Inverted/negative time counter |','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in r['tables'].items():audit.append(f"| {k} | {v['rows']} | {v['patients']} | {v['stays']} | {v['missing_timestamps']} | {v['after_discharge_entries']} | {v['negative_or_inverted_time_count']} |")
    audit+=['', 'All extracted clinical and vital rows have valid stay references; duplicate primary keys fail ingestion. Negative history offsets can be valid pre-ICU observations. Negative lab revision-minus-draw intervals are distinct inconsistencies and are blocked from causal lab features.',
            '', '## Disease evidence','', '| Documented condition | Patients | ICU stays |','|---|---:|---:|']
    for k,v in c['condition_counts'].items():audit.append(f"| {k} | {v['patients']} | {v['stays']} |")
    audit+=['', 'HF rule: affirmative “heart failure”, “congestive heart failure” or CHF in structured diagnosis/history/admission-Dx text; ICD-9 428.*, 402.*1, 404.*1/*3; ICD-10 I50.*, I11.0, I13.0, I13.2. Cardiomyopathy alone is not HF. Negated/uncertain text (without/not due to/no CHF, rule-out, possible, suspected, not obtainable) is excluded. The field named icd9code contains both ICD-9 and ICD-10 values. Exact mapping implementation is `conditions()` in src/data/adapters/eicu.py.',
            '', 'HF history requires an affirmative pastHistory HF entry; HF from active diagnoses alone can be acute and does not prove chronic HF. Hypertension excludes pulmonary/portal hypertension; diabetes uses explicit diabetes text or 250/E10/E11/E13 families; CKD uses explicit chronic renal disease or 585/N18 codes and excludes kidney stones. Broader CV includes affirmative cardiovascular diagnosis/history branches and HF/HTN. Definitions are documented-evidence phenotypes, not clinical adjudication.',
            '', '“Ever documented” counts above are retrospective ascertainment. Predictors and HF/CV evaluation risk sets use only entries available by T0. Future diagnosis/history entries cannot retroactively define an index-time HF feature. A missing entry remains unknown, not absence of disease.',
            '', '## Physiology: actual cadence and QC','', 'Periodic measurements are archived bedside-monitor medians and are not bedside-validated; monitor SpO2 is stored under sao2. The source distinguishes systemic invasive BP from cuff non-invasive BP. These are ICU monitors, not consumer wearables. [Periodic source]('+SOURCES['periodic']+'), [aperiodic source]('+SOURCES['aperiodic']+').',
            '', '| Table / field | Patients / stays | Nonmissing / raw rows | Usable | Missing % | Median interval min | Interval 1% / 99% / max min | Connected hours (gap ≤10 min) | Suspect / invalid |','|---|---:|---:|---:|---:|---:|---|---:|---|']
    for table,a in r['physiology'].items():
        for f,v in a['fields'].items():
            q=v['cadence_minutes_quantiles'];cad='—' if q is None else f'{q[3]:g}';gaps='—' if q is None else f'{q[1]:g} / {q[5]:g} / {q[6]:g}'
            audit.append(f"| {table}/{f} | {v['patients']} / {v['stays']} | {v['nonmissing']} / {v['raw_rows']} | {v['usable_values']} | {100*v['missing_fraction']:.2f} | {cad} | {gaps} | {v['usable_connected_hours_max_gap_10min']:.2f} | {v['quality_counts']['SUSPECT']} / {v['quality_counts']['INVALID']} |")
    audit+=['', 'Missingness is source-row missingness, not a known scheduled-sampling denominator. Connected hours sum adjacent usable intervals ≤10 minutes; calendar span and gap distributions are separate in report.json. Same-time duplicate values are counted once; conflicting values are blocked per signal, never averaged. Raw lexical values remain in immutable source gzip files; packed arrays retain source IDs, raw numeric values, usable values and quality states.',
            '', 'The actual usable invasive MAP median interval is five minutes. Cuff MAP median interval is twenty minutes with long gaps. One-minute averages are not present, so this does not reproduce the PhysioNet AHE endpoint. No gaps are filled, no interpolation occurs, and full-stay statistics never become individual forecast inputs.',
            '', 'Important QC: negative/zero arterial pressures and extreme values; inconsistent SBP/DBP/MAP triplets (MAP outside DBP–SBP with 5 mmHg tolerance or SBP≤DBP); duplicate/conflicting cuff records; measurements before admission/after ICU discharge; temperatures near 99–103 in a Celsius-labelled column; CVP values hundreds of units. No automatic Fahrenheit conversion, CVP rescaling or clipping is performed. CVP and unit-unknown ST/resistance channels are excluded from the initial model. Temperature is usable only within the explicit Celsius plausibility screen; remaining unit ambiguity limits its interpretation.',
            '', '## Static + dynamic overlap','', '| Cohort (adult; ever evidence is retrospective) | Patients / ICU stays | Context + any dynamic patients / stays | Context + arterial MAP patients / stays | Potential 6h MAP patients / stays | Connected MAP hours |','|---|---:|---:|---:|---:|---:|']
    for k,v in r['cohort_overlap'].items():audit.append(f"| {k} | {v['patients']} / {v['icu_stays']} | {v['static_and_any_dynamic_patients']} / {v['static_and_any_dynamic_stays']} | {v['static_and_invasive_map_patients']} / {v['static_and_invasive_map_stays']} | {v['potential_6h_map_patients']} / {v['potential_6h_map_stays']} | {v['invasive_map_connected_hours']:.2f} |")
    audit+=['', 'Potential six-hour coverage is a coarse screen (≥60 points and ≥6h span); complete prediction eligibility additionally checks local gap/coverage, stability, future ascertainment and lead-time rules. Labels and predictors are physically separate JSONL files. No model is trained.',
            '', '## Medications, labs and fluids','',
            'Medication rows represent orders, not proven administrations. Infusion rates have inconsistent dose/concentration units; first positive pressor recording is not proven initiation. Post-T0 therapies, stop/cancel states and APACHE summaries are excluded. [Medication semantics]('+SOURCES['medication']+'), [infusion semantics]('+SOURCES['infusion']+').',
            '',f"Lab rows: {r['tables']['lab']['rows']}; revision-minus-draw interval quantiles: {r['tables']['lab']['entry_lag_minutes_quantiles']}; inverted intervals: {r['tables']['lab']['negative_or_inverted_time_count']}. Selected labs require draw≤revision≤T0, recognized units, broad plausibility screens and no conflicting same-time result. Later revisions are not backfilled to draw time. [Lab timing]({SOURCES['lab']}).",
            '', 'Fluid records contain cell-specific values and repeated cumulative intake/output/net totals. Summing cumulative totals across rows would double-count. They are available for audit but excluded from initial model features until unit and aggregation review. Top medication/lab/IO terms and all recorded unit strings remain in the private aggregate report.',
            '', 'Artifacts and commands are specified in PHASE3_5_DECISION.md.']
    write_doc(docs,'EICU_DATA_AUDIT.md',audit)
    target=['# Hemodynamic target feasibility','',
            'This evaluates label/cohort feasibility, not predictive performance. No model accuracy, discrimination or improvement from EHR fusion has been established.',
            '', '## Candidate definitions and ranking','',
            '1. **Sustained recorded invasive MAP <65 for ≥30 observed minutes**: recommended moderate-hypotension research proxy. Seven or more strictly low archived observations, adjacent gaps ≤5 minutes, and first-to-last span ≥30 minutes. It uses the same strict quality/duration rules as the severe candidate and provides enough distinct positive patients for a cautious broader-ICU pilot. The clinical relevance of 65 has critical-care context, but a sepsis treatment target is not an HF-specific adverse-event diagnosis. [Clinical context]('+SOURCES['map65']+').',
            '2. **Sustained MAP ≤60**: severe-hypotension sensitivity endpoint; more specific severity but insufficient eligible positive patients in this demo.',
            '3. **Drop ≥20 to sustained MAP <65**: requires a prior-hour median ≥80, ≥10 prior points and adequate prior support; less common and depends on an unvalidated baseline/severity choice.',
            '4. **Repeated cuff MAP <65**: ≥3 readings over ≥30 minutes, adjacent gaps ≤15 minutes. Sparse/irregular spot readings cannot certify continuously sustained hypotension; rejected as the primary endpoint despite the highest episode count.',
            '', 'PhysioNet AHE uses one-minute MAP with ≥90% low measurements in a ≥30-minute period. Five-minute medians can conceal within-bin excursions; seven consecutive low archived values across 30 minutes are an eICU-specific proxy, not reproduction of that definition. [Original AHE definition]('+SOURCES['ahe']+').',
            '', '| Candidate | Episodes / affected patients | Qualifying low runs | HF-ever / HF-known-at-onset episodes | Recurrent patients | Observed low-run duration min: median / max |','|---|---:|---:|---:|---:|---|']
    for n,v in r['candidates'].items():
        q=v['run_observed_duration_minutes_quantiles'];target.append(f"| {n} | {v['episodes']} / {v['affected_patients']} | {v['qualifying_low_runs']} | {v['hf_ever_episodes']} / {v['hf_known_at_onset_episodes']} | {v['patients_with_recurrence']} | {q[3] if q else '—'} / {q[-1] if q else '—'} |")
    target+=['', '## Decision matrix (3h forecast / 6h history)', '', '| Rank / target | Eligible / positive patients | Positive windows / controls | Window prevalence | Temporal validity and label reliability | EHR + dynamic fusion | HF relevance / clinical importance | Leakage risk / honest evaluation / demonstration |','|---|---:|---:|---:|---|---|---|---|']
    judgments={
      'map65_30min':('1','Strict sustained 5-min recorded proxy; live latency uncertain','Same-person causal context + arterial MAP','Moderate all-cause hypotension; HF subgroup exploratory','Buffered as-of inputs; sufficient broad-pilot support; no clinical benefit evidence'),
      'map60_30min':('2','Same duration/cadence; severe recorded proxy','Same-person causal context + arterial MAP','More severe hypotension; too few eligible HF events','Same leakage controls; insufficient positive-patient support for credible held-out development'),
      'drop20_map65':('3','Prior-hour support required; baseline/drop choice unvalidated','Same-person causal context + arterial MAP','Relative fall may matter; not HF-specific','Causal baseline only; very small positive cohort, demonstration only'),
      'cuff65_30min':('4','Irregular spot values do not certify continuous low MAP','Same-person context + irregular cuff data','Hypotension relevant; label reliability weak','Duplicates/gaps especially consequential; reject primary despite highest episode count')}
    for n in ['map65_30min','map60_30min','drop20_map65','cuff65_30min']:
        a=r['candidates'][n]['window_matrix']['h3_l6']['full'];rank,valid,fusion,relevance,risk=judgments[n]
        prev='—' if a['window_prevalence'] is None else f"{a['window_prevalence']:.2%}"
        target.append(f"| {rank} / {n} | {a['patients']} / {a['positive_patients']} | {a['positive_windows']} / {a['controls']} | {prev} | {valid} | {fusion} | {relevance} | {risk} |")
    target+=['', 'Separate qualifying low runs within 120 minutes are clustered into one episode for recurrence accounting; an episode span is not continuous low duration. Events still near ICU admission or in sparse monitoring can be counted descriptively but lack eligible pre-onset windows. Unvalidated monitor medians, arterial-line selection, nuisance artefacts, treatment and individualized MAP targets limit label reliability. No organ injury or HF decompensation is adjudicated by these labels.',
             '', '## Window protocol','',
             'For each candidate evaluate 1/3/6-hour horizons and 6/12/24-hour histories. Landmarks are separated by their horizon length and anchored to ICU admission, so forecast intervals do not overlap within a configuration. A positive episode contributes at most one positive landmark per configuration; multiple episodes may share a forecast interval. Histories can overlap within a patient, so inference and uncertainty must remain patient/episode grouped.',
             '', 'Features end at T0−5 minutes; clinical context/labs must be available ≤T0. History MAP coverage ≥80%, future MAP coverage ≥90% through T0+horizon+30 minutes, and bounded gaps (≤10 minutes for arterial MAP). Label confirmation requires the extra 30 minutes even when onset is near the forecast boundary. Cuff sensitivity windows use 15-minute support/gap limits and are still spot-reading proxies.',
             '', 'No known MAP <65 in the previous 120 minutes (through the buffered feature cutoff): a causal conservative washout after even transient low readings. Already-underway episodes are excluded. Onset must be >T0+15 minutes and ≤T0+horizon; early-onset windows are excluded, never labelled negative. Missing future support or discharge before confirmation is unknown/censored, never a control. No event/recovery summaries enter predictors.',
             '', '## Full horizon/lookback matrix','', '| Candidate | Horizon / lookback h | Eligible patients | Positive patients | Positive windows | Controls | Window prevalence | HF-known patients / positives / positive windows | CV-known patients / positive windows |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for n,v in r['candidates'].items():
        for k,stats in v['window_matrix'].items():
            a=stats['full'];hf=stats['hf_known'];cv=stats['cv_known'];prevalence='—' if a['window_prevalence'] is None else f"{a['window_prevalence']:.2%}"
            target.append(f"| {n} | {k} | {a['patients']} | {a['positive_patients']} | {a['positive_windows']} | {a['controls']} | {prevalence} | {hf['patients']} / {hf['positive_patients']} / {hf['positive_windows']} | {cv['patients']} / {cv['positive_windows']} |")
    target+=['', 'All exclusion reasons, recurrence and duration distributions, retrospective HF counts and controls are in private report.json. Window prevalence is conditional on this observed stable risk set; it is not population incidence. Controls mean no qualifying target episode within an adequately observed horizon, not healthy physiology.',
             '', '## Explicit leakage findings','',
             '- AdmissionDx can be entered during the first 36 hours; admission naming does not imply known at ICU arrival. Only timestamped entries ≤T0 enter.',
             '- Future diagnoses/history entries inflate retrospective HF ascertainment. HF evaluation uses documented HF by T0; chronic-HF history is separately reported.',
             '- Lab draw time precedes availability, sometimes by hours/days; inverted revision times occur. Gate on revised entry time and reject inverted times.',
             '- APACHE first-day/untimed summaries, final mortality and discharge status leak future knowledge; all are excluded.',
             '- Medication orders are not administration; later stop/cancel fields and pressor responses are future treatment information. They are excluded from predictors, and first infusion record is not used as an adverse-event label.',
             '- Duplicate cuff rows/conflicting same-time readings can manufacture sustained events unless consolidated/blocked. No averaging or gap interpolation is allowed.',
             '- Repeated stays and overlapping histories inflate apparent sample size. Split by uniquepid, retain all stays in one partition, and bootstrap patients/episodes.',
             '- Requiring future observation/discharge support selects a monitored subgroup. This is ascertainment/selection bias, not a feature; it limits deployment generalizability.',
             '', f"Recommended primary configuration is {PROTOCOL['primary_candidate']}, 3h horizon / 6h lookback. It was selected on clinical/timing feasibility before any model score. The initial severe ≤60 configuration fails positive-patient support; moderate <65 retains identical temporal safeguards. Higher-count cuff episodes are rejected for lower reliability. No label was optimized against downstream performance."]
    write_doc(docs,'HEMODYNAMIC_TARGET_FEASIBILITY.md',target)
    mimic=['# MIMIC-IV demo integration','',f"Actual files contain {m['patients']} patients, {m['hospital_stays']} admissions, {m['icu_stays']} ICU stays and {m['patients_with_icu']} patients with ICU data. All checked subject/admission/stay references resolve. This is integration/schema evidence only, never evidence of clinical model performance.",
           '', '| Extracted table | Rows | Patients | Admissions | ICU stays | Available timestamp columns |','|---|---:|---:|---:|---:|---|']
    for name,v in m['tables'].items():mimic.append(f"| {name} | {v['rows']} | {v['patients']} | {v['hospital_stays']} | {v['icu_stays']} | {', '.join(v['timestamp_fields'] or [])} |")
    mimic+=['', '## Canonical migration map','', '| eICU concept | Canonical Twin representation | MIMIC-IV equivalent / availability |','|---|---|---|',
      '| uniquepid / hospital stay / ICU stay | Patient identity / Encounter links, separately namespaced | subject_id / hadm_id / stay_id |',
      '| age, gender | PatientClinicalContext, index-semantic demographics | patients.anchor_age/gender; anchor_year + admission year for age under documented deidentification policy; preserve topcoding |',
      '| timed diagnosis/history | Condition evidence with recorded availability | diagnoses_icd + d_icd_diagnoses are billing diagnoses without recorded availability: exclude current-admission codes from index features; consider genuinely prior completed admissions only |',
      '| unit/hospital admission | Encounter context | admissions.admittime and icustays.intime; derive ICU-relative minutes independently per stay |',
      '| lab draw / revised result | Observation with observed_time and available_time | labevents.charttime / storetime + itemid→d_labitems; preserve valueuom and source values |',
      '| invasive and cuff BP, HR, SpO2, respiration | Observation, separate modalities and units | icu.chartevents.charttime / storetime + itemid→d_items; do not pool invasive/cuff MAP |',
      '| medication orders | MedicationRequest, not administration | prescriptions/pharmacy; order/verification timing must be resolved |',
      '| medication/infusion administration | MedicationAdministration/intervention event | emar/emar_detail charttime/storetime; inputevents starttime/endtime/storetime. Future stop/status information must not enter historical snapshots |',
      '| intake/output | Measurement/event with explicit units | outputevents and inputevents; cumulative totals cannot be summed blindly |',
      '| death/discharge/APACHE outcomes | Outcome separate from features | admissions.dischtime/deathtime/hospital_expire_flag, patients.dod; always exclude future fields |',
      '', '## Actual physiological dictionary mappings','', '| Concept | Item ID | Dictionary label | Recorded units | Measurements | Patients / ICU stays |','|---|---:|---|---|---:|---:|']
    for v in m['physiology_mapping']:mimic.append(f"| {v['canonical_concept']} | {v['itemid']} | {v['label']} | {v['recorded_units']} | {v['measurements']} | {v['patients']} / {v['icu_stays']} |")
    mimic+=['',f"Chart-to-store delay quantiles (minutes): {m['chart_to_store_delay_minutes_quantiles']}; negative delays: {m['negative_store_delays']}. Storetime may precede charttime: ingestion must flag contradictions and require both timestamps ≤cutoff, not blindly trust a draw/chart timestamp. Explicit Fahrenheit/Celsius items can later be converted with provenance, unlike ambiguous eICU temperatures.",
            '', 'Compatibility auditing verifies actual columns and cross-table identities; it is not a complete FHIR/UCUM mapping, terminology certification or full MIMIC adapter. The demo omits clinical notes. Preserve source coded data, units, item dictionaries, timestamps, status fields and namespaces when upgrading to full data. [Official module documentation]('+SOURCES['mimic']+').']
    write_doc(docs,'MIMIC_DEMO_INTEGRATION.md',mimic)
    primary=decision['primary'];hf=decision['hf_known_primary'];fusion=decision['causal_clinical_plus_dynamic_overlap']
    lines=['# Phase 3.5 decision','',f"**{decision['status']}** — "+('sufficient structural evidence for cautious predictive-model development on the broader monitored adult ICU population.' if decision['status']=='GO' else 'insufficient positive-patient support for predictive development with the current configuration.'),
      '', 'This is a label/cohort feasibility decision. Whether historical EHR + dynamic physiology improves prediction remains unproven. No final event predictor has been trained; no predictive performance is claimed.',
      '', '| Configuration | Recommendation |','|---|---|',
      '| Condition | Heart failure remains the product/evaluation condition; broader adult monitored ICU population for physiological event-model development |',
      '| Target adverse event | Recorded sustained moderate invasive hypotension; all-cause hemodynamic proxy, not adjudicated HF deterioration or organ injury |',
      '| Exact label definition | Invasive systemicmean <65 mmHg at ≥7 consecutive usable timestamps, gaps ≤5 min, first-to-last low span ≥30 min; onset is first low entry, confirmed only after 30 min. Runs ≤120 min apart form one recurrent episode |',
      '| Primary prediction horizon | 3 hours, onset strictly >T0+15 min and ≤T0+180 min |',
      '| Secondary horizons | 1h and 6h; severe MAP ≤60 and rapid-drop variants are sensitivity targets, not replacement headline endpoints |',
      '| Lookback window | 6 hours ending T0−5 min; 12h/24h sensitivity analyses with their own eligible cohorts |',
      '| Eligible population | Age ≥18 or recorded >89; static demographics/admission context and adequately observed arterial MAP; complete observed future horizon +30min confirmation |',
      f"| HF evaluation subset | {hf['patients']} patients / {hf['icu_stays']} ICU stays with HF evidence available by T0; {hf['positive_patients']} positive patients. Chronic-HF history is a stricter separate phenotype; prospective clinical HF validation remains unsupported |",
      '| Static EHR features | Age with topcoding, gender, unit type/admission source; documented HF/history, HTN, diabetes, CKD and CV evidence by T0; latest causally available screened creatinine/K/Na/lactate/BUN/Hgb/glucose with units |',
      '| Dynamic physiological features | HR, invasive SBP/DBP/MAP, SpO2, respiration, quality-screened Celsius temperature; past-only summaries/trend, observation counts, missingness and staleness. CVP/unknown-unit channels and cuff pooling excluded |',
      '| Exclusions | Unknown/adolescent age; impossible/suspect/conflicting vital values; inadequate history/future support; already-underway/early-onset events; untimed APACHE, future diagnoses/labs, discharge/mortality, post-T0 therapies and final drug status |',
      '| Washout rule | No usable MAP <65 in previous 120 min through the buffered cutoff. This is conservative and excludes transient lows as well as confirmed events |',
      f"| Expected number of patients | {primary['patients']} eligible patients / {primary['icu_stays']} ICU stays / {primary['windows']} windows; windows are not independent patients |",
      f"| Expected positives | {primary['positive_patients']} distinct patients; {primary['positive_windows']} primary positive windows; conditional window prevalence {primary['window_prevalence']:.2%}, patient-positive prevalence {primary['patient_event_prevalence']:.2%} |",
      f"| Expected controls | {primary['controls']} adequately observed negative windows; {primary['negative_only_patients']} negative-only patients |",
      '| Split strategy | Fixed seed 2026; patient-level 60/20/20 stratified by any primary positive window; every ICU/hospital stay of a uniquepid inherits one role. No cross-partition window splitting |',
      '| Primary metrics | AUPRC alongside prevalence; patient/episode-averaged evaluation, AUROC, sensitivity/specificity/PPV at validation-chosen thresholds, false alerts per observed patient-hour and event sensitivity/lead time under a frozen alert protocol |',
      '| Calibration metrics | Brier, reliability curves, calibration intercept/slope; development-only calibration, explicit event prevalence and drift |',
      '| Uncertainty evaluation | Patient-level bootstrap of discrimination/calibration/operating points; episode-aware alert uncertainty; no independent-window bootstrap; small HF subgroup intervals labelled exploratory |',
      '| Major limitations | Only several hundred arterial-MAP patients; 5-min unvalidated medians, ambiguous live latency, selective arterial-line monitoring, treatment confounding, phenotype timing/absence, individual MAP targets, sparse HF held-out events and coded-site metadata contradiction |',
      '| Evidence level | Real-data temporal label/cohort construction and integration feasibility; no learned predictive performance, prospective clinical benefit, consumer-wearable validity or fused-twin validation |',
      '', '## Feasibility gates and frozen partitions','', f'Prespecified support thresholds: {decision["gate_thresholds"]}. These are engineering/pilot support gates, not a statistical power calculation. Gate results: {decision["gates"]}. HF support gate: {decision["hf_evaluation_gate"]}.',
      '', '| Partition | Patients | Positive patients | Positive windows | Controls | HF-known patients / positive patients |','|---|---:|---:|---:|---:|---|']
    for part,v in decision['partitions'].items():
        a=decision['hf_partitions'][part];lines.append(f"| {part} | {v['patients']} | {v['positive_patients']} | {v['positive_windows']} | {v['controls']} | {a['patients']} / {a['positive_patients']} |")
    lines+=['', 'Only cohort-support counts are examined; no model selection or test performance has occurred. Do not tune thresholds, eligibility or label definitions against later test performance. HF subgroup event support remains too small for strong clinical conclusions even if the broad-ICU pilot gate passes.',
      '', '## Dataset roles','', '| Dataset | Role |','|---|---|',
      '| Zigong | Historical HF/EHR benchmark and preserved negative result: AUROC 0.466, AUPRC 0.085, Brier 0.06515; no useful clinical risk claim |',
      '| CHIRON | Independent longitudinal HF summaries, causal personalization and forecasting; no adjudicated clinical deterioration label |',
      '| PerHeart | Home-measurement baselines/trajectories and behavior; no clinical HF-event validation |',
      '| Synthea | FHIR ingestion, synthetic EHR demonstration and engineering tests only |',
      '| eICU Demo | Same-person EHR + ICU monitor physiology, timestamped context and recorded hypotension proxy construction; no consumer-wearable claim |',
      '| MIMIC-IV Demo | Schema/reference integration and full-MIMIC migration preparation only; no clinical-performance evidence |',
      '', f"Beyond demographics, {fusion['patients']} eligible patients / {fusion['icu_stays']} stays / {fusion['windows']} primary windows have at least one causally documented positive condition or screened available lab alongside dynamic arterial MAP. Missing condition documentation remains unknown. This is genuine within-patient integration, not proof of predictive gain.",
      '', 'No patients are paired across these cohorts. The eICU data permit true same-person static/dynamic fusion, but the clinical utility of that fusion must be measured in Phase 4.',
      '', '## Exact next Phase 4','',
      'First perform blinded episode/trace review with a clinician, including normal controls, gaps, artefacts, pressure modality and treatment context. Verify clinical onset/receipt latency and adjudicate ambiguous cases before a final event predictor. Keep the recorded MAP target distinct from HF decompensation. Acquire full eICU or full MIMIC with sufficient timestamped history, arterial-monitor support, HF-positive patients and adjudicated outcomes for credible HF-specific validation.',
      '', 'Then, only under the frozen protocol, compare training-prevalence and past-MAP rule baselines, EHR-only regularized logistic regression, physiology-only baselines and true same-person EHR+physiology fusion. Fit imputation/scaling/calibration on development patients, use validation for operating rules, and touch the eventual test once. Compare patient/episode errors, calibration, alert burden and lead-time distributions; do not claim a useful fused twin from a high window AUROC.',
      '', '## Reproduce and preserve earlier evidence','', '```bash',
      '# Initial build only; use a fresh destination when output already exists:',
      '.venv/bin/python -m src.data.eicu_feasibility',
      '.venv/bin/python -m src.data.eicu_validate',
      '.venv/bin/python -m src.data.eicu_report',
      '.venv/bin/python -m unittest discover -s tests -v',
      '# For an independent repeat build, use a fresh ignored destination:',
      '.venv/bin/python -m src.data.eicu_feasibility --output data/processed/eicu_demo_reproduction',
      '```',
      '', 'Build refuses occupied destinations and preserves raw ZIP/extracted files. Canonical source arrays, clinical contexts, events, all candidate windows, primary predictors, separate labels, split manifest and checksums remain private under data/processed/eicu_demo. Existing Phase 1/2 cohorts and Phase 3 results are untouched.',
      '', 'Phase 3 was frozen before new modules were added: `data/reproducibility/phase3_source/` contains its exact source/tests/dependency version and links to unchanged datasets/docs. Its whole-source hash is expected to differ from the evolving worktree. Replay the original audit from that frozen working directory with the project virtual-environment Python and `-m src.models.audit`; this preserves the negative benchmark without refitting or repeating test evaluation.',
      '', 'See EICU_DATA_AUDIT.md, EICU_FEATURE_AVAILABILITY.json, HEMODYNAMIC_TARGET_FEASIBILITY.md and MIMIC_DEMO_INTEGRATION.md for complete evidence.']
    write_doc(docs,'PHASE3_5_DECISION.md',lines)
    print({'docs':5,'recommendation':decision['status'],'no_predictor_trained':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/processed/eicu_demo');p.add_argument('--docs',default='docs');a=p.parse_args();render(a.root,a.docs)
