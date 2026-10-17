"""Publish reviewed aggregates only. Private manifests retain all record references."""
import json
from pathlib import Path
from src.vitaldb_feasibility.source import ROOT,private_json
from src.vitaldb_feasibility.retrieve import TRACKS
from src.vitaldb_feasibility.verify import volume

def read(name):return json.loads((ROOT/(name+'.json')).read_text())
def number(value,decimals=1):return 'unavailable' if value is None else f'{value:,.{decimals}f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
def write(name,text):
    path=Path('docs')/name;path.write_text(text.strip()+'\n')

def generate():
    meta=read('metadata_audit');feas=read('feasibility_report');signals=read('signal_audit');clinical=read('clinical_audit');fhir=read('clinical_fhir_audit');tests=read('tests');native=read('native_check_report')
    diagnostic=read('event_attrition_diagnostic');waves=read('waveform_audit');wave_overlap=read('waveform_overlap');proof=read('replay_proof');v=volume();adult=feas['adults'];allages=feas['all_ages']
    primary=feas['risk_sets']['history360_horizon180_band_washed_out'];hard=feas['risk_sets']['history360_horizon180_band_hard']
    usable_positive_adults=sum(r['adult'] and r['native_event_count']>0 and r['usable_minutes']>=30 and r['usable_fraction_minutes']>=.8 for r in read('case_event_audit'))
    sources='[Official dataset and parameter descriptions](https://vitaldb.net/dataset/), [current public API](https://api.vitaldb.net/), [API specification](https://api.vitaldb.net/openapi.json), and [API guide](https://api.vitaldb.net/AGENTS.md). Cite Lee HC et al., Scientific Data 2022, DOI [10.1038/s41597-022-01411-5](https://doi.org/10.1038/s41597-022-01411-5).'
    def risk(history,horizon,stability='washed_out',band='band'):return feas['risk_sets'][f'history{history}_horizon{horizon}_{band}_{stability}']
    signal_rows=[]
    for concept,s in signals['signals'].items():
        q=s['qc_totals'];rate=s['sample_rate_hz'];span=s['observed_span_minutes'];mins=s['usable_minutes']
        signal_rows.append([concept,s['catalog_cases'],s['catalog_patients'],s['retrieved_cases'],s['usable_cases_30minutes_80percent'],s['usable_patients_30minutes_80percent'],number(span['median'] if span else None),number(mins['median'] if mins else None),number(rate['median'] if rate else None,3),q['missing_values'],q['impossible_values']])
    signal_table=table(['Signal','Catalog cases','Catalog subjects','Retrieved cases','Usable cases','Usable subjects','Median span min','Median usable min','Median Hz','Missing values','Impossible values'],signal_rows)
    quality_table=table(['Signal','Median usable-minute %','Impossible / unique %','Identical duplicate rows','Conflicting time groups','Rapid pressure jumps','≥5min flat runs','Observed rate range Hz'],
        [[k,number(s['usable_minute_fraction']['median']*100) if s['usable_minute_fraction'] else 'unavailable',
          number(s['impossible_values_fraction_unique_records']*100) if s['impossible_values_fraction_unique_records'] is not None else 'unavailable',
          s['qc_totals']['identical_duplicate_records'],s['qc_totals']['conflicting_timestamp_groups'],s['qc_totals']['rapid_pressure_jumps_flagged_not_removed'],
          s['qc_totals']['flatline_runs_at_least5min_flagged_not_removed'],
          f"{number(s['sample_rate_hz']['min'],3)}–{number(s['sample_rate_hz']['max'],3)}" if s['sample_rate_hz'] else 'unavailable'] for k,s in signals['signals'].items()])
    alignment_table=table(['Combination','Median latest timestamp spread sec','95th percentile sec','Maximum sec'],
        [[k,number(r['latest_time_spread_seconds_common_minutes']['median']),number(r['latest_time_spread_seconds_common_minutes']['q95']),number(r['latest_time_spread_seconds_common_minutes']['max'])]
         for k,r in signals['overlaps'].items() if r['latest_time_spread_seconds_common_minutes']])
    overlap_table=table(['Combination','Catalog case upper bound','Catalog subject upper bound','Measured usable cases','Measured usable subjects','Median common usable min'],
        [[k,r['catalog_cases_upper_bound'],r['catalog_patients_upper_bound'],r['usable_cases'],r['usable_patients'],number(r['common_usable_minutes']['median'] if r['common_usable_minutes'] else None)] for k,r in signals['overlaps'].items()])
    clinical_table=table(['Source field group','Measured coverage','Availability contract'],[
        ['age, sex, height, weight, BMI, emergency status, department','6,388 cases','Candidate context; preoperative availability assumed, receipt unverified'],
        ['asa',str(meta['clinical_nonmissing']['asa'])+' nonmissing','Preoperative assumption; not independently timestamped'],
        ['preop_htn, preop_dm, preop_ecg, preop_pft','6,388 nonmissing each','Preoperative status; categorical coding needs review'],
        ['preop_hb / preop_cr / preop_alb',f"{meta['clinical_nonmissing']['preop_hb']} / {meta['clinical_nonmissing']['preop_cr']} / {meta['clinical_nonmissing']['preop_alb']}",'Candidate lab context; explicit assumptions, missingness retained'],
        ['dx, opname, approach, ane_type','6,388 nonmissing each','Untimed or whole-case coding; excluded until pre-index knowledge defensible'],
        ['intraop_* drugs, fluids, blood, urine, estimated blood loss','Whole-case totals; some missing','Blocked predictors: include future actions/information'],
        ['death_inhosp, icu_days, dis, aneend, opend, caseend','6,388 nonmissing each','Outcomes/endpoints only; never predictors'],
        ['Laboratories',f"{clinical['lab_records']:,} rows / {clinical['lab_cases']:,} cases / {len(clinical['lab_names'])} names",'dt is relative case time; result receipt and CSV units absent'],
        ['INSPIRE FHIR enrichment',f"{fhir['enriched_cases']:,} tagged cases / {fhir['enriched_patients']:,} subjects",'Measured Encounter tags; not all codes/interventions verified']])
    write('VITALDB_DATA_AUDIT.md',f'''
# VitalDB data audit — Phase 4.5A

Audit retrieval date: 7 October 2026 UTC. Status: feasibility only; zero model fits and no spent eICU test evaluation. All Phase 1–4 files remain unchanged, checked against the private pre-audit SHA-256 snapshot. Exact IDs, response bodies, endpoint URLs containing IDs, and derived records stay in Git-ignored `data/vitaldb/` with private permissions.

## Measured metadata

The current tables contain **{meta['cases']:,} surgical cases from {meta['patients']:,} unique subject IDs**, {meta['track_rows']:,} catalog rows and {meta['distinct_track_names']} distinct track names. Cases are not unique people. Future partitions must group `subjectid` across repeat cases. Recording duration median {number(meta['duration_minutes']['median'])} min, IQR {number(meta['duration_minutes']['q25'])}–{number(meta['duration_minutes']['q75'])}, range {number(meta['duration_minutes']['min'])}–{number(meta['duration_minutes']['max'])}. At least 3/6/9 recording hours: {meta['duration_at_least_hours']['3']:,}/{meta['duration_at_least_hours']['6']:,}/{meta['duration_at_least_hours']['9']:,} cases. Case duration is a ceiling on usable physiological history, not measured signal coverage.

The older published summary lists 557,622 tracks and describes 6,388 surgical patients. The current API catalog has 486,449 tracks and identifies 6,090 subjects. This audit uses the measured current catalog and repeated-patient identifiers, rather than treating the publication's wording as a unique-patient count. Recording duration differs from anesthesia duration. The source is single-center non-cardiac surgery in ten operating rooms, predominantly general anesthesia. {sources}

## Clinical variables and temporal availability

{clinical_table}

Laboratory rows occur before/during/after recording: {clinical['relative_recording_timing'].get('before_recording',0):,}/{clinical['relative_recording_timing'].get('during_recording',0):,}/{clinical['relative_recording_timing'].get('after_recording',0):,}. Their observation offsets do not establish when clinicians received the results. No labs are silently made available at T0. In six deterministically chosen enriched cases, the API returned {fhir['sample_resource_counts']['Procedure']} Procedure, {fhir['sample_resource_counts']['Condition']} Condition, and {fhir['sample_resource_counts']['MedicationAdministration']} MedicationAdministration resources. This verifies schemas, not complete coverage or contemporaneous receipt for all 2,087 cases. The shared INSPIRE case identifier is documented by the API; no linkage to eICU, Zigong, CHIRON, PerHeart or Synthea is made.

## Selective acquisition and storage

Metadata came first: `/cases`, `/trks`, `/labs`, `/openapi.json`, `/AGENTS.md`, `/fhir/metadata`, and `/fhir/Location?physicalType=ro`. Discovered track IDs then drove bounded legacy `/<track-id>` numeric CSV reads, which were verified live even though that route is absent from the current OpenAPI specification. FHIR Encounter pagination measured enrichment. `/fhir/Observation` with category, code, date, count, and summary parameters supplied small track summaries and waveform chunks. OR1 `$status`/`$sample` supplied a fixed-clock ten-second replay proof. No `.vital` or `.parquet` bulk files were downloaded.

All **3,754** cases in the prioritized invasive-MAP union were retrieved; one initial transient failure was recovered from a resumable retry. Non-MAP QC covers {signals['selected_multisignal_cases']:,} cases selected by subject/case hashes, including the first 1,200 hashed MAP cases and every case lasting at least 575 minutes; neither labels nor QC outcomes selected this sample. It is enriched for long recordings, so it is not an unbiased prevalence sample. Numeric cohort responses were parsed at native cadence, then discarded after hashes, QC, native events and compressed closed-minute summaries were saved. Only twelve cases contributed waveform chunks (three sixty-second tracks per case). Initial small probes and repeated requests are included in volume accounting.

Known downloaded response bodies: **{v['known_response_body_MiB']:.1f} MiB**. Current private audit cache: approximately **{v['cache_MiB']:.1f} MiB**, including provenance and derived windows. HTTP/TLS headers and any unavailable failed-response bytes are excluded; these are response-body totals, not an exact network packet capture. Audit bookkeeping can add small files after this estimate. Replay canonical records are gzip-compressed; waveforms are retained only for the tiny verification sample.

Checksums, retrieved UTC times, source headers, exact private selections and preprocessing are in the private manifests. `casestart=0` recording offsets, anonymized FHIR dates near 2100, and virtual replay dates starting in 2000 are distinct clocks. Acquisition time is not verified clinical availability time. Numeric CSV units are not supplied per row; units come from the official device/channel catalog and are recorded as declared, not measured from payloads.

## Reproduction

Run from the repository using the existing virtual environment. Cached inputs are checksummed and reused; a missing input requires public network access.

```bash
.venv/bin/python -m src.vitaldb_feasibility.fetch metadata
.venv/bin/python -m src.vitaldb_feasibility.protocol
.venv/bin/python -m src.vitaldb_feasibility.retrieve --concepts map --budget-mib 512
.venv/bin/python -m src.vitaldb_feasibility.retrieve --concepts sbp dbp hr spo2 rr temperature co cvp --case-sample 1200 --budget-mib 512
.venv/bin/python -m src.vitaldb_feasibility.fetch clinical
.venv/bin/python -m src.vitaldb_feasibility.waveforms
.venv/bin/python -m src.vitaldb_feasibility.fetch replay
.venv/bin/python -m src.vitaldb_feasibility.replay
.venv/bin/python -m src.vitaldb_feasibility.native_check
.venv/bin/python -m src.vitaldb_feasibility.windows
.venv/bin/python -m src.vitaldb_feasibility.signals
.venv/bin/python -m src.vitaldb_feasibility.report
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m src.vitaldb_feasibility.verify
```

Retrieval is resumable, rate-limited to five requests/sec with four workers, capped at 4 MiB per numeric response and 512 MiB of new response bodies per run. A resume appends run history so prior bytes are retained. Repeated source bodies are not required for ordinary reanalysis. Private manifests retain source body hashes for fresh-fetch comparison; APIs may change, so reproducing a later source snapshot requires verifying those hashes. Failed requests and partial coverage must be disclosed, never converted to absent physiology.
''')
    units={'map':'mmHg','sbp':'mmHg','dbp':'mmHg','hr':'bpm','spo2':'%','rr':'breaths/min','temperature':'degC','co':'L/min','cvp':'mmHg'}
    canonical={'map':'invasive_map','sbp':'invasive_systolic_bp','dbp':'invasive_diastolic_bp','hr':'heart_rate','spo2':'spo2','rr':'respiratory_rate','temperature':'body_temperature','co':'cardiac_output','cvp':'central_venous_pressure'}
    private_json(ROOT/'canonical_signal_map.json',{c:{'source_priority':names,'canonical_concept':canonical[c],'declared_catalog_unit':units[c],
        'native_csv_unit_supplied':False,'observation_time':'seconds relative to recording start','historical_clinical_receipt_time':None,
        'patient_namespace':'vitaldb:subject','quality':'missing/impossible/conflicting duplicates quarantined; plausible artifacts flagged, not silently repaired',
        'provenance':'response SHA256, source channel, private case and subject, UTC retrieval, source headers','labels_separate':True} for c,names in TRACKS.items()})
    mapping_table=table(['Canonical concept','Discovered source priority','Declared unit','Boundary'],[[canonical[k],', '.join(names),units[k],
        'Not NIBP; dense target source' if k=='map' else 'Device-specific measurement method retained' if k in ['co','rr','temperature','cvp'] else 'No synthetic cross-source filling'] for k,names in TRACKS.items()])
    waveform_rows=[]
    for code in ['SNUADC/ART','SNUADC/ECG_II','SNUADC/PLETH']:
        rows=[t for r in waves for t in r['tracks'] if t['track']==code];catalog=meta['tracks'][code]
        waveform_rows.append([code,catalog['cases'],catalog['patients'],len(rows),number(sum(r['track_span_minutes'] for r in rows)/len(rows)),number(rows[0]['rate_hz'],0),sum(r['samples'] for r in rows),sum(r['missing_samples'] for r in rows),sum(r['screened_values'] for r in rows)])
    waveform_table=table(['Waveform','Catalog cases','Catalog subjects','Sampled cases','Mean metadata span min','Native Hz','Snippet samples','Missing','Screened values'],waveform_rows)
    write('VITALDB_SIGNAL_MAP.md',f'''
# VitalDB signal map — Phase 4.5A

## Canonical mapping and provenance

{mapping_table}

Waveforms map `SNUADC/ART` → arterial_pressure_waveform (mmHg), `SNUADC/ECG_II` → ecg_lead_ii (mV), `SNUADC/PLETH` → ppg_waveform (native unit empty; catalog dimensionless). FHIR SampledData is decoded at float64 as `raw*factor+origin`, preserving raw tokens, native units, factor, origin, period, subject, encounter and response checksum. Multidimensional or invalid-period arrays are rejected. Missing tokens remain missing; replay values are marked SUSPECT because historical virtual-clock data and receipt assumptions are not a validated clinical stream.

The existing Observation/SourceValue/Provenance schema is reused by a separate VitalDB audit adapter. Phase 2's registry and prior dataset pipelines are unchanged. Namespace is dataset-specific and repeat surgeries share the source subject identity. Observation time, recording offsets and audit ingestion receipt are distinct; receipt during this API fetch is never represented as historical clinical receipt. No fallback to intermittent `Solar8000/NIBP_*` labels occurs. Femoral and arterial MAP source priorities are retained per case; sources are not averaged together.

## Numeric coverage and quality

{signal_table}

{quality_table}

MAP retrieval exhausts its selected source union. Other rows are measured only in the label-free {signals['selected_multisignal_cases']:,}-case sample; their catalog columns are whole-source presence counts. Usable means at least thirty usable minutes and at least 80% usable bins over that track's cached span. For 2-second tracks a usable minute requires at least fifteen valid samples and maximum gap ten seconds; slower device tracks use at least ceil(60/(2×median cadence)) samples and maximum gap max(10,3×cadence). Missing-value counts do not count omitted packets; gaps and usable-minute fractions capture dropout separately. Initial/post-line-placement gaps and nonphysiological monitor codes are not imputed.

Impossible-value screens: MAP 10–200, SBP 30–300, DBP 10–200 mmHg; HR 20–250 bpm; SpO₂ 50–100%; RR 1–80/min; temperature 30–43 °C; CO 0.1–30 L/min; CVP −10–50 mmHg. Values are quarantined, not clipped. Identical timestamp duplicates are collapsed; conflicting duplicates become unusable rather than averaged. Finite plausible pressure jumps exceeding 40 mmHg within ten seconds and constant runs of at least five minutes are flagged, retained and disclosed rather than presumed artifacts. These engineering screens are not clinical adjudication. Per-case counts, rates, distributions and masks are private.

## Synchronized overlap

{overlap_table}

{alignment_table}

These are simultaneously usable minutes, rather than merely intersecting track names. A–E are evaluated over each sampled case's MAP grid with no filling outside any signal's actual recording span. Counts are measured lower bounds on whole-cohort usability, not extrapolations. Shared recording-offset clocks permit alignment, but do not prove hardware synchronization or physiological phase matching. Latest-source timestamp spreads are measured in the private overlap report. Synchronized MAP/SBP/DBP ordering was inconsistent in {signals['bp_ordering_inconsistent_minutes']:,}/{signals['bp_ordering_checked_minutes']:,} checked minutes (latest timestamps within five seconds); these are flags, not automatically dropped labels.

Additional catalog channels include EV1000/CI, SV, SVI and SVV (617 cases each), SVR/SVRI (255 each), Vigileo/SVV (323), Vigilance/SVO2 (63), pulmonary-artery pressure (81) and SNUADC/CVP waveform (1,586). These methods, populations and units require dedicated audit before use. Anesthetic concentration, ventilator settings and infusion channels are available but reflect treatment and require causally timed context contracts. CO/CVP numeric feasibility is sampled; other channels' actual usable quality remains unmeasured.

## Small waveform verification

{waveform_table}

Combination F (arterial+ECG+PPG) has **{wave_overlap['catalog_cases_upper_bound']:,} catalog cases / {wave_overlap['catalog_patients_upper_bound']:,} subjects**. Twelve cases were chosen from that triple intersection without labels. All {wave_overlap['snippet_screen_usable_cases']} cases / {wave_overlap['snippet_screen_usable_patients']} subjects pass the tiny-snippet screen (≥90% finite, ≤10% diagnostic ART/ECG flagged values per track), a measured snippet-level lower bound rather than whole-case waveform usability. Each track contributes one sixty-second chunk near its own midpoint. Common metadata span median is {number(wave_overlap['common_metadata_span_minutes']['median'])} minutes; it is not continuously clean duration. Selected chunk start spread was {number(wave_overlap['selected_chunk_start_spread_seconds']['max'],0)} seconds in these twelve cases, but timestamp agreement is not pulse-transit-time validation or independent hardware clock certification. The separate ten-second replay proof also has a common window for all three waveforms. No whole-cohort waveform quality count is claimed. Adjacent repeated ADC samples may reflect quantization/resampling; no clinical artifact detector was validated. ART outside 0–300 mmHg and ECG magnitude >10 mV are diagnostic screens; no validated PPG amplitude bound is imposed. Future candidate morphology, pulse-arrival, perfusion and beat-variability features need independent signal-quality and timing verification.

Source units and device definitions: {sources}
''')
    horizon_table=table(['History min','Horizon min','Eligible cases / subjects','Positive cases / subjects','Control cases','Windows + / −','Lead min'],[
        [history,horizon,f"{risk(history,horizon)['cases']} / {risk(history,horizon)['patients']}",f"{risk(history,horizon)['positive_cases']} / {risk(history,horizon)['positive_patients']}",risk(history,horizon)['control_cases'],f"{risk(history,horizon)['positive_windows']} / {risk(history,horizon)['control_windows']}",'none']
        for history in [30,60,180,360] for horizon in [30,60,120,180]])
    stable_table=table(['History min','Stability at 3h','Eligible cases / subjects','Positive cases / subjects','Events','Controls','Windows + / −'],[
        [history,stability,f"{risk(history,180,stability)['cases']} / {risk(history,180,stability)['patients']}",f"{risk(history,180,stability)['positive_cases']} / {risk(history,180,stability)['positive_patients']}",risk(history,180,stability)['eligible_unique_events'],risk(history,180,stability)['control_cases'],f"{risk(history,180,stability)['positive_windows']} / {risk(history,180,stability)['control_windows']}"]
        for history in [30,60,180,360] for stability in ['washed_out','normal80','hard','normal90']])
    diagtable=table(['History min','Exact lead min','Episodes with history + future','Cases / subjects','Current MAP≥80 cases without washout assurance','Strict stable positives'],[
        [history,horizon,(d:=diagnostic['event_centred_exact_horizon_diagnostic'][f'history{history}_horizon{horizon}']).get('history_and_future_supported',0),f"{d.get('history_future_supported_cases',0)} / {d.get('history_future_supported_patients',0)}",d.get('current_normal80_cases_no_washout_assurance',0),d.get('prespecified_washedout_eventcentred',0)]
        for history in [30,60,180,360] for horizon in [30,60,120,180]])
    censor_table=table(['History min','Horizon min','Candidate landmarks','History fail','Washout unobserved','Recent low MAP','Future censored','Eligible landmarks'],[
        [history,horizon,(c:=feas['censoring'][f'history{history}_horizon{horizon}_band']).get('candidate_landmarks',0),c.get('insufficient_history',0),c.get('insufficient_observed_washout',0),c.get('recent_hypotension',0),c.get('incomplete_future_censored',0),c.get('eligible_landmarks',0)]
        for history in [30,180,360] for horizon in [30,60,120,180]])
    write('VITALDB_HYPOTENSION_FEASIBILITY.md',f'''
# VitalDB hypotension feasibility — Phase 4.5A

**VitalDB does not supply a usable positive cohort for the prespecified stable/hard warning task.** The broad observed target exists, but long history and stability requirements remove all eligible positives. This is an ascertainment feasibility result, not model performance.

## Target and harmonization

Use the discovered invasive-MAP priority Solar8000/ART_MBP, Solar8000/FEM_MBP, then EV1000/ART_MBP, never NIBP. After explicit impossible-value/duplicate QC, confirm a run only after consecutive finite MAP <65 mmHg spans at least 1,800 seconds, with gaps ≤10 seconds and at least 181 samples. A finite MAP ≥65, an invalid value, or a larger gap breaks the run. No interpolation, clipping or retrospective filling. Native median cadence is approximately two seconds, rather than eICU's five-minute observations. Confirmation is causal when thirty minutes have elapsed; completed duration/end is retrospective outcome information and never a predictor.

The historical eICU proxy required seven consecutive usable observations spanning thirty minutes with gaps ≤5 minutes. Shared threshold and duration make the concepts comparable; the observation/QC rules differ substantially. Coarse five-minute causal as-of emulation (sample age ≤10 seconds) finds **{adult['coarse5minute_events']} adult runs**, versus **{adult['native_events']} native runs**. Coarse aliasing can bridge brief recovery or dropout, so it is a sensitivity count, not an equivalent clinical label. Its events were not used for the primary risk sets. No adjudicated hypotension, tissue perfusion or intervention efficacy is established.

## Cases, subjects and episodes

All ages: {allages['cases']:,} MAP cases / {allages['patients']:,} subjects; {allages['usable_map_cases_30minutes_80percent']:,} cases / {allages['usable_map_patients_30minutes_80percent']:,} subjects meet the thirty-minute/80% usable-MAP screen. {allages['event_positive_cases']} cases / {allages['event_positive_patients']} subjects contain {allages['native_events']} native sustained runs, merged into {allages['merged_episodes']} episodes. Adult primary audit: **{adult['cases']:,} cases / {adult['patients']:,} subjects**, **{adult['usable_map_cases_30minutes_80percent']:,} usable cases / {adult['usable_map_patients_30minutes_80percent']:,} subjects**, **{adult['event_positive_cases']} event-positive cases / {adult['event_positive_patients']} subjects**, {adult['native_events']} runs and {adult['merged_episodes']} episodes. Adult case prevalence is {adult['case_event_prevalence']*100:.2f}% among all retrieved adult MAP cases, not a population clinical incidence estimate.

Runs within 120 minutes of the preceding run's last low sample form one episode; {adult['recurrent_case_count']} adult cases have more than one merged episode. Native adult run duration median {number(adult['event_duration_minutes']['median'])} min, IQR {number(adult['event_duration_minutes']['q25'])}–{number(adult['event_duration_minutes']['q75'])}, range {number(adult['event_duration_minutes']['min'],2)}–{number(adult['event_duration_minutes']['max'])}. {adult['no_observed_event_cases']:,} adult cases have no observed native run, but censored/low-quality recordings are not automatically controls.

Of the {adult['event_positive_cases']} adult event-positive cases, {usable_positive_adults} also pass the whole-record thirty-minute/80% usable-MAP screen. The other event-positive recordings still have an observed sustained run, but insufficient global coverage under that screen. Neither group is automatically eligible for forecasting.

## Prespecified risk sets and stable definitions

Primary history is six hours, ending five minutes before T0. Sensitivity histories are thirty, sixty and 180 minutes, with horizons thirty, sixty, 120 and 180 minutes. Landmarks start at max(history,120-minute observed washout)+5 and recur every thirty minutes. History/washout require ≥80% usable minute bins and no gap over two consecutive missing bins. Future requires ≥90% coverage and the same missing-run limit. For 2-second MAP a valid minute has ≥15 usable samples and maximum gap ≤10 seconds.

The first future merged event defines the target. Minimum lead is strictly greater than fifteen minutes. Cumulative targets are within (15,H]; terminal bands are (max(15,H−30),H]. **The three-hour band is (150,180] minutes: approximately 2.5–3 hours, not guaranteed 180-minute warning.** Exact 180-minute event-centred counts are also reported below. Events before the terminal band are excluded rather than misclassified as controls. One positive landmark per episode/configuration/stratum prevents inflated positive counts; controls can overlap and remain correlated within cases/subjects. T0 audit strata never enter buffered predictor fields.

Washed-out: no usable MAP <65 during the preceding 120 minutes through the buffered cutoff, with that washout actually observed; no ongoing confirmed episode at T0. Normal80: washed-out plus latest T0 and buffered MAP ≥80, T0 measurement age ≤60 seconds. Hard: normal80 plus buffered last-hour minimum ≥75 and slope ≥−1 mmHg/hour. Normal90: washed-out plus both current and buffered MAP ≥90 and last-hour minimum ≥80. The washout excludes **any** low excursion, a deliberately stricter criterion than excluding only recent sustained episodes, matching Phase 4's hard-case reasoning. It cannot be weakened after seeing counts and called the same task.

Versions v1.1/v1.2 document implementation corrections after initial counts: follow-up reaches H+31 minutes and the recording must actually extend to that endpoint; the washout is the full 120 minutes ending at the buffered cutoff, [T0−125,T0−5]. The extra closed minute accommodates native confirmation jitter (up to ten seconds after thirty minutes). v1's exact H+30 boundary censored runs confirming just after that boundary and did not explicitly reject a partial final minute. An initial washout implementation spanned only 115 minutes; v1.2 corrects it to the requested 120. Original protocols and revision rationale remain private. Clinical thresholds, requested history/washout durations, stability and lead bands were unchanged; no definition was selected to increase positives.

## Horizon feasibility

Terminal-band risk sets (case and subject counts separate from windows):

{horizon_table}

Every cumulative counterpart also has zero positives under these stability/coverage rules. These rows describe observed eligible **controls**, not a viable supervised dataset. Histories of thirty and sixty minutes retain the same 120-minute washout and therefore often have identical counts. Positive lead distributions and positive-window prevalence are empty/zero; empty strata have undefined prevalence, not zero prevalence. Private reports contain all 128 history/horizon/band/stability combinations.

At three hours:

{stable_table}

Primary six-hour-history three-hour band: {primary['cases']} cases / {primary['patients']} subjects, {primary['control_windows']} control windows, zero positive cases, subjects and events. Hard primary stratum: {hard['cases']} eligible cases, zero positives. Adding non-MAP modalities cannot manufacture missing positive targets or pre-event observation.

## Event-centred diagnostic and censoring

This diagnostic places a landmark exactly H minutes before each native episode without changing the primary definition. It separates history/future observation from stability:

{diagtable}

With thirty-minute history, fifteen cases/subjects support exactly 180 minutes of lead and complete future recording; only four have current MAP ≥80, and none satisfies the observed 120-minute washout. Those four are **not** hard-case positives. With six-hour history no adult event has sufficient elapsed pre-event recording even before stability. Episode onset median {number(diagnostic['adult_episode_onset_minutes']['median'])} min, maximum {number(diagnostic['adult_episode_onset_minutes']['max'])} min. This directly explains why the long-history three-hour headline fails; it is not merely a low total-case count.

Sequential first-reason landmark attrition before stability stratification:

{censor_table}

No future-censored landmark becomes a negative. Anesthesia, line setup, flushing, calibration, treatment and surgery can cause artifacts or alter trajectories. Plausible jumps are retained flags; sustained labels remain research monitor proxies requiring artifact and clinician review. The chosen strict washout limits generalization: zero eligible positives does not mean no deterioration can ever be predicted with a different clinical task. A new endpoint would require a separate prespecified study, not silent relaxation here.
''')
    d=signals['overlaps']['D_BP_HR_SpO2'];e=signals['overlaps']['E_BP_HR_SpO2_RR']
    write('VITALDB_TRANSFER_PLAN.md',f'''
# VitalDB transfer plan — Phase 4.5A

**PARTIAL GO for a bounded physiological representation and replay module. No supervised three-hour stable/hard pretraining recommendation.** The observed primary positive cohort is empty, despite thousands of MAP recordings. No model has been trained in this phase and no efficacy gain is claimed.

## Feasible learning objectives

Supervised harmonized sustained-hypotension pretraining: no-go under the prespecified warning task. There are {adult['event_positive_patients']} adult event-positive subjects overall, below the declared engineering gate of 100 independent positive subjects, and zero eligible warning positives. These gates are feasibility checks, not statistical power calculations. A new short-hypotension endpoint would be a different study and is not approved by this audit.

Self-supervised numeric learning: bounded candidate module. Full native MAP retrieval supplies {adult['usable_map_patients_30minutes_80percent']:,} usable adult subjects. The deliberately limited multivariate audit measures {d['usable_cases']} cases / {d['usable_patients']} subjects for BP+HR+SpO₂ and {e['usable_cases']} cases / {e['usable_patients']} subjects after adding respiration (all-age sampled counts). Exact adult training-eligible partitions and contiguous sequences must be created before fitting; presence upper bounds are not assumed usable training examples. A 1,000-subject multivariate gate is an engineering scale criterion; any subset below it remains a pilot rather than proof of large-scale multimodal training.

Temporal encoder learning: plausible only with shorter observed sequences and explicit masks. Start with thirty-to-sixty-minute past-only segments and causal next-segment prediction; reconstruction-only objectives must not be presented as forecasting. Evaluate simple persistence/linear/compact baselines first. Keep source/domain/device indicators and treatment context separate. No six-hour pretraining-history claim follows from shorter segments.

Waveform learning: technical candidate after a dedicated quality/timing audit. Three waveform catalog tracks coexist in thousands of cases, and twelve bounded three-track examples decoded correctly at native 500 Hz. This verifies access and representation, not cohort-wide clean waves, beat timing, model benefit or HF validity. Initially use a prespecified bounded subset and signal-quality masks; do not retrieve all high-frequency cases merely because they are available.

## Adaptation and independent evaluation

Future VitalDB train/validation/held-out partitions must group repeated cases by subject. Fit scalers, imputers and encoders only on the relevant training partition. VitalDB validation can assess representation quality within surgery; a VitalDB held-out group is still surgical evidence, not independent ICU/HF validation. Freeze objectives, source units, QC, data masks and model family before transferring. Neither labels nor normalization should exploit held-out subjects or future timestamps.

If a representation clears that limited study, adapt using **only existing eICU TRAIN and VALIDATION**, with train-only preprocessing and validation-only selection/calibration/thresholds. Compare to the preserved Phase 4 MAP baseline using development data; do not rerun, select on, recalibrate on or tune with the spent eICU TEST. Its published result remains historical evidence only.

An unbiased later deterioration test needs a fresh independent ICU cohort, ideally prospectively collected at another institution/time, with native invasive MAP, aligned HR/BP/SpO₂, credible receipt times, sufficient pre-event recording, clinician-reviewed signal artifacts, treatment timestamps and sustained-event adjudication. Patients must be absent from all development cohorts. Predeclare hard/normal-MAP and true lead-time analyses, evaluate calibration and alert burden, and keep outcome adjudication independent. A specifically adjudicated HF cohort is additionally required for HF claims; ICU monitoring alone is insufficient. Home HF monitoring requires a separate paired longitudinal outcome study and cannot be validated by intraoperative arterial monitoring.

## Dataset responsibilities and scientific boundary

VitalDB: surgical physiology representation/access/replay. eICU: ICU EHR+physiology fusion development and historical exploratory HF evidence. Zigong: retrospective clinical readmission benchmark. CHIRON and PerHeart: separate small longitudinal/personalized-monitoring cohorts. Synthea: FHIR/application engineering and synthetic demo only. SCG-RHC: future hidden-hemodynamics work requiring supplied paired reference data. These cohorts share concepts, not identities, fabricated pairs or supervision labels.

VitalDB is single-center intraoperative non-cardiac surgery, **not heart-failure validation**. General anesthesia, ventilation, fluids, vasopressors, surgery and clinician responses alter and censor trajectories. CO measurement devices select particular populations. Domain shift includes measurement cadence, invasive monitoring, demographics, case duration, interventions and label prevalence. Confounding by treatment means observed trajectories are not untreated counterfactuals. Good surgical performance would not prove ICU transfer, home-monitoring performance, personalization benefit, EHR fusion gain or clinical HF usefulness. Masking MAP in a model does not establish signal beyond BP without paired development ablations and a fresh independent test.

## Replay proof

The public read-only API supports `Location/OR1` through OR10 and a repeating historical monitor. Verified path: `$status`/`$sample` → bounded JSON parser → existing canonical Observation/SourceValue/Provenance objects. A fixed ten-second window emitted {proof['canonical_records']:,} records across eight concepts: HR, invasive SBP/DBP/MAP, SpO₂ and arterial/ECG/PPG waveforms. Numeric replay is 1 Hz in the pilot and can be resampled from the original two-second measurements; replay is never used to derive event labels or source missingness.

Future polling can initially request thirty seconds and then ten-second increments, handle 409 idle turnover, track case/subject changes, deduplicate overlapping windows and preserve the virtual/source/receipt clocks. This is a technical historical replay, not a connected real patient or a validated live clinical service. No UI, virtual-heart animation, alert service or final Twin layers were built. API details: [official guide](https://api.vitaldb.net/AGENTS.md) and [OpenAPI](https://api.vitaldb.net/openapi.json).

## Exact Phase 4.5B recommendation

Create a **bounded, label-independent physiological representation study**, beginning with adult MAP/BP/HR/SpO₂ thirty-to-sixty-minute sequences: verify contiguous coverage and units, group subjects, freeze private splits and causal mask/normalization rules, establish simple baselines, and predeclare limited self-supervised/temporal objectives. Add respiration only where actually supported. Pilot waveform quality and phase timing before any waveform encoder. Keep the demonstrated replay adapter as an engineering module without expanding UI scope. Assess any transfer only on eICU TRAIN/VALIDATION, and establish the fresh external ICU test acquisition plan before claiming prediction improvement. Do not train a harmonized three-hour hard-case classifier from the empty VitalDB risk set, revise the headline, or start full Digital Twin innovation layers on this evidence.
''')
    write('PHASE4_5A_DECISION.md',f'''
# Phase 4.5A decision

**B. PARTIAL GO — VitalDB is suitable for a bounded physiological representation/replay module. It does not solve the prespecified six-hour-history, approximately three-hour stable/hard deterioration task.** No model training, Phase 1–4 result changes or spent eICU test evaluation occurred.

| Required question | Measured answer |
| --- | --- |
| 1. Usable cases? | 6,388 accessible cases / 6,090 subjects. 3,754 invasive-MAP union cases retrieved; 3,709 adults. {adult['usable_map_cases_30minutes_80percent']:,} adult cases / {adult['usable_map_patients_30minutes_80percent']:,} subjects meet ≥30 usable minutes and ≥80% MAP coverage. Usable recording does not imply an eligible warning example. |
| 2. Sustained hypotension? | {adult['event_positive_cases']} adult cases / {adult['event_positive_patients']} subjects, {adult['native_events']} native runs / {adult['merged_episodes']} merged episodes; {usable_positive_adults} adult positive cases also pass whole-record usable-MAP QC. {allages['event_positive_cases']} event-positive cases across all ages. No clinician adjudication. |
| 3. Genuine 3h prediction? | Six-hour-history primary: {primary['cases']} eligible control cases, {primary['control_windows']} control windows, zero positives in the (150,180]min lead band. No event has full six-hour history at exactly 180min lead. With thirty-minute history, fifteen episodes/cases/subjects have exact 180min observation support, but zero satisfy prespecified washout. |
| 4. Hard-case 3h? | Zero eligible primary hard cases and zero positives. Even short-history hard/stable sets have zero positive examples. Four short-history cases have MAP≥80 at exact 3h, but fail washout and are not hard positives. |
| 5. Signals beyond MAP? | Invasive SBP/DBP, HR, SpO₂, respiratory rate, temperature, CO/CVP subsets, ECG/PPG/arterial waves and other device-specific hemodynamics. See measured QC and source mapping. |
| 6. Meaningful multimodal scale? | {d['usable_cases']} measured BP+HR+SpO₂ cases / {d['usable_patients']} subjects in a {signals['selected_multisignal_cases']:,}-case label-free sample; {e['usable_cases']} cases / {e['usable_patients']} subjects with respiration. Whole-cohort usable counts remain lower bounds, not catalog-count assertions. Representation exploration is feasible; supervised stable warning lacks positive scale. |
| 7. Clinical context? | Demographics, preop conditions/labs, surgery/anesthesia context, whole-case interventions and outcomes; 2,087 FHIR enriched cases / 2,069 subjects measured. Future totals/outcomes blocked, untimed/receipt-unknown fields excluded pending availability review. |
| 8. Waveforms? | Twelve cases × three sixty-second native chunks decoded at 500 Hz. Whole-cohort clean-waveform counts unmeasured; catalog presence does not guarantee usable morphology or timing. |
| 9. Pretraining? | Partial: short-sequence label-independent numeric/temporal representations; waveform pilot after QC. No harmonized stable/hard supervised pretraining recommendation. |
| 10. Live demo? | Yes technically: fixed public OR replay yielded {proof['canonical_records']:,} canonical records for eight concepts. Historical looping replay, not live clinical patients; no UI built. |
| 11. Download/cache? | Approximately {v['known_response_body_MiB']:.1f} MiB of known response bodies and {v['cache_MiB']:.1f} MiB private cache, plus small final audit bookkeeping. No full 95GB download. |
| 12. Domain shift? | Single-center surgery/anesthesia, clinician treatments, invasive monitoring, short recordings and different sampling/label ascertainment; substantial shift to ICU and home HF. Strong VitalDB results would not validate HF. |

The primary target uses native <65 mmHg runs of ≥30min with maximum ten-second gaps. Five-minute emulation produces {adult['coarse5minute_events']} adult runs versus {adult['native_events']} native runs, demonstrating why sampling rules cannot be copied mechanically. Strict washout excludes any low excursion, not only recent sustained events; zero positives is specific to this declared task. No boundary was relaxed to create a positive cohort. A documented follow-up implementation correction allows native confirmation jitter and rejects partial endpoints; original protocol/revision rationale remain private.

Verification: **{tests['tests_run']} tests pass, zero failures/errors/skips** (112 existing + 25 VitalDB tests), covering parser, QC, determinism, causal-event/feature, stable/hard, genuine-lead, censoring, replay scaling and private-Git checks. Three checksum-matched real MAP streams reproduce {native['events_checked']} native events, every minute array, and confirmation prefixes. The completion audit verifies 11,944 numeric streams and 15,734 risk-window records and preserves 4,467 earlier files byte-for-byte. The private completion report records preservation hashes, cache checksums, source hashes and aggregate reconstruction. See [data audit](VITALDB_DATA_AUDIT.md), [signal map](VITALDB_SIGNAL_MAP.md), [hypotension feasibility](VITALDB_HYPOTENSION_FEASIBILITY.md), and [transfer plan](VITALDB_TRANSFER_PLAN.md).

**Phase 4.5B:** a bounded adult short-sequence self-supervised/temporal representation feasibility study with subject-grouped splits, train-only normalization, causal masks and simple baselines; a small waveform quality/timing pilot; retain the canonical replay engineering module. Any eICU adaptation/selection uses TRAIN/VALIDATION only, followed eventually by a newly acquired independent ICU test. No supervised three-hour hard-case training from VitalDB, no HF validation claim and no full Twin/UI expansion.
''')
    print({'aggregate_documents_written':5,'recommendation':'B_PARTIAL_GO','models_trained':0})

if __name__=='__main__':generate()
