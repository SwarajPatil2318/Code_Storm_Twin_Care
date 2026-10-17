"""Aggregate-only reports for long-horizon target feasibility, never performance."""
import argparse
import json
import math
from pathlib import Path


def wilson(positive, patients):
    if not patients: return None
    z = 1.959963984540054; p = positive / patients
    denom = 1 + z*z/patients
    center = (p + z*z/(2*patients))/denom
    spread = z * math.sqrt(p*(1-p)/patients + z*z/(4*patients*patients))/denom
    return [max(0., center-spread), min(1., center+spread)]


def get(report, kind, lower, upper, history=6, stable='phase35_washout'):
    return report['configurations'][f'{kind}_{lower}_{upper}_l{history}_{stable}']


def table(configurations):
    lines = ['| Horizon / band | Patients / stays | Positive patients | Windows / positive / controls | Prevalence | Censored / stable-at-risk windows | HF positive patients / events / controls | Median event lead h | Recommendation |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for config in configurations:
        full = config['full']; hf = config['hf_known']; counts = config['exclusions']
        label = f"{config['upper_hours']}h" if config['kind']=='cumulative' else f"({config['lower_hours']},{config['upper_hours']}]h"
        prev = '—' if full['window_prevalence'] is None else f"{full['window_prevalence']:.2%}"
        lead = '—' if full['time_to_event_hours'] is None else f"{full['time_to_event_hours']['median']:.2f}"
        rec = 'Exploratory cohort audit; preserve 3h primary'
        if full['positive_patients'] < 50: rec = 'Insufficient broad-pilot positive-patient support'
        if not full['positive_patients']: rec = 'No eligible positives'
        lines.append(f"| {label} | {full['patients']} / {full['icu_stays']} | {full['positive_patients']} | {full['windows']} / {full['positive_windows']} / {full['controls']} | {prev} | {counts.get('incomplete_future_windows',0)} / {counts.get('stable_at_risk_windows',0)} | {hf['positive_patients']} / {hf['unique_positive_events']} / {hf['controls']} | {lead} | {rec} |")
    return lines


def render(root='data/results/phase36', docs='docs'):
    root=Path(root); docs=Path(docs); report=json.loads((root/'report.json').read_text())
    configs=report['configurations']; protocol=report['protocol']
    lines=['# Long-horizon early-warning feasibility — Phase 3.6', '',
           '**No predictive models were trained.** This audit measures temporally observable risk sets and labels. It cannot demonstrate prediction, discrimination, calibration, clinical benefit or 48–72-hour warning skill.', '',
           'The sustained invasive MAP <65 mmHg definition is unchanged: ≥7 consecutive usable archived observations, ≥30-minute first-to-last span, adjacent gaps ≤5 minutes. Onset is the first low observation; confirmation requires 30 later minutes. Existing 120-minute episode clustering, raw-value/QC rules and the three-hour primary remain unchanged. The ICU-monitor proxy is not adjudicated HF decompensation or a consumer-wearable endpoint.', '',
           '## Landmark, history and label contract', '',
           'Use 6/12/24-hour histories ending T0−5 minutes. At least 80% observed arterial-MAP support and no gap >10 minutes are required. Clinical conditions and screened labs require observed and availability offsets ≤T0. Demographics/admission context retain the explicit Phase 3.5 index-semantic availability assumption. No full-stay baseline, interpolation or future diagnosis/lab value enters a baseline.', '',
           'Indices begin at the first whole ICU-relative hour after history+buffer: 7h, 13h or 25h for 6h, 12h or 24h histories. Subsequent indices are separated by the upper forecast endpoint (6/12/24/48/72h). This avoids requiring the first 72h landmark to be at ICU hour 72. The 6h comparison is newly anchored, so it need not equal the Phase 3.5 admission-grid 6h sensitivity count. This Phase 3.6 choice does not modify prior cohorts. Different histories start at different landmarks, so history cohorts are not nested: higher positive counts with longer history can result from anchor shifts, not predictive gain.', '',
           'Cumulative labels require first subsequent episode onset strictly >T0+15min and ≤T0+H. Bands require the **first** subsequent episode in (6,12], (12,24], (24,48] or (48,72] hours. If that first episode occurs at/before the lower band bound, exclude the window as an earlier event; never count it as a control or select a later recurrence to create a long-lead case. Controls have no event through the upper endpoint. Already-underway episodes are excluded.', '',
           '## Stability at T0', '',
           '- `phase35_washout`: no usable MAP <65 in the preceding 120 minutes through T0−5; exclude an episode already underway at T0.',
           '- `stable65_2h`: washout plus ≥90% recent two-hour buffered coverage, all recent values ≥65, and last usable MAP at/before T0 ≥65 with age ≤10min.',
           '- `stable70_6h`: stable65 plus ≥90% six-hour buffered coverage, all those MAP values ≥70, current MAP ≥70, six-hour slope ≥−1 mmHg/h and last two-hour median no more than 5 mmHg below the preceding two-hour median.', '',
           'The current MAP at/before T0 is an eligibility/audit value. Predictor histories still end T0−5. Stability thresholds are transparent engineering sensitivity restrictions, not validated clinical definitions of stability. They are fixed independently of predictive scores. Even a stable trace does not prove an absence of other deterioration or treatment dependence.', '',
           '## Cumulative horizons — 6h history, Phase 3.5 washout', '']
    lines += table([get(report,'cumulative',0,h) for h in protocol['cumulative_hours']])
    lines += ['', '## Strict first-event lead-time bands — 6h history, Phase 3.5 washout', '']
    lines += table([get(report,'band',a,b) for a,b in protocol['bands_hours']])
    lines += ['', '## All histories and stability restrictions', '',
              '| Kind / interval h | History h | Stability | Patients / stays | Positive patients / events | Windows / positive / controls | HF patients / positive patients / events / controls | HF clinical-fusion patients / stays / windows | CV positive patients | Clinical EHR+MAP patients / stays / windows | Median causal history span / connected h |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for config in configs.values():
        a=config['full']; hf=config['hf_known']; cv=config['cv_known']
        span=a['available_history_span_hours']; connected=a['requested_history_connected_hours']
        history='—' if span is None else f"{span['median']:.2f} / {connected['median']:.2f}"
        lines.append(f"| {config['kind']} ({config['lower_hours']},{config['upper_hours']}] | {config['lookback_hours']} | {config['stability']} | {a['patients']} / {a['icu_stays']} | {a['positive_patients']} / {a['unique_positive_events']} | {a['windows']} / {a['positive_windows']} / {a['controls']} | {hf['patients']} / {hf['positive_patients']} / {hf['unique_positive_events']} / {hf['controls']} | {hf['clinical_fusion_patients']} / {hf['clinical_fusion_stays']} / {hf['clinical_fusion_windows']} | {cv['positive_patients']} | {a['clinical_fusion_patients']} / {a['clinical_fusion_stays']} / {a['clinical_fusion_windows']} | {history} |")
    lines += ['', 'All patients counted above have same-person demographic/admission context and supported MAP. “Clinical EHR+MAP” requires at least one documented positive condition or screened, causally available lab beyond demographics. Lack of condition evidence remains unknown. The history span column is observed first-to-last past MAP, possibly with gaps; connected hours refer to the requested history only and sum intervals ≤10min. The private report separately records total connected past history, observed fractions, and quantiles.', '',
              '## Strict 48–72h focus', '',
              '| History / stability | Eligible patients / stays | Positive patients / events | Positive windows / controls | HF patients / positive patients / events / fusion patients | CV positive patients | Conditional positive-patient fraction, Wilson 95% interval |',
              '|---|---:|---:|---:|---:|---:|---|']
    for h in protocol['lookback_hours']:
        for s in protocol['stability']:
            config=get(report,'band',48,72,h,s); a=config['full']; hf=config['hf_known']; cv=config['cv_known']
            ci=wilson(a['positive_patients'],a['patients']); interval='—' if ci is None else f"{a['patient_event_prevalence']:.2%} [{ci[0]:.2%}, {ci[1]:.2%}]"
            lines.append(f"| {h}h / {s} | {a['patients']} / {a['icu_stays']} | {a['positive_patients']} / {a['unique_positive_events']} | {a['positive_windows']} / {a['controls']} | {hf['patients']} / {hf['positive_patients']} / {hf['unique_positive_events']} / {hf['clinical_fusion_patients']} | {cv['positive_patients']} | {interval} |")
    lines += ['', 'Wilson intervals describe conditional proportions of distinct patients with ≥1 eligible positive window, treating patients as the independent units. They are not CIs for prediction accuracy, population event incidence or a fixed individual 72h risk. Patient follow-up/landmark counts vary. Do not compute independent-window confidence intervals.', '',
              '## MAP and recent history at T0', '',
              '| Interval / stability, 6h history | Eligible MAP at T0 min / median | Recent 2h MAP minimum: median | Positive-case MAP at T0: median | Positive-case recent 2h MAP minimum: median | Retained events |',
              '|---|---|---:|---:|---:|---:|']
    for kind,lower,upper in [('cumulative',0,h) for h in protocol['cumulative_hours']]+[('band',a,b) for a,b in protocol['bands_hours']]:
        for stable in protocol['stability']:
            a=get(report,kind,lower,upper,6,stable)['full']
            def value(key,quantile='median'):
                return '—' if a[key] is None else f"{a[key][quantile]:.2f}"
            lines.append(f"| {kind} ({lower},{upper}] / {stable} | {value('map_at_t0','min')} / {value('map_at_t0')} | {value('recent_2h_map_min')} | {value('positive_map_at_t0')} | {value('positive_recent_2h_map_min')} | {a['unique_positive_events']} |")
    lines += ['', '## Censoring and competing events', '',
              'Required future follow-up is the entire T0→upper endpoint plus 30-minute event-confirmation tail. Require ICU discharge boundary and last usable MAP recording to reach that end; additionally require ≥90% observed MAP support and no gap >10min across the interval. A confirmed early event does not exempt an incomplete window from this conservative complete-observation comparison. An incomplete window is excluded/censored, never labelled negative.', '',
              'Death before completion is treated as a competing terminal event and excluded from this complete-case binary target; death is not automatically hypotension. ICU transfer/discharge ends this ICU-specific risk interval. Separate ICU stays are not stitched across transfers, and future discharge destination/status are audit-only. Sustained monitoring loss inside a long stay also censors a label. Exact transfer timing within separate hospital admissions is not reliably recoverable here. A later model may require competing-risk/survival methods and assumptions about informative loss of monitoring, rather than treating exclusions as random.', '',
              '| Interval, 6h history / washout | Candidate windows | History failed | Unstable / underway | Stable at-risk | Incomplete future | Discharge / recording end / interior gap | Death before completion | Earlier first event before band |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind,lower,upper in [('cumulative',0,h) for h in protocol['cumulative_hours']]+[('band',a,b) for a,b in protocol['bands_hours']]:
        c=get(report,kind,lower,upper)['exclusions']
        lines.append(f"| {kind} ({lower},{upper}] | {c.get('candidate_windows',0)} | {c.get('insufficient_past_history',0)} | {c.get('unstable_at_t0',0)} / {c.get('event_underway_at_t0',0)} | {c.get('stable_at_risk_windows',0)} | {c.get('incomplete_future_windows',0)} | {c.get('discharge_censored',0)} / {c.get('recording_end_censored',0)} / {c.get('missing_period_censored',0)} | {c.get('death_before_completion',0)} | {c.get('earlier_first_event_before_band',0)} |")
    lines += ['', 'Discharge and recording-end counters overlap; do not sum them to obtain censored totals. Interior-gap counts here cover complete calendar/recording ends but inadequate interval support. Death is a subset of discharge censoring. Destination distributions and all 81 configurations’ censoring/coverage diagnostics are in private report.json. Counts follow a deterministic waterfall; absence of sufficient data is not a negative clinical outcome.', '',
              '## Dependence, selection and honest evaluation', '',
              'Forecast intervals do not overlap within a configuration, apart from the extra 30-minute confirmation tail. Each episode has at most one positive landmark in that configuration. Rows from different horizons, bands, histories and stability sensitivities are alternative analyses—not independent additional subjects. Repeated stays and controls remain correlated within patient. HF membership uses documented evidence at T0, not future HF diagnoses. HF-positive counts are too small for a strong clinical claim.', '',
              'Landmarks are fixed without looking at episode times. Alternative anchors could change small-sample counts; no event-centered anchor optimization is permitted. A spacing of the upper horizon is recommended for eventual landmark modeling, with grouped patient/episode uncertainty. More frequent live alerts need a separately frozen alert policy and episode-level evaluation, not inflated window-level sample size.', '',
              'Complete future ascertainment selects patients with long ICU stays and invasive lines, and missingness/transfer/death are informative. These counts do not generalize to all HF patients, home wearables or newly admitted patients without long observation. The frozen Phase 3.5 splits remain unchanged; no final-test model performance was examined.', '',
              '## Reproduce', '', '```bash',
              '.venv/bin/python -m src.feasibility.long_horizon',
              '.venv/bin/python -m src.feasibility.long_horizon --output data/results/phase36_reproduction',
              '.venv/bin/python -m src.feasibility.long_horizon_audit',
              '```', '', 'All records and outputs remain under ignored data/results/phase36*. Fresh-build tooling refuses occupied destinations. The audit verifies old artifact hashes, recreates every risk set, runs current regression tests, replays prior validation using unchanged sources, compares independent builds and writes only the two new public aggregate documents.']
    (docs/'LONG_HORIZON_FEASIBILITY.md').write_text('\n'.join(lines)+'\n')
    baseline=get(report,'band',48,72)['full']; strict=get(report,'band',48,72,6,'stable70_6h')['full']; hf=get(report,'band',48,72)['hf_known']
    decision=['# Phase 3.6 decision', '', '**A. 3h remains primary.** The Phase 3.5 target/cohort/splits and earlier negative Zigong result remain preserved.', '',
              '**Can we honestly tell judges that the twin predicts hemodynamic deterioration 48–72 hours before onset? NO.**', '',
              f"The true first-event (48,72]h band with 6h history and Phase 3.5 washout contains {baseline['patients']} eligible patients, {baseline['positive_patients']} positive patients, {baseline['positive_windows']} positive windows and {baseline['controls']} controls. The HF subset has {hf['positive_patients']} positive patients. Under stronger stable70_6h criteria, {strict['patients']} eligible patients and {strict['positive_patients']} positive patients / {strict['unique_positive_events']} events remain.", '',
              'Across the three histories, the 48–72h configurations contain at most three positive patients each. History-specific index anchors can shift cases between bands; these sensitivity counts must never be summed as independent patients.', '',
              'Cumulative “event within 72h” counts do not demonstrate a warning 48–72h before onset. Cohort counts—even when a few real long-lead cases exist—do not establish that any model can predict them. No predictive models or performance estimates were produced. The long-horizon risk sets are selected by sustained monitoring and survival in the ICU; censored windows were never converted to controls.', '',
              'The existing 3h cohort has 263 patients, 74 positive patients, 127 positive windows and 2,648 controls, with real same-person fusion. Its feasibility evidence supports cautious broad-ICU research, not proven prediction or clinical benefit. Do not replace that primary with a small long-lead cohort or describe longer cumulative labels as earlier warnings.', '',
              '## Required evidence for a future 48–72h claim', '',
              'Acquire full eICU/full MIMIC or a prospective longitudinal HF cohort with enough independent HF-positive patients and adjudicated incident deterioration after at least 48h of stable, timestamped observation. It must contain causal historical EHR, continuous or well-characterized MAP monitoring through ≥72h plus confirmation, clinical onset/measurement-receipt times, treatment context, and death/transfer/discharge/loss-of-monitoring information. ICU monitors cannot validate a consumer-wearable claim; that needs paired target-population device and outcome data.', '',
              'Freeze the first-event lead-band, stable-at-T0 definition, landmark/alert spacing and competing-risk/censoring protocol before modeling. Plan sample size around desired event-sensitivity/PPV/false-alert precision, calibration and external validation—not an arbitrary window count. Compare causal EHR-only, physiology-only and fused baselines in patient-isolated development, then evaluate untouched external/prospective patients. Demonstrate episode-level detection with realized 48–72h lead, false alerts per patient-time, calibration, uncertainty and performance against simple past-MAP baselines. Only that evidence could support a predictive claim.', '',
              '## Next action', '',
              'Retain the frozen 3h target; undertake the Phase 4 trace/clinical-availability review and cautious broad-ICU pilot already specified in Phase 3.5. Treat 6/12/24h and longer bands as feasibility sensitivities until their support and future prediction evidence justify a change. HF results remain exploratory. No model training is authorized by this audit.', '',
              'See [LONG_HORIZON_FEASIBILITY.md](LONG_HORIZON_FEASIBILITY.md) for cumulative/band tables, all 81 histories/stability configurations, MAP summaries, HF/CV fusion, and censoring.']
    decision += ['', 'Final recommendation: **A. 3h remains primary.**']
    (docs/'PHASE3_6_DECISION.md').write_text('\n'.join(decision)+'\n')
    print({'public_docs':2, 'recommendation':'A. 3h remains primary', '48_72_prediction_claim':'NO'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default='data/results/phase36'); parser.add_argument('--docs',default='docs')
    args=parser.parse_args(); render(args.root,args.docs)
