import {
  Activity,
  ArrowRight,
  Database,
  Info,
  Layers,
  Radio,
} from "lucide-react";
import { Trend } from "./Chart";
import {
  difference,
  driverTitle,
  groupHelp,
  metricHelp,
  readingMeaning,
} from "./insights";
import { fmt, groups, names } from "./types";
import type { EHR, Metadata, Signal, TwinState } from "./types";

export function Details({
  view,
  state,
  frames,
  ehr,
  metadata,
  metric,
  setMetric,
  open,
}: {
  view: string;
  state: TwinState;
  frames: TwinState[];
  ehr: EHR;
  metadata: Metadata;
  metric: Signal;
  setMetric: (s: Signal) => void;
  open: (s: string) => void;
}) {
  const isPhysiology = Boolean(groups[view]) || view === "Clinical Details";
  const departure = difference(state, metric);
  const persistence = state.why_now.personal_deviations.find(
    (d) => d.signal === metric,
  )?.recent_48h_deviation_fraction;
  return (
    <>
      {isPhysiology && (
        <>
          <div className="view-intro">
            <span className="eyebrow">
              READING → PERSONAL COMPARISON → CONTEXT
            </span>
            <h2>
              {groupHelp[view]?.question ?? "The observations behind the Twin."}
            </h2>
            <p>
              {groupHelp[view]?.explanation ??
                "Explore one metric at a time, compare it with this patient’s own expected pattern, then inspect the historical clinical context."}
            </p>
          </div>
          <div className="detail-layout">
            <section className="content-card chart-card">
              <div className="card-heading">
                <label htmlFor="metric-select">Explore a metric</label>
                <select
                  id="metric-select"
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as Signal)}
                >
                  {(groups[view] ?? (Object.keys(names) as Signal[])).map(
                    (s) => (
                      <option key={s} value={s}>
                        {names[s]}
                      </option>
                    ),
                  )}
                </select>
              </div>
              <div className="metric-comparison">
                <div className="actual-value">
                  <span>ACTUAL · RECEIVED</span>
                  <strong>
                    {fmt(state.actual_state[metric])}
                    <small>{state.signal_units[metric]}</small>
                  </strong>
                </div>
                <div className="shadow-value">
                  <span>SHADOW · EXPECTED</span>
                  <strong>
                    {fmt(state.shadow_state[metric].value)}
                    <small>{state.signal_units[metric]}</small>
                  </strong>
                </div>
              </div>
              <Trend frames={frames} metric={metric} />
              <div className="legend">
                <span>
                  <i className="coral" />
                  Observed
                </span>
                <span>
                  <i className="cyan" />
                  Expected stable pattern
                </span>
              </div>
              <p className="chart-explanation">
                Each point uses data received before its own cutoff. The shaded
                band is an estimated personal range, not a clinical confidence
                guarantee.
              </p>
            </section>
            <section className="content-card interpretation-card">
              <span className="eyebrow">WHAT THIS READING MEANS</span>
              <h3>{departure.text}</h3>
              <p>{readingMeaning(state, metric)}</p>
              <div className="explanation-box">
                <strong>Why compare with their own pattern?</strong>
                <p>
                  The same value can be usual for one person and different for
                  another. This comparison makes the personal change visible; it
                  does not establish a clinical cause.
                </p>
              </div>
              <dl>
                <div>
                  <dt>Personal expected band</dt>
                  <dd>
                    {fmt(state.shadow_state[metric].lower)}–
                    {fmt(state.shadow_state[metric].upper)}{" "}
                    {state.signal_units[metric]}
                  </dd>
                </div>
                <div>
                  <dt>Historical personal reference</dt>
                  <dd>
                    {fmt(state.personal_baseline[metric].reference)}{" "}
                    {state.signal_units[metric]}
                  </dd>
                </div>
                <div>
                  <dt>Trusted historical readings</dt>
                  <dd>{state.personal_baseline[metric].trusted_count}</dd>
                </div>
                {persistence != null && (
                  <div>
                    <dt>Recent readings outside personal ±1.5 scale</dt>
                    <dd>{fmt(persistence * 100, 0)}%</dd>
                  </div>
                )}
              </dl>
              <p className="metric-definition">{metricHelp[metric]}</p>
            </section>
          </div>
          {view === "Clinical Details" && <EhrView ehr={ehr} />}
          <div className="view-bottom-note">
            <Info size={16} />
            <p>
              These readings describe recent changes. Compare recent changes with the patient’s personal pattern.
            </p>
          </div>
        </>
      )}
      {view === "Why this changed" && (
        <>
          <div className="view-intro">
            <span className="eyebrow">
              THE EVIDENCE BEHIND THE PERSONAL SHIFT
            </span>
            <h2>What changed, compared with what?</h2>
            <p>
              These are the largest observed departures from this patient’s
              expected stable physiology. They explain the comparison, rather
              than claiming to explain a medical cause.
            </p>
          </div>
          <div className="drivers-list">
            {state.why_now.personal_deviations.map((d, i) => (
              <article className="content-card driver-detail" key={d.signal}>
                <span className="driver-order">0{i + 1}</span>
                <div>
                  <h3>{driverTitle(state, d.signal)}</h3>
                  <p>{metricHelp[d.signal]}</p>
                  <div className="driver-values">
                    <span>
                      Observed{" "}
                      <strong>
                        {fmt(d.observed)} {state.signal_units[d.signal]}
                      </strong>
                    </span>
                    <span>
                      Expected{" "}
                      <strong>
                        {fmt(d.expected)} {state.signal_units[d.signal]}
                      </strong>
                    </span>
                    {d.recent_48h_deviation_fraction !== null && (
                      <span>
                        Outside personal scale{" "}
                        <strong>
                          {fmt(d.recent_48h_deviation_fraction * 100, 0)}% of
                          recent readings
                        </strong>
                      </span>
                    )}
                  </div>
                </div>
                <button
                  className="secondary-button"
                  onClick={() => {
                    setMetric(d.signal);
                    open("Clinical Details");
                  }}
                >
                  Explore trend <ArrowRight size={15} />
                </button>
              </article>
            ))}
          </div>
          <div className="view-bottom-note">
            <Info size={16} />
            <p>
              Changes are shown relative to the patient’s usual variation.
            </p>
          </div>
        </>
      )}
      {view === "Timeline" && (
        <>
          <div className="view-intro">
            <span className="eyebrow">ISSUED ESTIMATES, IN ORDER</span>
            <h2>How did the estimate change?</h2>
            <p>
              Each point shows the estimated risk at that time.
            </p>
          </div>
          <section className="content-card">
            <h3>Estimated deterioration risk within 72 hours</h3>
            <Trend frames={frames} metric="heart_rate" risk />
            <table>
              <thead>
                <tr>
                  <th>Snapshot</th>
                  <th>Within 24h</th>
                  <th>Within 48h</th>
                  <th>Within 72h</th>
                </tr>
              </thead>
              <tbody>
                {frames.map((s) => (
                  <tr key={s.timestamp}>
                    <td>
                      {s.replay_day === 0 ? "Today" : `Day ${s.replay_day}`}
                    </td>
                    {([24, 48, 72] as const).map((h) => (
                      <td key={h}>{fmt(s.risk[`next_${h}h`] * 100)}%</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <div className="evidence-highlights risk-explanations">
            <article className="content-card">
              <h3>What does {fmt(state.risk.next_72h * 100)}% mean?</h3>
              <p>
                This is the estimated probability of deterioration within the next 72 hours. It is an estimate, not a diagnosis or a certainty.
              </p>
            </article>
            <article className="content-card">
              <h3>Why three horizons?</h3>
              <p>
                The estimates are cumulative: 72 hours includes the first 48 and
                24 hours. Low, Moderate and Elevated describe the displayed risk level.
              </p>
            </article>
          </div>
          <div className="view-bottom-note">
            <Info size={16} />
            <p>
              These estimates do not predict an exact event time.
            </p>
          </div>
        </>
      )}
      {view === "Data quality" && (
        <>
          <div className="view-intro">
            <span className="eyebrow">DATA AVAILABILITY, NOT CERTAINTY</span>
            <h2>{state.data_quality.level} measurement availability.</h2>
            <p>
              The quality label reflects coverage, sensor staleness and
              available personal history. It is not the chance that a prediction
              is correct.
            </p>
          </div>
          <section className="content-card">
            <h3>What is missing or limited?</h3>
            {state.data_quality.missing.length ? (
              state.data_quality.missing.map((m) => (
                <div className="detail-row" key={m.signal}>
                  <span>{names[m.signal]}</span>
                  <strong>{m.received_recent_count} recent readings</strong>
                </div>
              ))
            ) : (
              <p>
                All required signal histories are available at this time.
              </p>
            )}
            <p className="chart-explanation">
              Availability reflects the completeness and recency of readings.
            </p>
          </section>
        </>
      )}
      {view === "Evidence" && (
        <>
          <div className="view-intro">
            <span className="eyebrow">VISIBLE EVIDENCE BOUNDARIES</span>
            <h2>What is supported—and where?</h2>
            <p>
              Personalized prediction and expected physiology are supported
              within synthetic POC experiments. There is no prospective clinical
              validation.
            </p>
          </div>
          <div className="evidence-highlights">
            <article className="content-card">
              <h3>Supported in the synthetic POC</h3>
              <p>
                Global personalization improved prediction over population
                fusion. Shadow stable-physiology MAE improved across all ten
                signals. Activity and HRV contributed predictive information.
              </p>
            </article>
            <article className="content-card">
              <h3>A conditional warning time</h3>
              <p>
                Phase 6 reported approximately 38 hours median warning among
                detected synthetic events. This is conditional on detection, not
                a promised lead time.
              </p>
            </article>
          </div>
          <p className="cohort-note">
            {metadata.cohort_separation} No unrelated patients were linked.
          </p>
          <div className="source-grid">
            {metadata.sources.map((s) => (
              <article className="content-card" key={s.name}>
                <Database size={22} />
                <h3>{s.name}</h3>
                <p>{s.role}</p>
                <span
                  className="evidence-label"
                  title={
                    s.label === "SYNTHETIC POC VALIDATED"
                      ? "Supported only within the synthetic research experiments."
                      : s.label === "REAL-DATA RESEARCH"
                        ? "Independent grounding source; no records for this demo patient."
                        : "Integration or descriptive research; incremental benefit not established."
                  }
                >
                  {s.label}
                </span>
              </article>
            ))}
          </div>
          <details className="content-card model-version">
            <summary>Inspect frozen model identity</summary>
            <p>
              {metadata.reference} · {metadata.version}
            </p>
            <label>Model source SHA-256</label>
            <code>{metadata.model_code_sha256}</code>
            <label>Model artifact SHA-256</label>
            <code>{metadata.model_sha256}</code>
            <label>Original calibration SHA-256</label>
            <code>{metadata.calibration_sha256}</code>
          </details>
          <button
            className="secondary-button"
            onClick={() => open("Limitations")}
          >
            Read the research limitations <ArrowRight size={15} />
          </button>
        </>
      )}
      {view === "Architecture" && (
        <>
          <div className="view-intro">
            <span className="eyebrow">STATIC CONTEXT + DYNAMIC PHYSIOLOGY</span>
            <h2>Two streams. One personal comparison.</h2>
            <p>
              Clinical history and health readings help you review the patient’s personal pattern.
            </p>
          </div>
          <section className="content-card architecture">
            <div className="architecture-inputs">
              <article>
                <Database />
                <h3>FHIR / EHR</h3>
                <p>
                  Historical demographics, diagnoses, medication context and
                  labs.
                </p>
              </article>
              <span>+</span>
              <article>
                <Radio />
                <h3>Wearables / home IoT</h3>
                <p>
                  Observed values, units, quality, measurement and receipt
                  times.
                </p>
              </article>
            </div>
            <div className="flow-arrow">↓</div>
            <article className="architecture-twin">
              <Layers />
              <h3>Personal Digital Twin</h3>
              <p>
                Own baseline → expected Shadow state → Actual / Expected
                comparison
              </p>
            </article>
            <div className="flow-arrow">↓</div>
            <div className="architecture-outputs">
              <article>
                <Activity />
                <h3>24 / 48 / 72h risk</h3>
                <p>Estimated risk over three time horizons</p>
              </article>
              <article>
                <HeartPulseIcon />
                <h3>Clinician experience</h3>
                <p>Personal history, readings and evidence</p>
              </article>
            </div>
          </section>
          <p className="cohort-note">
            Explore source information and model evidence in Evidence.
          </p>
          <button className="secondary-button" onClick={() => window.print()}>
            Export architecture / Print to PDF
          </button>
        </>
      )}
      {view === "Limitations" && (
        <>
          <div className="view-intro">
            <span className="eyebrow">
              RESEARCH PROTOTYPE · NOT CLINICALLY VALIDATED
            </span>
            <h2>A clear view of what remains unproved.</h2>
            <p>
              The demonstration illustrates a personal comparison. It is not a
              diagnosis, treatment recommendation or clinically validated
              monitoring service.
            </p>
          </div>
          <section className="content-card limits-list">
            {metadata.limitations.map((l) => (
              <div key={l}>
                <Info size={20} />
                <p>{l}</p>
              </div>
            ))}
          </section>
          <div className="view-bottom-note">
            <Info size={16} />
            <p>
              {metadata.adaptive_sensing} Experimental recalibration and
              recovery reliability are also excluded.
            </p>
          </div>
        </>
      )}
    </>
  );
}
function HeartPulseIcon() {
  return <Activity />;
}
function EhrView({ ehr }: { ehr: EHR }) {
  return (
    <section className="content-card ehr-block">
      <div className="card-heading">
        <div>
          <span className="eyebrow">STATIC HISTORICAL CONTEXT</span>
          <h3>Medical records</h3>
        </div>
        <Database size={21} />
      </div>
      
      <div className="ehr-grid">
        <div>
          <h4>Demographics & diagnoses</h4>
          <p>
            {ehr.demographics.age} years · {ehr.demographics.gender}
          </p>
          <div className="clinical-chips">
            {ehr.diagnoses.map((s) => (
              <span key={s}>
                {s === "hf"
                  ? "Heart failure"
                  : s === "ckd"
                    ? "Chronic kidney disease"
                    : s}
              </span>
            ))}
          </div>
          <h4>Recorded medication classes</h4>
          <p>
            {ehr.medications.length
              ? ehr.medications.map((s) => s.replaceAll("_", " ")).join(" · ")
              : "No medication classes recorded."}
          </p>
        </div>
        <div>
          <h4>Historical labs</h4>
          {ehr.labs.length ? (
            ehr.labs.map((l) => (
              <div className="detail-row" key={l.name}>
                <span>
                  {l.name}
                  <small>
                    {new Date(l.observed_time).toLocaleDateString("en-GB", {
                      timeZone: "UTC",
                    })}
                  </small>
                </span>
                <strong>
                  {fmt(l.value)} {l.unit}
                </strong>
              </div>
            ))
          ) : (
            <p>No lab records available.</p>
          )}
        </div>
      </div>
      <details>
        <summary>Source details</summary>
        <p>{ehr.scope}</p><pre>{JSON.stringify(ehr.fhir, null, 2)}</pre>
      </details>
    </section>
  );
}
