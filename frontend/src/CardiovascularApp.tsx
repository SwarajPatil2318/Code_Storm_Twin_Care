import { WearableMiniCard } from './platform/WearableMiniCard';
import { useReceivedSources } from './platform/useReceivedSources';
import React, { useEffect, useRef, useState } from "react";
import { usePlatformChrome } from "./platform/TwinLayout";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  ChartNoAxesCombined,
  Check,
  ChevronDown,
  Clock,
  GitBranch,
  HeartPulse,
  Home,
  Info,
  Pause,
  Play,
  Radio,
  Settings,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  UserRound,
  X,
} from "lucide-react";
import { HeartVisual, HEART_ASSET } from "./Heart";
import { Details } from "./Details";
import { difference } from "./insights";
import { Sources } from "./Sources";
import { MiniTrend } from "./MiniTrend";
import { fmt, groups, names } from "./types";
import type {
  EHR,
  Metadata,
  Observation,
  Patient,
  Signal,
  TwinState,
} from "./types";
import "./styles.css";

type Pack = { frames: TwinState[]; ehr: EHR; observations: Observation[] };
const nav = [
  ["Twin", Home],
  ["Timeline", ChartNoAxesCombined],
  ["Clinical Details", UserRound],
  ["Evidence", ShieldCheck],
  ["Architecture", GitBranch],
] as const;
const steps = [
  {
    title: "Meet the patient",
    text: "Review this patient’s history over the past seven days.",
    view: "Twin",
    day: 0,
  },
  {
    title: "Actual meets expected",
    text: "Cyan shows this patient’s expected pattern; coral shows received readings.",
    view: "Twin",
    day: 0,
  },
  {
    title: "Watch the week unfold",
    text: "Review daily updates and see how the patient’s patterns change.",
    view: "Twin",
    day: 3,
  },
  {
    title: "A personal shift",
    text: "A reading may look usual on its own yet differ from this patient’s normal pattern.",
    view: "Twin",
    day: 7,
  },
  {
    title: "Understand the risk estimate",
    text: "Review risk estimates over 24, 48 and 72 hours. They do not predict an exact event time.",
    view: "Timeline",
    day: 7,
  },
  {
    title: "Ask what changed",
    text: "Compare recent readings with the patient’s personal expected values.",
    view: "Why this changed",
    day: 7,
  },
  {
    title: "Read the personal context",
    text: "Explore heart rate and HRV one at a time, with a chart and a plain-language interpretation.",
    view: "Rhythm",
    day: 7,
  },
  {
    title: "Show the two streams",
    text: "Review clinical history and health readings alongside the personal comparison.",
    view: "Architecture",
    day: 7,
  },
  {
    title: "Explain the evidence",
    text: "Review the source data and evidence behind this Twin.",
    view: "Evidence",
    day: 7,
  },
  {
    title: "Make the boundaries clear",
    text: "False alerts and recovery ambiguity remain. There is no prospective clinical validation or treatment advice.",
    view: "Limitations",
    day: 7,
  },
];
async function get<T>(path: string): Promise<T> {
  const response = await fetch("/api" + path);
  if (!response.ok)
    throw new Error(
      "The Twin is unavailable right now. Please try again.",
    );
  return response.json();
}
export default function CardiovascularApp({ active }: { active: boolean }) {
  const [patients, setPatients] = useState<Patient[]>([]),
    [metadata, setMetadata] = useState<Metadata | null>(null),
    [cache, setCache] = useState<Record<string, Pack>>({});
  const [id, setId] = useState(""),
    [index, setIndex] = useState(7),
    [view, setView] = useState("Twin"),
    [metric, setMetric] = useState<Signal>("heart_rate");
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [playing, setPlaying] = useState(false),
    [percent, setPercent] = useState(false);
  const [demo, setDemo] = useState<number | null>(null),
    [live, setLive] = useState(false),
    [liveIndex, setLiveIndex] = useState(0),
    [fault, setFault] = useState("None");
  const heading = useRef<HTMLHeadingElement>(null);
  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const [p, m] = await Promise.all([
        get<Patient[]>("/patients"),
        get<Metadata>("/evidence"),
      ]);
      const items = await Promise.all(
        p.map(async (patient) => {
          const base = "/patients/" + encodeURIComponent(patient.id);
          const [frames, ehr, observations] = await Promise.all([
            get<TwinState[]>(base + "/timeline"),
            get<EHR>(base + "/ehr"),
            get<Observation[]>(base + "/observations"),
          ]);
          return [patient.id, { frames, ehr, observations }] as const;
        }),
      );
      setPatients(p);
      setMetadata(m);
      setCache(Object.fromEntries(items));
      setId(p[0].id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    void load();
  }, []);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setIndex((i) => {
          if (i >= 7) {
            setPlaying(false);
            return 7;
          }
          return i + 1;
        }),
      1250,
    );
    return () => clearInterval(timer);
  }, [playing]);
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => setLiveIndex((i) => i + 1), 1600);
    return () => clearInterval(timer);
  }, [live]);
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "instant" });
    const back = (e: KeyboardEvent) => {
      if (e.key === "Escape") setView("Twin");
    };
    window.addEventListener("keydown", back);
    return () => window.removeEventListener("keydown", back);
  }, [view]);
  const open = (name: string) => {
    setView(name);
    if (groups[name]) setMetric(groups[name][0]);
  };
  const choose = (patientId: string) => {
    setId(patientId);
    setIndex(7);
    setView("Twin");
    setPlaying(false);
    setLive(false);
    setLiveIndex(0);
    setDemo(null);
    setFault("None");
  };
  const present = (step: number) => {
    setDemo(step);
    setId(patients[0].id);
    setIndex(steps[step].day);
    setView(steps[step].view);
    setPlaying(false);
    setLive(false);
    setFault("None");
    setMetric("heart_rate");
  };
  usePlatformChrome(
    active,
    {
      view: view === "Twin" ? "Home" : view,
      onNavigate: (name) => open(name === "Home" ? "Twin" : name),
      header: (
        <>
          <div className="patient-selector">
            <label className="sr-only" htmlFor="patient-select">
              Patient
            </label>
            <select
              id="patient-select"
              value={id}
              onChange={(e) => choose(e.target.value)}
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <ChevronDown size={13} />
          </div>
          
        </>
      ),
    },
    [id, patients, view, loading],
  );
  useEffect(() => {
    if (active) document.title = "TwinCare · Cardiovascular Twin";
  }, [active]);
  useEffect(() => {
    if (!active) {
      setPlaying(false);
      setLive(false);
    }
  }, [active]);
  const [wearableEmphasis,setWearableEmphasis]=useState<string|null>(null);
  const emptyWearableObservations=useRef<Observation[]>([]).current;
  const wearableFrame=cache[id]?.frames[index];
  const wearableSources=useReceivedSources(cache[id]?.observations??emptyWearableObservations,wearableFrame?.timestamp??'');
  if (loading || error || !metadata || !cache[id])
    return (
      <div className="boot" role={error ? "alert" : "status"}>
        <HeartPulse size={44} />
        <h1>
          {error
            ? "Your Twin is taking a moment."
            : "Preparing your personal Twin"}
        </h1>
        <p>{error || "Loading patient history."}</p>
        {error && (
          <button className="primary-button" onClick={() => void load()}>
            Retry connection
          </button>
        )}
      </div>
    );
  const pack = cache[id],
    original = pack.frames[index],
    patient = original.patient;
  const state: TwinState = {
    ...original,
    actual_state: {
      ...original.actual_state,
      ...(fault === "Missing sensors" ? { hrv_rmssd: null, weight: null } : {}),
    },
    data_quality: {
      ...original.data_quality,
      level: ["Missing sensors", "Stale sensors"].includes(fault)
        ? "Limited"
        : original.data_quality.level,
    },
  };
  const frames = pack.frames
    .slice(0, index + 1)
    .map((s, i) => (i === index ? state : s));
  const ehr =
    fault === "Incomplete EHR"
      ? {
          ...pack.ehr,
          medications: [],
          labs: [],
          scope:
            "Display-only test: medication and lab context are unavailable.",
        }
      : pack.ehr;
  const modelMissing = fault === "Model unavailable";
  const driver =
    state.why_now.personal_deviations.find(
      (d) => d.signal === "hrv_rmssd" && state.actual_state[d.signal] !== null,
    ) ??
    state.why_now.personal_deviations.find(
      (d) => state.actual_state[d.signal] !== null,
    );
  const driverDifference = driver ? difference(state, driver.signal) : null;
  const riskBands = [24, 48, 72].map((h) => state.risk.bands[String(h)]);
  const sharedRiskBand =
    !modelMissing && riskBands.every((b) => b === riskBands[0])
      ? riskBands[0]
      : null;
  const firstName = patient.name.split(" ")[0];
  const title =
    patient.story === "Transient disturbance" && index >= 5
      ? "Their readings are moving back toward their pattern."
      : state.status === "Stable"
        ? "Your patient is close to their usual pattern."
        : "Your patient is drifting from their normal.";
  const observations = pack.observations.filter(
    (o) =>
      (o.available_time ?? o.metadata.scheduled_opportunity_time) <=
      state.timestamp,
  );
  const observation =
    observations[liveIndex % Math.max(observations.length, 1)];
  return (
    <div className="cv-module">
      <main>
        {fault !== "None" && (
          <div className="fault-banner" role="status">
            <Info size={17} />
            <span>
              <strong>Display-only test: {fault}.</strong> Cached scientific
              estimates remain unchanged.
            </span>
            <button onClick={() => setFault("None")}>Reset</button>
          </div>
        )}
        {demo !== null && (
          <section className="presenter">
            <span className="step-number">{demo + 1}/10</span>
            <div>
              <strong>{steps[demo].title}</strong>
              <p>{steps[demo].text}</p>
            </div>
            <button
              className="secondary-button"
              disabled={demo === 0}
              onClick={() => present(demo - 1)}
            >
              Back
            </button>
            <button
              className="primary-button"
              onClick={() =>
                demo === 9 ? (setDemo(null), open("Twin")) : present(demo + 1)
              }
            >
              {demo === 9 ? "Finish" : "Next"}
              <ArrowRight size={14} />
            </button>
            <button
              className="icon-button"
              aria-label="Exit guided tour"
              onClick={() => setDemo(null)}
            >
              <X size={17} />
            </button>
          </section>
        )}
        {view === "Twin" ? (
          <>
            <section className="hero-heading">
              <h1 ref={heading} tabIndex={-1}>
                {title}
              </h1>
              <span
                className={`status-chip ${state.status === "Stable" ? "stable" : "watch"}`}
              >
                {state.status === "Stable" ? (
                  <Check size={16} />
                ) : (
                  <TriangleAlert size={16} />
                )}{" "}
                {modelMissing ? "Estimate unavailable" : state.status}
              </span>
            </section>
            <section
              className="twin-overview"
              aria-label="Personal Digital Twin overview"
            >
              <div className="comparison-card cv-wearable-panel">
                <span className="eyebrow">YOUR PERSONAL COMPARISON</span>
                <h2>
                  {state.divergence.status === "Aligned"
                    ? `${firstName} is close to their usual pattern.`
                    : patient.story === "Transient disturbance" && index >= 5
                      ? `${firstName} is moving back toward their usual pattern.`
                      : `Today is drifting from ${firstName}’s normal.`}
                </h2>
                <div className="mini-hearts">
                  <figure>
                    <img
                      className="mini-shadow"
                      src={HEART_ASSET}
                      alt="Cyan expected heart illustration"
                    />
                    <figcaption>Expected twin</figcaption>
                  </figure>
                  <figure>
                    <img
                      src={HEART_ASSET}
                      alt="Coral observed heart illustration"
                    />
                    <figcaption>Observed today</figcaption>
                  </figure>
                </div>
                <p>
                  {state.population_context.broadly_in_core_bands
                    ? "Some readings look usual. The change from their own baseline tells us more."
                    : "Today’s readings compared with their own usual pattern."}
                </p>
                <button
                  className="card-link"
                  onClick={() => open("Clinical Details")}
                >
                  Understand the comparison <ArrowRight size={15} />
                </button>
                <WearableMiniCard mode="cardiovascular" timestamp={state.timestamp} heartRate={state.actual_state.heart_rate}
                  received={wearableSources.includes('Smartwatch')} onRegion={open} onEmphasis={setWearableEmphasis}
                  readings={[
                    ...(state.actual_state.hrv_rmssd!=null?[{label:'HRV',value:state.actual_state.hrv_rmssd,unit:state.signal_units.hrv_rmssd,region:'Rhythm'}]:[]),
                    ...(state.actual_state.steps!=null?[{label:'Steps',value:state.actual_state.steps,unit:state.signal_units.steps,region:'Activity'}]:[]),
                    ...(state.actual_state.sleep_hours!=null?[{label:'Sleep',value:state.actual_state.sleep_hours,unit:state.signal_units.sleep_hours,region:'Recovery'}]:[])
                  ]}/>
              </div>
              <div className="heart-column">
                <HeartVisual state={state} onRegion={open} emphasizedRegion={wearableEmphasis} />
                <p className="heart-caption">
                  Select an area to explore its readings.
                  <span>Personal physiology overview</span>
                </p>
              </div>
              <div className="overview-context">
                <section className="risk-card">
                  <div className="card-heading">
                    <h2>Estimated deterioration risk</h2>
                    <button
                      className="text-link"
                      aria-pressed={percent}
                      onClick={() => setPercent(!percent)}
                    >
                      {percent ? "Hide %" : "Show %"}
                    </button>
                  </div>
                  <p>Over the next 24, 48 and 72 hours.</p>
                  {sharedRiskBand && (
                    <span
                      className={`shared-risk-band ${sharedRiskBand.toLowerCase()}`}
                    >
                      {sharedRiskBand} across all three horizons
                    </span>
                  )}
                  <div
                    className={`risk-list ${sharedRiskBand ? "risk-list-shared" : ""}`}
                  >
                    {([24, 48, 72] as const).map((h) => (
                      <div className="risk-row" key={h}>
                        <span>{h}h</span>
                        {!sharedRiskBand && (
                          <span
                            className={`risk-band ${state.risk.bands[String(h)].toLowerCase()}`}
                          >
                            {modelMissing
                              ? "Unavailable"
                              : state.risk.bands[String(h)]}
                          </span>
                        )}
                        {sharedRiskBand && !percent && (
                          <span
                            className="risk-horizon-marker"
                            aria-hidden="true"
                          />
                        )}
                        {percent && !modelMissing && (
                          <strong>
                            {fmt(state.risk[`next_${h}h`] * 100)}%
                          </strong>
                        )}
                      </div>
                    ))}
                  </div>
                  <button
                    className="card-link"
                    onClick={() => open("Timeline")}
                  >
                    What does this estimate mean? <ArrowRight size={15} />
                  </button>
                </section>
                <section className="insight-card">
                  <span className="eyebrow">ONE CHANGE TO UNDERSTAND</span>
                  {driver &&
                  driverDifference?.percent !== null &&
                  driverDifference?.percent !== undefined ? (
                    <>
                      <div className="change-summary">
                        <div>
                          <h3>
                            {driver.signal === "hrv_rmssd"
                              ? "HRV"
                              : names[driver.signal]}{" "}
                            <span>
                              {driverDifference.percent < 0 ? "↓" : "↑"}{" "}
                              {fmt(Math.abs(driverDifference.percent), 0)}%
                            </span>
                          </h3>
                          <span className="change-direction">
                            {driverDifference.direction} expected
                          </span>
                        </div>
                        <MiniTrend frames={frames} signal={driver.signal} />
                      </div>
                      <p>
                        {driver.signal === "hrv_rmssd"
                          ? "Heart-rate variability"
                          : "Received 48-hour mean"}{" "}
                        compared with {firstName}’s expected stable value.
                      </p>
                    </>
                  ) : (
                    <h3>Some received readings are unavailable.</h3>
                  )}
                  <button
                    className="primary-button"
                    onClick={() => open("Why this changed")}
                  >
                    Why is this changing?
                    <ArrowRight size={16} />
                  </button>
                </section>
                <button
                  className="quality-summary"
                  onClick={() => open("Data quality")}
                >
                  <ShieldCheck size={18} />
                  <span>
                    Data availability{" "}
                    <strong>{state.data_quality.level}</strong>
                  </span>
                  <ArrowRight size={15} />
                </button>
              </div>
            </section>
            <Sources
              observations={pack.observations}
              cutoff={state.timestamp}
              onSelect={(name, signal) => {
                open(name);
                if (signal) setMetric(signal);
              }}
            />
            <Replay
              frames={pack.frames}
              index={index}
              playing={playing}
              setIndex={(i) => {
                setIndex(i);
                setPlaying(false);
              }}
              play={() => {
                if (index === 7) setIndex(0);
                setPlaying(!playing);
              }}
              disabled={fault === "Replay loading"}
            />
          </>
        ) : (
          <section
            className="center-view"
            aria-label={`${view} full information`}
          >
            <div className="view-heading">
              <button className="back-button" onClick={() => open("Twin")}>
                <ArrowLeft size={16} />
                Back to Twin
              </button>
              <span className="snapshot-label">
                {new Date(state.timestamp).toLocaleDateString("en-GB", {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                  timeZone: "UTC",
                })}{" "}
                · {index === 7 ? "Today" : `Day ${index - 7}`}
              </span>
            </div>
            <h1 className="view-title" ref={heading} tabIndex={-1}>
              {view}
            </h1>
            {view === "Settings" ? (
              <section className="content-card settings-card">
                <h2>Display settings</h2>
                <p>
                  Choose how information is presented.
                </p>
                <label htmlFor="resilience">
                  Display state
                </label>
                <select
                  id="resilience"
                  value={fault}
                  onChange={(e) => setFault(e.target.value)}
                >
                  {[
                    "None",
                    "Missing sensors",
                    "Stale sensors",
                    "Incomplete EHR",
                    "Model unavailable",
                    "Replay loading",
                    "API error",
                  ].map((f) => (
                    <option key={f}>{f}</option>
                  ))}
                </select>
                <button
                  className="secondary-button"
                  onClick={() => {
                    open("Live stream");
                    setLive(true);
                  }}
                >
                  <Radio size={17} />
                  View incoming readings
                </button>
              </section>
            ) : view === "Live stream" ? (
              <section className="content-card stream-card">
                <div className="card-heading">
                  <h2>Incoming readings</h2>
                  <button
                    className="secondary-button"
                    onClick={() => setLive(!live)}
                  >
                    {live ? "Pause stream" : "Resume stream"}
                  </button>
                </div>
                <p>
                  Review readings received through the selected time.
                </p>
                {observation && (
                  <div className="stream-reading">
                    <span>{names[observation.concept]}</span>
                    <strong>
                      {fmt(observation.value.usable_value)}{" "}
                      {observation.value.unit}
                    </strong>
                    <dl>
                      <div>
                        <dt>Observed</dt>
                        <dd>{observation.observed_time ?? "Missing"}</dd>
                      </div>
                      <div>
                        <dt>Received</dt>
                        <dd>{observation.available_time ?? "Not received"}</dd>
                      </div>
                      <div>
                        <dt>Quality</dt>
                        <dd>{observation.value.quality_status}</dd>
                      </div>

                    </dl><details><summary>Source details</summary><p>Historical synthetic observation records; no physical device pairing.</p><p>{observation.metadata.sensor_source}</p><p>Original value: {fmt(observation.value.raw_value)} · {observation.value.reasons.join(", ")}</p></details>
                  </div>
                )}
              </section>
            ) : (
              <Details
                view={view}
                state={state}
                frames={frames}
                ehr={ehr}
                metadata={metadata}
                metric={metric}
                setMetric={setMetric}
                open={(name) => setView(name)}
              />
            )}
          </section>
        )}
        <footer className="footer">
          <span>
            <ShieldCheck size={14} />
            TwinCare
          </span>
          <div>
            <button onClick={() => present(0)}>
              <Play size={12} />
              Guided tour
            </button>
            <span>·</span>
            <button onClick={() => open("Limitations")}>
              Evidence & limitations
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}
function Replay({
  frames,
  index,
  playing,
  setIndex,
  play,
  disabled,
}: {
  frames: TwinState[];
  index: number;
  playing: boolean;
  setIndex: (i: number) => void;
  play: () => void;
  disabled: boolean;
}) {
  // Both trajectories are descriptive personal divergence, never event probability.
  const points = frames.map(
      (s, i) => `${30 + i * 83},${62 - s.divergence.descriptive_tdi * 42}`,
    ),
    expected = frames.map((_, i) => `${30 + i * 83},62`);
  return (
    <section className="replay-card" aria-label="Seven-day replay">
      <button
        className="replay-play"
        disabled={disabled}
        onClick={play}
        aria-label={playing ? "Pause replay" : "Play seven-day replay"}
      >
        <span>{playing ? <Pause size={20} /> : <Play size={20} />}</span>
        <strong>
          {disabled
            ? "Loading replay…"
            : playing
              ? "Pause replay"
              : "Replay 7 days"}
        </strong>
      </button>
      <div className="replay-timeline">
        <svg
          viewBox="0 0 642 88"
          preserveAspectRatio="none"
          role="img"
          aria-label="Personal comparison over the past week. Coral shows observed changes; cyan shows the expected pattern."
        >
          <path
            d={"M" + expected.join(" L")}
            stroke="#21aec4"
            strokeWidth="2"
            strokeDasharray="4 4"
            fill="none"
          />
          <path
            d={"M" + points.slice(0, index + 1).join(" L")}
            stroke="#ed7169"
            strokeWidth="2.5"
            fill="none"
          />
          {frames.map((s, i) => (
            <g key={s.timestamp}>
              {i <= index && (
                <circle
                  cx={30 + i * 83}
                  cy={62 - s.divergence.descriptive_tdi * 42}
                  r={i === index ? 5 : 3.5}
                  fill="#ed7169"
                  stroke="white"
                  strokeWidth="2"
                />
              )}
              <circle cx={30 + i * 83} cy="62" r="3" fill="#21aec4" />
            </g>
          ))}
        </svg>
        <div className="timeline-points">
          {frames.map((s, i) => (
            <button
              key={s.timestamp}
              aria-label={`Replay ${i === 7 ? "today" : `day ${i - 7}`}`}
              aria-pressed={i === index}
              className={i === index ? "current" : ""}
              onClick={() => setIndex(i)}
            >
              <i />
              <span>{i === 7 ? "Today" : `Day ${i - 7}`}</span>
            </button>
          ))}
        </div>
        <label className="sr-only" htmlFor="replay-range">
          Selected day
        </label>
        <input
          id="replay-range"
          type="range"
          min="0"
          max="7"
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
        />
        <p className="replay-caption">
          Coral: descriptive personal divergence · Cyan: expected stable
          reference · Eight daily updates
        </p>
      </div>
    </section>
  );
}
