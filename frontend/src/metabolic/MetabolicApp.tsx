import {RiskDetails,RiskEvidence} from './RiskDetails';
import {
  TechnicalEvidence,
  HowTwinWorks,
  DataLineage,
  sourceRoles,
} from "./ProductEvidence";
import "./product-evidence.css";
import { useLiveTwin } from "./useLiveTwin";
import { LiveDetails } from "./LiveDetails";
import "./live-twin.css";
import { WhyDetails, stateEvidence } from "./StoryDetails";
import { eventText, readable, topEvidence, type DemoSpeed } from "./story";
import { usePlatformChrome } from "../platform/TwinLayout";
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  CalendarDays,
  ChartNoAxesColumn,
  ChevronRight,
  Clock3,
  Database,
  Droplet,
  FileText,
  FlaskConical,
  Footprints,
  Heart,
  Home,
  Info,
  Lightbulb,
  Moon,
  Pill,
  Radio,
  Settings,
  ShieldCheck,
  TriangleAlert,
  UserRound,
  Utensils,
  Watch,
  X,
} from "lucide-react";
import { FlowHero, ForecastChart, ReplayStrip, WatchVisual } from "./Visuals";
import {
  caseNames,
  clock,
  date,
  horizons,
  num,
  primaryReasons,
  reason,
  type Case,
  type Evidence,
  type Pack,
  type State,
} from "./types";
import "./metabolic.css";

async function api<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch("/metabolic-api" + path, { signal });
  if (!response.ok)
    throw new Error(
      "The Metabolic Twin is unavailable right now. Please try again.",
    );
  return response.json();
}
const sourceIcons = [
  ["CGM", Droplet],
  ["Smartwatch", Watch],
  ["EHR / FHIR", Database],
  ["Meal logger", Utensils],
] as const;
type Panel =
  | "technical"
  | "risk"
  | "inspect"
  | "events"
  | "flow"
  | "lineage"
  | "clinical"
  | "why"
  | "settings"
  | "forecast"
  | "shadow"
  | "timeline"
  | (typeof sourceIcons)[number][0];
export default function MetabolicApp({ active }: { active: boolean }) {
  const [cases, setCases] = useState<Case[]>([]),
    [packs, setPacks] = useState<Record<string, Pack>>({}),
    [evidence, setEvidence] = useState<Evidence | null>(null),
    [performance, setPerformance] = useState<Record<
      string,
      { median: number; p95: number }
    > | null>(null);
  const [id, setId] = useState(""),
    [index, setIndex] = useState(0),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [reload, setReload] = useState(0);
  const [panel, setPanel] = useState<Panel | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const load = async () => {
      try {
        const [list, model, timings] = await Promise.all([
          api<Case[]>("/cases", controller.signal),
          api<Evidence>("/evidence", controller.signal),
          api<Record<string, { median: number; p95: number }>>(
            "/performance",
            controller.signal,
          ),
        ]);
        const items = await Promise.all(
          list.map(async (c) => {
            const replay = await api<Pack["replay"]>(
              `/cases/${c.case}/replay`,
              controller.signal,
            );
            return [
              c.case,
              { frames: replay.map((update) => update.state), replay },
            ] as const;
          }),
        );
        if (!list.length || items.some(([, pack]) => !pack.frames.length))
          throw new Error("No cached Metabolic Twin cases are available.");
        setCases(list);
        setPacks(Object.fromEntries(items));
        setEvidence(model);
        setPerformance(timings);
        setId(list[0].case);
        setIndex(items[0][1].frames.length - 1);
      } catch (e) {
        if (!controller.signal.aborted) setError((e as Error).message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, [reload]);
  const runtime = useLiveTwin({
    id,
    packs,
    active,
    index,
    onIndex: setIndex,
    onCase: setId,
  });
  const pack = packs[id],
    state =
      runtime.result &&
      pack &&
      runtime.result.state.patient.id === pack.frames[0].patient.id
        ? runtime.result.state
        : pack?.frames[index];
  const presentation = runtime.result?.presentation;
  const humanEvidence =
    presentation?.evidence ?? (state ? stateEvidence(state) : []);
  const demoActivity = presentation?.activity;
  const [sourceNotice, setSourceNotice] = useState("");
  const [highlightedSources, setHighlightedSources] = useState<string[]>([]);
  useEffect(() => {
    const events = runtime.result?.presentation?.new_events ?? [];
    setHighlightedSources([...new Set(events.map((event) => event.source))]);
    const event =
      events.find((e) => e.source === "meal_logger") ??
      events.find((e) => e.concept === "activity") ??
      events.find((e) => e.source === "cgm");
    setSourceNotice(event ? eventText(event) : "");
    const pulse = setTimeout(() => setHighlightedSources([]), 1000);
    const notice = setTimeout(() => setSourceNotice(""), 1800);
    return () => {
      clearTimeout(pulse);
      clearTimeout(notice);
    };
  }, [runtime.result?.update.trace_id]);
  useEffect(() => {
    if (panel) {
      dialog.current?.showModal();
    } else if (dialog.current?.open) dialog.current.close();
  }, [panel]);
  useEffect(() => {
    if (active) document.title = "TwinCare · Metabolic Twin";
  }, [active]);
  const select = (caseId: string) => {
    if (caseId === id) runtime.seek(packs[caseId].frames.length - 1);
    else setId(caseId);
    setIndex(packs[caseId].frames.length - 1);
    runtime.pause();
    setPanel(null);
  };
  const startLive = () => {
    runtime.start();
    setPanel(null);
  };
  const clinical = state?.recent_context.ehr;
  const risk = state?.multimodal_risk?.probability;
  const watchRisk = risk != null && risk >= state!.spike.threshold;
  const personal = state?.status === "PERSONALLY_ELEVATED_RESPONSE";
  const status = watchRisk
    ? "Needs attention"
    : personal
      ? "Different from usual"
      : state?.status === "INSUFFICIENT_DATA"
        ? "More data needed"
        : "Near usual pattern";
  const slope = state?.why_now.find((w) => w.signal === "recent_slope")?.value;
  const slopeDeparture = state?.why_now.find(
    (w) => w.signal === "slope_departure_from_personal_norm",
  )?.value;
  const subtitle = watchRisk
    ? slopeDeparture != null && slopeDeparture > 0.5
      ? "Glucose response is rising faster than expected."
      : "The model estimates a higher chance of a glucose spike."
    : personal
      ? "This response is above their personal expectation."
      : state?.shadow.available
        ? "Glucose response is close to the expected pattern."
        : "Building a personal glucose expectation.";
  const age =
    clinical?.demographics.birth_date && state
      ? Math.floor(
          (Date.parse(state.timestamp) -
            Date.parse(clinical.demographics.birth_date)) /
            (365.2425 * 86400000),
        )
      : null;
  const lab = clinical?.labs.find((l) =>
    l.text.toLowerCase().includes("hba1c"),
  )?.quantity;
  const medication = state?.recent_context.medication
    ?.map((m) => m.text.split(" — ")[0])
    .join(" + ");
  const peakAt = state?.predicted_peak.approximate_horizon
    ? clock(
        new Date(
          Date.parse(state.timestamp) +
            state.predicted_peak.approximate_horizon * 60000,
        ).toISOString(),
      )
    : null;
  usePlatformChrome(
    active,
    {
      header: (
        <>
          {" "}
          <span className="mt-demo-id" title="Patient">
            {id === "normal_response"
              ? "DT-018"
              : id === "personally_atypical"
                ? "DT-019"
                : "DT-017"}
          </span>
          <span
            className="mt-pill mt-mint"
            title="Glucose monitoring"
          >
            <i />
            {runtime.mode === "VERIFIED REPLAY"
              ? "Verified replay"
              : runtime.mode === "LIVE TWIN"
                ? runtime.running
                  ? "Twin Live"
                  : "Twin paused"
                : "History"}
          </span>
          
          <button
            className="mt-date"
            onClick={() => setPanel("timeline")}
            title="Replay source timestamps in UTC"
          >
            <CalendarDays size={17} />
            {pack
              ? `${date(pack.frames[0].timestamp)} · ${date(pack.frames.at(-1)!.timestamp)}`
              : "Loading dates"}
            <ChevronRight size={15} />
          </button>
          <button
            aria-label="Settings"
            className="mt-settings"
            onClick={() => setPanel("settings")}
          >
            <Settings size={21} />
          </button>
        </>
      ),
      view:
        panel === "clinical"
          ? "Clinical Details"
          : panel === "technical"
            ? "Evidence"
            : panel === "settings"
              ? "Settings"
              : panel === "timeline"
                ? "Timeline"
                : "Home",
      onNavigate: (target) =>
        setPanel(
          (
            {
              Home: null,
              "Clinical Details": "clinical",
              Timeline: "timeline",
              Evidence: "technical",
              Architecture: "technical",
              Settings: "settings",
            } as Record<string, Panel | null>
          )[target] ?? null,
        ),
    },
    [id, pack, runtime.mode, runtime.running, panel],
  );
  useEffect(() => {
    if (!active) {
      runtime.pause();
      setPanel(null);
    }
  }, [active]);
  return (
    <div className="metabolic-module">
      {loading || error || !state ? (
        <main className="mt-loading" aria-live="polite">
          <h1>
            {error
              ? "Metabolic Twin unavailable"
              : "Loading your metabolic twin"}
          </h1>
          <p>
            {error || "Preparing the cached forecast and personal comparison."}
          </p>
          {error && (
            <button
              className="mt-action"
              onClick={() => setReload((r) => r + 1)}
            >
              Retry connection
            </button>
          )}
        </main>
      ) : (
        <main className="mt-main">
          <div className="mt-grid">
            <div className="mt-left">
              <section className="mt-card mt-patient">
                <div className="mt-person">
                  <span className="mt-avatar">
                    {state.patient.name.slice(0, 2).toUpperCase()}
                  </span>
                  <div>
                    <label className="sr-only" htmlFor="metabolic-case">
                      Patient
                    </label>
                    <select
                      id="metabolic-case"
                      value={id}
                      onChange={(e) => select(e.target.value)}
                    >
                      {cases.map((c) => (
                        <option key={c.case} value={c.case}>
                          {c.patient.name}
                        </option>
                      ))}
                    </select>
                    <p>
                      {clinical?.demographics.gender
                        ? clinical.demographics.gender.replace(/^./, (c) =>
                            c.toUpperCase(),
                          )
                        : "Gender unavailable"}{" "}
                      · {age ?? "—"} years
                    </p>
                    <p className="mt-condition">
                      {clinical?.diagnoses
                        .map((d) => d.text.split(" — ")[0])
                        .join(" · ")}
                    </p>
                  </div>
                </div>
                <dl className="mt-patient-facts">
                  {lab?.value != null && (
                    <div>
                      <dt>
                        <Droplet size={19} aria-hidden="true" />
                        <span>HbA1c</span>
                      </dt>
                      <dd>
                        {num(lab?.value, 1)}
                        {lab?.unit}
                      </dd>
                    </div>
                  )}
                  {medication && (
                    <div>
                      <dt>
                        <Pill size={19} aria-hidden="true" />
                        <span>Medication</span>
                      </dt>
                      <dd>{medication}</dd>
                    </div>
                  )}
                </dl>
                <button
                  className="mt-card-hit"
                  aria-label="Open patient clinical details"
                  onClick={() => setPanel("clinical")}
                />
              </section>
              <div className="mt-wearable-slot">
                <WatchVisual state={state} activity={demoActivity}
                  received={highlightedSources.includes("smartwatch")}
                  heartRateDetail={<WearableTrend frames={pack.frames.slice(0, index + 1)} />} />
              </div>
              <section className="mt-sources">
                <h2>Connected sources</h2>
                <div>
                  {sourceIcons.map(([name, Icon]) => (
                    <button
                      key={name}
                      className={
                        highlightedSources.includes(
                          {
                            CGM: "cgm",
                            Smartwatch: "smartwatch",
                            "Meal logger": "meal_logger",
                            "EHR / FHIR": "ehr",
                          }[name],
                        )
                          ? "mt-source-active"
                          : ""
                      }
                      data-active={highlightedSources.includes(
                        {
                          CGM: "cgm",
                          Smartwatch: "smartwatch",
                          "Meal logger": "meal_logger",
                          "EHR / FHIR": "ehr",
                        }[name],
                      )}
                      title={name === "CGM" ? "Glucose monitor" : name === "Smartwatch" ? "Wearable" : name === "EHR / FHIR" ? "Medical records" : "Meal Logger"}
                      data-trace={
                        name === "CGM"
                          ? runtime.result?.update.trace_id
                          : undefined
                      }
                      onClick={() => setPanel(name)}
                    >
                      <Icon size={19} />
                      <span>{name === "CGM" ? "Glucose monitor" : name === "Smartwatch" ? "Wearable" : name === "EHR / FHIR" ? "Medical records" : "Meal Logger"}</span>
                      <i
                        key={
                          name === "CGM"
                            ? runtime.result?.update.trace_id
                            : name
                        }
                        className={
                          name === "CGM" && runtime.result
                            ? "mt-source-received"
                            : undefined
                        }
                        title={
                          name === "CGM"
                            ? `Glucose reading received · reading age ${state.data_quality.observation_age_minutes ?? "unknown"} min · receipt age ${state.data_quality.receipt_age_minutes ?? "unknown"} min`
                            : name === "EHR / FHIR"
                              ? "Synced · historical clinical context"
                              : "Recent readings"
                        }
                      />
                    </button>
                  ))}
                </div>
              </section>
            </div>
            <div className="mt-center">
              <section className="mt-hero">
                <h1>Your metabolic twin</h1>
                <p>{subtitle}</p>
                <span
                  className={`mt-status ${watchRisk ? "watch" : personal ? "personal" : "stable"}`}
                >
                  {watchRisk ? (
                    <TriangleAlert size={16} />
                  ) : (
                    <Activity size={16} />
                  )}{" "}
                  {status}
                </span>
                <button
                  className="mt-twin-update"
                  aria-label="Inspect latest Twin update"
                  title="Review the latest Twin update"
                  onClick={() => setPanel("inspect")}
                >
                  {runtime.busy
                    ? "Updating Twin…"
                    : runtime.mode === "VERIFIED REPLAY"
                      ? "Verified replay"
                      : runtime.result
                        ? "Twin updated"
                        : "Initializing Twin"}{" "}
                  ·{" "}
                  <time dateTime={state.timestamp}>
                    {clock(state.timestamp)} UTC
                  </time>
                  {sourceNotice ? (
                    <span className="mt-event-notice" role="status">
                      {" "}
                      · {state.multimodal_risk ? "Glucose and risk updated" : "Reading received"}
                    </span>
                  ) : (
                    runtime.mode === "LIVE TWIN" && " · Monitoring"
                  )}
                </button>
                {runtime.warning && (
                  <p
                    className="mt-runtime-warning mt-live-status"
                    role="status"
                  >
                    The latest update is unavailable. Showing saved history.
                  </p>
                )}
                <FlowHero state={state} />
                <button
                  className="mt-flow-info"
                  aria-label="Explain expected twin"
                  onClick={() => setPanel("shadow")}
                >
                  <Info size={15} />
                </button>
              </section>
              <ForecastChart
                state={state}
                observedHistory={runtime.result?.observed_history}
              />
            </div>
            <div className="mt-right">
              <section className="mt-card mt-forecast">
                <header>
                  <h2>
                    <Droplet size={24} fill="#39baf5" stroke="#159de3" />
                    Glucose &amp; forecast
                  </h2>
                  <button
                    className="mt-info"
                    aria-label="Explain glucose forecast"
                    onClick={() => setPanel("forecast")}
                  >
                    <span>
                      Data quality:{" "}
                      {state.data_quality.status === "Ready"
                        ? "Good"
                        : "Limited"}
                    </span>
                    <Info size={16} />
                  </button>
                </header>
                <div className="mt-current">
                  <span>Current glucose</span>
                  <div>
                    <strong data-testid="current-glucose">
                      {num(state.current_glucose)}
                    </strong>
                    <b>mg/dL</b>
                    <span
                      className={`mt-trend ${slope != null && slope > 0.5 ? "rising" : ""}`}
                    >
                      <ArrowUpRight size={24} />
                      {slope == null
                        ? "Unavailable"
                        : slope > 0.5
                          ? "Rising"
                          : slope < -0.5
                            ? "Falling"
                            : "Steady"}
                    </span>
                  </div>
                  <small>
                    {clock(
                      state.data_quality.last_observed_time ?? state.timestamp,
                    )}{" "}
                    <span>UTC</span>
                  </small>
                </div>
                <div
                  className="mt-four-forecasts mt-refresh-values"
                  key={runtime.result?.update.trace_id}
                >
                  {horizons.map((h) => (
                    <div key={h}>
                      <span>+{h} min</span>
                      <strong data-testid={`forecast-${h}`}>
                        {num(state.forecast[`plus_${h}m`])}
                      </strong>
                      <small>mg/dL</small>
                    </div>
                  ))}
                </div>
                <div className="mt-summaries">
                  <div>
                    <Clock3 size={27} />
                    <span>
                      Predicted peak
                      <strong>
                        ~ {state.predicted_peak.approximate_horizon ?? "—"} min
                      </strong>
                      <small>
                        {num(state.predicted_peak.glucose)} mg/dL ·{" "}
                        {peakAt ?? "—"}
                      </small>
                    </span>
                  </div>
                  <button onClick={() => setPanel("risk")}>
                    <TriangleAlert size={27} />
                    <span>
                      Spike risk
                      <strong className={watchRisk ? "mt-coral-text" : ""}>
                        {risk == null
                          ? "Unavailable"
                          : `${num(risk * 100, 1)}%`}
                      </strong>
                      <small>
                        {watchRisk ? "Above threshold" : "Below threshold"} ·
                        next 2h
                      </small>
                    </span>
                  </button>
                </div>
                <button
                  className="mt-inspect-button"
                  onClick={() => setPanel("risk")}
                >
                  Inspect risk
                </button>
              </section>
              <section className="mt-card mt-why">
                <header>
                  <h2>
                    <Lightbulb size={23} className="mt-bulb" />
                    Why now?
                  </h2>
                  <button onClick={() => setPanel("why")}>
                    View all <ChevronRight size={13} />
                  </button>
                </header>
                <div>
                  {topEvidence(humanEvidence).map((item, i) => {
                    const { title, summary: text } = readable(item);
                    return (
                      <button
                        key={item.type}
                        onClick={() => setPanel("why")}
                        data-evidence={item.type}
                      >
                        <b>{i + 1}</b>
                        <span>
                          <strong>{title}</strong>
                          <small>{text}</small>
                        </span>
                        <ChevronRight size={16} />
                      </button>
                    );
                  })}
                </div>
              </section>
            </div>
          </div>
          <ReplayStrip
            frames={pack.frames}
            index={index}
            playing={runtime.running}
            mode={
              runtime.mode === "LIVE TWIN" && runtime.speed === "demo"
                ? "Monitoring"
                : runtime.mode === "HISTORICAL / DEMO REPLAY" ? "History" : runtime.mode === "VERIFIED REPLAY" ? "Saved history" : "Monitoring"
            }
            knownThrough={index}
            sourceEvents={presentation?.source_events}
            controls={
              <div className="mt-live-controls">
                <button onClick={startLive} disabled={runtime.busy}>
                  {runtime.running ? "Pause" : "Start Live Twin"}
                </button>
                <button onClick={runtime.reset} disabled={runtime.busy}>
                  Reset
                </button>
                <details><summary>Source details</summary>
                <select
                  aria-label="Live Twin speed"
                  value={runtime.speed}
                  onChange={(e) =>
                    runtime.setSpeed(e.target.value as DemoSpeed)
                  }
                >
                  {[1, 2, 4, 8].map((speed) => (
                    <option key={speed} value={speed}>
                      {speed}×
                    </option>
                  ))}
                  <option value="demo">Accelerated</option>
                </select>
                </details>
                <button
                  onClick={() => {
                    runtime.jury();
                    setPanel(null);
                  }}
                  disabled={runtime.busy}
                >
                  Guided playback
                </button>
              </div>
            }
            onPlay={() => runtime.replay(index)}
            onSelect={runtime.seek}
          />
        </main>
      )}
      <dialog
        ref={dialog}
        className="mt-dialog"
        onCancel={() => setPanel(null)}
        onClose={() => setPanel(null)}
      >
        <header>
          <div>
            <small>TwinCare · Metabolic Twin</small>
            <h2>
              {panel === "technical"
                ? "Technical evidence"
                : panel === "risk"
                  ? "Inspect risk prediction"
                  : panel === "inspect"
                  ? "Inspect prediction"
                  : panel === "events"
                    ? "Live event log"
                    : panel === "flow"
                      ? "How this Twin works"
                      : panel === "lineage"
                        ? "Data lineage"
                        : panel === "clinical"
                          ? "Clinical details"
                          : panel === "why"
                            ? "Why now?"
                            : panel === "settings"
                              ? "Monitoring settings"
                              : panel === "forecast"
                                ? "Understanding the forecast"
                                : panel === "shadow"
                                  ? "The expected twin"
                                  : panel === "timeline"
                                    ? "Replay timeline"
                                    : panel}
            </h2>
          </div>
          <button aria-label="Close details" onClick={() => setPanel(null)}>
            <X size={22} />
          </button>
        </header>
        {state &&
          panel &&
          (panel === "inspect" || panel === "events" ? (
            <LiveDetails
              result={runtime.result}
              state={state}
              eventLog={panel === "events"}
              onEvidence={() => setPanel("technical")}
              onLog={() => setPanel("events")}
            />
          ) : panel === "risk" ? <RiskDetails state={state}/> : panel === "why" ? (
            <WhyDetails evidence={humanEvidence} state={state} />
          ) : panel === "settings" ? (
            <div className="mt-panel-body">
              <p>
                Manage monitoring and review recent updates.
              </p>
              <div className="mt-evidence-actions">
                <button onClick={startLive}>Start Live Twin</button>
                <button
                  onClick={() => {
                    runtime.jury();
                    setPanel(null);
                  }}
                >
                  Guided playback
                </button>
                <button
                  onClick={() => {
                    runtime.jury(false);
                    setPanel(null);
                  }}
                >
                  Reset guided playback
                </button>
                <button onClick={() => setPanel("events")}>
                  Live event log
                </button>
                <button onClick={() => setPanel("flow")}>
                  How this Twin works
                </button>
              </div>
            </div>
          ) : (
            <PanelContent
              panel={panel}
              state={state}
              evidence={evidence}
              performance={performance}
              pack={pack}
              startLive={startLive}
              onPanel={setPanel}
              selectIndex={(i) => {
                runtime.seek(i);
                setPanel(null);
              }}
            />
          ))}
      </dialog>
    </div>
  );
}

function WearableTrend({ frames }: { frames: State[] }) {
  const values = frames.map(
      (f) => f.recent_context.activity?.heart_rate?.value ?? null,
    ),
    seen = values.filter((v): v is number => v !== null);
  if (seen.length < 2) return null;
  const low = Math.min(...seen),
    span = Math.max(1, Math.max(...seen) - low);
  const path = values
    .map((v, i) =>
      v === null
        ? ""
        : `${i === 0 || values[i - 1] === null ? "M" : "L"}${(i / (values.length - 1)) * 48} ${17 - ((v - low) / span) * 14}`,
    )
    .join(" ");
  return (
    <svg
      viewBox="0 0 50 20"
      role="img"
      aria-label="Heart rate readings through the selected time"
    >
      <path d={path} />
    </svg>
  );
}

function PanelContent({
  panel,
  state,
  evidence,
  performance,
  pack,
  startLive,
  selectIndex,
  onPanel,
}: {
  panel: Panel;
  onPanel: (panel: Panel) => void;
  state: State;
  evidence: Evidence | null;
  performance: Record<string, { median: number; p95: number }> | null;
  pack: Pack;
  startLive: () => void;
  selectIndex: (i: number) => void;
}) {
  if (panel === "flow") return <HowTwinWorks />;
  if (panel === "lineage") return <DataLineage state={state} />;
  if (panel === "settings")
    return (
      <div className="mt-panel-body">
        <p>
          View incoming readings and Twin updates.
        </p>
        <button className="mt-action" onClick={startLive}>
          <Radio size={17} />
          View incoming readings
        </button>
        <p className="mt-note">
          Times are shown in UTC.
        </p>
        <a href="/">Open Cardiovascular Twin</a>
      </div>
    );
  if (panel === "timeline")
    return (
      <div className="mt-panel-body">
        <p>
          Review a reading. This history covers{" "}
          {num(
            (Date.parse(pack.frames.at(-1)!.timestamp) -
              Date.parse(pack.frames[0].timestamp)) /
              3600000,
          )}{" "}
          hours.
        </p>
        <div className="mt-time-options">
          {pack.frames.map((f, i) => (
            <button key={f.timestamp} onClick={() => selectIndex(i)}>
              {clock(f.timestamp)} UTC
              <span>{num(f.current_glucose)} mg/dL</span>
            </button>
          ))}
        </div>
      </div>
    );
  if (panel === "why")
    return (
      <div className="mt-panel-body">
        <p>
          Readings and personal comparisons explain what changed. They do
          not establish a medical cause.
        </p>
        {state.why_now.map((item) => {
          const [title, text] = reason(item);
          return (
            <article className="mt-explanation" key={item.signal}>
              <h3>{title}</h3>
              <p>{text}</p>
              
            </article>
          );
        })}
      </div>
    );
  if (panel === "clinical" || panel === "EHR / FHIR")
    return (
      <div className="mt-panel-body">
        
        <p className="mt-source-role">CLINICAL CONTEXT</p>
        
        <h3>Type 2 Diabetes</h3>
        <p>
          Clinical history for {state.patient.name}.
        </p>
        <dl>
          <div>
            <dt>Demographics</dt>
            <dd>
              {state.recent_context.ehr?.demographics.gender} · birth date{" "}
              {state.recent_context.ehr?.demographics.birth_date}
            </dd>
          </div>
          <div>
            <dt>BMI</dt>
            <dd>Not recorded</dd>
          </div>
          {state.recent_context.ehr?.labs.map((l) => (
            <div key={l.text}>
              <dt>{l.text.split(" — ")[0]}</dt>
              <dd>
                {num(l.quantity?.value, 1)} {l.quantity?.unit}
              </dd>
            </div>
          ))}
          <div>
            <dt>Medication orders</dt>
            <dd>
              {state.recent_context.medication?.map((m) => m.text).join(", ") ||
                "Not supplied"}
            </dd>
          </div>
          <div>
            <dt>Recorded condition</dt>
            <dd>
              {state.recent_context.ehr?.diagnoses
                .map((d) => d.text)
                .join(", ")}
            </dd>
          </div>
        </dl>
        <p className="mt-note">
          Medication orders may differ from medicines currently taken.
        </p>
        <details>
          <summary>Source details</summary>
          <p>Generated clinical records for this product environment; not linked to a ShanghaiT2DM participant.</p><pre>{JSON.stringify(state.recent_context.ehr, null, 2)}</pre>
        </details>
      </div>
    );
  if (panel === "shadow")
    return (
      <div className="mt-panel-body">
        <p>
          The cyan curve is a personal historical expectation, separate from the
          coral model prediction.
        </p>
        <h3>Personal expected glucose</h3>
        <dl>
          {[0, ...horizons].map((h) => (
            <div key={h}>
              <dt>{h === 0 ? "Expected now" : `Expected +${h} minutes`}</dt>
              <dd data-testid={`shadow-${h}`}>
                {num(
                  h === 0
                    ? state.shadow.current
                    : state.shadow[`plus_${h}m` as keyof typeof state.forecast],
                  1,
                )}{" "}
                mg/dL
              </dd>
            </div>
          ))}
          <div>
            <dt>Descriptive divergence</dt>
            <dd>
              {num(state.divergence.magnitude, 2)} · {state.divergence.label}
            </dd>
          </div>
        </dl>
        <p className="mt-note">
          Personal expectations describe your previous glucose patterns.
        </p>
      </div>
    );
  if (panel === "forecast")
    return (
      <div className="mt-panel-body">
        <p>
          The forecast estimates glucose at four future times. The displayed peak is approximate.
        </p>
        <h3>
          Estimated spike probability:{" "}
          {num(
            state.spike.probability == null
              ? null
              : state.spike.probability * 100,
            1,
          )}
          %
        </h3>
        <p>
          Alert threshold:{" "}
          {num(state.spike.threshold * 100, 2)}%.
        </p>
        <p>
          {state.spike.definition}. {state.spike.timing_scope}.
        </p>
        <p>
          Data quality: {state.data_quality.status}. Reading age:{" "}
          {num(state.data_quality.observation_age_minutes, 1)} minutes.
        </p>
        <p className="mt-note">
          No individual confidence bounds are provided. A probability below the
          alert threshold does not establish medical safety.
        </p>
      </div>
    );
  if (panel === "technical")
    return (
      <><RiskEvidence evaluation={evidence?.multimodal_risk}/><TechnicalEvidence
        state={state}
        performance={performance}
        onFlow={() => onPanel("flow")}
        onLineage={() => onPanel("lineage")}
      /></>
    );
  const copy: Record<string, [string, string]> = {
    CGM: [
      "Received glucose",
      "Review glucose readings and forecasts over time.",
    ],
    Smartwatch: [
      "Wearable",
      "Review the latest heart-rate reading. Other signals appear when available.",
    ],
    "Meal logger": [
      "Meal Logger",
      "Review logged meals alongside glucose changes.",
    ],
  };
  const [title, text] = copy[panel] ?? [
    "Source context",
    "Available source information",
  ];
  return (
    <div className="mt-panel-body">
      
      
      <details><summary>Source details</summary><p>{sourceRoles[panel as keyof typeof sourceRoles]?.text}</p></details>
      <h3>{title}</h3>
      <p>{text}</p>
      {panel === "CGM" && (
        <p>
          {num(state.current_glucose)} mg/dL ·{" "}
          {clock(state.data_quality.last_observed_time ?? state.timestamp)} UTC
        </p>
      )}
      {panel === "CGM" && (
        <dl>
          <div>
            <dt>Recent history</dt>
            <dd>
              Recent glucose readings over two hours.
            </dd>
          </div>
          <div>
            <dt>Forecast input status</dt>
            <dd>
              {state.data_quality.status}
            </dd>
          </div>
        </dl>
      )}
      {panel === "Smartwatch" && (
        <dl>
          {Object.entries(state.recent_context.activity ?? {})
            .filter(([, reading]) => reading.value != null)
            .map(([name, reading]) => (
              <div key={name}>
                <dt>{name.replaceAll("_", " ")}</dt>
                <dd>
                  {num(reading.value)} {reading.unit} ·{" "}
                  {clock(reading.observed_time)} UTC
                </dd>
              </div>
            ))}
        </dl>
      )}
      {panel === "Meal logger" && (
        <p>
          {state.recent_context.meal
            ? `${num(state.recent_context.meal.minutes_ago)} minutes ago · ${state.recent_context.meal.meal_type ?? "meal type unknown"}`
            : "No recent meal received at this cutoff."}
        </p>
      )}
      
    </div>
  );
}
