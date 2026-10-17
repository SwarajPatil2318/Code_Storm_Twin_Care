import { WearableMiniCard } from '../platform/WearableMiniCard';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import {
  Heart,
  Footprints,
  Moon,
  Play,
  Pause,
  Activity,
  Droplet,
  Utensils,
  X,
} from "lucide-react";
import { eventText, type Presentation, type SourceEvent } from "./story";
import { clock, horizons, num, type State } from "./types";

export function FlowHero({ state }: { state: State }) {
  const spread = Math.min(
    1,
    Math.max(0, (state.divergence.magnitude ?? 0) / 3),
  );
  return (
    <div
      className="mt-flow"
      style={
        {
          "--spread": spread,
          "--motion": `${Math.max(7, 13 - (state.render_hints.current_flow ?? 1) * 2)}s`,
        } as CSSProperties
      }
    >
      <img
        src="/assets/metabolic-flow.png"
        alt="Illustrative coral predicted and cyan expected glucose flows"
        fetchPriority="high"
      />
      <svg viewBox="0 0 650 300" aria-hidden="true" className="mt-particles">
        <path
          id="mt-red-flow"
          d="M0 145 C140 70 190 235 320 150 S510 40 650 80"
          fill="none"
        />
        <path
          id="mt-blue-flow"
          d="M0 165 C140 115 200 260 330 180 S500 225 650 230"
          fill="none"
        />
        {[0, 1, 2].map((i) => (
          <g key={i}>
            <circle r="2" fill="#ff706e">
              <animateMotion
                dur="12s"
                begin={`${i * -4}s`}
                repeatCount="indefinite"
              >
                <mpath href="#mt-red-flow" />
              </animateMotion>
            </circle>
            <circle r="2" fill="#31b5ed">
              <animateMotion
                dur="15s"
                begin={`${i * -5}s`}
                repeatCount="indefinite"
              >
                <mpath href="#mt-blue-flow" />
              </animateMotion>
            </circle>
          </g>
        ))}
      </svg>
      <div className="mt-flow-label mt-today">
        <span>Today</span>
        <i />
      </div>
      <div className="mt-flow-label mt-expected">
        <i />
        <span>Expected twin</span>
      </div>
      <small>Illustrative glucose response · not measured anatomy</small>
    </div>
  );
}

export function WatchVisual({state, activity, received = false, heartRateDetail}: {
  state: State; activity?: Presentation["activity"]; received?: boolean; heartRateDetail?: ReactNode;
}) {
  const reading = state.recent_context.activity;
  return <WearableMiniCard mode="metabolic" timestamp={state.timestamp}
    heartRate={reading?.heart_rate?.value} received={received} heartRateDetail={heartRateDetail}
    readings={[
      ...(activity ? [{label: 'Activity', value: activity.active_minutes, unit: 'active min',
        detail: `${activity.level} · past ${activity.window_minutes} min`}] : []),
      ...(reading?.steps?.value != null ? [{label: 'Steps', value: reading.steps.value, unit: reading.steps.unit}] : []),
      ...(reading?.sleep_hours?.value != null ? [{label: 'Sleep', value: reading.sleep_hours.value, unit: reading.sleep_hours.unit}] : []),
    ]} />;
}

export function ForecastChart({
  state,
  observedHistory,
}: {
  state: State;
  observedHistory?: { timestamp: string; glucose_mg_dl: number }[];
}) {
  const svg = useRef<SVGSVGElement>(null),
    [size, setSize] = useState({ width: 600, height: 209 });
  useLayoutEffect(() => {
    if (!svg.current) return;
    const element = svg.current;
    const observer = new ResizeObserver(() => {
      const width = element.clientWidth,
        height = element.clientHeight;
      if (width && height)
        setSize((old) =>
          old.width === width && old.height === height
            ? old
            : { width, height },
        );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const data = useMemo(
    () => ({
      today: [
        state.current_glucose,
        ...horizons.map((h) => state.forecast[`plus_${h}m`]),
      ],
      expected: [
        state.shadow.current,
        ...horizons.map((h) => state.shadow[`plus_${h}m`]),
      ],
    }),
    [state],
  );
  const values = [
    ...data.today,
    ...data.expected,
    ...(observedHistory?.map((e) => e.glucose_mg_dl) ?? []),
  ].filter((n): n is number => n !== null);
  const lo = Math.max(0, Math.floor((Math.min(...values, 90) - 15) / 50) * 50),
    hi = Math.ceil((Math.max(...values, 180) + 15) / 50) * 50;
  const bottom = size.height - 29,
    right = size.width - 18;
  const minimum = observedHistory ? -120 : 0;
  const at = (minute: number) =>
    65 + ((minute - minimum) / (120 - minimum)) * (right - 65);
  const x = (i: number) => at(i * 30),
    y = (v: number) => bottom - ((v - lo) / (hi - lo)) * (bottom - 15);
  const path = (array: (number | null)[]) =>
    array
      .map((v, i) =>
        v === null
          ? ""
          : `${i === 0 || array[i - 1] === null ? "M" : "L"}${x(i)} ${y(v)}`,
      )
      .join(" ");
  return (
    <section
      className="mt-card mt-chart"
      aria-label="Glucose response forecast chart"
    >
      <header>
        <h2>Glucose response — next 2 hours</h2>
        <div className="mt-legend">
          <span>
            <i className="red" />
            {observedHistory ? "Observed / predicted" : "Today (predicted)"}
          </span>
          <span>
            <i className="blue" />
            Expected twin
          </span>
        </div>
      </header>
      <svg
        ref={svg}
        viewBox={`0 0 ${size.width} ${size.height}`}
        role="img"
        aria-label={
          observedHistory
            ? "Received CGM over the past 120 minutes, solid; future forecast, dashed; personal expected glucose, cyan. No uncertainty bands."
            : "Observed now and predicted versus personal expected glucose at 30, 60, 90 and 120 minutes, no uncertainty bands"
        }
      >
        {Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4).map(
          (v) => (
            <g key={v}>
              <line
                x1="65"
                x2={right}
                y1={y(v)}
                y2={y(v)}
                className="mt-gridline"
              />
              <text x="55" y={y(v) + 4} textAnchor="end">
                {num(v)}
              </text>
            </g>
          ),
        )}
        {(observedHistory ? [-120, -60, 0, 60, 120] : [0, 30, 60, 90, 120]).map(
          (h, i) => (
            <g key={h}>
              <line
                x1={at(h)}
                x2={at(h)}
                y1="15"
                y2={bottom}
                className="mt-gridline"
              />
              <text
                x={at(h)}
                y={bottom + 18}
                textAnchor={i === 4 ? "end" : "middle"}
              >
                {h === 0 ? "Now" : `${h} min`}
              </text>
            </g>
          ),
        )}
        <text x="0" y={bottom / 2 - 2} style={{ fontSize: 9 }}>
          Glucose
        </text>
        <text x="0" y={bottom / 2 + 11} style={{ fontSize: 9 }}>
          (mg/dL)
        </text>
        {observedHistory && (
          <>
            <line
              x1={at(0)}
              x2={at(0)}
              y1="15"
              y2={bottom}
              stroke="#adc9da"
              strokeDasharray="3 3"
            />
            <path
              data-testid="observed-history-path"
              d={observedHistory
                .map(
                  (e, i) =>
                    `${i === 0 ? "M" : "L"}${at((Date.parse(e.timestamp) - Date.parse(state.timestamp)) / 60000)} ${y(e.glucose_mg_dl)}`,
                )
                .join(" ")}
              className="mt-observed-line"
            />
          </>
        )}
        <path d={path(data.expected)} className="mt-expected-line" />
        <path
          data-testid="current-forecast-path"
          d={path(data.today)}
          className={`mt-today-line ${observedHistory ? "mt-live-forecast-line" : ""}`}
        />
        {Object.entries(data).map(([key, array]) =>
          array.map(
            (v, i) =>
              v !== null && (
                <circle
                  key={`${key}-${i}`}
                  data-kind={key}
                  data-horizon={i * 30}
                  data-value={v}
                  cx={x(i)}
                  cy={y(v)}
                  r="4.1"
                  fill={key === "today" ? "#ff6368" : "#159ce7"}
                  stroke="white"
                  strokeWidth="1.5"
                >
                  <title>
                    {key === "today"
                      ? i === 0
                        ? "Observed now"
                        : "Predicted"
                      : "Personal expected"}{" "}
                    +{i * 30} min: {num(v, 1)} mg/dL
                  </title>
                </circle>
              ),
          ),
        )}
        <text
          x={size.width / 2}
          y={size.height - 1}
          textAnchor="middle"
          style={{ fontSize: 10 }}
        >
          Time from now
        </text>
      </svg>
    </section>
  );
}

export function ReplayStrip({
  frames,
  index,
  playing,
  onPlay,
  onSelect,
  controls,
  mode,
  knownThrough,
  sourceEvents,
}: {
  frames: State[];
  index: number;
  playing: boolean;
  onPlay: () => void;
  onSelect: (i: number) => void;
  controls?: ReactNode;
  mode?: string;
  knownThrough?: number;
  sourceEvents?: SourceEvent[];
}) {
  const [selectedEvent, setSelectedEvent] = useState<SourceEvent | null>(null);
  useEffect(() => setSelectedEvent(null), [frames]);
  useEffect(() => {
    if (
      selectedEvent &&
      Date.parse(selectedEvent.receipt_time) >
        Date.parse(frames[index].timestamp)
    )
      setSelectedEvent(null);
  }, [index, frames, selectedEvent]);
  const seen = (
    knownThrough == null ? frames : frames.slice(0, knownThrough + 1)
  )
    .flatMap((f) => [f.current_glucose, f.shadow.current])
    .filter((n): n is number => n !== null);
  const lo = Math.min(...seen),
    span = Math.max(10, Math.max(...seen) - lo);
  const x = (i: number) => 15 + (i / Math.max(1, frames.length - 1)) * 910,
    y = (v: number) => 47 - ((v - lo) / span) * 33;
  const curve = (field: "actual" | "expected") =>
    frames
      .map((f, i) => {
        const v =
          knownThrough != null && i > knownThrough
            ? null
            : field === "actual"
              ? f.current_glucose
              : f.shadow.current;
        return v === null ? "" : `${i === 0 ? "M" : "L"}${x(i)} ${y(v)}`;
      })
      .join(" ");
  const hours =
    (Date.parse(frames.at(-1)!.timestamp) - Date.parse(frames[0].timestamp)) /
    3600000;
  const markers = Array.from(
    new Set([
      0,
      Math.round((frames.length - 1) / 4),
      Math.round((frames.length - 1) / 2),
      Math.round(((frames.length - 1) * 3) / 4),
      frames.length - 1,
    ]),
  );
  return (
    <section
      className="mt-card mt-replay"
      aria-label={
        mode === "VERIFIED REPLAY"
          ? "Saved glucose history"
          : "Glucose history"
      }
    >
      <button
        className="mt-play"
        onClick={onPlay}
        aria-label={playing ? "Pause replay" : "Play replay"}
      >
        {playing ? (
          <Pause size={22} fill="currentColor" />
        ) : (
          <Play size={22} fill="currentColor" />
        )}
      </button>
      <div className="mt-replay-label">
        <strong>History · {num(hours)} hours</strong>
        <small>{mode ?? "Received CGM · personal expected"}</small>
        {controls}
      </div>
      <div className="mt-replay-track">
        <div className="mt-event-markers" aria-label="Received source events">
          {(sourceEvents ?? [])
            .filter((event) => event.source !== "ehr")
            .map((event, i) => {
              const position =
                ((Date.parse(event.timestamp) -
                  Date.parse(frames[0].timestamp)) /
                  (Date.parse(frames.at(-1)!.timestamp) -
                    Date.parse(frames[0].timestamp))) *
                100;
              if (position < 0 || position > 100) return null;
              const Icon =
                event.source === "cgm"
                  ? Droplet
                  : event.source === "meal_logger"
                    ? Utensils
                    : Activity;
              return (
                <button
                  key={`${event.source}-${event.timestamp}-${i}`}
                  data-source={event.source}
                  style={{ left: `${position}%` }}
                  title={`${clock(event.timestamp)} UTC · ${eventText(event)}`}
                  aria-label={`${clock(event.timestamp)} UTC · ${eventText(event)}`}
                  onClick={() => setSelectedEvent(event)}
                >
                  <Icon size={9} />
                </button>
              );
            })}
        </div>
        {selectedEvent && (
          <div className="mt-event-popover" role="status">
            <button
              aria-label="Close event"
              onClick={() => setSelectedEvent(null)}
            >
              <X size={13} />
            </button>
            <strong>{clock(selectedEvent.timestamp)} UTC</strong>
            <p>{eventText(selectedEvent)}</p>
            <small>
              {selectedEvent.model_input
                ? "Glucose reading"
                : "Logged activity or meal"}
            </small>
          </div>
        )}
        <svg viewBox="0 0 940 60" preserveAspectRatio="none" aria-hidden="true">
          <path d={curve("actual")} className="mt-today-line" />
          <path d={curve("expected")} className="mt-expected-line" />
          <line
            x1={x(index)}
            x2={x(index)}
            y1="4"
            y2="55"
            stroke="#adc9da"
            strokeDasharray="3 3"
          />
          {frames[index].current_glucose !== null && (
            <circle
              cx={x(index)}
              cy={y(frames[index].current_glucose!)}
              r="5"
              fill="#ff6368"
              stroke="white"
              strokeWidth="2"
            />
          )}
        </svg>
        <input
          type="range"
          min="0"
          max={frames.length - 1}
          value={index}
          onChange={(e) => onSelect(+e.target.value)}
          aria-label="Review reading"
        />
        <div className="mt-replay-ticks">
          {markers.map((i) => (
            <button
              key={i}
              onClick={() => onSelect(i)}
              aria-pressed={index === i}
              aria-label={`Review reading ${i + 1}`}
              style={{ left: `${(i / (frames.length - 1)) * 100}%` }}
            >
              {i === frames.length - 1 ? "Latest" : clock(frames[i].timestamp)}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
