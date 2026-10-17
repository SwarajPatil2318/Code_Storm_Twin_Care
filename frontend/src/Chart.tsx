import type { Signal, TwinState } from "./types";
import { fmt, names } from "./types";
export function Trend({
  frames,
  metric,
  risk = false,
}: {
  frames: TwinState[];
  metric: Signal;
  risk?: boolean;
}) {
  const series = frames.map((s) => ({
    day: s.replay_day,
    actual: risk ? s.risk.next_72h * 100 : s.actual_state[metric],
    expected: risk ? null : s.shadow_state[metric].value,
    low: risk ? null : s.shadow_state[metric].lower,
    high: risk ? null : s.shadow_state[metric].upper,
  }));
  const vals = series
    .flatMap((s) => [s.actual, s.expected, s.low, s.high])
    .filter((x): x is number => x !== null && Number.isFinite(x));
  const min = Math.min(...vals),
    max = Math.max(...vals),
    pad = Math.max((max - min) * 0.2, risk ? 1 : 1),
    lo = min - pad,
    hi = max + pad;
  const x = (day: number) => 54 + (day + 7) * 66;
  const y = (v: number) => 196 - ((v - lo) / (hi - lo)) * 164;
  const line = (key: "actual" | "expected") =>
    series
      .map((s, i) =>
        s[key] === null
          ? ""
          : `${i === 0 || series[i - 1][key] === null ? "M" : "L"}${x(s.day)},${y(s[key]!)}`,
      )
      .join(" ");
  const band = series.every((s) => s.low !== null && s.high !== null)
    ? series
        .map((s) => `${x(s.day)},${y(s.high!)}`)
        .concat([...series].reverse().map((s) => `${x(s.day)},${y(s.low!)}`))
        .join(" ")
    : "";
  return (
    <figure className="trend">
      <svg
        viewBox="0 0 548 236"
        role="img"
        aria-label={`${risk ? "Estimated 72-hour risk" : names[metric]} through selected replay day. Actual coral, expected cyan; band estimated, not clinical confidence.`}
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="54"
              x2="520"
              y1={y(lo + (hi - lo) * t)}
              y2={y(lo + (hi - lo) * t)}
              stroke="#e6e9e6"
            />
            <text
              x="43"
              y={y(lo + (hi - lo) * t) + 4}
              textAnchor="end"
              className="chart-label"
            >
              {fmt(lo + (hi - lo) * t, risk ? 1 : 0)}
            </text>
          </g>
        ))}
        {band && <polygon points={band} fill="#d4e8ec" opacity=".5" />}
        {!risk && (
          <path
            d={line("expected")}
            fill="none"
            stroke="#469faf"
            strokeWidth="2"
            strokeDasharray="5 4"
          />
        )}
        <path
          d={line("actual")}
          fill="none"
          stroke="#c96d5d"
          strokeWidth="2.7"
          strokeLinejoin="round"
        />
        {series.map((s) => (
          <g key={s.day}>
            {s.actual !== null && (
              <circle cx={x(s.day)} cy={y(s.actual)} r="3.5" fill="#c96d5d" />
            )}
            <text
              x={x(s.day)}
              y="225"
              textAnchor="middle"
              className="chart-label"
            >
              {s.day === 0 ? "Today" : `${s.day}d`}
            </text>
          </g>
        ))}
      </svg>
      <figcaption>
        {risk
          ? "Cached original Phase 6 cumulative risk."
          : `48-hour received usable means · ${frames[0]?.signal_units[metric] ?? ""}. Earlier points are forecasts issued at their own cutoffs.`}
      </figcaption>
    </figure>
  );
}
