import type { Signal, TwinState } from "./types";
import { names } from "./types";
export function MiniTrend({
  frames,
  signal,
}: {
  frames: TwinState[];
  signal: Signal;
}) {
  const values = frames.map((f) => f.actual_state[signal]),
    seen = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (!seen.length) return null;
  const lo = Math.min(...seen),
    span = Math.max(Math.max(...seen) - lo, 0.01);
  const x = (i: number) =>
      5 + (frames.length === 1 ? 90 : (i / (frames.length - 1)) * 90),
    y = (v: number) => 38 - ((v - lo) / span) * 29;
  const path = values
    .map((v, i) =>
      v === null
        ? ""
        : `${i === 0 || values[i - 1] === null ? "M" : "L"}${x(i)} ${y(v)}`,
    )
    .join(" ");
  const last = values[values.length - 1];
  return (
    <svg
      className="mini-trend"
      viewBox="0 0 102 46"
      role="img"
      aria-label={`${names[signal]} observed trend through the selected replay day`}
    >
      <path
        d={path}
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {last !== null && (
        <circle
          cx={x(values.length - 1)}
          cy={y(last)}
          r="3.5"
          fill="currentColor"
          stroke="white"
          strokeWidth="1.5"
        />
      )}
    </svg>
  );
}
