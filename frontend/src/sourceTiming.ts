import type { Observation, Signal } from "./types";

export function sourceReceipt(
  observations: Observation[],
  signals: Signal[],
  cutoff: string,
) {
  const until = Date.parse(cutoff);
  const received = observations.filter(
    (o) =>
      signals.includes(o.concept) &&
      o.value.usable_value !== null &&
      o.available_time !== null &&
      Date.parse(o.available_time) <= until &&
      o.observed_time !== null &&
      Date.parse(o.observed_time) <= until,
  );
  const latest = received.reduce<Observation | null>(
    (best, o) =>
      !best || Date.parse(o.available_time!) > Date.parse(best.available_time!)
        ? o
        : best,
    null,
  );
  if (!latest)
    return {
      label: "No received reading",
      detail: "No usable receipt is available at this replay cutoff.",
      available: false,
    };
  const hours = (until - Date.parse(latest.available_time!)) / 3600000;
  const age =
    hours < 1
      ? "<1h ago"
      : hours < 24
        ? `${Math.floor(hours)}h ago`
        : `${Math.floor(hours / 24)}d ago`;
  return {
    label: `Synced ${age}`,
    detail: `Reading received ${latest.available_time}; observed ${latest.observed_time}. Age is relative to the selected time.`,
    available: true,
  };
}
