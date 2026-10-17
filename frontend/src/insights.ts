import { fmt, names } from "./types";
import type { Signal, TwinState } from "./types";

export const metricHelp: Record<Signal, string> = {
  heart_rate:
    "The average of received heart-rate readings over the preceding 48 hours, rather than a single reading or a measured resting rate.",
  hrv_rmssd:
    "Variation between heartbeats, summarized as RMSSD. The useful comparison here is with this person’s expected value, not someone else’s reading.",
  systolic_bp:
    "The upper blood-pressure reading. Compare the observed home-cuff mean with the patient’s expected stable pattern.",
  diastolic_bp:
    "The lower blood-pressure reading. This is observed cuff data; filling pressure is not inferred.",
  weight:
    "Received scale readings averaged over 48 hours. The difference describes a change in recorded weight; it does not establish its cause.",
  steps:
    "Recorded activity in steps per hour. This is an hourly activity summary, not a daily step total.",
  sleep_hours:
    "Recorded sleep duration. Missing nights remain missing; they are not filled with an assumed sleep duration.",
  spo2: "Received oxygen-saturation readings. The comparison describes the observed value and its personal expectation.",
  respiration:
    "Received breathing-rate readings, compared with the personal expected stable state.",
  temperature:
    "Received temperature readings with their original units and quality status.",
};
export const groupHelp: Record<
  string,
  { question: string; explanation: string }
> = {
  Rhythm: {
    question: "Is their heart rhythm behaving like their usual pattern?",
    explanation:
      "Look at heart rate and heart-rate variability together. The chart shows what was received and what the Shadow Twin expected at each cutoff.",
  },
  "Fluid & pressure": {
    question: "How have weight and blood-pressure readings changed?",
    explanation:
      "These are recorded home-scale and cuff readings. A personal change is useful context, but it does not tell us the cause or estimate filling pressure.",
  },
  Recovery: {
    question: "What do sleep, oxygen and breathing readings show?",
    explanation:
      "Compare the available readings with the patient’s own expected values. Improving readings do not guarantee recovery; alerts may still require review.",
  },
  Activity: {
    question: "Has movement changed from their personal pattern?",
    explanation:
      "Activity is shown in steps per hour. Compare this received 48-hour average with the expected personal pattern, rather than a universal daily step goal.",
  },
};
export function difference(state: TwinState, signal: Signal) {
  const actual = state.actual_state[signal],
    expected = state.shadow_state[signal].value;
  if (actual === null || expected === null)
    return {
      delta: null,
      percent: null,
      text: "Not enough received data",
      direction: "unavailable",
    };
  const delta = actual - expected,
    percent =
      Math.abs(expected) > 1e-6 ? (delta / Math.abs(expected)) * 100 : null;
  return {
    delta,
    percent,
    text: `${fmt(Math.abs(delta))} ${state.signal_units[signal]} ${delta < 0 ? "below" : "above"} expected`,
    direction: delta < 0 ? "below" : "above",
  };
}
export function readingMeaning(state: TwinState, signal: Signal) {
  const { lower, upper } = state.shadow_state[signal],
    a = state.actual_state[signal];
  if (a === null)
    return "No usable received value is available at this cutoff. Missing data is shown explicitly.";
  if (lower === null || upper === null)
    return "The expected range is unavailable. This reading cannot be compared with a personal band.";
  if (a < lower)
    return "Below the estimated personal band. It is a departure from the expected stable pattern, not a diagnosis.";
  if (a > upper)
    return "Above the estimated personal band. It is a departure from the expected stable pattern, not a diagnosis.";
  return "Within the estimated personal band. Small differences are part of the comparison; this alone does not rule out deterioration.";
}
export function driverTitle(state: TwinState, signal: Signal) {
  const d = difference(state, signal);
  return d.percent === null
    ? `${names[signal]}: data unavailable`
    : `${names[signal]} is ${fmt(Math.abs(d.percent), 0)}% ${d.direction} expected`;
}
