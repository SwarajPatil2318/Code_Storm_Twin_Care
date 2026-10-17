export const horizons = [30, 60, 90, 120] as const;
export type HorizonKey = `plus_${(typeof horizons)[number]}m`;
export type Trajectory = Record<HorizonKey, number | null>;
export type Why = {
  signal: string;
  value: number;
  unit: string;
  source: string;
  meaning: string;
  observed?: number;
  expected?: number;
};
export type ContextReading = {
  value: number | null;
  unit: string;
  observed_time: string;
  source: string;
  synthetic: boolean;
  predictive_use: false;
};
export type State = {
  multimodal_risk?: {
    model_version:string;sha256:string;trace_id:string;inference_ms:number;
    static_input:Record<string,string|number|null>;dynamic_input:Record<string,number>;
    probability:number;threshold:number;personal_inputs_used:boolean;shadow_predictive_use:boolean;
    evidence_label:string;input_evidence:string;missing_static_fields:string[];
  };
  reference_spike?: State['spike'];
  patient: {
    id: string;
    name: string;
    synthetic: boolean;
    label: string;
    context: string;
  };
  timestamp: string;
  current_glucose: number | null;
  forecast: Trajectory;
  shadow: Trajectory & {
    current: number | null;
    available: boolean;
    method: string;
    scope: string;
    support: unknown;
  };
  divergence: {
    magnitude: number | null;
    label: string;
    components: Record<string, unknown> | null;
  };
  predicted_peak: {
    glucose: number | null;
    approximate_horizon: number | null;
    scope: string;
  };
  spike: {
    probability: number | null;
    status: string;
    threshold: number;
    expected_time_window_minutes: number[];
    definition: string;
    timing_scope: string;
  };
  why_now: Why[];
  status: string;
  recent_context: {
    meal: {
      minutes_ago: number;
      meal_type: string | null;
      source: string;
    } | null;
    activity: Record<string, ContextReading> | null;
    medication: { text: string; kind: string }[] | null;
    ehr: {
      label: string;
      scope: string;
      demographics: { birth_date: string | null; gender: string | null };
      diagnoses: { text: string }[];
      labs: {
        text: string;
        quantity: { value: number; unit: string } | null;
        observed_time: string;
      }[];
      resources: unknown[];
    } | null;
  };
  data_sources: {
    field: string;
    category: string;
    source: string;
    method: string;
    synthetic_input: boolean;
  }[];
  data_quality: {
    status: string;
    last_observed_time?: string;
    observation_age_minutes?: number;
    receipt_age_minutes?: number;
    warm_prefix?: boolean;
  };
  evidence: {
    model_version: string;
    evidence_label: string;
    input_evidence: string;
    clinical_status: string;
    model_hashes: Record<string, string>;
    forecast_mode: string | null;
  };
  render_hints: {
    current_flow: number | null;
    expected_flow: number | null;
    predicted_flow: Trajectory;
    stream_label: string;
    flow_scale_scope: string;
  };
};
export type Case = { case: string; patient: State["patient"]; frames: number };
export type Replay = {
  type: string;
  label: string;
  state: State;
  history?: unknown[];
};
export type Evidence = {
  multimodal_risk?: import('./RiskDetails').RiskEvaluation;
  name: string;
  model_hashes: Record<string, string>;
  historical_validation: {
    label: string;
    dataset: string;
    holdout_participants: number;
    MAE_mg_dl: number[];
    spike_AUPRC: number;
    spike_AUROC: number;
    scope: string;
  };
};
export type Pack = { frames: State[]; replay: Replay[] };
export const num = (v: number | null | undefined, digits = 0) =>
  v == null
    ? "Unavailable"
    : v.toLocaleString("en-US", { maximumFractionDigits: digits });
export const clock = (stamp: string) =>
  new Date(stamp).toLocaleTimeString("en-GB", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
  });
export const date = (stamp: string) =>
  new Date(stamp).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
export const caseNames: Record<string, string> = {
  predicted_spike: "Predicted spike",
  normal_response: "Aligned response",
  personally_atypical: "Personally atypical",
};
export function reason(item: Why) {
  const value = num(Math.abs(item.value), 1);
  const known: Record<string, [string, string]> = {
    slope_departure_from_personal_norm: [
      item.value > 0.5
        ? "Rising faster than their usual pattern"
        : item.value < -0.5
          ? "Falling faster than their usual pattern"
          : "Slope close to their personal pattern",
      `${value} ${item.unit} ${item.value >= 0 ? "above" : "below"} their historical slope.`,
    ],
    departure_from_expected_now: [
      item.value >= 0
        ? "Above their expected glucose"
        : "Below their expected glucose",
      `${value} mg/dL ${item.value >= 0 ? "above" : "below"} the personal expectation now.`,
    ],
    future_personal_departure: [
      "Predicted response differs from expected",
      `${value} mg/dL ${item.value >= 0 ? "above" : "below"} expected on average over the next 2 hours.`,
    ],
    recent_slope: [
      item.value > 0 ? "Glucose is rising" : "Glucose is steady or falling",
      `${num(item.value, 2)} mg/dL per minute over the last 30 minutes.`,
    ],
    recent_acceleration: [
      "Recent glucose acceleration",
      `${num(item.value, 3)} ${item.unit} from received CGM readings.`,
    ],
    recent_meal: [
      "A meal was recently logged",
      `${num(item.value)} minutes ago. Timing is context, not a medical cause.`,
    ],
    current_glucose: [
      "Latest observed glucose",
      `${num(item.value)} mg/dL from received CGM.`,
    ],
  };
  return (
    known[item.signal] ?? [
      item.signal.replaceAll("_", " "),
      `${num(item.value, 2)} ${item.unit}`,
    ]
  );
}
export function primaryReasons(state: State) {
  const order = [
    "slope_departure_from_personal_norm",
    "departure_from_expected_now",
    "future_personal_departure",
    "recent_slope",
    "current_glucose",
  ];
  return [...state.why_now]
    .sort(
      (a, b) =>
        (order.indexOf(a.signal) < 0 ? 99 : order.indexOf(a.signal)) -
        (order.indexOf(b.signal) < 0 ? 99 : order.indexOf(b.signal)),
    )
    .slice(0, 2);
}
