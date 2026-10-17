export type Signal =
  | "heart_rate"
  | "hrv_rmssd"
  | "systolic_bp"
  | "diastolic_bp"
  | "spo2"
  | "respiration"
  | "weight"
  | "steps"
  | "sleep_hours"
  | "temperature";
export type Patient = {
  id: string;
  name: string;
  age: number;
  synthetic: true;
  label: string;
  story: string;
  context: string;
  note: string;
};
export type Deviation = {
  signal: Signal;
  observed: number | null;
  expected: number | null;
  difference: number | null;
  relative_percent: number | null;
  standardized_residual: number | null;
  recent_48h_deviation_fraction: number | null;
  recent_count: number;
};
export type EHR = {
  demographics: { age: number; gender: string };
  diagnoses: string[];
  medications: string[];
  labs: {
    name: string;
    value: number;
    unit: string;
    observed_time: string;
    available_time: string;
  }[];
  scope: string;
  fhir?: unknown;
};
export type TwinState = {
  patient: Patient;
  timestamp: string;
  replay_day: number;
  status: string;
  risk: {
    next_24h: number;
    next_48h: number;
    next_72h: number;
    likely_window: string;
    bands: Record<string, string>;
    source: string;
  };
  divergence: {
    descriptive_tdi: number;
    status: string;
    components: Record<Signal, number>;
  };
  actual_state: Record<Signal, number | null>;
  shadow_state: Record<
    Signal,
    { value: number | null; lower: number | null; upper: number | null }
  >;
  personal_baseline: Record<
    Signal,
    { reference: number | null; scale: number; trusted_count: number }
  >;
  why_now: {
    personal_deviations: Deviation[];
    interpretation: string;
    missing_data: { signal: Signal; received_recent_count: number }[];
  };
  data_quality: {
    level: string;
    scope: string;
    availability_score: number;
    missing: { signal: Signal; received_recent_count: number }[];
  };
  signal_units: Record<Signal, string>;
  state_aggregation: string;
  ehr_context: EHR;
  evidence: Record<string, string>;
  population_context: { definition: string; broadly_in_core_bands: boolean };
};
export type Metadata = {
  version: string;
  reference: string;
  model_code_sha256: string;
  model_sha256: string;
  config_sha256: string;
  calibration_sha256: string;
  sources: { name: string; role: string; label: string }[];
  limitations: string[];
  cohort_separation: string;
  adaptive_sensing: string;
  selection: string;
  display_bands: string;
};
export type Observation = {
  observation_id: string;
  patient_id: string;
  concept: Signal;
  observed_time: string | null;
  available_time: string | null;
  value: {
    raw_value: number | null;
    usable_value: number | null;
    unit: string;
    quality_status: string;
    reasons: string[];
    provenance: { dataset: string; original_unit: string };
  };
  metadata: { sensor_source: string; scheduled_opportunity_time: string };
};
export const names: Record<Signal, string> = {
  heart_rate: "Heart rate",
  hrv_rmssd: "Heart-rate variability",
  systolic_bp: "Systolic BP",
  diastolic_bp: "Diastolic BP",
  spo2: "Oxygen saturation",
  respiration: "Respiration",
  weight: "Weight",
  steps: "Activity",
  sleep_hours: "Sleep",
  temperature: "Temperature",
};
export const groups: Record<string, Signal[]> = {
  Rhythm: ["heart_rate", "hrv_rmssd"],
  "Fluid & pressure": ["systolic_bp", "diastolic_bp", "weight"],
  Recovery: ["sleep_hours", "spo2", "respiration"],
  Activity: ["steps"],
};
export const fmt = (n: number | null | undefined, digits = 1) =>
  n == null
    ? "Unavailable"
    : n.toLocaleString("en-US", { maximumFractionDigits: digits });
