export type SourceEvent = {
  source: string;
  concept: string;
  timestamp: string;
  receipt_time: string;
  value: number | null;
  unit: string | null;
  synthetic: boolean;
  meal_type?: string | null;
  carbohydrate_g?: number | null;
  activity_level?: string;
  origin: string;
  model_input: boolean;
};
export type WhyEvidence = {
  type: string;
  sources: string[];
  timestamp: string;
  classification: string;
  evidence_level: string;
  causal_claim_allowed: false;
  scientific_signal: string;
  meal_age_minutes?: number;
  meal_type?: string | null;
  carbohydrate_g?: number | null;
  glucose_change_since_meal?: number | null;
  comparison_timestamp?: string | null;
  current_glucose?: number;
  expected_glucose?: number;
  difference_mg_dl?: number;
  change_mg_dl?: number;
  window_minutes?: number;
  activity_level?: string;
  active_minutes?: number;
  label?: string;
  model_input?: boolean;
  previous_prediction_time?: string;
  forecast_changes?: Record<string, number>;
};
export type Presentation = {
  evidence: WhyEvidence[];
  source_events: SourceEvent[];
  new_events: SourceEvent[];
  activity: {
    level: string;
    active_minutes: number;
    window_minutes: number;
    timestamp: string;
    label: string;
    model_input: false;
  } | null;
  ehr_status: string;
  llm_used: false;
};
const n = (v: number) =>
  Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 1 });
export const clinicianConsiderations = [
  "Continue monitoring subsequent CGM readings.",
  "Review the recent meal and activity context when available.",
  "Verify the reading if it conflicts with symptoms or sensor quality.",
  "Follow the patient’s existing clinician-defined care plan if glucose crosses its action threshold.",
];
export function readable(item: WhyEvidence) {
  const personal = item.type === "personal_divergence";
  const meal = item.type === "recent_meal";
  const activity = item.type === "recent_activity";
  let title = "Glucose changed between readings",
    what = "",
    why = "",
    comparison = "";
  if (meal) {
    const change = item.glucose_change_since_meal;
    title =
      change != null && change > 0
        ? "Rise followed a recent meal"
        : "A recent meal was logged";
    what = `${item.meal_type ?? "Meal"} logged ${n(item.meal_age_minutes ?? 0)} min ago.`;
    if (item.carbohydrate_g != null)
      what += ` ${n(item.carbohydrate_g)} g carbohydrate recorded.`;
    if (item.meal_age_minutes === 0)
      what += " A later glucose response has not yet been observed.";
    else if (change === 0)
      what +=
        " Glucose is unchanged from the last reading at or before the meal.";
    else if (change != null)
      what += ` Glucose ${change >= 0 ? "rose" : "fell"} ${n(change)} mg/dL since the last reading at or before the meal.`;
    why = "Meal timing is context for review; it does not establish causation.";
  } else if (personal) {
    const diff = item.difference_mg_dl ?? 0;
    title =
      diff > 0
        ? "Above their personal expectation"
        : diff < 0
          ? "Below their personal expectation"
          : "Close to their personal expectation";
    what = `Current glucose is ${n(diff)} mg/dL ${diff >= 0 ? "above" : "below"} the Expected Twin.`;
    why =
      "The personal comparison helps show how today differs from the historical pattern.";
    comparison = `Observed ${n(item.current_glucose ?? 0)} · expected ${n(item.expected_glucose ?? 0)} mg/dL. `;
  } else if (activity) {
    title = "Activity context is available";
    what = `${item.activity_level} activity · ${n(item.active_minutes ?? 0)} active min / ${n(item.window_minutes ?? 0)} min. `;
    why =
      "Activity helps you review the recent glucose pattern.";
  } else if (item.type === "forecast_update") {
    title = "The forecast updated with new CGM history";
    what = Object.entries(item.forecast_changes ?? {})
      .map(
        ([key, value]) =>
          `${key.replace("plus_", "+").replace("m", " min")}: ${value >= 0 ? "+" : "−"}${n(value)} mg/dL`,
      )
      .join(" · ");
    why =
      "Compare successive forecasts to see how the expected response is changing.";
  } else {
    const change = item.change_mg_dl ?? 0;
    title =
      change > 0
        ? "Glucose is rising between readings"
        : change < 0
          ? "Glucose is falling between readings"
          : "Glucose is steady between readings";
    what = `${n(change)} mg/dL ${change >= 0 ? "increase" : "decrease"} over ${n(item.window_minutes ?? 0)} min.`;
    why =
      "New glucose readings help update the forecast.";
  }
  const summary =
    item.type === "forecast_update"
      ? "The forecast updated with new glucose readings."
      : meal
        ? `${item.meal_type ?? "Meal"} logged ${n(item.meal_age_minutes ?? 0)} min ago.${item.carbohydrate_g == null ? "" : ` ${n(item.carbohydrate_g)} g carbohydrate.`}`
        : personal
          ? `${n(item.difference_mg_dl ?? 0)} mg/dL ${(item.difference_mg_dl ?? 0) >= 0 ? "above" : "below"} the Expected Twin.`
          : activity
            ? `${item.activity_level} activity.`
            : `${n(item.change_mg_dl ?? 0)} mg/dL ${(item.change_mg_dl ?? 0) >= 0 ? "increase" : "decrease"} in ${n(item.window_minutes ?? 0)} min.`;
  return { title, what, why, comparison, summary };
}
export const topEvidence = (items: WhyEvidence[]) =>
  [...items]
    .sort(
      (a, b) =>
        [
          "recent_meal",
          "personal_divergence",
          "recent_change",
          "recent_activity",
          "forecast_update",
        ].indexOf(a.type) -
        [
          "recent_meal",
          "personal_divergence",
          "recent_change",
          "recent_activity",
          "forecast_update",
        ].indexOf(b.type),
    )
    .slice(0, 2);
export type DemoSpeed = "1" | "2" | "4" | "8" | "demo";
export const releaseDelay = (speed: DemoSpeed, transitions: number) =>
  speed === "demo" ? 75000 / Math.max(1, transitions) : 8000 / Number(speed);
export function eventText(e: SourceEvent) {
  if (e.source === "meal_logger")
    return `Meal Logger · ${e.meal_type ?? "Meal"} recorded${e.carbohydrate_g == null ? "" : ` · ${n(e.carbohydrate_g)} g carbohydrate`}`;
  if (e.concept === "activity")
    return `Wearable · ${e.activity_level} activity`;
  if (e.source === "smartwatch")
    return "Wearable · Heart rate updated";
  return `CGM · ${e.value == null ? "Reading" : `${n(e.value)} mg/dL`} received · Twin recalculated`;
}
