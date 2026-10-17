import type { State } from "./types";
const observed = "OBSERVED / REPLAYED",
  synthetic = "SYNTHETIC DEMO PATIENT",
  predicted = "MODEL PREDICTED",
  personal = "PERSONAL EXPECTED",
  real = "REAL OPEN-DATA MODEL EVIDENCE";
export function fieldCategory(field: string, state: State): string {
  if (
    field.startsWith("patient") ||
    field.startsWith("recent_context") ||
    field.startsWith("evidence.input_evidence") ||
    field === "render_hints.stream_label"
  )
    return synthetic;
  if (field.startsWith("multimodal_risk.static_input")) return synthetic;
  if (field.startsWith("multimodal_risk.dynamic_input.glucose_lag")) return observed;
  if (field.startsWith("evidence.risk")) return "DEVELOPMENT-VALIDATED MULTIMODAL RISK HEAD";
  if (field.startsWith("evidence")) return real;
  if (
    field.startsWith("shadow") ||
    field.startsWith("divergence") ||
    field === "render_hints.expected_flow"
  )
    return personal;
  if (
    field === "render_hints.current_flow" ||
    field.startsWith("current_glucose") ||
    field.startsWith("timestamp") ||
    field.startsWith("data_quality") ||
    field.startsWith("data_sources")
  )
    return observed;
  if (field.startsWith("why_now")) {
    const item = state.data_sources
      .filter((s) => field === s.field || field.startsWith(s.field + "."))
      .sort((a, b) => b.field.length - a.field.length)[0];
    return item?.category === "PERSONAL EXPECTED" ? personal : observed;
  }
  return predicted;
}
export function lineageRows(state: State) {
  const fields = [
    ...new Set([
      ...Object.keys(state),
      ...state.data_sources.map((s) => s.field),
      "evidence.input_evidence",
      "render_hints.current_flow",
      "render_hints.expected_flow",
      "render_hints.stream_label",
    ]),
  ];
  return fields.map((field) => ({
    field,
    category: fieldCategory(field, state),
    source:
      state.data_sources.find((s) => s.field === field)?.source ??
      "Inherited field lineage",
    method:
      state.data_sources.find((s) => s.field === field)?.method ??
      "Descriptive metadata; inspect parent origin",
  }));
}
