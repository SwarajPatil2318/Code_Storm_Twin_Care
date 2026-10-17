import { clock, num, type State } from "./types";
import {
  clinicianConsiderations,
  eventText,
  readable,
  type Presentation,
  type WhyEvidence,
} from "./story";
export function stateEvidence(state: State): WhyEvidence[] {
  const evidence: WhyEvidence[] = [];
  if (state.current_glucose != null && state.shadow.current != null)
    evidence.push({
      type: "personal_divergence",
      sources: ["cgm", "shadow_twin"],
      timestamp: state.timestamp,
      classification: "MODEL-DERIVED",
      evidence_level: "direct_comparison",
      causal_claim_allowed: false,
      scientific_signal: "current glucose minus causal D3 Shadow",
      current_glucose: state.current_glucose,
      expected_glucose: state.shadow.current,
      difference_mg_dl: state.current_glucose - state.shadow.current,
    });
  const meal = state.recent_context.meal;
  if (meal && meal.minutes_ago <= 120)
    evidence.push({
      type: "recent_meal",
      sources: ["meal_logger"],
      timestamp: new Date(
        Date.parse(state.timestamp) - meal.minutes_ago * 60000,
      ).toISOString(),
      classification: "CONTEXTUAL",
      evidence_level: "contextual",
      causal_claim_allowed: false,
      scientific_signal: "received meal context only",
      meal_age_minutes: meal.minutes_ago,
      meal_type: meal.meal_type,
    });
  return evidence;
}
export function WhyDetails({
  evidence,
  state,
}: {
  evidence: WhyEvidence[];
  state: State;
}) {
  return (
    <div className="mt-panel-body mt-story-details">
      <p>
        Observed changes and personal comparisons support review. They do not
        establish a medical cause.
      </p>
      {evidence.map((item) => {
        const text = readable(item);
        return (
          <article key={item.type} data-evidence={item.type}>
            <h3>{text.title}</h3>
            
            <h4>What happened</h4>
            <p>{text.what}</p>
            <h4>Why it may matter</h4>
            <p>{text.why}</p>
            {text.comparison && (
              <>
                <h4>Personal comparison</h4>
                <p>{text.comparison}</p>
              </>
            )}
            <small>
              Updated {clock(item.timestamp)} UTC
            </small>
            <details>
              <summary>Inspect prediction</summary>
              <p>{item.scientific_signal}</p>
              <pre>{JSON.stringify(item, null, 2)}</pre>
            </details>
          </article>
        );
      })}
      <details>
        <summary>Inspect original prediction</summary>
        {state.why_now.map((item) => (
          <p key={item.signal}>
            {item.signal} · {num(item.value, 3)} {item.unit} · {item.meaning}
          </p>
        ))}
      </details>
      <section>
        <h3>Clinician considerations</h3>
        <p>
          For review within the existing care plan; no treatment or dosing
          changes are generated.
        </p>
        <ul>
          {clinicianConsiderations.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
export function SourceTrace({
  presentation,
  trace,
}: {
  presentation: Presentation;
  trace: string;
}) {
  return (
    <section className="mt-story-trace">
      <h3>Source → event → Twin update</h3>
      <p>
        Only events received by this cutoff are shown. Context events accompany
        the CGM-triggered update; they do not independently rerun the
        forecaster.
      </p>
      {(presentation.new_events.length
        ? presentation.new_events
        : presentation.source_events.slice(-6)
      ).map((event, i) => (
        <p key={`${event.source}-${event.timestamp}-${i}`}>
          <strong>{eventText(event)}</strong>
          <br />
          <small>
            {clock(event.timestamp)} UTC · received {clock(event.receipt_time)}{" "}
            UTC ·{" "}
            {event.origin === "independent_presentation_schedule"
              ? "Independent simulated context; excluded from model"
              : event.model_input
                ? "CGM model input"
                : "Context; excluded from frozen forecast inputs"}
          </small>
        </p>
      ))}
      <p>
        CGM → METABOLIC_D2_REFERENCE → D3 personal Twin →{" "}
        <strong>{trace}</strong> → forecast updated. Meal, wearable and EHR
        remain parallel context, excluded from the frozen forecast inputs.
      </p>
    </section>
  );
}
