import { clock, horizons, num } from "./types";
import type { State } from "./types";
import type { LiveResult } from "./useLiveTwin";
import { SourceTrace } from "./StoryDetails";

export function LiveDetails({
  result,
  state,
  eventLog,
  onEvidence,
  onLog,
}: {
  result: LiveResult | null;
  state: State;
  eventLog: boolean;
  onEvidence: () => void;
  onLog: () => void;
}) {
  if (!result)
    return (
      <div className="mt-panel-body">
        <p>
          VERIFIED REPLAY · cached D3 TwinState. No runtime inference is claimed
          for this view. Start Live Twin to inspect an actual backend update.
        </p>
        <button className="mt-action" onClick={onEvidence}>
          View technical evidence
        </button>
      </div>
    );
  const { update, forecast_history, event_log } = result;
  return (
    <div className="mt-panel-body mt-live-details">
      <p>
        <strong>{update.mode}</strong> · SIMULATED CGM STREAM ·{" "}
        {update.trace_id}
      </p>
      <div className="mt-evidence-actions">
        <button className="mt-action" onClick={onEvidence}>
          View technical evidence
        </button>
        {!eventLog && <button onClick={onLog}>Live event log</button>}
      </div>
      {result.presentation && (
        <SourceTrace
          presentation={result.presentation}
          trace={update.trace_id}
        />
      )}
      {eventLog ? (
        <section>
          <h3>Received updates</h3>
          <p>
            Session-only; up to 256 updates retained. All timings are measured
            locally.
          </p>
          {[...event_log].reverse().map((event) => (
            <article key={event.trace_id}>
              <strong>
                {clock(event.timestamp)} UTC · CGM {num(event.glucose_mg_dl)}{" "}
                mg/dL
              </strong>
              <p>{event.stages.join(" → ")}</p>
              <small>
                {event.trace_id} · D2 inference{" "}
                {num(event.timings_ms.model_inference_ms, 3)} ms
              </small>
            </article>
          ))}
        </section>
      ) : (
        <>
          <section>
            <h3>Input state</h3>
            <dl className="mt-live-facts">
              <dt>Current glucose</dt>
              <dd>{num(update.input.current_glucose)} mg/dL</dd>
              <dt>CGM history</dt>
              <dd>
                {update.input.usable_cgm_observations} observations ·{" "}
                {update.input.history_window_minutes} min
              </dd>
              <dt>Recent slope</dt>
              <dd>{num(update.input.recent_slope, 3)} mg/dL/min</dd>
              <dt>Personal baseline</dt>
              <dd>
                {update.input.personal_baseline_available
                  ? "Available"
                  : "Insufficient personal history"}{" "}
                · {update.input.personal_prefix_observations} earlier
                observations
              </dd>
              <dt>Observation / receipt</dt>
              <dd>
                {update.input.event_time}
                <br />
                {update.input.receipt_time}
              </dd>
              <dt>Recent meal</dt>
              <dd>
                {state.recent_context.meal
                  ? `${num(state.recent_context.meal.minutes_ago)} minutes ago · synthetic context`
                  : "No recent received meal"}
              </dd>
              <dt>Clinical context</dt>
              <dd>
                {update.input.static_clinical_context_available
                  ? "Synthetic demo EHR available; excluded from D2 inputs"
                  : "No received EHR context"}
              </dd>
            </dl>
          </section>
          <section>
            <h3>Engine</h3>
            <p>{update.pipeline.join(" → ")}</p>
            <p>
              {update.model_version} · horizons 30 / 60 / 90 / 120 minutes
              <br />
              Inference at {update.inference_timestamp}
            </p>
            <table>
              <thead>
                <tr>
                  <th scope="col">Measured stage</th>
                  <th scope="col">Milliseconds</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(update.timings_ms).map(([name, value]) => (
                  <tr key={name}>
                    <td>{name.replaceAll("_", " ")}</td>
                    <td>{num(value, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-note">
              Endpoint timing excludes network transfer and browser rendering.
              No artificial processing delay is included.
            </p>
            <details>
              <summary>Frozen model hashes</summary>
              {Object.entries(update.model_hashes).map(([name, value]) => (
                <p key={name}>
                  {name}: <code>{value}</code>
                </p>
              ))}
            </details>
          </section>
          <section>
            <h3>Output</h3>
            <table>
              <thead>
                <tr>
                  <th scope="col">Horizon</th>
                  <th scope="col">D2 predicted</th>
                  <th scope="col">D3 expected</th>
                </tr>
              </thead>
              <tbody>
                {horizons.map((h) => (
                  <tr key={h}>
                    <td>+{h} min</td>
                    <td>{num(state.forecast[`plus_${h}m`], 3)}</td>
                    <td>{num(state.shadow[`plus_${h}m`], 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>
              Peak {num(state.predicted_peak.glucose, 3)} mg/dL · approximately
              +{state.predicted_peak.approximate_horizon} min.
            </p>
            <p>
              Spike probability{" "}
              {state.spike.probability === null
                ? "Unavailable"
                : `${num(state.spike.probability * 100, 3)}%`}{" "}
              · {state.spike.status}
            </p>
            <p>
              {state.divergence.label} · divergence{" "}
              {num(state.divergence.magnitude, 3)}
            </p>
          </section>
          <section>
            <h3>Forecast history and later observations</h3>
            <p>
              Up to 128 session snapshots retained. Comparisons appear only
              after the target observation is received. These synthetic examples
              are not model validation.
            </p>
            {[...forecast_history].reverse().map((snapshot) => (
              <details key={snapshot.trace_id} data-testid="forecast-snapshot">
                <summary>
                  {clock(snapshot.prediction_time)} UTC · {snapshot.trace_id} ·{" "}
                  {snapshot.comparisons.length} matured comparisons
                </summary>
                <p>
                  trajectory {snapshot.model_version} · risk {snapshot.risk_model_version ?? "historical D2 risk"} · spike{" "}
                  {num(snapshot.spike_probability * 100, 1)}% · peak{" "}
                  {num(snapshot.peak.glucose, 1)} mg/dL
                </p>
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Horizon</th>
                      <th scope="col">Predicted</th>
                      <th scope="col">Later observed</th>
                      <th scope="col">Absolute error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {horizons.map((h) => {
                      const comparison = snapshot.comparisons.find(
                        (c) => c.horizon_minutes === h,
                      );
                      return (
                        <tr key={h}>
                          <td>+{h} min</td>
                          <td>{num(snapshot.forecast[`plus_${h}m`], 3)}</td>
                          <td>
                            {comparison
                              ? num(comparison.observed, 3)
                              : "Not yet observed"}
                          </td>
                          <td>
                            {comparison
                              ? `${num(comparison.absolute_error_mg_dl, 3)} mg/dL`
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </details>
            ))}
          </section>
        </>
      )}
      <section>
        <h3>Provenance</h3>
        <p>
          Observed/replayed: synthetic CGM. Predicted: frozen D2 model,
          separately validated on real open data. Expected: D3 causal personal
          Shadow. Clinical context: synthetic demo EHR. Wearable: simulated live
          context, excluded from D2.
        </p>
      </section>
    </div>
  );
}
