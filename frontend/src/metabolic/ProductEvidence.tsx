import { lineageRows } from "./provenance";
import { useEffect, useState } from "react";
import { Activity, Database, History, Info, Watch } from "lucide-react";
import { clock, horizons, num, type State } from "./types";

export const sourceRoles = {
  CGM: {
    label: "SIMULATED CGM STREAM",
    role: "PRIMARY VALIDATED FORECAST INPUT",
    text: "Recent CGM dynamics are the primary validated input to the short-horizon forecast. This public replay uses synthetic CGM observations; real model validation is separate.",
  },
  "EHR / FHIR": {
    label: "SYNTHETIC DEMO EHR",
    role: "SAFE HISTORY INPUT TO MULTIMODAL RISK",
    text: "Clinical history enters the separate two-hour spike-risk model. Short-horizon glucose forecasting is driven primarily by recent CGM patterns. Development evaluation does not show an accuracy gain over dynamic-only risk. Displayed EHR is synthetic and does not belong to a real Shanghai participant.",
  },
  Smartwatch: {
    label: "SIMULATED LIVE STREAM",
    role: "SIMULATED BEHAVIORAL CONTEXT",
    text: "Simulated wearable context for the product demonstration. It supplies patient context; its predictive contribution is not independently validated.",
  },
  "Meal logger": {
    label: "DEMO / RECENT CONTEXT",
    role: "RECENT EVENT CONTEXT",
    text: "Recent logged meal timing provides context. It does not establish nutritional causal effects or enter the frozen production forecast.",
  },
} as const;
export type ProductEvidence = {
  model_version: string;
  claim: string;
  d2: {
    participants: number;
    dataset: string;
    holdout_participants: number;
    holdout_windows: number;
    holdout_status: string;
    MAE_mg_dl: number[];
    persistence_MAE_mg_dl: number[];
    spike_AUPRC: number;
    spike_AUROC: number;
  };
  fusion: {
    participants: number;
    folds: number;
    windows: number;
    scope: string;
    static_features_used: string[];
    dynamic_MAE_mg_dl: number[];
    fused_MAE_mg_dl: number[];
    delta_MAE_mg_dl: number[];
    bootstrap_ci95: number[][];
    bootstrap_replicates: number;
    shuffled_static_120m_MAE: number;
    fused_minus_shuffled_120m_ci95: number[];
    static_permutation_120m_MAE_increase: number;
    dynamic_permutation_120m_MAE_increase: number;
    conclusion: string;
    negative_control_explanation: string;
    permutation_scope: string;
    decision: string;
    fusion_supported: boolean;
  };
  provenance_categories: Record<string, string>;
  limitations: string[];
};
export function DataLineage({ state }: { state: State }) {
  return (
    <div className="mt-panel-body mt-lineage">
      <p>
        Real model evidence, demo inputs and derived estimates have different
        origins. Current CGM replay is synthetic, even though the forecasting
        model has real open-data evidence.
      </p>
      <table>
        <caption>
          Every TwinState field has an origin; nested values inherit their
          closest mapped parent.
        </caption>
        <thead>
          <tr>
            <th>Field</th>
            <th>Provenance</th>
            <th>Source / method</th>
          </tr>
        </thead>
        <tbody>
          {lineageRows(state).map((row) => (
            <tr key={row.field}>
              <td>{row.field}</td>
              <td>{row.category}</td>
              <td>
                {row.source}
                <small>{row.method}</small>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-note">
        Observed/replayed does not mean real clinical inputs: this patient,
        wearable and EHR context are synthetic demo data. Expected state is
        exploratory, and predictions remain model estimates.
      </p>
    </div>
  );
}
export function HowTwinWorks() {
  return (
    <div className="mt-panel-body">
      <p>
        The Twin combines clinical context and live sensor state while keeping
        the validated glucose forecaster unchanged.
      </p>
      <div className="mt-information-flow">
        <article>
          <Database size={20} />
          <h3>Historical context</h3>
          <p>Clinical history and CGM patterns inform two-hour spike risk.</p>
          <small>
            Demographics, diagnosis and medications · synthetic demo context;
            clinical history is incorporated in the two-hour risk model.
          </small>
        </article>
        <article>
          <Activity size={20} />
          <h3>Live physiology</h3>
          <p>Sensor streams tell the Twin what is happening now.</p>
          <small>
            Received CGM · primary validated forecast input; demo replay is
            synthetic.
          </small>
        </article>
        <article>
          <Watch size={20} />
          <h3>Behavioral context</h3>
          <p>Wearable and meal streams supply recent context.</p>
          <small>
            Simulated heart rate and meal timing · no demonstrated nutritional
            causal effects.
          </small>
        </article>
        <article>
          <History size={20} />
          <h3>Personal history</h3>
          <p>
            Personal history tells the Twin what is normal for this patient.
          </p>
          <small>
            Historical CGM supports eligible prefix features and the separate
            causal Shadow expectation.
          </small>
        </article>
      </div>
      <div className="mt-flow-output">
        <span aria-hidden="true">↓</span>
        <h3>Metabolic Twin</h3>
        <p>The Glucose Forecast Model estimates the trajectory. Clinical history, CGM patterns and that trajectory inform two-hour spike risk.</p>
        <small>
          Current state · Expected state · +30 / +60 / +90 / +120 forecast ·
          Multimodal spike risk · WhyNow
        </small>
      </div>
      <p className="mt-note">
        Clinical and behavioral context are integrated into the patient view;
        their presence does not mean they enter the production forecasting
        model. No model runs in the browser.
      </p>
    </div>
  );
}
let cached: ProductEvidence | undefined,
  pending: Promise<ProductEvidence> | undefined;
function loadEvidence() {
  if (cached) return Promise.resolve(cached);
  if (!pending)
    pending = fetch("/metabolic-api/evidence/metabolic")
      .then(async (response) => {
        if (!response.ok)
          throw new Error("Technical evidence could not be loaded.");
        return response.json() as Promise<ProductEvidence>;
      })
      .then((data) => {
        cached = data;
        return data;
      })
      .finally(() => {
        pending = undefined;
      });
  return pending;
}
export function TechnicalEvidence({
  state,
  performance,
  onFlow,
  onLineage,
}: {
  state: State;
  performance: Record<string, { median: number; p95: number }> | null;
  onFlow: () => void;
  onLineage: () => void;
}) {
  const [data, setData] = useState(cached),
    [error, setError] = useState(""),
    [retry, setRetry] = useState(0);
  useEffect(() => {
    let cancelled = false;
    loadEvidence()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setError("");
        }
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [retry]);
  if (!data)
    return (
      <div className="mt-panel-body" role="status">
        <p>{error || "Loading model evidence…"}</p>
        {error && (
          <button className="mt-action" onClick={() => setRetry((i) => i + 1)}>
            Retry evidence
          </button>
        )}
      </div>
    );
  const f = data.fusion;
  return (
    <div className="mt-panel-body mt-technical">
      <p>Glucose forecasting is evaluated on ShanghaiT2DM participant data. Source provenance and evaluation scope are shown below.</p>
      <div className="mt-evidence-actions">
        <button onClick={onFlow}>
          <Info size={15} />
          How this Twin works
        </button>
        <button onClick={onLineage}>
          <Database size={15} />
          Data lineage
        </button>
      </div>
      <section aria-label="Real-data forecasting">
        <h3>Real-data forecasting</h3>
        <p>
          Glucose Forecast Model · {data.d2.dataset} · {data.d2.participants} Type
          2 Diabetes participants.
        </p>
        <p>
          Independent participant-level holdout evaluation: {data.d2.holdout_participants}{" "}
          participants · {num(data.d2.holdout_windows)} forecast windows.{" "}
          Training and evaluation participants were separated.
        </p>
        <table data-testid="d2-metrics">
          <caption>
            Participant-level holdout MAE (mg/dL)
          </caption>
          <thead>
            <tr>
              <th>Horizon</th>
              <th>Forecast model</th>
              <th>Persistence</th>
            </tr>
          </thead>
          <tbody>
            {horizons.map((h, i) => (
              <tr key={h}>
                <td>+{h} min</td>
                <td>{data.d2.MAE_mg_dl[i].toFixed(2)}</td>
                <td>{data.d2.persistence_MAE_mg_dl[i].toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          All horizons beat persistence. Spike ranking: AUPRC{" "}
          {data.d2.spike_AUPRC.toFixed(3)} · AUROC{" "}
          {data.d2.spike_AUROC.toFixed(3)}.
        </p>
        <p className="mt-note">
          Historical model evidence does not validate this synthetic patient,
          its displayed EHR, wearable streams or Shadow response.
        </p>
      </section>
      <section aria-label="Fusion evidence">
        <h3>Clinical-history forecasting evaluation</h3>
        <p>
          {f.participants} development participants · {f.folds}-fold
          participant-grouped CV · {num(f.windows)} windows.
        </p>
        <p>
          Safe historical fields: age, sex, diabetes duration, smoking history,
          alcohol history and coarse historical medication context.
        </p>
        <p>
          Excluded: labs, BMI/weight/height with ambiguous timing, undated
          diagnoses/complications and synthetic EHR.
        </p>
        <table data-testid="fusion-metrics">
          <caption>Participant-level cross-validation MAE (mg/dL)</caption>
          <thead>
            <tr>
              <th>Variant</th>
              {horizons.map((h) => (
                <th key={h}>+{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["Dynamic", f.dynamic_MAE_mg_dl],
              ["Fused", f.fused_MAE_mg_dl],
              ["Δ MAE", f.delta_MAE_mg_dl],
            ].map(([name, values]) => (
              <tr key={name as string}>
                <th scope="row">{name as string}</th>
                {(values as number[]).map((value, i) => (
                  <td key={i}>
                    {name === "Δ MAE" && value > 0 ? "+" : ""}
                    {value.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p data-testid="fusion-ci">
          95% participant-bootstrap CI at +120m: [
          {f.bootstrap_ci95[3][0].toFixed(3)},{" "}
          {f.bootstrap_ci95[3][1].toFixed(3)}] mg/dL ·{" "}
          {num(f.bootstrap_replicates)} participant resamples.
        </p>
        <p className="mt-evidence-conclusion">Recent CGM patterns drive short-horizon forecasting. This evaluation does not establish a forecasting accuracy benefit from clinical history.</p>
        <details><summary>Developer evaluation details</summary><p>{f.conclusion}</p><p>Decision: {f.decision}. {data.d2.holdout_status}.</p></details>
        <details>
          <summary>Inspect shuffled-static negative control</summary>
          <dl>
            <div>
              <dt>Shuffled-static +120 MAE</dt>
              <dd>{f.shuffled_static_120m_MAE.toFixed(3)} mg/dL</dd>
            </div>
            <div>
              <dt>Correctly linked fused</dt>
              <dd>{f.fused_MAE_mg_dl[3].toFixed(3)} mg/dL</dd>
            </div>
            <div>
              <dt>Dynamic-only</dt>
              <dd>{f.dynamic_MAE_mg_dl[3].toFixed(3)} mg/dL</dd>
            </div>
          </dl>
          <p>{f.negative_control_explanation}</p>
          <p>
            Fused-minus-shuffled +120 MAE CI: [
            {f.fused_minus_shuffled_120m_ci95
              .map((v) => v.toFixed(3))
              .join(", ")}
            ] mg/dL. One deterministic whole-profile training permutation per
            fold.
          </p>
        </details>
        <details>
          <summary>Inspect feature reliance</summary>
          <dl>
            <div>
              <dt>Static-profile permutation MAE increase</dt>
              <dd>{f.static_permutation_120m_MAE_increase.toFixed(3)} mg/dL</dd>
            </div>
            <div>
              <dt>Dynamic-feature permutation MAE increase</dt>
              <dd>
                {f.dynamic_permutation_120m_MAE_increase.toFixed(3)} mg/dL
              </dd>
            </div>
          </dl>
          <p>Dynamic physiology dominates short-horizon forecasting.</p>
          <p className="mt-note">
            {f.permutation_scope} Dynamic perturbation includes the
            current-glucose offset. These comparisons are not causal effects or
            percentages of model gain.
          </p>
        </details>
      </section>
      <section aria-label="Data provenance">
        <h3>Data provenance</h3>
        <dl>
          {Object.entries(data.provenance_categories).map(
            ([category, text]) => (
              <div key={category}>
                <dt>{category}</dt>
                <dd>{text}</dd>
              </div>
            ),
          )}
        </dl>
        <button className="mt-action" onClick={onLineage}>
          Inspect TwinState lineage
        </button>
      </section>
      <section aria-label="Technical limitations">
        <h3>Technical limitations</h3>
        <ul>
          {data.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
      <details>
        <summary>Developer details: model identity and measured latency</summary>
        <pre>{JSON.stringify(state.evidence.model_hashes, null, 2)}</pre>
        {performance && (
          <dl>
            {Object.entries(performance).map(([name, v]) => (
              <div key={name}>
                <dt>{name.replaceAll("_", " ")}</dt>
                <dd>
                  Median {num(v.median, 2)} ms · p95 {num(v.p95, 2)} ms
                </dd>
              </div>
            ))}
          </dl>
        )}
      </details>
    </div>
  );
}
