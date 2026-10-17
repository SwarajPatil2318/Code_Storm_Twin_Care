import {num,type State} from './types';
export type RiskEvaluation={decision:string;scope:string;participants:number;folds:number;metrics:Record<string,{AUPRC:number;AUROC:number;Brier:number}>;final?:{sha256:string;threshold:number}};
export function RiskDetails({state}:{state:State}) {
 const risk=state.multimodal_risk;
 if(!risk)return <div className="mt-panel-body"><p>Multimodal risk is unavailable in this historical reference state. Start the multimodal runtime to inspect both input streams.</p></div>;
 return <div className="mt-panel-body" data-testid="risk-details">
  <p>Live physiology + patient history → multimodal risk model → next 2-hour incident spike probability.</p>
  <p className="mt-source-role">SYNTHETIC DEMO EHR · SYNTHETIC CGM</p>
  <h3>Live physiology</h3><p>✓ CGM · recent trajectory · frozen D2 future forecast</p>
  <dl>{Object.entries(risk.dynamic_input).map(([name,value])=><div key={name}><dt>{name.replaceAll('_',' ')}</dt><dd>{num(value,3)}</dd></div>)}</dl>
  <h3>Patient history</h3><p>These values actually enter the risk head. Missing values retain missingness indicators and training-only imputation.</p>
  <dl>{Object.entries(risk.static_input).map(([name,value])=><div key={name}><dt>{name.replaceAll('_',' ')}</dt><dd>{value==null?'Not supplied (explicit missingness)':String(value)}</dd></div>)}</dl>
  <h3>Personal twin</h3><p>Expected response and personal baseline remain visible comparisons. Shadow and personal-prefix features are not inputs to this risk model.</p>
  <h3>Multimodal risk model</h3><dl>
   <div><dt>Engine / version</dt><dd>{risk.model_version}</dd></div><div><dt>Model SHA256</dt><dd className="mt-mono">{risk.sha256}</dd></div>
   <div><dt>Trace</dt><dd>{risk.trace_id}</dd></div><div><dt>Inference latency</dt><dd>{num(risk.inference_ms,2)} ms</dd></div>
   <div><dt>Probability</dt><dd data-testid="exact-risk">{risk.probability}</dd></div><div><dt>Threshold</dt><dd>{risk.threshold}</dd></div>
   <div><dt>Status</dt><dd>{state.spike.status}</dd></div></dl>
  <p>{risk.evidence_label}. Historical context and dynamic physiology are jointly ingested; short-horizon signal remains dominated by CGM. No incremental accuracy benefit from EHR is claimed.</p>
 </div>;
}
export function RiskEvidence({evaluation}:{evaluation?:RiskEvaluation}) {
 if(!evaluation)return null;
 return <section className="mt-fusion-evidence"><h3>Multimodal adverse-event risk</h3><p>{evaluation.scope}</p>
  <p>{evaluation.participants} development participants · {evaluation.folds} participant folds · {evaluation.decision}</p>
  <table><thead><tr><th>Inputs</th><th>AUPRC</th><th>AUROC</th><th>Brier</th></tr></thead><tbody>
   {Object.entries(evaluation.metrics).map(([name,m])=><tr key={name}><th>{name.replaceAll('_',' ')}</th><td>{num(m.AUPRC,4)}</td><td>{num(m.AUROC,4)}</td><td>{num(m.Brier,4)}</td></tr>)}
  </tbody></table><p>Both streams are computational inputs to the final adverse-event risk head.</p>
  <p>Fusion did not improve the point estimates over dynamic-only. Correct linkage outperformed shuffled static; CGM dominates the short-horizon signal. This is development evidence, not untouched-holdout risk validation.</p></section>;
}
