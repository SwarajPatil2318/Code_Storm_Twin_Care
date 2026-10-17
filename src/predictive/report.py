"""Reviewed aggregate documentation and private publication/demo figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/twin-patient-matplotlib")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
from src.data.io import read_jsonl,file_hash
from src.predictive.artifacts import DOCS,write_json


def fmt(v):return 'not estimable' if v is None else f'{v:.4f}'

def interval(record,key):
    ci=record['patient_bootstrap95'].get(key,{}).get('interval95')
    return 'not supported' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]'


def table(names,results,validation):
    lines=['| Model | Validation raw AP | Test AP | Test ROC | Brier | Balanced sensitivity | Balanced PPV |','|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        r=results[name];m=r['metrics'];o=r['operating_points']['balanced']
        lines.append(f"| {name} | {fmt(validation[name]['validation_raw']['auprc'])} | {fmt(m['auprc'])} | {fmt(m['auroc'])} | {fmt(m['brier'])} | {fmt(o['sensitivity'])} | {fmt(o['ppv'])} |")
    return '\n'.join(lines)


def plots(root,result,predictions,traces,validation):
    target=root/'figures';target.mkdir(exist_ok=True,mode=0o700);y=np.array([r['label'] for r in predictions]);selected=result['selected'];names=list(result['models'])
    for kind in ['precision_recall','roc','calibration']:
        fig,ax=plt.subplots(figsize=(12,6))
        for name in names:
            p=np.array([r['predictions'][name]['calibrated'] for r in predictions])
            if kind=='precision_recall':a,b,_=precision_recall_curve(y,p);ax.plot(b,a,label=name)
            elif kind=='roc':a,b,_=roc_curve(y,p);ax.plot(a,b,label=name)
            else:
                bins=result['models'][name]['calibration']['bins'];ax.plot([r['predicted'] for r in bins],[r['observed'] for r in bins],marker='o',label=name)
        if kind=='precision_recall':ax.axhline(y.mean(),color='grey',ls='--');ax.set(xlabel='Recall',ylabel='Precision')
        elif kind=='roc':ax.plot([0,1],[0,1],'k--');ax.set(xlabel='False positive rate',ylabel='Sensitivity')
        else:ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Mean forecast probability',ylabel='Observed event frequency')
        ax.set_title(f'Frozen test comparison: {kind.replace("_"," ")}');ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1));fig.tight_layout();fig.savefig(target/(kind+'.png'),dpi=250);plt.close(fig)
    winners=validation['winners']
    for plotname,modalities in [('modality_comparison',['ehr','phys','fused','personalized']),('personalization_ablation',['fused','personalized']),
                               ('map_ablation',['map_only','bp_only','phys_no_map','phys_no_bp','ehr','phys','fused','personalized']),('temporal_comparison',['phys','fused','temporal_phys','temporal_fused','late_fusion'])]:
        fig,ax=plt.subplots(figsize=(9,5));values=[result['models'][winners[m]]['metrics']['auprc'] for m in modalities]
        ax.bar(modalities,values);ax.axhline(y.mean(),color='grey',ls='--',label='Test prevalence');ax.set(ylabel='Test average precision',title=plotname.replace('_',' '));ax.tick_params(axis='x',rotation=25);ax.legend();fig.tight_layout();fig.savefig(target/(plotname+'.png'),dpi=250);plt.close(fig)
    strata=['below_threshold','near_threshold','moderate','clearly_normal','hard'];fig,ax=plt.subplots(figsize=(9,5))
    values=[result['subsets'][s][selected]['metrics']['auprc'] for s in strata]
    ax.bar(strata,[v if v is not None else 0 for v in values]);ax.set(ylabel='Selected model average precision',title='Predefined MAP strata; zero-height unsupported bars are not estimable');ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(target/'hard_cases.png',dpi=250);plt.close(fig)
    leads=[360,240,180,120,60,30,0];summary=json.loads((root/'trajectory_summary.json').read_text());fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
    for ax,key,label in [(axes[0],'median_risk','Median eligible risk (within next 3h)'),(axes[1],'median_map','Median buffered MAP mmHg')]:
        values=[summary[str(l)][key] for l in leads];ax.plot(leads,[np.nan if v is None else v for v in values],marker='o');ax.set_ylabel(label)
    axes[1].set_xlabel('Minutes before eventual event (diagnostic alignment)');axes[1].invert_xaxis();fig.suptitle('Eligibility-filtered event-aligned risk; sparse retrospective support');fig.tight_layout();fig.savefig(target/'risk_before_events.png',dpi=250);plt.close(fig)
    grouped={}
    for row in traces:grouped.setdefault((row['icu_stay_id'],row['event_onset']),[]).append(row)
    # Deterministic representative paths, no choosing the nicest prediction.
    for i,key in enumerate(sorted(grouped)[:3]):
        records=sorted(grouped[key],key=lambda r:r['minutes_before_onset'],reverse=True);x=[r['minutes_before_onset'] for r in records]
        fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True);risks=[r['prediction_state']['calibrated_risk'] for r in records];axes[0].plot(x,risks,'o-',label='Risk in next3h')
        intervals=[r['prediction_state']['uncertainty_interval90'] for r in records]
        if all(v is not None for v in intervals):axes[0].fill_between(x,[v[0] for v in intervals],[v[1] for v in intervals],alpha=.2,label='Conditional bootstrap90%')
        axes[0].set_ylabel('Risk');axes[0].legend();axes[1].plot(x,[r['physiology']['systemicmean'] for r in records],'o-',label='Buffered MAP');axes[1].axhline(65,color='r',ls='--')
        axes[1].plot(x,[r['map_reference']['median'] for r in records],ls=':',label='Past personal reference');axes[1].set_ylabel('MAP mmHg');axes[1].legend()
        for field in ['heartrate','sao2','respiration']:axes[2].plot(x,[r['physiology'][field] for r in records],marker='o',label=field)
        axes[2].legend();axes[2].set_xlabel('Minutes before eventual onset');axes[2].invert_xaxis();fig.suptitle(f'Private deterministic representative {i+1}; pre-eligibility/onset probes are diagnostic only');fig.tight_layout();fig.savefig(target/f'private_trace_{i+1}.png',dpi=250);plt.close(fig)
    for p in target.glob('*'):p.chmod(0o600)


def report(root='data/results/phase4',docs_dir='docs'):
    root=Path(root);result=json.loads((root/'test_metrics.json').read_text());validation=json.loads((root/'validation_metrics.json').read_text());metadata=json.loads((root/'metadata.json').read_text());selection=json.loads((root/'selection.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    selected=result['selected'];record=result['models'][selected];m=record['metrics'];winners=validation['winners'];candidates=validation['candidates'];pred=list(read_jsonl(root/'test_predictions.jsonl'));traces=list(read_jsonl(root/'event_trajectories.jsonl'))
    caveats=('Research-only retrospective ICU monitor proxy. Sustained invasive MAP<65 requires at least seven consecutive usable observations spanning at least30 minutes with gaps<=5 minutes. Forecast is event onset within the next180 minutes; six-hour history, five-minute input buffer, minimum15-minute lead,120-minute washout. This does not guarantee three hours of warning. Source availability offsets are proxies for live receipt, and no clinician adjudication or prospective clinical utility is established. HF is exploratory. No48–72h or consumer wearable claim. Earlier negative Zigong readmission and long-horizon findings remain unchanged.')
    alltable=table(list(result['models']),result['models'],candidates)
    explanation_path=root/'validation_feature_explanations.json';associations=json.loads(explanation_path.read_text()) if explanation_path.exists() else {}
    association_lines=['| Validation model/feature | Mean AP drop on patient-donor permutation |','|---|---:|']
    for modality,detail in associations.items():
        for key,value in sorted(detail['feature_permutation'].items(),key=lambda item:item[1]['validation_auprc_drop'],reverse=True)[:8]:association_lines.append(f"| {modality}: {key} | {value['validation_auprc_drop']:.5f} |")
    association_table='\n'.join(association_lines)
    op=record['operating_points'];ops=['| Frozen operating point | Threshold | Sensitivity | Specificity | PPV | NPV | F1 | False alerts | False alerts/patient | False alerts/stay | False alerts/eligible hour | Median observed lead hours |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,o in op.items():ops.append(f"| {name} | {fmt(o['threshold'])} | {fmt(o['sensitivity'])} | {fmt(o['specificity'])} | {fmt(o['ppv'])} | {fmt(o['npv'])} | {fmt(o['f1'])} | {fmt(o['false_alerts'])} | {fmt(o['false_alerts_per_patient'])} | {fmt(o['false_alerts_per_stay'])} | {fmt(o['false_alerts_per_eligible_hour'])} | {fmt(o['lead_time_hours']['median'] if o['lead_time_hours'] else None)} |")
    partition_table='\n'.join(['| Partition | Patients | Positive patients | Stays | Windows | Positive windows | Prevalence |','|---|---:|---:|---:|---:|---:|---:|']+[f"| {role} | {v['patients']} | {v['positive_patients']} | {v['icu_stays']} | {v['windows']} | {v['positive_windows']} | {fmt(v['window_prevalence'])} |" for role,v in metadata['partitions'].items()])
    # Metadata source keys are fixed by Phase35; don't infer missing counts.
    totals={key:sum(v[key] for v in metadata['partitions'].values()) for key in ['patients','icu_stays','windows','positive_patients','positive_windows']}
    trajectory_summary=json.loads((root/'trajectory_summary.json').read_text());trajectory_lines=['| Minutes before onset | Available event probes | Eligible event probes | Median eligible risk | Median buffered MAP |','|---|---:|---:|---:|---:|']
    for lead in protocol['risk_trajectories_minutes_before_onset']:
        r=trajectory_summary[str(lead)];trajectory_lines.append(f"| {lead} | {r['available_event_probes']} | {r['eligible_event_probes']} | {fmt(r['median_risk'])} | {fmt(r['median_map'])} |")
    trajectory_table='\n'.join(trajectory_lines)
    modeltext=f'''# Phase4 model card

Selected primary: **{selected}**, frozen using validation data only. Test AP={fmt(m['auprc'])}, patient-cluster95% interval {interval(record,'auprc')}; ROC={fmt(m['auroc'])},95% interval {interval(record,'auroc')}; Brier={fmt(m['brier'])},95% interval {interval(record,'brier')}. Test prevalence={fmt(m['prevalence'])}. Descriptive calibration intercept={fmt(record['calibration']['intercept'])}, slope={fmt(record['calibration']['slope'])}, ECE={fmt(record['calibration']['ece_equal_width_10_bins'])}.

{caveats}

Full frozen cohort: {totals['patients']} patients, {totals['icu_stays']} stays, {totals['windows']} windows, {totals['positive_patients']} positive patients and {totals['positive_windows']} positive windows.

{partition_table}

{alltable}

Top validation feature associations (three repeats, patient donors; correlated inputs limit marginal interpretation):

{association_table}

{'\n'.join(ops)}

False-alert denominators: {result['false_alert_exposure']}. Alerts are retained-window decisions, not deduplicated deployed notifications. Censoring-selected controls limit transportability. Test has only15 independent positive patients; intervals are conditional on fixed models and this cohort.

Train-only encoders retain missing indicators and unknown categories; numeric medians/means/scales and class weights use training patients. Inputs use approved context, lab availability, and buffered physiology. Documented-positive history remains positive-or-unknown, never documented absence. Engineered duration features count only observed supported intervals; no physiological interpolation. Personal reference is past observed physiology, not a healthy counterfactual. Input features exclude audit-current MAP, event labels, outcomes, therapies and IDs. No feature is derived from full-stay summaries.

Calibration selected among none/positive-slope Platt/isotonic by three-fold patient-isolated validation OOF Brier; final method={selection['calibration'][selected]}. Thresholds use OOF validation probabilities. Test intercept/slope are descriptive calibration diagnostics, never used to recalibrate predictions. [Official calibration documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) describes disjoint calibration data. Reliability plots use10 equal-width bins, sparse bins are descriptive.

Conditional probability uncertainty uses{selection['uncertainty_refits']} training-patient bootstrap refits plus validation-patient bootstrap recalibration, with fixed selected hyperparameters and5th/95th probability quantiles. It does not capture model selection, measurement bias, cohort shift or clinical correctness. Metric intervals use{result['bootstrap_replicates']} whole-patient resamples preserving windows/stays; undefined class-degenerate replicates are excluded and intervals suppressed below80% valid resamples.

Private artifacts: `{root}/pipeline.joblib`, selection/evaluation seals, validation registry, private prediction states, test predictions, event trajectories and figures. Public docs contain aggregate results only. PredictionState_v1 includes private ID/time, raw/calibrated risk, uncertainty, quality/operating state, current MAP, personal deviation, horizon and association explanations. Outside training eligibility is explicitly flagged; this is not a clinical decision service.

Seed2026; source hash `{metadata['source_code_sha256']}`; dataset manifest SHA `{file_hash('data/processed/eicu_demo/manifest.json')}`. Dependency versions: `{json.dumps(metadata['dependencies'],sort_keys=True)}`.
'''
    fusion=f'''# Multimodal fusion results

{caveats}

{table([winners[x] for x in ['ehr','phys','fused','personalized','late_fusion']],result['models'],candidates)}

Paired frozen-test AP differences (whole-patient bootstrap):

```json
{json.dumps(result['paired_comparisons'],indent=2)}
```

Validation complexity gates and selection are stored in the frozen private validation_metrics.json; test comparison does not revise selection. Late fusion uses three training-patient OOF base-model logits and a regularized training-only stacking head; test and validation never fit the stacker. Early fusion concatenates causal approved EHR and physiology, with missingness retained. Any fusion improvement here is exploratory and does not establish independent clinical benefit.
'''
    personalized=f'''# Personalization ablation

Personal reference uses the first five hours of the same six-hour buffered history, endingT0-65, at least30 observed points and three connected hours with gaps<=10 minutes. Mean/median/variability, recent/current deviation, standardized deviation, deviation persistence and support/absence flags are past-only. Zero variability leaves z-score undefined. This is an observed reference, not a premorbid healthy baseline.

{table([winners[x] for x in ['fused','personalized']],result['models'],candidates)}

Paired test comparison: `{json.dumps(result['paired_comparisons']['personal_minus_fused'])}`. No individual personal baseline is learned from a full stay. Limited reference availability and44 independent positive training patients restrict evidence for personalization.
'''
    temporal=f'''# Temporal model results

A trained compact causal temporal convolution network uses73 five-minute bins, seven signal values, seven observed masks and seven elapsed-time channels. No interpolation or backward fill. Three tanh causal convolution layers have kernel3, four channels, dilations1/4/16; mean and last pooling feed a scalar probability head. The fused variant adds a four-unit EHR representation. Missing signal values are replaced only after training-only normalization, alongside explicit masks/age. Derived normalized inputs are clipped to+/-8; original source values remain unchanged. Adam150 epochs, L2 .01/.1, training patient weights and train-only class weighting. Convolution gradients were checked against finite differences and prefixes against future perturbation. This small custom NumPy implementation avoids introducing unsupported heavyweight dependencies.

{table([winners[x] for x in ['phys','fused','temporal_phys','temporal_fused','late_fusion']],result['models'],candidates)}

Only44 independent positive training patients. Validation determines temporal hyperparameters and rejects added complexity absent paired AP support. A trained neural experiment is not evidence that deep learning is preferred. Causal convolutions are inspired by the [original TCN study](https://arxiv.org/abs/1803.01271); our small architecture does not reproduce its benchmarks.
'''
    hard=['# Hard-case and MAP-ablation analysis','',caveats,'',table([winners[x] for x in ['map_only','bp_only','phys_no_map','phys_no_bp','ehr','phys','fused','personalized']],result['models'],candidates),'',
          'Hard subset is predefined: audit MAP atT0 and buffered latest>=80, buffered last60-minute minimum>=75, slope>=-1mmHg/hour, no buffered MAP<65 in120 minutes, current audit MAP age<=10 minutes. Audit MAP atT0 is used only to assess strata; model physiology endsT0-5. No test-driven boundary tuning.','',
          '| Stratum | Patients | Positive patients | Windows | Positive windows | AP | AP95% | Sensitivity (balanced) | Brier |','|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for name in ['below_threshold','near_threshold','moderate','clearly_normal','hard']:
        r=result['subsets'][name][selected];n=r['counts'];hard.append(f"| {name} | {n['patients']} | {n['positive_patients']} | {n['windows']} | {n['positive_windows']} | {fmt(r['metrics']['auprc'])} | {interval(r,'auprc')} | {fmt(r['operating_points']['balanced']['sensitivity'])} | {fmt(r['metrics']['brier'])} |")
    hard+=['','All comparator strata, calibration, lead and cluster intervals are private test_metrics.json. When classes or patient support are insufficient, metrics/intervals are not estimable. Near-threshold performance can represent impending threshold crossing rather than a distinct upstream biological forecast; normal-MAP hard cases test this limitation.','',
           'Explanations use validation patient-donor group permutations (five repeats); local prediction-state occlusion replaces one group with a training reference. Risk changes are associations, not causal interventions. MAP and beyond-MAP groups are explicit; no test importance informs feature/model selection.']
    hard_record=result['subsets']['hard'][selected];normal_record=result['subsets']['clearly_normal'][selected]
    for title,r in [('clearly normal MAP',normal_record),('predefined hard cases',hard_record)]:
        ap=r['metrics']['auprc'];prev=r['metrics']['prevalence'];sens=r['operating_points']['balanced']['sensitivity']
        finding='AP is not estimable in this subgroup.' if ap is None else 'AP does not exceed subgroup prevalence.' if ap<=prev else 'AP exceeds subgroup prevalence numerically; sparse support prevents a clinical claim.'
        if sens==0:finding+=' The frozen balanced operating point detects none of its positive windows: sensitivity collapses to zero away from near-threshold cases.'
        hard+=['',f'{title}: {finding}']
    hf=result['subsets']['hf'][selected];hftext=f'''# HF subgroup analysis — EXPLORATORY

Selected frozen broader-ICU model, unchanged thresholds and calibrator; no HF-specific tuning. HF means documented-positive evidence available byT0, not cardiomyopathy inference or inferred negative absence.

Counts: `{json.dumps(hf['counts'])}`. Prevalence={fmt(hf['metrics']['prevalence'])}; AP={fmt(hf['metrics']['auprc'])},95% {interval(hf,'auprc')}; ROC={fmt(hf['metrics']['auroc'])},95% {interval(hf,'auroc')}; Brier={fmt(hf['metrics']['brier'])}. Balanced sensitivity={fmt(hf['operating_points']['balanced']['sensitivity'])}; PPV={fmt(hf['operating_points']['balanced']['ppv'])}. Descriptive calibration: `{json.dumps(hf['calibration'])}`.

Small independent-patient support makes all HF findings exploratory. These results establish no clinical HF product effectiveness, no hospitalization prediction and no long-horizon risk claim. Group bootstrap intervals are descriptive when estimable and do not remedy selection, limited support, undocumented HF or ICU-to-home shift.
'''
    decision=f'''# Phase4 decision

Primary selected from validation: **{selected}**. The frozen selection rule rewards validation precision–recall, acceptable OOF calibration and hard-case behavior when supported, and rejects fusion/personal/temporal added complexity unless paired validation AP confidence supports improvement. Test results never change the primary, features, hyperparameters, calibration or thresholds.

Test AP={fmt(m['auprc'])} ({interval(record,'auprc')}), ROC={fmt(m['auroc'])} ({interval(record,'auroc')}), Brier={fmt(m['brier'])}; prevalence={fmt(m['prevalence'])}. See [model card](PHASE4_MODEL_CARD.md) for all comparators/operating points; [MAP/hard cases](HARD_CASE_ANALYSIS.md), [fusion](MULTIMODAL_FUSION_RESULTS.md), [personalization](PERSONALIZATION_ABLATION.md), [temporal](TEMPORAL_MODEL_RESULTS.md) and [exploratory HF](HF_SUBGROUP_ANALYSIS.md) for evidence and limits.

{caveats}

{trajectory_table}

Event-aligned risks are diagnostic probes at6h/4h/3h/2h/1h/30m/onset; every forecast still concerns the next3h. Probes failing history/washout eligibility or the minimum-lead condition at onset are flagged and excluded from aggregate eligible-risk plots. They are not a formal time-to-event model or evidence for prediction at6h. Representative private traces are chosen deterministically by stay/event order.

Phase5 may consume the structured research PredictionState and assess broader, external/prospective data and clinician-adjudicated endpoints. Implementing shadow simulations, therapy-response inference, neighbours, formal time-to-event, adaptive sensing, self-evolution, hidden hemodynamics or a final UI is outside this phase. Preserve the simplest frozen model and all negative ablations; do not claim clinically useful personalized multimodal risk without larger independent validation.

Reproduce development only: `.venv/bin/python -m src.predictive.train --output data/results/phase4_reproduction --uncertainty-refits 20`. Final evaluation is an exclusive one-time transaction: `.venv/bin/python -m src.predictive.evaluate`. Repeated evaluation returns sealed outputs without running predictions. Run `.venv/bin/python -m src.predictive.audit` for integrity, role, serialization, reproduction and preservation checks. Historical frozen sources support replay of earlier audits without altering their hash scopes.
'''
    contents=[modeltext,fusion,personalized,temporal,'\n'.join(hard)+'\n',hftext,decision]
    Path(docs_dir).mkdir(parents=True,exist_ok=True)
    for name,content in zip(DOCS[:7],contents):Path(docs_dir,name).write_text(content)
    Path(docs_dir,'PHASE4_PROTOCOL.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
    plots(root,result,pred,traces,validation)
    write_json(root/'report_manifest.json',{'public_docs':{name:file_hash(Path(docs_dir,name)) for name in DOCS},'private_figures':{str(p.relative_to(root)):file_hash(p) for p in sorted((root/'figures').glob('*.png'))}})
    return {'status':'REPORTED','documents':7,'private_figures':len(list((root/'figures').glob('*.png')))}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/results/phase4');args=parser.parse_args();print(report(args.root))
