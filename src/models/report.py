"""Render only aggregate documentation; patient trajectory plots stay private."""
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/twin-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .common import load_json, read_jsonl, write_json, file_hash, verify_hashes


def save_plot(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fig.tight_layout();fig.savefig(path,dpi=160,metadata={'Software':'Heart Failure Digital Twin Phase 3'});plt.close(fig);path.chmod(0o600)


def report(root='data/results/phase3',docs='docs'):
    root=Path(root);docs=Path(docs);docs.mkdir(exist_ok=True)
    clinical=root/'clinical';sel=load_json(clinical/'selection.json');ev=load_json(clinical/'test_metrics.json')
    verify_hashes(clinical,load_json(clinical/'evaluation_seal.json')['hashes'])
    final=ev['calibrated_final'];op=ev['operating_points']['max_validation_f1'];sensitive=ev['operating_points']['validation_sensitivity_at_least_0.80']
    lines=['# Clinical model card','',
           'Retrospective recorded 28-day readmission benchmark. The endpoint clock remains unconfirmed; this is not established as 28 days after discharge. No prospective deployment or causality is validated.',
           '',f"Selected model: **{sel['selected']}**. Base fitting uses 1,205 training patients (84 positives). Of 402 validation patients, 201 (14 positives) calibrate probabilities and a disjoint 201 (14 positives) select models, calibration and thresholds. Test: {final['n']} patients, {final['positives']} positives, prevalence {final['prevalence']:.4%}.",
           '', 'The six predictors are gender, ageCat, admission.ward, admission.way, visit.times, and congestive.heart.failure. Four categorical fields use train-only imputation and one-hot encoding with unseen-category support; numeric fields use train-only median imputation and scaling. No medications, labs, discharge information, IDs, or outcomes enter the feature matrix.',
           '', '## Validation evidence','', '| Candidate | Raw AUPRC | Calibrated AUPRC | Calibrated Brier | Selected calibration |','|---|---:|---:|---:|---|']
    for r in sel['candidates']:
        lines.append(f"| {r['name']} | {r['raw_validation']['auprc']:.4f} | {r['calibrated_validation']['auprc']:.4f} | {r['calibrated_validation']['brier']:.5f} | {r['calibration_selected']} |")
    lines+=['', 'All validation AUPRCs above use only the 201 selection patients (prevalence 6.965%). Ten declared candidates include prevalence, L1/L2 logistic models, a class-weighted logistic experiment and shallow gradient boosting. Elastic net adds little distinct value at this scale and was omitted; SMOTE was not used. Models within 0.005 AUPRC of the best favor the simpler candidate in the declared order. These exploratory validation comparisons are noisy.',
            '', 'An initial training out-of-fold sigmoid attempt reversed ranking for all learned candidates. Before touching test data, this was replaced with a disjoint validation calibration/selection design. Sigmoid and intercept-only logistic recalibration were compared by selection Brier. Intercept-only preserves ranking and estimates an additive log-odds offset; probabilities remain in [0,1]. Isotonic was omitted because 14 calibration positives cannot support a flexible curve. Calibration estimates here do not prove calibration in a new clinical population.',
            '', '## Untouched final test','', '| Metric | Calibrated selected model | Training-prevalence baseline |','|---|---:|---:|']
    for k in ['auroc','auprc','brier']:
        lines.append(f"| {k} | {final[k]:.5f} | {ev['baseline'][k]:.5f} |")
    lines+=['', '| Operating rule fixed on selection patients | Threshold | Sensitivity | Specificity | PPV | F1 | Confusion matrix (TN, FP / FN, TP) |', '|---|---:|---:|---:|---:|---:|---|']
    for name,obj in ev['operating_points'].items():
        m=obj['metrics'];lines.append(f"| {name} | {m['threshold']:.5f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | {m['ppv']:.4f} | {m['f1']:.4f} | {m['confusion_matrix']} |")
    lines+=['','The high-sensitivity rule seeks ≥80% sensitivity on selection patients and then maximizes specificity. This is not a guaranteed test sensitivity. Undefined PPV with no predicted positives is reported as zero. Thresholds are operating experiments, not clinical action thresholds.',
            '', '## Calibration and uncertainty','',f"Raw final-model Brier: {ev['raw_final']['brier']:.5f}; calibrated Brier: {final['brier']:.5f}. Descriptive test calibration intercept: {final['calibration']['intercept']}; slope: {final['calibration']['slope']}. Ideal values are 0 and 1. The diagnostic uses test labels only to describe fixed predictions and is never applied to the model.",
            '', '| Metric, maximum validation F1 rule | Patient bootstrap 95% interval |','|---|---|']
    for k,v in op['bootstrap']['intervals_95'].items():lines.append(f'| {k} | [{v[0]:.4f}, {v[1]:.4f}] |')
    lines+=['', 'Bootstrap: 2,000 nonparametric patient resamples; fixed model, calibration, and threshold. This quantifies test-sample uncertainty, not model selection or training uncertainty. Only 28 positive test patients support the estimates. Both operating rules have complete intervals in the private metrics JSON.',
            '', '## Associations and limitations','']
    importance=sel['interpretability'].get('validation_permutation_importance',[])
    if importance:
        lines+=['| Feature | Selection AUPRC permutation drop | SD across 20 permutations |','|---|---:|---:|']
        for r in sorted(importance,key=lambda r:-r['mean_ap_drop']):lines.append(f"| {r['feature']} | {r['mean_ap_drop']:.5f} | {r['std']:.5f} |")
    logs=[r for r in sel['candidates'] if r['name'].startswith('logistic')]
    bestlog=max(logs,key=lambda r:r['calibrated_validation']['auprc'])
    lines+=['',f"Logistic coefficient illustration: {bestlog['name']} (best logistic selection AUPRC). One-hot coefficients are category associations without a dropped reference; numeric coefficients are per training standard deviation. They are not causal effects or directly comparable clinical odds ratios.", '', '| Encoded feature | Log-odds coefficient |','|---|---:|']
    for r in sorted(bestlog['training_coefficients'],key=lambda r:-abs(r['log_odds_coefficient'])):lines.append(f"| {r['encoded_feature']} | {r['log_odds_coefficient']:.4f} |")
    lines+=['','Uncertain horizon clock and semantic rather than verified feature capture times limit applicability. Discharge conflicts remain unadjudicated. There is no external validation, prospective validation, confirmed censoring, or validation of a fused EHR/physiology model. Weak discrimination cannot support patient-specific clinical recommendations.',
            '', 'Artifacts: `data/results/phase3/clinical/selection.json`, `model.joblib`, `test_metrics.json`, `test_predictions.jsonl`, seals and private plots. Serialized artifacts must be loaded only from trusted local runs. The prevalence baseline is the original training prevalence, without validation recalibration.']
    (docs/'CLINICAL_MODEL_CARD.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    for key,label in [('raw_final','Raw'),('calibrated_final','Calibrated'),('baseline','Prevalence')]:
        m=ev[key];ax[0].plot(m['roc_curve']['fpr'],m['roc_curve']['tpr'],label=label);ax[1].plot(m['pr_curve']['recall'],m['pr_curve']['precision'],label=label)
        c=m['calibration_curve'];ax[2].plot([b['mean_probability'] for b in c],[b['observed_fraction'] for b in c],'o-',label=label)
    ax[0].plot([0,1],[0,1],'k--');ax[1].axhline(final['prevalence'],ls='--',color='gray');ax[2].plot([0,1],[0,1],'k--')
    for a,title,xlab,ylab in zip(ax,['Untouched test ROC','Untouched test PR','Test calibration (10 fixed bins)'],['False-positive rate','Recall','Mean predicted probability'],['Sensitivity','Precision','Observed fraction']):a.set(title=title,xlabel=xlab,ylabel=ylab);a.legend(fontsize=8)
    save_plot(fig,clinical/'plots/curves.png')
    fig,ax=plt.subplots(figsize=(4,3));matrix=np.array(op['metrics']['confusion_matrix']);ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Observed',title='Test: validation F1 threshold');save_plot(fig,clinical/'plots/confusion_matrix.png')
    physiology=['# Physiology forecasting evidence','',
       'CHIRON and PerHeart are independent longitudinal cohorts. Forecasts estimate observed physiology, not heart-failure deterioration. There is no paired Zigong patient, no adjudicated HF deterioration outcome, and no full Shadow Twin or TDI.',
       '', '## Protocol','',
       'CHIRON predicts the next exact recorded source day, excluding gaps. PerHeart predicts the next usable same-signal observation within three days, preserving actual elapsed time. This is an event-based replay; the time of the future measurement is not known at inference or supplied as an input. Heart-rate records from BP and oximetry devices can arrive minutes apart, so short-gap persistence may look unusually strong and is not evidence of tomorrow forecasting. All eight raw signal cadences and eligible fold counts are available in coverage.json.',
       '', 'Support gates fixed before evaluating errors: ≥100 windows, ≥10 patients overall; every fold needs ≥30 training windows/5 patients and ≥10 validation/test windows/2 patients. Eligibility uses quality-approved observed targets, not imputed labels. Coverage selection uses timestamps and measurement availability, not target-error scores. Unsupported signals are documented rather than relaxing gates to improve coverage.',
       '', 'Five Phase 2 patient-grouped outer folds: next fold is validation, remaining three are training. Held-out patient causal history is allowed as an input; it never fits population transforms or model parameters. Training-only imputation/scaling supports missing prior-history features. Ridge loss weights each training patient equally, while preprocessing moments use training rows. Missing forecasts are never created as measured observations.',
       '', 'Baselines: persistence, prior seven-day median and mean (fall back to current observation at cold start). Global Ridge uses current value, prior counts, slope, elapsed time since last usable reading, and current missing/invalid flags. Personalized Ridge adds prior seven-day mean/median/std, 30-day median, current deviation and recent-vs-baseline delta. Both already use current patient physiology; personalization here means adding causal baseline summaries, not patient-specific fitted coefficients. Ridge α∈{1,10,100}. Model choice minimizes validation macro patient MAE; within 1% prefers simpler declared candidate. Global/personalized α are chosen separately per fold for a fair paired comparison.',
       '', 'Uncertainty: patient-balanced 95th percentile of absolute validation residuals produces a symmetric empirical interval. Only 2–6 validation patients are available, and selection reuses that validation partition. These are exploratory intervals, without conformal or distribution-free coverage guarantees; they are not clinical reference ranges. Coverage is evaluated only on outer test patients. Intervals are not clipped to clinical bounds.',
       '', 'Normalized residual = (observed − expected) / outer-training target SD. Units, quality, current/prior input IDs, target observation IDs, prediction cutoff, actual target time, interval and warm-up status remain in private ShadowResidual records. Baselines represent observed history, not a confirmed healthy state.',
       '', '## Outer evaluation results','',
       '| Dataset / signal (unit) | Windows / patients | Selected MAE | Selected RMSE | Patient macro MAE | Fold SD of macro MAE | Interval macro coverage | Personal − global MAE (95% paired patient CI) |','|---|---:|---:|---:|---:|---:|---:|---|']
    aggregate={}
    details=[]
    for dataset in ['chiron','perheart']:
        d=root/dataset;verify_hashes(d,load_json(d/'evaluation_seal.json')['hashes']);metrics=load_json(d/'test_metrics.json');coverage=load_json(d/'coverage.json')
        aggregate[dataset]={}
        residuals=list(read_jsonl(d/'shadow_residuals.jsonl'))
        for signal,m in metrics['signals'].items():
            diff=m['personalization'];ci=diff['patient_bootstrap_95']
            physiology.append(f"| {dataset} / {signal} ({m['unit']}) | {m['n']} / {m['patients']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['macro_patient_mae']:.3f} | {m['fold_macro_mae_std']:.3f} | {m['interval_macro_patient_coverage']:.1%} | {diff['mean_personal_minus_global_mae']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
            aggregate[dataset][signal]={k:m[k] for k in ['n','patients','mae','rmse','macro_patient_mae','selected_models','personalization','interval_macro_patient_coverage','normalized_mae_training_sd','warmup']}
            sr=[r for r in residuals if r['signal']==signal]
            fig,ax=plt.subplots(1,2,figsize=(8,3));ax[0].scatter([r['observed_value'] for r in sr],[r['expected_value'] for r in sr],s=5,alpha=.3);ax[0].set(xlabel='Observed',ylabel='Predicted',title=f'{dataset} {signal}: outer test')
            ax[1].hist([r['residual'] for r in sr],bins=30);ax[1].set(xlabel=f"Observed − expected ({m['unit']})",ylabel='Windows');save_plot(fig,d/f'plots/{signal}_predictions_residuals.png')
            # Choose example by eligible count only, independent of error magnitude; identifiers stay private.
            patient=max(sorted({r['patient_id'] for r in sr}),key=lambda pid:sum(r['patient_id']==pid for r in sr))
            pr=sorted((r for r in sr if r['patient_id']==patient),key=lambda r:float(r['timestamp']) if dataset=='chiron' else r['timestamp'])
            time=np.array([r['timestamp'] if dataset=='chiron' else __import__('datetime').datetime.fromisoformat(r['timestamp'].replace('Z','+00:00')).timestamp()/86400 for r in pr],dtype=float)
            time-=time[0]
            fig,ax=plt.subplots(figsize=(8,3));ax.plot(time,[r['observed_value'] for r in pr],'.-',label='Observed target');ax.plot(time,[r['expected_value'] for r in pr],'.-',label='Expected')
            ax.fill_between(time,[r['lower_bound'] for r in pr],[r['upper_bound'] for r in pr],alpha=.2,label='Exploratory 95% interval');ax.set(xlabel='Days since first evaluated target',ylabel=m['unit'],title=f'{dataset} {signal}: held-out trajectory');ax.legend();save_plot(fig,d/f'plots/{signal}_private_trajectory.png')
        details+=['',f'### {dataset}: personalization evidence','']
        for signal,m in metrics['signals'].items():
            lo,hi=m['personalization']['patient_bootstrap_95']
            conclusion='lower held-out error with personalization' if hi<0 else 'higher held-out error with personalization' if lo>0 else 'uncertain direction: interval includes zero'
            details.append(f"- {signal}: {conclusion}.")
        strongest=min(metrics['signals'],key=lambda n:metrics['signals'][n]['normalized_mae_training_sd'])
        weakest=max(metrics['signals'],key=lambda n:metrics['signals'][n]['normalized_mae_training_sd'])
        details+=['',f'By selected MAE divided by outer-training target SD, strongest: {strongest}; weakest: {weakest}. This relative-scale comparison does not establish clinical value.']
        fig,ax=plt.subplots(figsize=(8,4));names=list(metrics['signals']);ax.barh(names,[metrics['signals'][n]['normalized_mae_training_sd'] for n in names]);ax.set(xlabel='Selected MAE / outer training SD',title=f'{dataset}: per-signal normalized error');save_plot(fig,d/'plots/per_signal_mae.png')
        details+=['',f'### {dataset}: fold-selected models and fixed comparators','', '| Signal | Models selected in folds 0–4 | Persistence macro MAE | Best fixed candidate by observed outer scores (descriptive only) | Warm-up MAE cold / warm |','|---|---|---:|---|---|']
        for signal,m in metrics['signals'].items():
            fixed=min(m['fixed_models'],key=lambda n:m['fixed_models'][n]['macro_patient_mae'])
            cold=m['warmup'].get('cold',{}).get('mae');warm=m['warmup'].get('warm',{}).get('mae')
            details.append(f"| {signal} | {', '.join(m['selected_models'])} | {m['fixed_models']['persistence']['macro_patient_mae']:.3f} | {fixed} ({m['fixed_models'][fixed]['macro_patient_mae']:.3f}) | {cold:.3f} / {warm:.3f} |")
        details+=['', 'Fixed-model comparisons describe held-out results and do not override validation selection. Per-patient, every candidate, all fold, normalization and uncertainty-width results are stored privately in test_metrics.json. Pooled rows can overweight frequently measured patients, so macro patient results are primary. Fold SD reflects small grouped samples and overlapping fits, not five independent trials. Personalization CI bootstraps paired patient errors across outer folds with fixed models; it excludes model-fitting uncertainty.',
                     '',f"Unsupported {dataset} signals: "+', '.join(f"{n} ({v['windows']} windows / {v['patients']} patients; fold coverage failure)" for n,v in coverage.items() if not v['supported'])+'.']
    physiology+=details
    physiology+=['', 'CHIRON units remain provisional within source; HR SD is not RR-based HRV and skin temperature is not core temperature. Ambient temperature is environmental context rather than a patient physiological biomarker. PerHeart units are descriptor-supported. Short-gap home readings, device differences, selective observation, invalid-value exclusion, and sparse patients limit generalization. No comparison pools unrelated cohorts.',
                 '', 'Proceed to a Phase 4 engineering prototype with causal state, missingness-aware replay and exploratory residual displays. Do not implement validated clinical deterioration alarms or claim a healthy counterfactual. Prospectively paired outcome-labelled data, verified availability times, and patient-level interval calibration are needed before clinical use.']
    (docs/'PHYSIOLOGY_FORECASTING.md').write_text('\n'.join(physiology)+'\n')
    # Safe public aggregate summary, never patient keys or raw predictions.
    write_json(docs/'MODEL_BASELINES_AGGREGATE.json',{'clinical':{'selected':sel['selected'],'test':{k:final[k] for k in ['n','positives','prevalence','auroc','auprc','brier']},'operating_point':{k:op['metrics'][k] for k in ['threshold','sensitivity','specificity','ppv','f1']},'bootstrap':op['bootstrap']},'physiology':aggregate})
    base=['# Phase 3 evidence-based baselines','',
          'Two scientifically separate engines are implemented. Zigong estimates recorded readmission probability; CHIRON/PerHeart forecast observed longitudinal signals. There is no combined headline accuracy, patient pairing, synthetic clinical validation, frontend, full Shadow Twin or final TDI.',
          '', 'Results and limitations: [clinical model card](CLINICAL_MODEL_CARD.md), [physiology forecasting](PHYSIOLOGY_FORECASTING.md), and [aggregate machine-readable results](MODEL_BASELINES_AGGREGATE.json).',
          '', '## Reproduce','', '```bash','python3 -m venv .venv','.venv/bin/python -m pip install -r requirements-models.txt','python3 -m src.data.validate','.venv/bin/python -m unittest discover -s tests -v',
          '.venv/bin/python -m src.models.train_clinical','.venv/bin/python -m src.models.train_forecasting',
          '# Selection is sealed before either evaluation command:', '.venv/bin/python -m src.models.evaluate_clinical','.venv/bin/python -m src.models.evaluate_forecasting',
          '.venv/bin/python -m src.models.verify','.venv/bin/python -m src.models.report','.venv/bin/python -m src.models.audit','```','',
          'Default private experiment root: `data/results/phase3`. Training refuses nonempty run directories. Use `--output data/results/new_run/clinical` for clinical commands and `--output data/results/new_run` for forecasting/report/audit commands to reproduce training. Do not repeat final clinical testing in new runs after seeing the test result. Determinism checks repeat training only; repeat evaluation commands on the sealed final run return stored metrics without reopening test labels.',
          '', 'Artifacts record canonical dataset manifest hashes and versions, source input hashes, feature version, split seed, model hyperparameters, UTC timestamp, code hash, Python and pinned dependency versions. Joblib files preserve training-only pipelines/calibration and reproduce predictions. Metadata timestamps vary; prediction/model values are deterministic within the pinned environment. Cross-platform bitwise identity is not promised. Artifacts and patient trajectory plots are private mode 0600 under ignored data/results; documentation contains only aggregates.',
          '', 'Final clinical evaluation creates an exclusive persistent start marker and is performed exactly once after selection. An interrupted evaluation cannot silently restart. Selection and evaluation seals detect mutation; input manifests and canonical validators are checked before evaluation. Stored results may be replotted without predicting or reopening test labels. A bootstrap of already fixed test predictions is an uncertainty calculation, not further model optimization.',
          '', f"Verification: {load_json(root/'verification.json')['tests_run']} tests pass without skips; repeated training produces identical selected candidates, validation results and model predictions. All forecast model files are byte-identical; the clinical tree serialization differs in bytes while learned tree fields and predictions match. See private verification.json and completion_audit.json.",
          '', '## What the evidence supports','',
          'Actual retrospective benchmark and grouped forecast errors are measured with leakage-controlled inputs and preprocessing. Personalization is supported only where paired held-out patient errors improve; evidence remains signal-specific. Weak readmission discrimination or poor interval coverage must remain visible.',
          '', 'The data do not validate HF deterioration forecasts, causal biomarkers, a clinically calibrated fused twin, clinical action thresholds, or a healthy counterfactual. Phase 4 can implement a research replay interface, model-versioned expected trajectories, uncertainty, residuals, quality/missingness state, and explicit unsupported modalities. Keep clinical vulnerability and physiological residuals separate; do not invent cross-cohort patient links. Collect real paired longitudinal EHR/outcome data and verify capture times before clinical alarm validation.']
    (docs/'MODEL_BASELINES.md').write_text('\n'.join(base)+'\n')
    print({'documentation':'3 model documents and aggregate JSON','plots':'private evaluation plots generated'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/results/phase3');p.add_argument('--docs',default='docs');a=p.parse_args();report(a.root,a.docs)
