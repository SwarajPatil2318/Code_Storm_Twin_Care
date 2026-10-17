"""Six necessary figures, reviewed aggregate documents and private UI examples."""
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/twin_phase65_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.io import file_hash
from src.twin.protocol import PRIMARY_SIGNALS,EVIDENCE
from src.twin.inference import finite
from src.synthetic.config import UNITS,SIGNALS
from src.reliability.access import root,save_json,phase6_identity,PHASE6
from src.reliability.sensing import CANDIDATES

LABEL='SYNTHETIC POC VALIDATION — NOT CLINICALLY VALIDATED'


def number(x):return 'undefined' if x is None else f'{x:.5f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def decision(result):
    a=result['reference'];b=result['selected'];sense=result['adaptive_sensing']
    old=a['subgroups72']['recovery']['alert_fraction'];new=b['subgroups72']['recovery']['alert_fraction']
    recovery=old is not None and old>0 and new<=.9*old
    sensitivity=b['72']['metrics']['sensitivity']>=a['72']['metrics']['sensitivity']-.03
    improved=[]
    for h in ['24','48','72']:
        x=a[h]['calibration'];y=b[h]['calibration']
        improved.append(x['intercept'] is not None and y['intercept'] is not None and x['slope'] is not None and y['slope'] is not None
            and abs(y['intercept'])+abs(y['slope']-1)<abs(x['intercept'])+abs(x['slope']-1))
    calibration=sum(improved)>=2 and np.mean([b[h]['metrics']['brier'] for h in ['24','48','72']])<np.mean([a[h]['metrics']['brier'] for h in ['24','48','72']]) and b['72']['metrics']['auprc']>=.95*a['72']['metrics']['auprc']
    sensing=sense['mean_observed_net_utility_requested'] is not None and sense['mean_observed_net_utility_requested']>0 and sense['decisions_corrected']>sense['decisions_harmed'] and sense['decisions_corrected']+sense['decisions_harmed']>0
    gates={'recovery':bool(recovery),'sensitivity':bool(sensitivity),'calibration':bool(calibration),'adaptive_sensing':bool(sensing)}
    value='GO' if all(gates.values()) else 'PARTIAL GO' if recovery or calibration or sensing else 'NO-GO'
    return {'decision':value,'gates':gates,'recommendation_phase7':
        'STOP core ML research. Build the product around the frozen Phase6 reference with measured limitations. Expose only replicated reliability components; present sensing as an experimental decision-clarification feature, never a treatment or validated confidence recommendation.'}


def private_cases(config,data,risk,features,cases,updates,rec):
    threshold=cases['threshold72'][0];before=updates['before'][:,2];after=updates['after'][:,2];y=cases['labels'].sum(1)
    requested=updates['requested'];choice=updates['choice']
    masks={
        'confirm_deterioration':requested&(choice==0)&(y>0)&(after>before)&(after>=threshold),
        'suppress_false_alert':requested&np.isin(choice,[0,1])&(y==0)&(before>=threshold)&(after<threshold)}
    examples=[]
    # First eligible by patient/time, not largest favorable change. Missing exact
    # demo patterns are a result; closest factual cases are separately labelled.
    for name,mask in masks.items():
        eligible=np.flatnonzero(mask)
        if len(eligible):
            order=np.lexsort((cases['landmark_hours'][eligible],cases['patient_ids'][eligible]));i=int(eligible[order[0]])
            readings=json.loads(str(cases['reading_values_json'][i]))
            examples.append({'case':name,'exact_pattern_available':True,'patient_id':str(cases['patient_ids'][i]),
                'landmark_hour':int(cases['landmark_hours'][i]),'requested_measurement':CANDIDATES[int(choice[i])],
                'measurement_values':readings[int(choice[i])],'risk_before':before[i].item(),'risk_after':after[i].item(),
                'event_within72h':bool(y[i]),'decisions_corrected':bool(updates['corrected'][i]),
                'selection_rule':'First patient/time satisfying predeclared pattern, not greatest favorable change'})
        else:examples.append({'case':name,'exact_pattern_available':False,'reason':'No replication benchmark case matched the required observable policy/outcome pattern; no fabricated measurement or risk'})
    no_request=np.flatnonzero(rec['sufficient_evidence']&~rec['measurement_needed'])
    if len(no_request):
        order=np.lexsort((data['landmark_hours'][no_request],data['patient_ids'][no_request]));i=int(no_request[order[0]])
        examples.append({'case':'sufficient_evidence','exact_pattern_available':True,'patient_id':str(data['patient_ids'][i]),
            'landmark_hour':int(data['landmark_hours'][i]),'risk72':float(risk[i,2]),'requested_measurement':None,
            'reason':'Persistent multisystem evidence is already available; no additional measurement required'})
    else:examples.append({'case':'sufficient_evidence','exact_pattern_available':False,'reason':'No exact sufficient-evidence case present'})
    # Include harmed and unavailable/nonuseful requests, so demos do not conceal failures.
    for name,mask in [('harmed_decision',updates['harmed']),('unhelpful_request',requested&(updates['net_utility']<=0))]:
        found=np.flatnonzero(mask)
        if len(found):
            i=int(found[0]);examples.append({'case':name,'patient_id':str(cases['patient_ids'][i]),'landmark_hour':int(cases['landmark_hours'][i]),
                'risk_before':float(before[i]),'risk_after':float(after[i]),'event_within72h':bool(y[i]),
                'requested_measurement':CANDIDATES[int(choice[i])]})
    save_json(root(config)/'phase7_demo_cases.json',{'cases':examples,'selection_is_post_evaluation':True,
        'missing_patterns_preserved':True,'deterioration_is_synthetic_outcome_not_clinical_diagnosis':True})
    chosen=np.linspace(0,len(risk)-1,12,dtype=int)
    _,seal=phase6_identity();engine_path=PHASE6/'shadow_engines.joblib'
    if file_hash(engine_path)!=seal['artifacts']['shadow_engines.joblib']:raise ValueError('Frozen Shadow engine changed')
    engines=joblib.load(engine_path);small={k:v[chosen] for k,v in data.items()};forecast=engines['shadow'].predict(small)
    divergence=engines['divergence'].transform(small,forecast)
    states=[]
    for j,i in enumerate(chosen):
        k=int(rec['choice'][i]);needed=bool(rec['measurement_needed'][i]);f=features[i]
        states.append({'patient_id':str(data['patient_ids'][i]),'landmark_hour':int(data['landmark_hours'][i]),
            'timestamp_basis':'Patient monitoring origin plus landmark_hour; chronology retained in replication patient table',
            'risk':dict(zip(['next_24h','next_48h','next_72h'],risk[i].tolist())),
            'observed_state':{s:finite(data['recent'][i,z]) for z,s in enumerate(PRIMARY_SIGNALS)},
            'expected_shadow_state':{s:{'value':finite(forecast['expected'][j,z]),'lower':finite(forecast['lower'][j,z]),'upper':finite(forecast['upper'][j,z])} for z,s in enumerate(PRIMARY_SIGNALS)},
            'signal_units':{s:UNITS[SIGNALS.index(s)] for s in PRIMARY_SIGNALS},
            'divergence':{'tdi':finite(divergence['tdi'][j]),'risk_is_separate':True},
            'reliability':{'recovery_score':finite(f[-2]),'persistence_score':finite(f[-1]),'adjusted_risk':risk[i].tolist()},
            'adaptive_sensing':{'measurement_needed':needed,'requested_measurement':CANDIDATES[k] if needed else None,
                'reason':'Positive expected measurement utility in ambiguous/stale state' if needed else 'Sufficient evidence, nonpositive expected utility, outside ambiguity gate, or cooldown',
                'expected_information_gain':finite(rec['expected_information_gain'][i]),'current_risk':risk[i].tolist(),
                'current_uncertainty':finite(rec['current_uncertainty'][i]),'current_data_quality':finite(rec['current_data_quality'][i]),
                'request_priority':'recommended' if needed else 'none'},
            'calibration':dict(zip(['calibrated_risk_24h','calibrated_risk_48h','calibrated_risk_72h'],risk[i].tolist())),
            'uncertainty_claim':'Predictive entropy and quality heuristic only; no validated confidence interval',**EVIDENCE})
    save_json(root(config)/'phase7_twin_states.json',{'states':states,'state_aggregation':'Received usable preceding48h means',
        'demo_hide_reveal_scenarios_are_separate_from_intact_snapshot_states':True})
    return examples


def figures(config,data,result,risk,reference,features,cases,updates):
    directory=root(config)/'figures';directory.mkdir(exist_ok=True,mode=0o700);names=[]
    def save(fig,name):
        fig.suptitle(LABEL,fontsize=9);fig.tight_layout(rect=[0,0,1,.95]);p=directory/(name+'.png');fig.savefig(p,dpi=160);plt.close(fig);p.chmod(0o600);names.append(name)
    fig,ax=plt.subplots(figsize=(7,4))
    for group,color in [('recovery','tab:blue'),('deterioration','tab:red')]:
        idx=np.flatnonzero(data[group])[:400]
        if len(idx):
            z=abs((data['sequence'][idx,-8:,:,0]-data['reference'][idx,None,:])/np.maximum(data['scale'][idx,None,:],1e-6))
            import warnings
            with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);curve=np.nanmean(z,axis=(0,2))
            ax.plot(np.arange(-42,1,6),curve,label=group,color=color)
    ax.set(xlabel='Hours before current cutoff (retrospective bins)',ylabel='Mean absolute personal standardized deviation');ax.legend();save(fig,'recovery_vs_deterioration')
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,h in zip(axes,['24','48','72']):
        for name in ['reference','selected']:
            bins=result[name][h]['calibration']['bins'];ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=name)
        ax.plot([0,.08],[0,.08],'k--',lw=.6);ax.set(title=h+'h',xlabel='Mean prediction',ylabel='Observed event fraction');ax.legend(fontsize=8)
    save(fig,'calibration_before_after')
    fig,ax=plt.subplots(figsize=(7,4));groups=['recovery','decoy','ordinary_stable'];x=np.arange(3)
    for n,name in enumerate(['reference','selected']):ax.bar(x+(n-.5)*.35,[result[name]['subgroups72'][g]['alert_fraction'] for g in groups],width=.35,label=name)
    ax.set_xticks(x,groups);ax.set_ylabel('72h false-alert fraction');ax.legend();save(fig,'recovery_false_alert_comparison')
    fig,ax=plt.subplots(figsize=(7,4));mask=updates['requested'];ax.hist(updates['observed_brier_gain'][mask],bins=30);ax.axvline(0,color='black');ax.set(xlabel='Observed Brier gain from requested measurement',ylabel='Cases');save(fig,'adaptive_sensing_information_gain')
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter(updates['before'][mask,2],updates['after'][mask,2],s=12,alpha=.5);mx=max(.05,float(np.max(updates['before'][:,2])));ax.plot([0,mx],[0,mx],'k--');ax.set(xlabel='Risk72 before measurement',ylabel='Risk72 after requested measurement');save(fig,'risk_before_after_measurement')
    fig,axes=plt.subplots(1,3,figsize=(10,3.6))
    for ax,metric in zip(axes,['auprc','auroc','brier']):
        for name in ['reference','selected']:ax.plot([24,48,72],[result[name][h]['metrics'][metric] for h in ['24','48','72']],'o-',label=name)
        ax.set(xlabel='Horizon (h)',ylabel=metric);ax.legend(fontsize=8)
    save(fig,'replication_performance_comparison');save_json(root(config)/'figure_manifest.json',{'figures':names,'count':len(names)})


def generate(config,data,risk,reference,features,cases,updates,rec):
    out=root(config);r=json.loads((out/'replication_results.json').read_text());d=decision(r);save_json(out/'decision.json',d)
    examples=private_cases(config,data,risk,features,cases,updates,rec)
    figures(config,data,r,risk,reference,features,cases,updates)
    full=json.loads((out/'full_development_complete.json').read_text());fast=json.loads((out/'fast_selection_final.json').read_text());audit=json.loads((Path(config['replication'])/'replication_audit.json').read_text())
    header='**'+LABEL+'.**\n\n';common='This is one simulator-specific synthetic replication. It is not clinical evidence, an intervention study or a validated acquisition policy. The original Phase6 TEST was never reopened.\n\n'
    perf=table(['System','Horizon','Prevalence','AUPRC','AUROC','Brier','Sensitivity','Specificity','PPV'],[[name,h]+[number(r[name][h]['metrics'][k]) for k in ['prevalence','auprc','auroc','brier','sensitivity','specificity','ppv']] for name in ['reference','selected'] for h in ['24','48','72']])
    cal=table(['System','Horizon','Brier','Slope','Intercept','ECE'],[[name,h,number(r[name][h]['metrics']['brier'])]+[number(r[name][h]['calibration'][k]) for k in ['slope','intercept','ece_equal_count']] for name in ['reference','selected'] for h in ['24','48','72']])
    groups=table(['System','Group','Rows','Alert fraction'],[[name,g,r[name]['subgroups72'][g]['rows'],number(r[name]['subgroups72'][g]['alert_fraction'])] for name in ['reference','selected'] for g in ['deterioration','recovery','decoy','ordinary_stable']])
    documents={
        'RECOVERY_DISCRIMINATION.md':'# Recovery discrimination\n\n'+header+common+
            'Twenty-one observable features describe the preceding48h received bins: seven returns toward personal reference, seven current absolute standardized deviations, deviation fraction, consecutive deviating bins, systems, agreement, worsening fraction, recovery score and persistence score. No oracle annotations enter predictions.\n\n'+
            f"Two candidates were tested on one1,800patient subset: offset-logistic and one fixed temporal rule. Selected core: **{full['winner']}**. Both candidate results, including failed recovery gates, are retained privately. No strength sweep or favorable subset resampling occurred.\n\n"+groups+'\n\n'+
            f"Paired replication recovery alert-fraction difference: {r['paired_recovery_false_alert_fraction']}. These groups use evaluation-only simulator annotations and can overlap. Sensitivity is landmark-level; alert burden uses eligible fixed daily landmarks.\n",
        'CALIBRATION_RELIABILITY.md':'# Calibration reliability\n\n'+header+common+
            f"Selected conditional-interval methods: {full['calibration_methods']}. Existing, Platt and isotonic were the only candidates. Calibration fits used600validation-calibration patients; thresholds used300different operating patients. Choices were selected on the fixed subset and locked before the full development pass. Cumulative probabilities retain24h≤48h≤72h ordering.\n\n"+cal+'\n\n'+
            'Calibration diagnostics on replication are evaluation only. Ranking retention, intercept/slope trade-offs and failed gates are retained; no replication recalibration was performed.\n',
        'ADAPTIVE_SENSING.md':'# Adaptive sensing\n\n'+header+common+
            'Candidates: weight, paired home BP, SpO2 and device-capable HRV. No invasive MAP. TRAIN-only Ridge estimates hide/reveal Brier improvement plus0.001times correct-minus-harmed decisions, minus configurable relative burden. Relative burden weights are weight1, BP1.5, SpO2 1, HRV2, scaled by0.00001Brier units; these are engineering assumptions, not prices or medical utility.\n\n'+
            'Request the highest positive expected-utility measurement when near the frozen alert threshold or when risk is elevated with stale/poor-quality sensors. Persistent multisystem high-risk evidence can suppress requests. At most one request is issued per72h per patient on the daily schedule.\n\n'+
            'The exact benchmark jointly hides the latest reading for each candidate newly received since the previous cutoff, branches the causal baseline before its current update, then reveals one actually observed reading. Future/unreceived readings cannot be revealed. Utility inputs exclude revealed values and outcomes.\n\n'+
            table(['Replication measure','Value'],[[k,str(v)] for k,v in r['adaptive_sensing'].items() if not isinstance(v,dict)])+'\n\n'+
            'Policy frequency covers all127,500intact daily landmarks. Observed correction/gain estimates cover a fixed1,000landmark benchmark from500patients; these are distinct estimands. Predictive entropy changes are descriptive, not validated uncertainty reduction.\n',
        'PHASE6_5_REPLICATION.md':'# Phase6.5 independent replication\n\n'+header+common+
            f"Seed2027;2,500patients;60days;127,500landmarks. Event patients: {audit['event_patients']}; event-patient prevalence: {number(audit['event_patient_prevalence'])}. Essential audit: {sum(audit['gates'].values())}/{len(audit['gates'])} gates passed. Simulator source/configuration unchanged. Reproducibility is demonstrated on two actual generated patient histories; all artifacts have checksums.\n\n"+perf+'\n\n'+
            f"Paired72h AUPRC difference: {r['paired_auprc72']}.\n\n"+
            table(['System','Horizon','Metric','Patient95%CI','Valid resamples'],[[name,h,k,str(v['interval95']),str(v['valid_resamples'])+'/'+str(v['requested'])] for name in r['intervals'] for h in ['24','48','72'] for k,v in r['intervals'][name][h].items()])+'\n\n'+
            f"One frozen evaluation event: `{r['event_id']}`, `{r['timestamp_utc']}`. No choices were changed after replication generation/evaluation.\n",
        'PHASE6_5_DECISION.md':'# Phase6.5 decision\n\n'+header+f"**{d['decision']}**\n\n"+
            table(['Frozen replication gate','Passed'],[[k,str(v)] for k,v in d['gates'].items()])+'\n\n'+d['recommendation_phase7']+'\n\n'+common+
            'The Phase6 findings remain: personalization and stable Shadow MAE gains supported; incremental TDI/EHR benefit unproven; uncertainty weak. Two recovery candidates failed development selection; that negative result was retained. Calibration and sensing are evaluated on a new seed, within the same simulator—not external clinical validation.\n\n'+
            'Private `phase7_demo_cases.json` records required pattern availability plus harmed/unhelpful cases. An unavailable exact demonstration is reported honestly rather than manufactured. Private `phase7_twin_states.json` contains12unit-aware intact snapshots with reliability, calibration and sensing fields.\n'}
    for name,text in documents.items():(Path('docs')/name).write_text(text)
    save_json(out/'report_manifest.json',{'documentation_sha256':{name:file_hash(Path('docs')/name) for name in ['PHASE6_5_COMPUTE_PLAN.md']+list(documents)},
        'figure_count':6,'representative_pattern_availability':{x['case']:x.get('exact_pattern_available') for x in examples if 'exact_pattern_available' in x}})
