"""Fit/tune only TRAIN+VALIDATION; atomically freeze every comparator before test."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.predictive.protocol import PROTOCOL,CANDIDATES,MODALITIES
from src.predictive.data import Cohort,weights,columns
from src.predictive.models import Tabular,Rule,LateFusion
from src.predictive.temporal import CompactTCN
from src.predictive.inference import background,explain_validation,feature_associations
from src.predictive.metrics import basic,select_calibration,thresholds,patient_folds,Calibrator,paired_bootstrap,operating,bootstrap
from src.predictive.artifacts import source_hash,previous_files,verify_previous,versions,freeze,write_json,write_rows,save_model,sealed_directory


def candidate_name(modality,i):return f'{modality}__candidate{i}'


def refit(model,rows,assignments,multiplicity):
    if isinstance(model,Tabular):return Tabular(model.modality,model.config).fit(rows,assignments,multiplicity)
    if isinstance(model,Rule):return Rule(model.kind).fit(rows,assignments,multiplicity)
    if isinstance(model,CompactTCN):return CompactTCN(model.fusion,model.l2,model.epochs).fit(rows,assignments,multiplicity)
    if isinstance(model,LateFusion):
        ehr=refit(model.ehr,rows,assignments,multiplicity);phys=refit(model.phys,rows,assignments,multiplicity)
        return LateFusion(ehr,phys).fit(rows,assignments,model.inner_folds,multiplicity)
    raise ValueError('Unsupported bootstrap model')


def model_stats(model,validation,assignments):
    p=model.predict(validation);y=[r['label'] for r in validation]
    cal,oof,trials=select_calibration(p,validation,assignments)
    hard=np.array([r['audit']['hard'] for r in validation]);hard_metrics=basic(np.array(y)[hard],oof[hard]) if any(hard) else None
    stats={'config':model.config,'modality':model.modality,'converged':model.converged,
           'validation_raw':basic(y,p),'validation_calibrated_oof':basic(y,oof),'calibration':cal.method,
           'calibration_candidates':trials,'hard_validation':hard_metrics,'complexity':model.complexity,'trainable_parameter_count':sum(v.size for v in model.params.values()) if isinstance(model,CompactTCN) else None}
    return stats,p,oof,cal


def pick(names,stats):
    eligible=[n for n in names if stats[n]['converged'] and stats[n]['validation_raw']['auprc'] is not None]
    return max(eligible,key=lambda n:(stats[n]['validation_raw']['auprc'],-stats[n]['complexity'],-stats[n]['validation_calibrated_oof']['brier']))


def choose_primary(names,stats,models,validation,incremental):
    supported=[n for n in names if n not in incremental or (incremental[n]['interval95'] is not None and incremental[n]['interval95'][0]>0)]
    names=supported or names
    best=max(stats[n]['validation_raw']['auprc'] for n in names)
    eligible=[n for n in names if stats[n]['validation_raw']['auprc']>=best-.01]
    # Calibration and hard-case checks use validation only, never eventual test.
    baseline=stats['prevalence']['validation_calibrated_oof']['brier']
    calibrated=[n for n in eligible if stats[n]['validation_calibrated_oof']['brier']<=baseline+.005]
    if calibrated:eligible=calibrated
    hard_rows=[r for r in validation if r['audit']['hard']]
    if len({r['patient_id'] for r in hard_rows if r['label']})>=3:
        prevalence=np.mean([r['label'] for r in hard_rows])
        robust=[n for n in eligible if stats[n]['hard_validation'] and stats[n]['hard_validation']['auprc'] is not None and stats[n]['hard_validation']['auprc']>=prevalence]
        if robust:eligible=robust
    return min(eligible,key=lambda n:(stats[n]['complexity'],stats[n].get('validation_balanced_operating',{}).get('false_alerts_per_eligible_hour',1),len(getattr(getattr(models[n],'encoder',None),'names',[])),stats[n]['validation_calibrated_oof']['brier'],-stats[n]['validation_raw']['auprc'],n))


def train(output='data/results/phase4',uncertainty_refits=20):
    root=sealed_directory(output);snapshot=previous_files();version=source_hash()
    c=Cohort();training=c.batch('train');validation=c.batch('validation');models={};stats={};pred={};oof={};calibrators={};winners={}
    write_json(root/'protocol.json',PROTOCOL);write_json(root/'preservation.json',snapshot)
    write_json(root/'feature_definitions.json',{'version':'phase4_causal_v1','causal_contract':PROTOCOL,'modality_columns':{m:columns(m,training[0]['features']) for m in MODALITIES},'training_fit_patients':sorted({r['patient_id'] for r in training}), 'source_data':'authoritative Phase35 usable/QC arrays and as-of context; original units preserved there','numeric_missingness':'train-only median replacement plus an indicator for every numeric feature; no physiological interpolation'})
    write_json(root/'splits.json',json.loads((c.root/'splits.json').read_text()))
    write_json(root/'metadata.json',{'seed':2026,'dependencies':versions(),'source_code_sha256':version,'source_manifest':c.manifest,
                                  'partitions':json.loads((c.root/'report.json').read_text())['decision']['partitions']})
    def add(name,model):
        models[name]=model;stats[name],pred[name],oof[name],calibrators[name]=model_stats(model,validation,c.assignments)
        print({'candidate':name,'validation_auprc':stats[name]['validation_raw']['auprc'],'converged':model.converged},flush=True)
    with threadpool_limits(limits=1):
        for kind in ['prevalence','map_proximity','map_trend']:
            model=Rule(kind).fit(training,c.assignments)
            if kind=='prevalence':
                models[kind]=model;p=model.predict(validation);pred[kind]=p;oof[kind]=p
                calibrators[kind]=Calibrator('none').fit(p,validation,c.assignments)
                stats[kind]={'config':model.config,'modality':model.modality,'converged':True,'validation_raw':basic([r['label'] for r in validation],p),
                             'validation_calibrated_oof':basic([r['label'] for r in validation],p),'calibration':'none','calibration_candidates':{},
                             'hard_validation':basic([r['label'] for r in validation if r['audit']['hard']],p[[r['audit']['hard'] for r in validation]]) if any(r['audit']['hard'] for r in validation) else None,'complexity':0}
            else:add(kind,model)
        for modality in MODALITIES:
            names=[]
            for i,config in enumerate(CANDIDATES):
                name=candidate_name(modality,i);add(name,Tabular(modality,config).fit(training,c.assignments));names.append(name)
            winners[modality]=pick(names,stats)
        for fusion in [False,True]:
            prefix='temporal_fused' if fusion else 'temporal_phys';names=[]
            for i,l2 in enumerate([.01,.1]):
                name=prefix+f'__candidate{i}';add(name,CompactTCN(fusion,l2).fit(training,c.assignments));names.append(name)
            winners[prefix]=pick(names,stats)
        late=LateFusion(models[winners['ehr']],models[winners['phys']]).fit(training,c.assignments,patient_folds(training))
        add('late_fusion',late);winners['late_fusion']='late_fusion'
        final_names=list(dict.fromkeys(['prevalence','map_proximity','map_trend']+list(winners.values())+[candidate_name(m,i) for m in ['ehr','phys','fused'] for i in [0,1]]))
        operating_rules={n:thresholds(validation,oof[n],c.assignments) for n in final_names}
        for n in final_names:
            stats[n]['validation_balanced_operating']=operating(validation,oof[n],operating_rules[n]['balanced'])
        paired={name:paired_bootstrap(validation,oof[winners[left]],oof[winners[right]],n=1000) for name,left,right in
                [('fusion_minus_phys','fused','phys'),('personal_minus_fused','personalized','fused'),('fusion_minus_ehr','fused','ehr')]}
        incremental={winners['fused']:paired['fusion_minus_phys'],winners['personalized']:paired['personal_minus_fused'],
                     winners['late_fusion']:paired_bootstrap(validation,oof[winners['late_fusion']],oof[winners['phys']],n=1000),
                     winners['temporal_phys']:paired_bootstrap(validation,oof[winners['temporal_phys']],oof[winners['phys']],n=1000),
                     winners['temporal_fused']:paired_bootstrap(validation,oof[winners['temporal_fused']],oof[winners['fused']],n=1000)}
        selected=choose_primary(final_names,stats,models,validation,incremental)
        # Both resampling populations are development-only and kept separate.
        rng=np.random.default_rng(2026);train_ids=sorted({r['patient_id'] for r in training});val_ids=sorted({r['patient_id'] for r in validation});ensemble=[]
        for i in range(uncertainty_refits):
            tm=Counter(rng.choice(train_ids,len(train_ids),replace=True));vm=Counter(rng.choice(val_ids,len(val_ids),replace=True))
            model=refit(models[selected],training,c.assignments,tm)
            cal=Calibrator(calibrators[selected].method).fit(model.predict(validation),validation,c.assignments,vm)
            ensemble.append((model,cal));print({'uncertainty_refit':i+1,'of':uncertainty_refits},flush=True)
        reference=background(training,c.assignments);explanations=explain_validation(models[selected],validation,reference)
        write_json(root/'validation_explanations.json',explanations)
        write_json(root/'validation_feature_explanations.json',{m:feature_associations(models[winners[m]],validation) for m in ['ehr','map_only','fused','personalized']})
        bundle={'background':reference,'explanations':explanations,'models':{n:models[n] for n in final_names},'calibrators':{n:calibrators[n] for n in final_names},'winners':winners,
                'selected':selected,'thresholds':operating_rules,'ensemble':ensemble,'protocol':PROTOCOL,'assignments':c.assignments}
        save_model(root/'pipeline.joblib',bundle);save_model(root/'validation_registry.joblib',models)
        write_json(root/'validation_metrics.json',{'candidates':stats,'winners':winners,'selected':selected,'paired_comparisons':paired,'incremental_complexity_checks':incremental})
        write_rows(root/'validation_predictions.jsonl',[{'window_id':r['window_id'],'patient_id':r['patient_id'],'icu_stay_id':r['icu_stay_id'],'t0':r['t0'],
                   'label':r['label'],'audit':r['audit'],'raw':{n:float(pred[n][i]) for n in final_names},'calibrated_oof':{n:float(oof[n][i]) for n in final_names}} for i,r in enumerate(validation)])
        write_json(root/'selection.json',{'selected':selected,'winners':winners,'thresholds':operating_rules,'model_selection_basis':'validation only',
                   'calibration':{n:calibrators[n].method for n in final_names},'training_prevalence':models['prevalence'].prevalence,
                   'validation_patient_folds':patient_folds(validation),'uncertainty_refits':uncertainty_refits,
                   'test_evaluated':False,'selection_rule':PROTOCOL['selection']})
    verify_previous(snapshot)
    if source_hash()!=version:raise ValueError('Source changed while fitting')
    freeze(root,{'selected':selected,'comparators':final_names})
    return {'status':'FROZEN','selected':selected,'test_accessed':False,'comparators':len(final_names)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/results/phase4');p.add_argument('--uncertainty-refits',type=int,default=20);a=p.parse_args();print(train(a.output,a.uncertainty_refits))
