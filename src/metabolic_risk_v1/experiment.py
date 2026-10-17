"""Reproducible nested participant evaluation; only new risk artifacts may be saved."""
import json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
from src.metabolic_d2.protocol import digest,write_json
from src.metabolic_d2.data import subset
from src.metabolic_d2.models import ForecastModel,subject_weights
from src.metabolic_d2.metrics import event_metrics
from .model import RiskHead,VARIANTS,DYNAMIC_NAMES
from .protocol import ROOT,SEED,load_inputs,folds,profile_permutation,preserve_snapshot,assert_preserved


def trajectories(train,allowed,test,seed,checkpoint):
    """Independent exact D2-family clones; never overwrite/reference-fit D2 artifacts."""
    checkpoint.mkdir(parents=True,exist_ok=True,mode=0o700)
    models={}
    for variant in ['population','personal']:
        path=checkpoint/(variant+'.pkl');meta=path.with_suffix('.json')
        if path.exists():
            record=json.loads(meta.read_text())
            if record['participants']!=sorted(allowed) or record['hash']!=digest(path) or record['seed']!=seed:
                raise RuntimeError('Trajectory checkpoint membership/hash mismatch')
            model=ForecastModel.load(path)
        else:
            begin=time.perf_counter();model=ForecastModel('boosting',variant).fit(train,set(allowed),seed)
            model.save(path);write_json(meta,{'participants':sorted(allowed),'seed':seed,'hash':digest(path),
                                            'seconds':time.perf_counter()-begin})
        models[variant]=model
    p=models['population'].predict(test)
    warm=test['warm']
    if warm.any():p[warm]=models['personal'].predict(subset(test,warm))
    return p


def threshold(y,p,pid):
    fpr,tpr,t=roc_curve(y,p,sample_weight=subject_weights(pid))
    valid=np.flatnonzero(np.isfinite(t)&(t>=0)&(t<=1))
    best=max(valid,key=lambda i:(round(float((tpr[i]+1-fpr[i])/2),12),1-fpr[i],t[i]))
    return float(t[best])


def metrics(y,p,pid,operating):
    result=event_metrics(y,p,.5,pid)
    y=np.asarray(y,bool);positive=np.asarray(operating,bool)
    tp=int((y&positive).sum());fp=int((~y&positive).sum());tn=int((~y&~positive).sum());fn=int((y&~positive).sum())
    div=lambda a,b:float(a/b) if b else None
    result.update({'threshold':'fold-specific calibration-selected operating point',
                   'confusion':{'tp':tp,'fp':fp,'tn':tn,'fn':fn},'sensitivity':div(tp,tp+fn),
                   'specificity':div(tn,tn+fp),'PPV':div(tp,tp+fp),'NPV':div(tn,tn+fn),'F1':div(2*tp,2*tp+fp+fn)})
    return result


def bootstrap(y,p,pid,operating,repeats=1000):
    """Whole participant weighted resamples, shared draws across comparators."""
    ids,inv=np.unique(pid,return_inverse=True);rng=np.random.default_rng(SEED)
    order=np.argsort(-p,kind='stable');scores=p[order];truth=np.asarray(y,dtype=float)[order];owner=inv[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
    out={k:[] for k in ['AUPRC','AUROC','Brier','sensitivity','specificity','PPV','NPV','F1']}
    yy=np.asarray(y,bool);op=np.asarray(operating,bool)
    categories=[yy&op,~yy&op,~yy&~op,yy&~op]
    for _ in range(repeats):
        count=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids));w=count[inv];ws=count[owner]
        tp=np.cumsum(ws*truth)[ends];fp=np.cumsum(ws*(1-truth))[ends]
        if tp[-1]>0 and fp[-1]>0:
            recall=tp/tp[-1];precision=tp/(tp+fp)
            # Drop zero-mass prefixes to avoid0/0 after resampling.
            precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
            out['AUPRC'].append(float(np.sum(np.diff(np.r_[0,recall])*precision)))
            out['AUROC'].append(float(np.trapezoid(np.r_[0,recall],np.r_[0,fp/fp[-1]])))
        out['Brier'].append(float(np.average((p-y)**2,weights=w)))
        a,b,c,d=[float(w[v].sum()) for v in categories]
        for k,num,den in [('sensitivity',a,a+d),('specificity',c,c+b),('PPV',a,a+b),('NPV',c,c+d),('F1',2*a,2*a+b+d)]:
            if den:out[k].append(num/den)
    return {k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in out.items()}


def run():
    started=time.perf_counter();ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (ROOT/'results.json').exists():raise FileExistsError('Completed risk experiment is immutable; inspect existing results')
    cfg=json.loads(Path('configs/metabolic_risk_v1.json').read_text())
    fingerprint={str(p):digest(p) for p in Path('src/metabolic_risk_v1').glob('*.py')}
    fingerprint['configs/metabolic_risk_v1.json']=digest('configs/metabolic_risk_v1.json')
    fingerprint['data/metabolic_d35/participant_static.parquet']=digest('data/metabolic_d35/participant_static.parquet')
    frozen=ROOT/'protocol_frozen.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['fingerprint']!=fingerprint:raise RuntimeError('Risk protocol changed during fitting')
    else:write_json(frozen,{'fingerprint':fingerprint,'config':cfg,'started_monotonic':started},exclusive=True)
    before=preserve_snapshot();data,profiles,scope=load_inputs();outer=folds(data['pid']);rowfold=np.array([outer[p] for p in data['pid']])
    predictions={v:np.zeros(len(data['pid'])) for v in VARIANTS};operating={v:np.zeros(len(data['pid']),bool) for v in VARIANTS}
    records=[]
    for fold in range(5):
        validmask=rowfold==fold;trainids=sorted(set(data['pid'][~validmask]));scope.assert_allowed(trainids)
        shuffled=np.random.default_rng(SEED+fold).permutation(trainids)
        calids=set(shuffled[:16]);fitids=set(shuffled[16:]);validids=set(data['pid'][validmask])
        assert not (calids&fitids or validids&fitids or validids&calids)
        fitmask=np.isin(data['pid'],list(fitids));calmask=np.isin(data['pid'],list(calids))
        train=subset(data,fitmask);cal=subset(data,calmask);valid=subset(data,validmask)
        directory=ROOT/'folds'/str(fold);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        checkpoint=directory/'complete.json'
        if checkpoint.exists():
            record=json.loads(checkpoint.read_text());pfile=directory/'predictions.npz'
            if record['prediction_sha256']!=digest(pfile):raise RuntimeError('Risk predictions changed')
            with np.load(pfile,allow_pickle=False) as saved:
                for v in VARIANTS:predictions[v][validmask]=saved[v];operating[v][validmask]=saved[v+'_positive']
            records.append(record);print(json.dumps({'fold':fold+1,'checkpoint_reused':True}),flush=True);continue
        inner=folds(train['pid'],4,SEED+fold);innerrow=np.array([inner[p] for p in train['pid']]);oof=np.zeros_like(train['y'])
        for j in range(4):
            mask=innerrow==j;allowed=set(train['pid'][~mask])
            oof[mask]=trajectories(subset(train,~mask),allowed,subset(train,mask),SEED+fold*10+j,directory/f'inner{j}')
            print(json.dumps({'fold':fold+1,'inner_trajectory_fold':j+1,'fit_participants':len(allowed)}),flush=True)
        combined={k:np.concatenate([cal[k],valid[k]]) for k in cal}
        p=trajectories(train,fitids,combined,SEED+100+fold,directory/'outer_trajectory')
        pcal,pvalid=p[:len(cal['pid'])],p[len(cal['pid']):]
        donor=profile_permutation(fitids,SEED+fold)
        record={'fold':fold,'fit_participants':sorted(fitids),'calibration_participants':sorted(calids),
                'validation_participants':sorted(validids),'shuffled_donors':donor,'metrics':{},'thresholds':{}}
        arrays={}
        for variant in VARIANTS:
            model=RiskHead(variant).fit(train,oof,profiles,fitids,donor if variant=='shuffled_static' else None)
            model.calibrate(cal,pcal,profiles,calids)
            probs=model.probabilities(cal,pcal,profiles);point=threshold(cal['spike'],probs,cal['pid'])
            model.threshold=point
            prob=model.probabilities(valid,pvalid,profiles);positive=prob>=point
            predictions[variant][validmask]=prob;operating[variant][validmask]=positive
            record['metrics'][variant]=metrics(valid['spike'],prob,valid['pid'],positive);record['thresholds'][variant]=point
            model.save(directory/(variant+'.pkl'));arrays[variant]=prob;arrays[variant+'_positive']=positive
        pfile=directory/'predictions.npz';np.savez_compressed(pfile,**arrays);pfile.chmod(0o600)
        record['prediction_sha256']=digest(pfile);write_json(checkpoint,record);records.append(record)
        print(json.dumps({'fold':fold+1,'completed':True,'fused_AUPRC':record['metrics']['fused']['AUPRC']}),flush=True)
    aggregate={v:metrics(data['spike'],predictions[v],data['pid'],operating[v]) for v in VARIANTS}
    f,d,s=aggregate['fused'],aggregate['dynamic_only'],aggregate['shuffled_static'];gate=cfg['go_gate']
    checks={'AUPRC_nondegradation':f['AUPRC']>=d['AUPRC']-gate['maximum_pooled_AUPRC_degradation'],
            'AUROC_nondegradation':f['AUROC']>=d['AUROC']-gate['maximum_pooled_AUROC_degradation'],
            'Brier_nondegradation':f['Brier']<=d['Brier']+gate['maximum_pooled_Brier_degradation'],
            'fold_stability':sum(r['metrics']['fused']['AUPRC']>=r['metrics']['dynamic_only']['AUPRC']-.03 for r in records)>=4,
            'directional_linkage':f['AUPRC']>s['AUPRC'] or f['Brier']<s['Brier'],
            'calibration_usable':f['ECE_5bins']<=gate['maximum_ECE_5bins'],'no_leakage':True}
    intervals={v:bootstrap(data['spike'],predictions[v],data['pid'],operating[v],cfg['bootstrap_replicates']) for v in VARIANTS}
    result={'version':'METABOLIC_MULTIMODAL_RISK_V1','participants':80,'windows':len(data['pid']),'folds':5,
            'fold_counts':{'classifier_fit':48,'calibration':16,'validation':16},'metrics':aggregate,'ci95':intervals,
            'fold_metrics':[{'fold':r['fold'],'metrics':r['metrics'],'thresholds':r['thresholds']} for r in records],
            'go_checks':checks,'decision':'GO' if all(checks.values()) else 'NO-GO','bootstrap_replicates':cfg['bootstrap_replicates'],
            'scope':'DEVELOPMENT-VALIDATED MULTIMODAL RISK HEAD; nested participant folds, no spent holdout evaluation',
            'preservation':assert_preserved(before),'runtime_seconds':time.perf_counter()-started}
    if result['decision']=='GO':
        # Original frozen forecaster training64 + reserved calibration16, never refit D2.
        fitting=np.isin(data['pid'],scope.manifest['model_development']);train=subset(data,fitting);cal=subset(data,~fitting)
        cacheids=np.load('data/metabolic_d2/cache/development.npz',allow_pickle=False)['pid'];assert np.array_equal(cacheids,train['pid'])
        with np.load('data/metabolic_d2/cv/oof_predictions.npz',allow_pickle=False) as saved:
            oof=saved['boosting_population'].copy();oof[train['warm']]=saved['boosting_personal'][train['warm']]
        from src.metabolic_d3.reference import Reference
        reference=Reference();pcal=reference.cold.predict(cal);warm=cal['warm']
        pcal[warm]=reference.warm.predict(subset(cal,warm))
        model=RiskHead('fused').fit(train,oof,profiles,set(train['pid']))
        model.calibrate(cal,pcal,profiles,set(cal['pid']));probs=model.probabilities(cal,pcal,profiles)
        model.threshold=threshold(cal['spike'],probs,cal['pid']);path=ROOT/'model.pkl';model.save(path)
        result['final']={'path':str(path),'sha256':digest(path),'threshold':model.threshold,'feature_order':model.feature_order,
                         'fit_participants':64,'calibration_participants':16,'classifier_params':model.estimator.get_params(),
                         'calibration_params':model.calibrator.get_params(),'preprocessing':'Embedded independent dynamic/static train-only preprocessors and StaticEncoder',
                         'static_profile_sha256':digest('data/metabolic_d35/participant_static.parquet')}
    write_json(ROOT/'results.json',result,exclusive=True)
    public=Path('artifacts/metabolic_risk_v1');public.mkdir(parents=True,exist_ok=True)
    write_json(public/'manifest.json',result)
    text='# Multimodal risk V1 development results\n\n'+json.dumps(result,indent=2)+'\n'
    Path('docs/METABOLIC_MULTIMODAL_RISK_RESULTS.md').write_text(text)
    print(json.dumps({'decision':result['decision'],'metrics':aggregate,'seconds':result['runtime_seconds']}),flush=True)

if __name__=='__main__':run()
