"""Private, labelled scientific figures from frozen synthetic predictions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
from src.twin.protocol import load,PRIMARY_SIGNALS,EVIDENCE
from src.vitaldb_feasibility.source import private_json


def generate(config=None):
    c=config or load();out=Path(c['output']);root=out/'figures';root.mkdir(exist_ok=True,mode=0o700);result=json.loads((out/'test_results.json').read_text());files=[]
    def save(fig,name):
        fig.suptitle('SYNTHETIC POC VALIDATION · NOT CLINICALLY VALIDATED',fontsize=9);fig.tight_layout(rect=(0,0,1,.96));path=root/(name+'.png');fig.savefig(path,dpi=160);plt.close(fig);path.chmod(0o600);files.append(path.name)
    predictions={}
    for name in c['model_variants']:
        with np.load(out/f'test_predictions_{name}.npz',allow_pickle=False) as z:predictions[name]=(np.cumsum(z['labels'],axis=1),z['probabilities'])
    comparisons=['ehr_only','wearable_only','population_fusion','personalized_fusion','full_twin','consumer_twin']
    for mode in ['pr','roc']:
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for j,ax in enumerate(axes):
            for name in comparisons:
                y,p=predictions[name]
                if mode=='pr':vertical,horizontal,_=precision_recall_curve(y[:,j],p[:,j]);ax.plot(horizontal,vertical,label=name)
                else:horizontal,vertical,_=roc_curve(y[:,j],p[:,j]);ax.plot(horizontal,vertical,label=name)
            ax.set(title=f'{(j+1)*24}h',xlabel='Recall' if mode=='pr' else 'False positive rate',ylabel='Precision' if mode=='pr' else 'True positive rate');ax.legend(fontsize=6)
        save(fig,mode+'_curves')
    for horizon in ['24','48','72']:
        fig,ax=plt.subplots(figsize=(6,5))
        for name in comparisons:
            bins=result['models'][name]['horizons'][horizon]['calibration']['bins'];ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],marker='o',label=name)
        ax.plot([0,1],[0,1],linestyle='--',color='gray');ax.set(title=f'{horizon}h equal-count reliability',xlabel='Predicted risk',ylabel='Observed event fraction');ax.legend(fontsize=7);save(fig,'calibration_'+horizon)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,['auprc','auroc','brier']):
        for name in comparisons:ax.plot([24,48,72],[result['models'][name]['horizons'][h]['metrics'][metric] for h in ['24','48','72']],marker='o',label=name)
        ax.set(title=metric,xlabel='Prediction horizon (h)');ax.legend(fontsize=6)
    save(fig,'horizon_performance')
    names=c['model_variants'];fig,ax=plt.subplots(figsize=(11,5));ax.barh(names,[result['models'][name]['horizons']['72']['metrics']['auprc'] for name in names]);ax.set(xlabel='72h AUPRC',title='Frozen model ablations');save(fig,'model_ablations')
    pairs=result['paired_auprc72_differences'];fig,ax=plt.subplots(figsize=(11,5));selected=list(pairs)
    for i,key in enumerate(selected):
        d=pairs[key]['difference'];ci=pairs[key]['interval95'];ax.plot(d,i,'o')
        if ci:ax.plot(ci,[i,i],color='gray')
    ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(len(selected)),selected);ax.set(xlabel='Paired patient-bootstrap difference in72h AUPRC',title='Personalization and component benefit');save(fig,'personalization_benefit')
    states=json.loads((out/'twin_states.json').read_text())['states'];first=states[0];template=first['retrospective_7day_template'];fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,signal in zip(axes.flat,['heart_rate','hrv_rmssd','spo2','weight']):
        j=PRIMARY_SIGNALS.index(signal);t=template['relative_hours'];actual=np.array(template['actual_six_hour_bin_values'],float)[:,j];expected=np.array(template['expected'],float)[:,j];lo=np.array(template['lower'],float)[:,j];hi=np.array(template['upper'],float)[:,j];ax.plot(t,actual,'o-',label='Received observed');ax.plot(t,expected,label='Stable Shadow template');ax.fill_between(t,lo,hi,alpha=.2);ax.set(title=signal,xlabel='Hours relative to selected cutoff',ylabel=first['signal_units'][signal]);ax.legend(fontsize=7)
    save(fig,'shadow_vs_actual')
    aligned=json.loads((out/'event_aligned.json').read_text());agg=aligned['aggregate'];lags=sorted((int(k) for k in agg),reverse=True);fig,axes=plt.subplots(1,3,figsize=(14,4))
    for key in ['risk24','risk48','risk72']:axes[0].plot([-h for h in lags],[agg[str(h)][key]['mean'] for h in lags],marker='o',label=key)
    axes[1].plot([-h for h in lags],[agg[str(h)]['tdi']['mean'] for h in lags],marker='o')
    for s in ['heart_rate','hrv_rmssd','weight','steps']:axes[2].plot([-h for h in lags],[agg[str(h)]['physiological_z'][s]['mean'] for h in lags],marker='o',label=s)
    for ax,title in zip(axes,['Risk','TDI','Signed physiological residual']):ax.set(title=title,xlabel='Nominal hours to event (last daily issued prediction)');ax.legend(fontsize=7) if ax!=axes[1] else None
    save(fig,'event_aligned')
    curves=aligned['representative_private_curves'];fig,axes=plt.subplots(2,1,figsize=(9,7))
    for curve in curves:axes[0].plot(-np.array(curve['lead_hours']),curve['tdi'],alpha=.6);axes[1].plot(-np.array(curve['lead_hours']),np.array(curve['risk'])[:,2],alpha=.6)
    axes[0].set(title='Individual pre-event TDI trajectories',ylabel='TDI');axes[1].set(title='Individual pre-event Risk72 trajectories',ylabel='Risk72',xlabel='Hours to event');save(fig,'tdi_and_risk_trajectories')
    decoys=result['decoys']['groups'];fig,ax=plt.subplots(figsize=(11,5));ax.barh(list(decoys),[decoys[k]['false_alert_fraction72'] or 0 for k in decoys]);ax.set(title='Balanced72h false alerts on nonevent decoy/recovery landmarks',xlabel='False alert fraction');save(fig,'decoy_false_alerts')
    sensor=result['sensor_ablations']['experiments'];stress=result['stress']['variants'];fig,axes=plt.subplots(1,2,figsize=(14,5));axes[0].barh(list(sensor),[sensor[k]['metrics']['72']['metrics']['auprc'] for k in sensor]);axes[0].axvline(result['models']['full_twin']['horizons']['72']['metrics']['auprc'],color='black',linestyle='--');axes[0].set(title='Whole-history sensor deletion',xlabel='72h AUPRC');axes[1].barh(list(stress),[stress[k]['models']['full_twin']['metrics']['72']['metrics']['auprc'] for k in stress]);axes[1].set(title='Matched128-patient stress cohorts',xlabel='72h AUPRC');save(fig,'missing_sensor_robustness')
    uq=result['uncertainty']['full_twin'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['model_sd72','data_quality']):ax.bar(range(4),[b['brier'] for b in uq[key]['quartile_brier']]);ax.set(title=key,xlabel='Uncertainty quartile',ylabel='Mean3-horizon Brier')
    save(fig,'uncertainty_behavior')
    challenge=result['challenge'];fig,ax=plt.subplots(figsize=(7,4));cn=list(challenge['models']);ax.bar(cn,[challenge['models'][n]['horizons']['72']['metrics']['auprc'] for n in cn]);ax.set(title='Observable personal-abnormal / population-range challenge',ylabel='72h AUPRC');save(fig,'personalization_challenge')
    private_json(root/'manifest.json',{'figure_files':files,'types':['PR','ROC','calibration24/48/72','horizon performance','model ablations','personalization benefit','Shadow vs Actual','TDI/risk trajectories','event-aligned','decoy/recovery','sensor robustness','uncertainty','challenge'],'private':True,**EVIDENCE});return files

if __name__=='__main__':generate()
