"""Standalone private scientific figures; no patient identifiers in figure labels."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/twin_synthetic_mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.synthetic.config import SIGNALS
from src.synthetic.generate import iter_rows
from src.vitaldb_feasibility.source import private_json


def plot(root='data/synthetic/pilot'):
    root=Path(root);figures=root/'figures';figures.mkdir(mode=0o700,exist_ok=True);manifest=json.loads((root/'manifest.json').read_text());metrics=json.loads((root/'realism_metrics.json').read_text());ground=json.loads(Path('data/synthetic_grounding/grounding.json').read_text())
    truth=list(iter_rows(root/'ground_truth/records.jsonl.gz'));saved=[]
    with np.load(root/'validation_arrays.npz') as z:ids=z['patient_ids'].tolist();daily=z['daily'];latent=z['latent_daily'];missing=z['missing']
    lookup={pid:i for i,pid in enumerate(ids)}
    def finish(fig,name):
        fig.text(.5,.005,'SYNTHETIC POC — simulator evaluation; not clinical validation',ha='center',fontsize=8)
        
        if name=='correlations':fig.subplots_adjust(left=.15,right=.86,bottom=.30,top=.88,wspace=.65)
        else:fig.tight_layout(rect=[0,.03,1,.96])
        path=figures/(name+'.png');fig.savefig(path,dpi=150,metadata={'Description':'SYNTHETIC POC; not clinical validation'});path.chmod(0o600);plt.close(fig);saved.append(name+'.png')
    stable=next(r for r in truth if r['event_time_hours'] is None and not r['episodes']);event=next(r for r in truth if r['event_time_hours'] is not None)
    for row,name,title in [(stable,'stable_example','Stable virtual patient with independent decoys'),(event,'deteriorating_example','Virtual patient with latent-generated event')]:
        fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=True);data=daily[lookup[row['patient_id']]]
        for ax,signal in zip(axes,['heart_rate','systolic_bp','spo2','weight']):
            ax.plot(np.arange(manifest['days']),data[:,SIGNALS.index(signal)],lw=1);ax.set_ylabel(signal.replace('_',' '));ax.grid(alpha=.2)
            if row['event_time_hours'] is not None:ax.axvline(row['event_time_hours']/24,color='crimson',ls='--',label='simulated event')
        axes[-1].set_xlabel('Days since virtual monitoring began');fig.suptitle(title);finish(fig,name)
    fig,axes=plt.subplots(5,1,figsize=(10,12),sharex=True)
    for ax,path in zip(axes,['autonomic_first','congestion_like','pressure_hemodynamic','mixed','noisy_atypical']):
        candidates=[r for r in truth if any(e['path']==path and e['event_hour'] is not None for e in r['episodes'])]
        if not candidates:ax.text(.1,.5,'No event example available',transform=ax.transAxes);continue
        row=candidates[0];data=daily[lookup[row['patient_id']]];ed=row['event_time_hours']/24;days=np.arange(len(data));mask=(days>=ed-10)&(days<=ed+1)
        for signal in ['heart_rate','hrv_rmssd','systolic_bp','spo2','weight']:
            j=SIGNALS.index(signal);ref=np.nanmedian(data[:7,j]);scale=max(np.nanstd(data[:7,j]),1);ax.plot(days[mask]-ed,(data[mask,j]-ref)/scale,label=signal.replace('_',' '),lw=1)
        ax.axvline(0,color='crimson',ls='--');ax.set_title(path.replace('_',' '));ax.set_ylabel('Prior-reference SD');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Days relative to simulated event');fig.suptitle('Different observable deterioration patterns — selected synthetic examples');finish(fig,'deterioration_phenotypes')
    fig,axes=plt.subplots(2,3,figsize=(12,7));levels=[.01,.05,.25,.5,.75,.95,.99]
    for ax,(signal,item) in zip(axes.flat,metrics['comparison'].items()):
        real=item['real']['daily_values']['quantiles_01_05_25_50_75_95_99'];synth=item['synthetic']['daily_values']['quantiles_01_05_25_50_75_95_99'];ax.plot(levels,real,'o-',label='PerHeart exploratory');ax.plot(levels,synth,'o-',label='Synthetic POC');ax.set_title(signal.replace('_',' '));ax.set_xlabel('Quantile');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Measured home distributions versus synthetic home summaries');finish(fig,'real_vs_synthetic_distributions')
    core=['heart_rate','systolic_bp','diastolic_bp','spo2','weight','temperature'];real=np.eye(6);synth=np.eye(6)
    for i,a in enumerate(core):
        for j,b in enumerate(core):
            if j<=i:continue
            key=a+'__'+b if a+'__'+b in ground['perheart']['daily_correlations'] else b+'__'+a;real[i,j]=real[j,i]=ground['perheart']['daily_correlations'].get(key) or 0
            x=daily[:,:,SIGNALS.index(a)].ravel();y=daily[:,:,SIGNALS.index(b)].ravel();good=np.isfinite(x)&np.isfinite(y);synth[i,j]=synth[j,i]=np.corrcoef(x[good],y[good])[0,1]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,m,title in zip(axes,[real,synth],['PerHeart observed daily (exploratory)','Synthetic observed daily']):
        image=ax.imshow(m,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(6),core,rotation=70,fontsize=8);ax.set_yticks(range(6),core,fontsize=8);ax.set_title(title)
    color_axis=fig.add_axes([.91,.33,.02,.50]);fig.colorbar(image,cax=color_axis);finish(fig,'correlations')
    fig,ax=plt.subplots(figsize=(11,5));image=ax.imshow(missing[:200],aspect='auto',vmin=0,vmax=1,cmap='magma');ax.set_xticks(range(len(SIGNALS)),SIGNALS,rotation=65);ax.set_ylabel('Virtual patient index (no identifier)');ax.set_title('Missing/invalid fraction at scheduled opportunities');fig.colorbar(image,ax=ax,label='Fraction');finish(fig,'missingness')
    fig,ax=plt.subplots(figsize=(10,4));events=[r['event_time_hours']/24 for r in truth if r['event_time_hours'] is not None];ax.hist(events,bins=20,edgecolor='white');ax.set_xlabel('Event day');ax.set_ylabel('Virtual patients');ax.set_title('Variable latent-generated event timing');finish(fig,'event_timing')
    fig,axes=plt.subplots(2,2,figsize=(11,7));lags=np.arange(-7,1)
    for ax,signal in zip(axes.flat,['heart_rate','systolic_bp','spo2','weight']):
        j=SIGNALS.index(signal);curves=[]
        for r in truth:
            if r['event_time_hours'] is None:continue
            day=int(r['event_time_hours']//24);index=day+lags
            if min(index)<0 or max(index)>=manifest['days']:continue
            series=daily[lookup[r['patient_id']],:,j]
            if not np.isfinite(series[:7]).any():continue
            ref=np.nanmedian(series[:7]);curves.append(series[index]-ref)
        curves=np.asarray(curves);lo,med,hi=np.nanquantile(curves,[.1,.5,.9],axis=0);ax.fill_between(lags,lo,hi,alpha=.2);ax.plot(lags,med);ax.set_title(signal.replace('_',' '));ax.set_xlabel('Days before simulated event');ax.set_ylabel('Change from first-week median');ax.grid(alpha=.2)
    fig.suptitle('P10–P90 observable trajectory diversity before events');finish(fig,'pre_event_trajectories')
    private_json(figures/'manifest.json',{'figures':saved,'evidence':'SYNTHETIC POC','evidence_label':'SYNTHETIC POC VALIDATED','scope':'private synthetic trajectories/aggregate reference comparisons; no clinical claim'})
    print({'private_scientific_figures_created':len(saved)},flush=True)

if __name__=='__main__':plot()
