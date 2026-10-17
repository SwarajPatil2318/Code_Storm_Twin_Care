"""Received seven-day trajectories, sparse clocks and personal deviations."""
import numpy as np
from src.synthetic.config import SIGNALS
from src.twin.protocol import ABS_STATS,PERSONAL_STATS,ABS_FEATURES,PERSONAL_FEATURES,EHR_COLUMNS,PRIMARY_SIGNALS
from src.twin.baseline import PersonalBaseline


def trend(t,v):
    if len(v)<2:return np.nan
    x=(t-t.mean())/24;return float(np.dot(x,v-v.mean())/np.dot(x,x)) if np.dot(x,x)>0 else np.nan

def representation(context,stream,landmark,baseline,config):
    value=stream['value'];hours=np.arange(len(value));history=(hours>landmark-168)&(hours<=landmark)
    received=(stream['available_hour']<=landmark)&~stream['missingness_mask']&np.isfinite(value)
    absolute=[];personal=[];recent=[];recent_counts=[];baseline_expected=[];earlier=[];persistence=[];sequence=np.full((28,len(PRIMARY_SIGNALS),4),np.nan,np.float32)
    for k,s in enumerate(PRIMARY_SIGNALS):
        j=SIGNALS.index(s);good=history&received[:,j];t=hours[good];v=value[good,j].astype(float);scheduled=stream['scheduled'][:,j]&history
        recent_good=t>landmark-48;rv=v[recent_good];rt=t[recent_good];prior=v[~recent_good];count=len(v)
        stats={'mean':float(v.mean()) if count else np.nan,'sd':float(v.std()) if count else np.nan,'last':float(v[-1]) if count else np.nan,'slope_per_day':trend(t,v),
          'recent_change':float(rv.mean()-prior.mean()) if len(rv) and len(prior) else np.nan,'coverage':count/max(int(scheduled.sum()),1),
          'staleness_hours':min(168,float(landmark-t[-1])) if count else 168,'recent_mean':float(rv.mean()) if len(rv) else np.nan,'recent_count':len(rv),
          'suspect_fraction':float(np.mean(stream['quality_status'][good,j]==2)) if count else np.nan}
        expected=baseline['clock'][j,(rt%24)//6] if len(rt) else np.array([]);reference=float(np.nanmean(expected)) if len(expected) and np.isfinite(expected).any() else baseline['median'][j]
        residual=rv-expected;z=residual/baseline['sd'][j];persist=float(np.mean(abs(z)>config['tdi']['deviation_z'])) if len(z) and np.isfinite(z).any() else np.nan
        deviation=float(np.nanmean(residual)) if len(residual) and np.isfinite(residual).any() else np.nan
        clock=baseline['clock'][j];ps={'mean':baseline['mean'][j],'median':baseline['median'][j],'sd':baseline['sd'][j],'q10':baseline['q10'][j],'q90':baseline['q90'][j],
          'count':baseline['count'][j],'confidence':baseline['confidence'][j],'frozen':int(baseline['frozen']),'recent_deviation':deviation,'recent_z':deviation/baseline['sd'][j],
          'persistence':persist,'circadian_recent_reference':reference,'day_reference':float(np.nanmean(clock[1:3])) if np.isfinite(clock[1:3]).any() else np.nan,
          'night_reference':float(np.nanmean(clock[[0,3]])) if np.isfinite(clock[[0,3]]).any() else np.nan}
        absolute.extend(stats[key] for key in ABS_STATS);personal.extend(ps[key] for key in PERSONAL_STATS);recent.append(stats['recent_mean']);recent_counts.append(len(rv));baseline_expected.append(reference);persistence.append(persist)
        old=value[received[:,j]&(hours>landmark-96)&(hours<=landmark-48),j];earlier.append(float(old.mean()) if len(old) else np.nan)
        for bin_id,end in enumerate(range(landmark-162,landmark+1,6)):
            bin_good=(t>end-6)&(t<=end);bv=v[bin_good];bt=t[bin_good]
            sequence[bin_id,k,0]=float(np.median(bv)) if len(bv) else np.nan;sequence[bin_id,k,1]=float(bool(len(bv)))
            past=t[t<=end];sequence[bin_id,k,2]=min(1,float(end-past[-1])/168) if len(past) else 1
            sequence[bin_id,k,3]=float(np.mean(stream['quality_status'][bt,j]==2)) if len(bt) else 0
    ehr=np.array([np.nan if context[n[5:]] is None else context[n[5:]] for n in EHR_COLUMNS],np.float32)
    return {'absolute':np.array(absolute,np.float32),'personal':np.array(personal,np.float32),'ehr':ehr,'sequence':sequence,
      'recent':np.array(recent,np.float32),'recent_count':np.array(recent_counts,np.int16),'reference':np.array(baseline_expected,np.float32),'prior_recent':np.array(earlier,np.float32),
      'scale':baseline['sd'][[SIGNALS.index(s) for s in PRIMARY_SIGNALS]].astype(np.float32),'persistence':np.array(persistence,np.float32),
      'baseline_count':baseline['count'][[SIGNALS.index(s) for s in PRIMARY_SIGNALS]],'confidence':baseline['confidence'][[SIGNALS.index(s) for s in PRIMARY_SIGNALS]].astype(np.float32),'baseline_frozen':baseline['frozen']}

def patient_history(context,stream,config):
    baseline=PersonalBaseline(config);start,end,stride=config['landmark_hours']
    for hour in range(start,end+1,stride):
        ref=baseline.update(stream,hour);yield hour,representation(context,stream,hour,ref,config)
