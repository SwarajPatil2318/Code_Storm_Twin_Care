"""Additional unlabelled acute/surgical dynamics; never clinical test access.

Keeps these aggregate references separate from frozen generation inputs. Adjacent
window bins and within-window MAP declines are descriptive observations, not
new event labels or a learned HF deterioration mechanism.
"""
import json
from pathlib import Path
import numpy as np
from src.synthetic.grounding import summary,correlation
from src.data.io import file_hash
from src.vitaldb_feasibility.source import private_json


def run():
    result={'evidence_label':'REAL DATA EXPLORATORY','no_event_labels_opened':True,'spent_test_clinical_access':False,'not_used_to_fit_hf_hazard':True,'sources':{}}
    names=['map','sbp','dbp','hr','spo2','rr']
    for dataset,filename,minutes in [('eicu_development','eicu_development.npz',5),('vitaldb','vitaldb_60.npz',5)]:
        path=Path('data/results/phase45b')/filename
        with np.load(path,allow_pickle=False) as z:a=z['sequence'][:,:,:6]
        delta=a[:,1:]-a[:,:-1];first=a[:,:3];last=a[:,-3:]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);change=np.nanmedian(last,axis=1)-np.nanmedian(first,axis=1)
        map_declining=np.isfinite(change[:,0])&(change[:,0]<=-10)
        result['sources'][dataset]={'path':str(path),'sha256':file_hash(path),'windows':len(a),'window_bins':a.shape[1],'bin_minutes':minutes,
          'adjacent_change':{s:summary(delta[:,:,j]) for j,s in enumerate(names)},
          'adjacent_lag_correlation':{s:correlation(a[:,:-1,j].ravel(),a[:,1:,j].ravel()) for j,s in enumerate(names)},
          'within_window_change_correlations':{names[i]+'__'+names[j]:correlation(change[:,i],change[:,j]) for i in range(6) for j in range(i+1,6)},
          'descriptive_map_decline_windows':int(map_declining.sum()),'other_changes_in_map_decline_windows':{s:summary(change[map_declining,j]) for j,s in enumerate(names)},
          'caveat':'Overlapping windows are dependent. MAP decline is a descriptive subset, not an outcome. ICU/intraoperative physiology cannot calibrate 72h home HF events.'}
    private_json('data/synthetic_grounding/reference_dynamics.json',result);print({'acute_surgical_reference_dynamics_complete':True,'clinical_test_access':False},flush=True)
    return result

if __name__=='__main__':run()
