"""Audit interpolation evidence without claiming an inverse reconstructs measured samples."""
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.metabolic.audit import times
from src.metabolic.windows import MINUTE
from .protocol import ROOT,write_json,digest,utc

SOURCE=Path('Datasets/CGMacros/CGMacros_dateshifted365.zip')

def run():
    rows=[];totals={name:{'positive_grid_values':0,'contiguous_triples':0,'linear_triples':0,
        'slope_change_candidates':0,'noninteger_values':0,'candidate_minute_phase_counts':[0]*period}
        for name,period in [('Dexcom',5),('Libre',15)]}
    with zipfile.ZipFile(SOURCE) as archive:
        names=archive.namelist()
        bio=pd.read_csv(archive.open(next(n for n in names if Path(n).name=='bio.csv')))
        ids=set(bio.loc[pd.to_numeric(bio['A1c PDL (Lab)'],errors='coerce')>6.4,'subject'].astype(int))
        original_exports=[n for n in names if any(v in n.lower() for v in ['raw_export','original_cgm','measurement_mask','observed_mask'])]
        for name in names:
            if not name.endswith('.csv') or 'CGMacros-' not in Path(name).stem:continue
            sid=int(Path(name).stem.split('-')[-1])
            if sid not in ids:continue
            frame=pd.read_csv(archive.open(name));t=np.asarray(times(frame['Timestamp']),dtype='int64')
            for device,field,period in [('Dexcom','Dexcom GL',5),('Libre','Libre GL',15)]:
                v=pd.to_numeric(frame[field],errors='coerce').to_numpy(float)
                good=np.isfinite(v)&(v>0)&(t!=np.iinfo(np.int64).min)
                adjacent=(np.diff(t)==MINUTE)
                triples=good[:-2]&good[1:-1]&good[2:]&adjacent[:-1]&adjacent[1:]
                linear=np.isclose(v[2:]-2*v[1:-1]+v[:-2],0,atol=1e-8,rtol=0)&triples
                vertices=(triples&~linear)
                phases=((t[1:-1]//MINUTE)%period)[vertices].astype(int)
                phase_counts=np.bincount(phases,minlength=period)
                row={'participant_id':'CGMacros:'+str(sid),'device':device,'positive_grid_values':int(good.sum()),
                    'contiguous_triples':int(triples.sum()),'linear_triples':int(linear.sum()),
                    'slope_change_candidates':int(vertices.sum()),'noninteger_values':int((good&~np.isclose(v,np.round(v),atol=1e-8,rtol=0)).sum()),
                    'candidate_minute_phase_counts':phase_counts.tolist()}
                rows.append(row)
                for key in totals[device]:
                    if key=='candidate_minute_phase_counts':totals[device][key]=(np.array(totals[device][key])+phase_counts).tolist()
                    else:totals[device][key]+=row[key]
    private=ROOT/'cgmacros_provenance_private.json'
    write_json(private,{'per_participant':rows})
    report={'created_at_utc':utc(),'release_sha256':digest(SOURCE),'T2D_participants':len(ids),
        'population_basis':'screening HbA1c >6.4% per original publication; not independent diagnosis record',
        'devices':totals,'explicit_original_export_or_observation_mask_files':len(original_exports),
        'measured_samples_certified':0,'interpolated_samples_individually_certified':0,
        'per_sample_glucose_provenance':'unknown measured/interpolated membership; source publication confirms mixed linear-interpolated 1-minute grid',
        'derived_samples':'all rounding, downsampling and slope-change reconstructions are derived; none is relabelled measured',
        'reconstruction_result':'Not identifiable: collinear native measurements leave no slope break, unknown sampling phase/gaps and original observation mask prevent measured-only certification.',
        'external_set_status':'WITHHELD: original device exports or authoritative per-sample measurement mask required',
        'external_evaluation_spent':False,'private_detail_sha256':digest(private),
        'publication':'https://psi.engr.tamu.edu/wp-content/uploads/2025/09/anurag2025CGMacros.pdf',
        'release':'https://physionet.org/content/cgmacros/1.0.0/'}
    write_json(ROOT/'cgmacros_provenance.json',report)
    print(__import__('json').dumps({'CGMacros_T2D_participants':len(ids),'measured_only_certified':False,'external_evaluation_spent':False}),flush=True)
    return report

if __name__=='__main__':run()
