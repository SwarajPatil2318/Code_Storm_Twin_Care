"""Verify private Phase45B artifacts without opening old test clinical contents."""
import hashlib,json,platform,subprocess,tempfile,shutil
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from src.ssl_physiology.protocol import ROOT,PROTOCOL
from src.ssl_physiology.transfer import references
from src.vitaldb_feasibility.source import private_json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    prior=json.loads((ROOT/'prior_preservation.json').read_text());public_changes=[];private_changes=[]
    for name,expected in prior['public_sha256'].items():
        path=Path(name)
        if not path.exists() or digest(path)!=expected:public_changes.append(name)
    for name,expected in prior['private_file_metadata'].items():
        path=Path(name)
        if not path.exists():private_changes.append(name);continue
        stat=path.stat()
        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:private_changes.append(name)
    if public_changes or private_changes:raise ValueError('Previously completed phase files changed')
    access=json.loads((ROOT/'development_access.json').read_text());rows=json.loads((ROOT/'eicu_development_rows.json').read_text());allowed=set(access['allowed_patients'])
    assert {r['patient_id'] for r in rows}<=allowed and len(allowed)==210
    assert all(r['original_role'] in {'train','validation'} for r in rows)
    assert all(r['audit']['max_physiology_offset']<=r['t0']-5 for r in rows)
    assert all((r['audit']['maximum_available_clinical_offset'] or 0)<=r['t0'] for r in rows)
    with np.load(ROOT/'eicu_development.npz') as z:seq=z['sequence']
    selected=json.loads((ROOT/'ssl_results.json').read_text())['selected_task'];bundles={m:joblib.load(ROOT/(selected+'__'+m+'.joblib')) for m in ['all','no_map','no_bp']}
    with np.load(ROOT/'development_latents.npz') as z:latent={k:z[k] for k in z.files}
    max_latent_error=0.
    for mode,bundle in bundles.items():
        assert not set(bundle['fit_patients'])&set(bundle['validation_patients'])
        assert set(bundle['normalizer'].fit_patients)==set(bundle['fit_patients'])
        prediction=bundle['encoder'].embedding(bundle['normalizer'].transform(seq,mode))
        max_latent_error=max(max_latent_error,float(np.max(abs(prediction-latent[mode]))))
    oof=json.loads((ROOT/'development_oof.json').read_text());names=sorted(oof[0]['raw']);maximum_error=0.;checked=0
    expected_early=bundles['all']['encoder'].params
    for seed in [2026,2031]:
        for fold in range(5):
            out=sorted([r for r in oof if r['seed']==seed and r['fold']==fold],key=lambda r:r['row_index']);indices=np.array([r['row_index'] for r in out]);held=[rows[i] for i in indices]
            for name in names:
                artifact=joblib.load(ROOT/'fold_models'/f'seed{seed}_fold{fold}'/(name+'.joblib'));model=artifact['model'];fit=set(artifact['train_patients']);held_ids=set(artifact['held_patients'])
                assert fit|held_ids==allowed and not fit&held_ids
                assert {r['patient_id'] for r in held}==held_ids
                assert set(artifact['calibrator'].fit_patients)==fit
                if name in ['map_only','bp_only','handcrafted','handcrafted_ehr','map_logistic']:p=model.predict(held)
                elif name in ['latent_finetuned','latent_finetuned_ehr','supervised_scratch']:
                    head,ehr=model;p=head.predict(seq[indices],ehr.transform(held) if ehr is not None else None)
                    assert head.__class__.__module__=='src.ssl_physiology.transfer' and set(head.fit_patients)==fit
                    if name!='supervised_scratch':
                        for key in ['w0','b0','w1','b1']:assert np.array_equal(head.encoder.params[key],expected_early[key])
                    else:assert set(head.norm.fit_patients)==fit
                else:
                    mode={'latent_no_map':'no_map','latent_no_bp':'no_bp','random_encoder':'random'}.get(name,'all')
                    x=latent[mode][indices]
                    if name.startswith('latent_delta'):x=latent['delta'][indices]
                    elif name.startswith('latent_recent'):x=np.concatenate([latent['recent'][indices],latent['reference_available'][indices,None]],axis=1)
                    if name.endswith('_ehr') or name=='latent_plus_map':
                        head,ehr=model;x=np.concatenate([x,ehr.transform(held)],axis=1);p=head.predict(x)
                    else:p=model.predict(x)
                difference=float(np.max(abs(p-np.array([r['raw'][name] for r in out]))));maximum_error=max(maximum_error,difference)
                cp=artifact['calibrator'].predict(p);maximum_error=max(maximum_error,float(np.max(abs(cp-np.array([r['calibrated'][name] for r in out])))))
                checked+=1
    assert maximum_error<1e-12 and max_latent_error<1e-12
    for path in ROOT.rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
    ROOT.chmod(0o700)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    with tempfile.TemporaryDirectory() as temp:
        shutil.copyfile('.gitignore',Path(temp)/'.gitignore');subprocess.run(['git','init','-q',temp],check=True)
        result=subprocess.run(['git','-C',temp,'check-ignore','--stdin'],input='\n'.join(str(p) for p in files)+'\n',text=True,capture_output=True,check=True)
        assert set(result.stdout.splitlines())==set(str(p) for p in files)
    report={'previous_public_files_sha256_unchanged':len(prior['public_sha256']),
      'previous_private_files_size_mtime_unchanged':len(prior['private_file_metadata']),
      'old_private_verification':'metadata only, not clinical content hashes; unchanged modification metadata does not prove absence of reads',
      'spent_test_access_boundary':'identity-only routing through mixed source bytes; rejected clinical JSON/CSV fields not deserialized; no original test predictions or whole-cohort arrays opened',
      'clinical_test_rows_decoded':access['test_clinical_rows_decoded'],'test_labels_decoded':access['test_labels_decoded'],
      'test_prediction_files_opened':access['test_predictions_files_opened'],
      'vitaldb_subject_split_disjoint':True,'eicu_patient_folds_disjoint':True,'early_pretrained_layers_unchanged':True,
      'saved_fold_models_reproduced':checked,'max_oof_reload_error':maximum_error,'max_latent_reload_error':max_latent_error,
      'private_files_git_excluded':len(files),'permissions':'Phase45B directories700/files600',
      'runtime':{'python':platform.python_version(),'numpy':np.__version__},
      'realized_protocol_notes':['Fixed20epochs fine-tuning used as declared; planned inner early selection was not implemented, and no outer-fold early stopping occurred.',
        'Additional matched MAP-logistic, recent-only and recent-only+EHR heads control classifier capacity and personalization context.',
        'CLI serialization corrected to importable class names; existing180 fold artifacts reserialized without refitting and all OOF predictions verified.'],
      'independent_validation_still_required':True}
    private_json(ROOT/'verification.json',report)
    source=sorted(Path('src/ssl_physiology').glob('*.py'))+[Path('tests/test_ssl_physiology.py')]
    manifest={'protocol':PROTOCOL,'source_sha256':{str(p):digest(p) for p in source},
      'private_artifact_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files if p.name not in ['reproducibility.json','verification.json']},
      'commands':['.venv/bin/python -m src.ssl_physiology.cohort','.venv/bin/python -m src.ssl_physiology.pretrain','.venv/bin/python -m src.ssl_physiology.transfer','.venv/bin/python -m src.ssl_physiology.analysis','.venv/bin/python -m src.ssl_physiology.verify','.venv/bin/python -m unittest discover -s tests -q'],
      'reexecution':'Always use the same development allowlist and frozen config; never call legacy whole-cohort training/evaluation readers; fresh runs regenerate private artifacts only.',
      'determinism':'Fixed seeds, one BLAS thread, deterministic grouped splits; saved pipeline predictions reproduced to tolerance1e-12. Rerun audit hashes can differ when metadata changes.'}
    private_json(ROOT/'reproducibility.json',manifest)
    print({k:v for k,v in report.items() if k in ['previous_public_files_sha256_unchanged','previous_private_files_size_mtime_unchanged','saved_fold_models_reproduced','max_oof_reload_error','private_files_git_excluded']},flush=True)
    return report

if __name__=='__main__':
    with threadpool_limits(limits=1):verify()
