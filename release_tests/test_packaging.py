import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.runtime_assets import ASSETS, MANIFEST, safe_asset, sha, verify, portable_reference
from tools.github_release import selected, audit
from tools.runtime_bundle import restore
from tools.run import load_environment


class SourceBoundaryTests(unittest.TestCase):
    def test_public_source_and_required_assets_selected(self):
        for name in ['src/metabolic_d2/models.py','patient_experience/server.py','experience_ui/app.js',
                     'frontend/src/platform/Wearable.tsx','frontend/public/assets/anatomical-heart.png',
                     '.env.example','requirements.txt','frontend/package-lock.json','docs/SETUP.md']:
            self.assertTrue(selected(name),name)

    def test_private_and_generated_paths_excluded(self):
        for name in ['.env','data/private_checks/result.json','data/patient_domain/patients.sqlite3',
                     'artifacts/productization_p6/model_contract.json','Datasets/cohort.zip',
                     'frontend/node_modules/test/index.js','frontend/dist/assets/app.js',
                     'src/models/weights.pkl','src/__pycache__/model.pyc','.aws/credentials',
                     'scripts/productization_p9_discovery.py','docs/FINAL_WORKED_PREDICTION_EXAMPLE.md']:
            self.assertFalse(selected(name),name)

    def test_credential_audit_rejects_secret_without_returning_it(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);key='AIza'+'a'*35;(root/'README.md').write_text(key)
            result=audit(root)
            self.assertFalse(result['passed']);self.assertNotIn(key,json.dumps(result))

    def test_deployment_is_explicit_opt_in(self):
        with patch.dict(os.environ,{},clear=True):self.assertIsNone(portable_reference())


class RuntimeBundleTests(unittest.TestCase):
    def make_archive(self,folder,extra=None,tamper=False):
        root=folder/'source';root.mkdir()
        for name in ASSETS:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'private integration fixture')
        manifest={'schema_version':'twincare-runtime/1','assets':{name:sha(root/name)for name in ASSETS},'reference':{}}
        archive=folder/'assets.zip'
        with zipfile.ZipFile(archive,'w')as z:
            for name in ASSETS:z.writestr(name,b'tampered'if tamper and name==ASSETS[0]else(root/name).read_bytes())
            z.writestr(MANIFEST,json.dumps(manifest))
            if extra:z.writestr(extra,b'not allowed')
        return archive

    def test_round_trip_and_ignored_destination(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);archive=self.make_archive(root);dest=root/'checkout'
            restore(archive,dest);self.assertEqual(len(verify(dest)['assets']),len(ASSETS))
            self.assertEqual((dest/ASSETS[0]).stat().st_mode&0o777,0o600)
            restore(archive,dest)  # Idempotent only for byte-identical assets.

    def test_hash_tampering_refused_before_restore(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);archive=self.make_archive(root,tamper=True);dest=root/'checkout'
            with self.assertRaises(ValueError):restore(archive,dest)
            self.assertFalse(dest.exists())

    def test_extra_or_traversal_entries_refused(self):
        for extra in ['../../escape.txt','.env','data/patient_domain/patients.sqlite3']:
            with self.subTest(path=extra),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);archive=self.make_archive(root,extra=extra)
                with self.assertRaises(ValueError):restore(archive,root/'checkout')

    def test_different_existing_asset_is_never_overwritten(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);archive=self.make_archive(root);dest=root/'checkout';p=dest/ASSETS[-1];p.parent.mkdir(parents=True);p.write_bytes(b'local unchanged')
            with self.assertRaises(FileExistsError):restore(archive,dest)
            self.assertEqual(p.read_bytes(),b'local unchanged');self.assertFalse((dest/MANIFEST).exists())

    def test_symlink_outside_root_refused(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'checkout';target=root/ASSETS[0];target.parent.mkdir(parents=True);outside=Path(tmp)/'outside';outside.write_bytes(b'local')
            target.symlink_to(outside)
            with self.assertRaises(ValueError):safe_asset(root,ASSETS[0])


class EnvironmentTests(unittest.TestCase):
    def test_dotenv_is_data_and_never_executes_shell(self):
        with tempfile.TemporaryDirectory()as tmp,patch.dict(os.environ,{},clear=True):
            path=Path(tmp)/'.env';sentinel=Path(tmp)/'executed'
            path.write_text("MEAL_ANALYSIS_PROVIDER=gemini\nGEMINI_API_KEY='example-only'\nTWINCARE_MEAL_VISION_MODEL=$(touch "+str(sentinel)+")\nUNRELATED=ignored\n")
            load_environment(path)
            self.assertEqual(os.environ['GEMINI_API_KEY'],'example-only');self.assertNotIn('UNRELATED',os.environ);self.assertFalse(sentinel.exists())

    def test_existing_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory()as tmp,patch.dict(os.environ,{'MEAL_ANALYSIS_PROVIDER':'disabled'},clear=True):
            p=Path(tmp)/'.env';p.write_text('MEAL_ANALYSIS_PROVIDER=gemini\n');load_environment(p)
            self.assertEqual(os.environ['MEAL_ANALYSIS_PROVIDER'],'disabled')

if __name__=='__main__':unittest.main()
