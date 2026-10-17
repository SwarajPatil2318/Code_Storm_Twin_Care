"""Check a checkout before starting TwinCare; never print credentials or records."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPENDENCIES = {'numpy':'2.2.6','pandas':'2.2.3','scipy':'1.15.3','scikit-learn':'1.6.1',
                'xgboost-cpu':'3.0.5','torch':'2.6.0+cpu','Pillow':'12.3.0','pypdf':'6.19.0'}


def check(source_only=False):
    checks = {'python_3_12': sys.version_info[:2] == (3, 12),
              'node_available': shutil.which('node') is not None,
              'npm_available': shutil.which('npm') is not None}
    if not source_only:
        checks['runtime_dependencies'] = True
        for name, version in DEPENDENCIES.items():
            try:
                good = importlib.metadata.version(name) == version
            except importlib.metadata.PackageNotFoundError:
                good = False
            checks['dependency_' + name] = good
            checks['runtime_dependencies'] &= good
        checks['frontend_built'] = (ROOT/'frontend/dist/index.html').is_file() and (ROOT/'frontend/dist/assets/product-wearable.js').is_file()
        try:
            from src.runtime_assets import verify
            verify()
            from src.metabolic_d3.reference import register
            original = os.environ.get('TWINCARE_RUNTIME_MANIFEST')
            os.environ['TWINCARE_RUNTIME_MANIFEST'] = str(ROOT/'data/runtime/manifest.json')
            try:
                register()
            finally:
                if original is None: os.environ.pop('TWINCARE_RUNTIME_MANIFEST', None)
                else: os.environ['TWINCARE_RUNTIME_MANIFEST'] = original
            checks['private_runtime_verified'] = True
        except (OSError, ValueError, KeyError, RuntimeError, ImportError):
            checks['private_runtime_verified'] = False
    return {'ready': all(checks.values()), 'checks': checks}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--source-only', action='store_true');a=p.parse_args()
    result=check(a.source_only);print(json.dumps(result, indent=2))
    if not result['ready']:
        print('See README: install requirements, restore the trusted private runtime bundle, and build the frontend.', file=sys.stderr)
        raise SystemExit(1)
