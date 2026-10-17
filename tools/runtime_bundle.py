"""Create/restore a PRIVATE allowlisted runtime bundle. Never upload it to GitHub."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runtime_assets import ASSETS, OPTIONAL_ASSETS, MANIFEST, safe_asset, sha, verify


def create(output, with_judge=False):
    os.chdir(ROOT)
    # Original verification reads hashes only, never reopens holdout trajectories.
    from src.metabolic_d3.reference import register
    reference = register()
    names = (*ASSETS, *OPTIONAL_ASSETS) if with_judge else ASSETS
    hashes = {name: sha(safe_asset(ROOT, name)) for name in names}
    manifest = {'schema_version': 'twincare-runtime/1', 'private': True,
                'distribution': 'Local trusted deployment only; excluded from Git.',
                'assets': hashes, 'reference': reference}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing runtime bundle')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as raw, zipfile.ZipFile(raw, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in names:
                z.write(ROOT / name, name)
            z.writestr(MANIFEST, json.dumps(manifest, indent=2))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print('Private runtime bundle created: ' + str(output))


def restore(archive, destination=ROOT):
    root = Path(destination).resolve()
    # Validate the complete archive in a temporary directory before changing the checkout.
    with tempfile.TemporaryDirectory(prefix='twincare-assets-') as td:
        stage = Path(td)
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)) or MANIFEST not in names:
                raise ValueError('Invalid runtime archive inventory')
            allowed = set((*ASSETS, *OPTIONAL_ASSETS, MANIFEST))
            if set(names) - allowed or sum(i.file_size for i in z.infolist()) > 100 * 1024 * 1024:
                raise ValueError('Archive outside runtime allowlist or size limit')
            for name in names:
                if name != MANIFEST:
                    safe_asset(stage, name)
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
                target.chmod(0o600)
        value = verify(stage)
        if set(names) != set(value['assets']) | {MANIFEST}:
            raise ValueError('Archive contains unverified files')
        # Refuse to overwrite local assets with different bytes.
        for name in names:
            target = root / name
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError('Destination leaves checkout')
            if target.exists() and sha(target) != sha(stage / name):
                raise FileExistsError('Different local asset exists: ' + name)
        for name in names:
            target = root / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write((stage / name).read_bytes())
            target.chmod(0o600)
    print('Runtime bundle verified and restored; all runtime assets remain ignored by Git.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    make = sub.add_parser('create');make.add_argument('--output', default='data/release/twincare-runtime-private.zip');make.add_argument('--with-judge', action='store_true')
    put = sub.add_parser('restore');put.add_argument('archive');put.add_argument('--destination', default=str(ROOT))
    args = p.parse_args()
    try:
        create(args.output, args.with_judge) if args.action == 'create' else restore(args.archive, args.destination)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError):
        print('Runtime bundle operation failed. Check the trusted bundle, file permissions, and documented prerequisites; existing differing assets are never overwritten.', file=sys.stderr)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
