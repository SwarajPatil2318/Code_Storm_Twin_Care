# GitHub source packaging

The active workspace contains private datasets, patient state, credentials, generated evidence and historical experiments. Do not upload the workspace as a ZIP. The prepared GitHub folder is produced by `tools/github_release.py` from an explicit source allowlist.

```bash
python tools/github_release.py --destination release/TwinCare --archive release/TwinCare-GitHub.zip
```

The destination must be new; existing releases are never silently overwritten. The exporter includes core Python source, necessary UI directories, React source and product illustrations, configuration, dependency pins/lockfile, setup tools, focused release tests and these current documents. It leaves the original research workspace and all private files in place.

Excluded: `AGENTS.md`, `.env`, `.venv`, `node_modules`, compiled bundles, `.git` history, assistant/cloud configuration, raw cohorts, patient databases, uploads, photos, all `artifacts/`, experiment caches, serialized models, generated records, logs, private screenshots, submission-video assets and old phase-specific QA/finalization scripts. Historical documentation remains in the working project; the lean package uses the reviewed current setup/architecture documents.

## Private companion bundle

`data/release/twincare-runtime-private.zip` is separate from the source ZIP. It contains private frozen models and selected runtime caches/contracts. The repository packaging policy requires model artifacts and record-level data to remain out of Git. The private bundle enables the full local app after restoration, without publishing those assets or the raw research archive. Keep it local or distribute privately only to authorized collaborators. SHA-256 checks are integrity checks, not authentication of an unknown sender.

## Publishing

The task prepares local files; it does not create a GitHub repository or push anything. In the prepared folder:

```bash
git init -b main
git add .
git status --short
# Review the staged files; configure your own remote and commit when ready.
```

Even after local runtime restoration and frontend installation, Git must ignore `data/`, `artifacts/`, `.env`, dependency directories and `frontend/dist/`. Ignore rules do not remove already tracked files from an existing repository. Check `git ls-files` before publishing. This workspace arrived without usable Git metadata, so no pre-existing remote/history is assumed or rewritten.

The original project code is released under the team-approved MIT license in `LICENSE`. Runtime bundles, datasets, media and third-party components retain their own rights and restrictions; the project license does not grant additional distribution rights for them.

## Local release verification

The prepared copy was checked with a fresh Python 3.12 virtual environment and a fresh `npm ci` from its own lockfile. All 14 release checks passed with the private bundle restored, including genuine cold/warm forecasts, personal baseline availability, risk execution, causal input cutoffs, owner isolation, manual meal persistence, integrity rejection and archive boundaries. Frontend build, source timing, chart and presentation tests passed. Browser smoke passed the patient workspace plus both twin routes, and desktop/tablet/mobile patient-shell widths. These are integration checks, not clinical validation or a claim of exhaustive browser coverage.
