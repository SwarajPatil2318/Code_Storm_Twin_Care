# Setup and troubleshooting

Run commands from the checkout root. Python 3.12 and Node 22 are the checked baseline. Install the runtime pins in `requirements.txt`, use `npm ci --prefix frontend`, and build the frontend. The CPU PyTorch extra index is needed because the frozen model's Python module imports PyTorch, even when the chosen forecaster is XGBoost. Raw cohorts, research training dependencies and old experiment reports are not needed for the packaged runtime.

## Private runtime assets

The owner creates a bundle from the existing verified project:

```bash
.venv/bin/python tools/runtime_bundle.py create --output data/release/twincare-runtime-private.zip
```

Optional internal scenario playback can be included with `--with-judge`. The bundle contains the required frozen inference objects, deployment contracts, alert policy, aggregate evidence metadata, and preserved replay caches. It does not copy the raw research archive, prediction exports, uploads, meal photos, patient database, credentials or dependency installations. Replay records and serialized models still remain private and must never be added to Git.

On the receiving checkout:

```bash
python tools/runtime_bundle.py restore /private/path/twincare-runtime-private.zip
python tools/doctor.py
```

Only restore a bundle from a trusted owner: the existing Python pickle models can execute code when loaded. Internal SHA-256 checks detect changes relative to the supplied manifest; they are not a signature or proof of who created it. Restoration validates the complete archive, rejects extra/traversal/duplicate entries, verifies hashes, and refuses different existing asset files.

The launcher explicitly selects deployment verification through `TWINCARE_RUNTIME_MANIFEST`. The original research path remains selected when this variable is absent. Research source/config/model selection and sealed holdout evaluation are not retrained, resealed or repeated by packaging.

## Services and state

`python tools/run.py` starts three loopback processes. The patient shell proxies the twin dashboards and assets through the existing cardiovascular origin. The cardiovascular origin proxies metabolic API requests to the metabolic service. Upstream ports are passed explicitly, allowing another local instance on different ports.

Use `--state-dir data/another-local-instance` to isolate a new patient database, uploads and meal images. Existing patient state is not distributed. Optional `--with-judge` starts the isolated internal scenario launcher on port 8116; its selected replay inputs must be present in the private bundle. This optional tool is not required for normal new-patient onboarding.

## Common problems

| Symptom | Check |
|---|---|
| Preflight fails | Run `python tools/doctor.py`; install pinned dependencies, restore the private bundle, and build the frontend. |
| Missing private runtime manifest | A public clone contains source only. Obtain and restore the trusted private bundle. |
| Frozen model/config mismatch | Restore the matching source and bundle. Do not edit hashes or disable verification. |
| Occupied port | Stop your own older session or pass distinct alternate ports. The launcher does not stop other processes. |
| Watch missing / `/assets/product-wearable.js` unavailable | Build `frontend/`; start all services through the launcher. |
| Forecast unavailable | Confirm clinical records and supply complete, recent, declared 15-minute CGM history. A watch alone does not provide glucose. |
| Personal baseline learning | Load prior synthetic history for testing, or wait for the existing historical-support rules to be met. |
| Stale historical replay | Its original observation times are preserved. Syncing does not relabel old values as new measurements. Use live simulation for current input testing. |
| Photo analysis unavailable | Configure the private root `.env`, use an account-accessible Gemini model, restart, or use manual entry. |
| Scanned PDF has no text | Automatic OCR is not implemented. Use text-based PDF, supported JSON/FHIR, or manual correction. |
| Playwright executable missing | Install its Chromium browser, or set `CHROME_PATH` to a compatible local browser for the release smoke test. |

The local HTTP prototype has no production authentication, HTTPS termination or commercial device integration. Keep it bound to loopback; internet deployment requires additional engineering. The release does not claim validation for diagnosis or treatment.
