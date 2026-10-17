# TwinCare

**Personal Health Intelligence — a personalized healthcare digital twin research prototype.**

TwinCare combines a patient workspace, medical-record review, glucose forecasting, a personal expected glucose pattern, meal logging, a shared wearable view, and alerts. The Metabolic Twin supports newly created patients with received observations. The Cardiovascular Twin provides the preserved synthetic research demonstration.

The app runs locally. Wearable and CGM connections currently use explicit synthetic simulation or historical replay; commercial watch pairing is not implemented. Predictions and meal estimates are research outputs, not clinically validated care recommendations.

## What is included

- Python patient services and the React/Vite twin dashboards.
- The patient overview, record ingestion/review, source connections, live simulator, meals, alerts, and wearable UI.
- Frozen-model inference code with separate integrity verification for deployment.
- A Gemini-first environment template; manual meal logging works without an API key.
- Dependency pins, frontend lockfile, setup checks, sample-record generator, and focused release tests.

**The GitHub source package deliberately excludes credentials, patient databases, medical uploads, research datasets, generated records, and model weights.** Full model inference requires the trusted **private runtime bundle** described below. A source checkout alone cannot produce genuine model predictions. The owner can create this bundle from the verified local project without retraining or distributing raw cohorts.

## Quick start

Tested baseline: **Python 3.12**, **Node.js 22**, npm, and Linux. WSL2 is the suggested Windows route. Native Windows/macOS runtime installation has not been verified; the pinned CPU PyTorch wheel targets the tested environment.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
```

Obtain `twincare-runtime-private.zip` directly from the project owner through a private channel, then restore it locally:

```bash
python tools/runtime_bundle.py restore /absolute/path/to/twincare-runtime-private.zip
python tools/doctor.py
python tools/run.py
```

Open **http://127.0.0.1:8113**. The launcher also starts the existing cardiovascular service on 8097 and metabolic service on 8098. **Ctrl+C stops only the processes this launcher created.** It refuses occupied ports and does not terminate an existing project session.

For an alternative set of ports:

```bash
python tools/run.py --port 8213 --product-port 8297 --metabolic-port 8298
```

The root `.env` is optional and is read safely by this launcher. Environment variables already set in the shell take precedence. Starting service modules directly does not automatically load `.env`.

## Test a new patient

```bash
python tools/create_sample_records.py
```

1. Open **Patients → Create New Twin** and enter fictional test demographics.
2. Upload `data/sample_records/clinical_history.json` or the generated PDF. Review the extracted facts and explicitly confirm them. Use the JSON example for the most repeatable structured integration check.
3. Connect **Simulated Live CGM** and **Simulated Wearable**. Acknowledge the synthetic source. Choose **Load prior history** to test forecasting and the personal baseline immediately.
4. Start monitoring. The overview should open directly, with the current glucose, four forecast horizons, personal expected pattern when ready, wearable context, and trajectory.
5. Log a meal manually, or configure Gemini below. Review and confirm the meal proposal.
6. Scroll down while updates arrive; check that your scroll position and the dashboard remain stable. Pause/resume and review saved predictions.

Starting without prior history requires enough actual received 15-minute observations to fill the 120-minute input window. Personalization needs substantially more historical support, including the existing 50-hour eligibility rule and observation-count checks. A synthetic history load is a testing convenience, not a clinical validation result.

See [manual testing](docs/MANUAL_TESTING.md) for expected states, cold/warm behavior, alerts, and both twin dashboards.

## Optional Gemini meal-photo analysis

```bash
cp .env.example .env
```

Edit the private root `.env`:

```dotenv
MEAL_ANALYSIS_PROVIDER=gemini
GEMINI_API_KEY=
TWINCARE_MEAL_VISION_MODEL=
```

Enter your key and an image-capable Gemini model ID available to your account. Restart the launcher after changing these values. The key stays on the backend; do not put it in frontend variables or commit `.env`. Photos selected for analysis are sent to the configured provider after image preprocessing. The user must review and confirm the approximate food/nutrition proposal. Provider failures retain the manual-entry fallback.

For manual meal entry only, set `MEAL_ANALYSIS_PROVIDER=disabled`. Optional HTTP/OpenAI adapter settings remain documented in `.env.example`.

## Checks

```bash
python -m unittest discover -s release_tests -v
npm run test:sources --prefix frontend
npm run test:chart --prefix frontend
npm run test:presentation --prefix frontend
```

With the full local runtime running:

```bash
npm exec --prefix frontend -- playwright install chromium
TWINCARE_URL=http://127.0.0.1:8113 npm run qa --prefix frontend
```

The lean GitHub package maps `qa` to the release browser smoke test. Screenshots and runtime evidence are written under ignored `data/release/`. CI runs source tests and the frontend build/unit checks; private-model integration is checked locally after restoring the bundle.

## Project layout

```text
frontend/                 React twin dashboards, shared wearable, visual assets
experience_ui/            Patient overview and onboarding shell
patient_*/                Patient records, connections, meals, readiness, inference, alerts
live_simulation/          Explicit synthetic CGM/wearable input generation
src/                      Scientific implementations, frozen-model adapters, cached twin APIs
tools/                    Setup checks, launcher, private assets, sample records, source exporter
configs/                  Versioned protocols and runtime configuration
release_tests/            Packaging, environment and integrity regression tests
docs/                     Setup, architecture, manual QA and publication boundaries
data/                     Local generated/private state; excluded from Git
artifacts/                Local runtime metadata and reports; excluded from Git
```

- [Detailed setup and troubleshooting](docs/SETUP.md)
- [Architecture and model boundaries](docs/ARCHITECTURE.md)
- [GitHub packaging and private assets](docs/GITHUB_PREPARATION.md)

No open-source license has been selected for this project. Add the owner's chosen license before representing it as licensed for reuse.
