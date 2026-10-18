# TwinCare — Personalized Healthcare Digital Twin

**Personal Health Intelligence · Digital Twin Challenge 2026 · Team Code Storm**

TwinCare combines a patient workspace, medical-record review, glucose forecasting, a personal expected glucose pattern, meal logging, a shared wearable view, and alerts. The Metabolic Twin supports newly created patients with received observations. The Cardiovascular Twin provides the preserved synthetic research demonstration.

The app runs locally. Wearable and CGM connections currently use explicit synthetic simulation or historical replay; commercial watch pairing is not implemented. Predictions and meal estimates are research outputs, not clinically validated care recommendations.

## Team and institution

| Detail | Information |
| --- | --- |
| Team | **Code Storm** |
| Members | **Swaraj Patil · Rutika Shripati** |
| College | **Vishwakarma Institute of Technology** |
| Challenge | **Digital Twin Challenge 2026 — Prototype & Code Submission** |
| Public repository | [Code_Storm_Twin_Care](https://github.com/SwarajPatil2318/Code_Storm_Twin_Care) |

## Explore TwinCare

Watch the product walkthrough and explore the design, architecture and intelligence behind TwinCare.

| Resource | Format | Link |
| --- | --- | --- |
| Product walkthrough | 15–20 minute unlisted YouTube video | [Watch the walkthrough](https://youtu.be/-8qKDi5uB3g) |
| Project presentation and outcomes | PDF/PPT | [View presentation](https://drive.google.com/file/d/10YFaL3yA_6Of2AKsIUNUEldb9NuliauE/view?usp=sharing) |
| System architecture | PDF/PPT | [View architecture](https://drive.google.com/file/d/1RCI218JNKtIm2BcXLpIph_OH7CCrRnKa/view?usp=sharing) |
| AI/ML framework | Supporting document | [View AI/ML framework](https://drive.google.com/file/d/1fZSdvW7SoX-eOYV8UjxYOK_O-fDAwZvS/view?usp=sharing) |

For a closer look at the implementation, see the [runtime architecture](docs/ARCHITECTURE.md) and the AI/ML details below.

## Problem statement

A current glucose value shows what is happening now, but it does not explain how that value compares with a person's historical pattern or what may happen next. Clinical records, glucose streams, meals and activity also arrive through separate workflows. TwinCare brings these inputs into a patient-owned longitudinal workspace, forecasts future glucose, and explains departures from the patient's expected pattern while preserving source times and uncertainty.

## Healthcare use case

The main implemented use case is **research monitoring for a person with Type 2 diabetes**:

1. Create a patient and review/confirm their clinical information.
2. Receive glucose observations and contextual wearable/meal information.
3. Estimate glucose at four future horizons and compare it with an eligible personal expected trajectory.
4. Inspect contributing signals, review the separate two-hour spike-risk estimate, and acknowledge policy-triggered alerts in the dashboard or wearable view.

The product supports research review and explanation; it does not recommend treatment or automate clinical decisions. The cardiovascular screen preserves the existing synthetic research demonstration rather than claiming new-patient cardiovascular prediction.

## Technical stack

| Layer | Implementation |
| --- | --- |
| Patient workspace | JavaScript modules, HTML/CSS, Python HTTP APIs |
| Twin dashboards | React 19, TypeScript, Vite 6, Lucide icons; shared wearable component |
| Backend and storage | Python 3.12, loopback HTTP services, SQLite, owner-scoped records and predictions |
| Forecasting and risk | XGBoost, scikit-learn, NumPy, pandas and SciPy; CPU PyTorch for research-model compatibility |
| Clinical documents and photos | pypdf, Pillow; supported structured clinical JSON/FHIR, text-based PDFs, reviewed image uploads |
| Optional food-photo AI | Gemini REST vision adapter with structured JSON output; manual review/fallback |
| Verification | Python unittest, frontend source/chart tests, Playwright browser checks and GitHub Actions source/build CI |

Exact Python runtime versions are in [requirements.txt](requirements.txt); frontend dependencies are locked in [package-lock.json](frontend/package-lock.json).

## AI/ML model and framework details

| Component | Implemented method and role |
| --- | --- |
| Glucose forecast model | Frozen **XGBoost multi-output gradient-boosted trees**: 180 trees, maximum depth 4, learning rate 0.05, L2 regularization 5 and a custom pseudo-Huber objective. Predicts glucose at **+30, +60, +90 and +120 minutes**. |
| Causal input construction | Nine native 15-minute CGM observations over the previous 120 minutes produce **21 core features**: lags, slopes, variability, distribution statistics, acceleration and time. Eligibility and missingness checks precede inference. |
| Personalization | Cold inference uses the population model; eligible warm inference adds **12 causal personal-history features**. Eligibility includes at least 192 prefix observations and 50 elapsed hours. The prefix excludes the current input window. |
| Shadow Twin / personal expectation | A historical expected trajectory for the same patient. Predicted-versus-expected separation describes personal divergence; it is **not a spike probability or a causal intervention simulator**. |
| Multimodal spike risk | A **separate regularized, calibrated logistic model** receives 7 historical clinical fields and 26 dynamic fields: 21 core CGM features, 4 forecast residuals and predicted peak. Its 33 raw fields are encoded/preprocessed for the two-hour event estimate. |
| Gemini meal analysis | Optional image-based food and approximate nutrition proposals. These require correction/confirmation and do not replace the forecaster, generate clinical diagnoses or directly change frozen model parameters. |

The primary research cohort contains 100 Type 2 diabetes participants, split by whole participant into 80 development and 20 held-out participants. Within development, 64 supply forecast fitting and 16 supply reserved calibration. Preprocessing fits training participants only; runtime features exclude observations or receipts after the prediction cutoff. Unrelated datasets are never linked as if they were the same patient.


## Recorded outcomes and evidence boundaries

The existing one-time D2 held-out evaluation reported the following **participant-macro MAE**, in mg/dL, for the frozen deployed cold/warm dispatch. These are previously recorded retrospective research results, not a new evaluation or a claim of clinical benefit.

| Forecast horizon | MAE (mg/dL) |
| --- | ---: |
| +30 minutes | 9.354 |
| +60 minutes | 16.360 |
| +90 minutes | 21.022 |
| +120 minutes | 23.796 |

Source: [reviewed submission evidence](docs/SUBMISSION_EVIDENCE.md), transcribed from the existing aggregate D2 forecasting report. The separate multimodal risk model was development-validated; adding EHR context is **not claimed to improve the direct glucose forecast or to establish clinical superiority**. Synthetic scenarios demonstrate software integration, not clinical validation.

The clean source release also passed a fresh Python environment install, `npm ci`, frontend build, **14 release tests**, and browser smoke checks for the patient workspace and both twin dashboards. Those tests include real cold/warm model execution, baseline/risk availability, owner isolation and integrity checks. The public CI configuration runs source/frontend checks; private-model acceptance requires the local runtime bundle.

## Open-source license details

The project's original code is released under the **[MIT License](LICENSE)**, copyright **2026 Swaraj Patil and Rutika Shripati**. The license permits use, modification and redistribution subject to retaining its notice and terms, and includes its standard warranty disclaimer. See the complete license text for the grant and the [official MIT license reference](https://opensource.org/license/mit).

Third-party dependencies retain their own licenses. The project MIT grant does not relicense research datasets, external media, supplied assets or private model artifacts. Product illustrations are supplied for this prototype; no blanket dataset/model/media redistribution right is asserted. No credentials, patient records, raw cohorts or model weights are published by this source release.

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
tools/                    Launcher, runtime asset setup, environment checks, sample records
configs/                  Versioned protocols and runtime configuration
release_tests/            Packaging, environment and integrity regression tests
docs/                     Setup, architecture, manual QA and publication boundaries
data/                     Local generated/private state; excluded from Git
artifacts/                Local runtime metadata and reports; excluded from Git
```

- [Detailed setup and troubleshooting](docs/SETUP.md)
- [Architecture and model boundaries](docs/ARCHITECTURE.md)
- [GitHub packaging and private assets](docs/GITHUB_PREPARATION.md)
