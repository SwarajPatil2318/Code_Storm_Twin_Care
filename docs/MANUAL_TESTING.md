# Manual product checks

Start the complete runtime with `python tools/run.py`. Use fictional test data. `python tools/create_sample_records.py` creates upload examples under ignored `data/sample_records/` without changing your patient database.

## New-patient workflow

1. Open `/patients`, create a fictional patient, and check the name/age/sex in the header.
2. Upload the generated clinical JSON, inspect each extracted field, accept/correct the intended facts, and confirm. Nothing should become a confirmed clinical fact just because it was uploaded.
3. Connect Simulated Live CGM with prior history and acknowledge the synthetic source. Connect Simulated Wearable. Start monitoring.
4. Verify direct navigation to the overview, the hero illustration, current glucose, all four horizons, a personal-baseline state, explanatory factors and a populated graph. When ready, the historical expected trajectory should be distinct from the forecast.
5. Let updates arrive while scrolled down. Check that the page stays in position and does not rebuild/flicker. Switch graph ranges, then pause/resume monitoring.
6. Open wearable details. Check compact dimensions, received values and units, no overlapping labels, and consistent style with the cardiovascular/metabolic demo screens.
7. Log a manual meal, correct the draft, confirm it, then edit/remove it. If Gemini is configured, repeat with a photo; provider failure should retain manual entry.
8. Inspect prediction evidence and saved history. Forecast timestamps, observed timestamps and availability timestamps should remain distinct. Review a stored prediction without turning it into a new measurement.

## Readiness and ownership

Create another fictional patient without prior history. A new connection does not guarantee immediate forecasting: complete declared-cadence CGM history and confirmed context are required. Personal expectation remains unavailable until the separate historical support rules are met. Loading synthetic history is an explicit integration-test option.

Switch patients and check that records, measurements, meals, alerts and watch values remain owned by the correct patient. Do not assume a Type 2 Diabetes diagnosis makes two unrelated dataset subjects the same person.

## Preserved dashboards

Open `/twin/cardiovascular` and `/twin/metabolic`. Check the heart/flow assets, replay controls, central details and shared watch. These screens use the packaged replay examples. Repeat at desktop, tablet and mobile widths and confirm no horizontal document overflow.

## Alerts

An alert only appears when the frozen policy is met; it is not forced by clicking a meal or a preset. If it appears during simulation or playback, inspect its prediction evidence, acknowledge it from the dashboard/watch, and verify that both views show the same acknowledged state. Alert testing is integration testing, not clinical performance validation.

The optional `/demo` internal scenario launcher is a separate service started with `--with-judge` and its private inputs. Normal patient onboarding does not require that launcher or editing database rows.

## Automated smoke

Run the commands in README. The browser smoke covers all three main route families and patient-shell widths. Its screenshots stay private under `data/release/browser/`. Focused source tests cover public-package boundaries, private-bundle tampering/restore safety, deployment opt-in and dotenv handling. Full real-model acceptance is run locally after restoring the bundle; public CI cannot load private weights.
