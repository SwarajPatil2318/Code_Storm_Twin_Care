# Reviewed submission evidence

This document transcribes previously recorded aggregate results for the submission README. It contains no patient identifiers, individual trajectories, model serialization, or raw clinical records. Updating this document did not train a model or repeat the held-out evaluation.

## D2 glucose forecast

The original research cohort has 100 Type 2 diabetes participants. The participant-level allocation is 80 development and 20 held out; within development, 64 participants supply final forecast fitting and 16 supply reserved event calibration. The held-out evaluation contains 21,820 complete four-target windows from 20 participants. Overlapping windows are not independent patients.

The primary frozen dispatch uses the selected population forecaster at cold cutoffs and the selected personalized forecaster at eligible warm cutoffs. Previously reported equal-participant macro errors are:

| Horizon (minutes) | MAE (mg/dL) | MAE 95% participant-bootstrap CI (mg/dL) |
| --- | ---: | --- |
| 30 | 9.354 | 8.806–9.934 |
| 60 | 16.360 | 15.246–17.477 |
| 90 | 21.022 | 19.504–22.614 |
| 120 | 23.796 | 21.975–25.795 |

Source: the existing `docs/GLUCOSE_FORECAST_RESULTS.md` section “One-time untouched holdout” and `docs/METABOLIC_D2_MODEL_CARD.md` in the research workspace. Those longer historical reports remain outside the lean source package. The underlying research protocol assumes observation-time availability for source recordings lacking receipt timestamps; live runtime checks both observation and received times.

These errors are retrospective cohort results, not prospective clinical validation, proof of intervention benefit, or a guarantee for a new patient/device/population. The additional multimodal risk model was evaluated in development folds; its results must not be relabelled as new held-out evidence or evidence that EHR directly improves the short-horizon glucose forecast.

## Engineering verification

The clean source release was tested using a freshly installed Python 3.12 environment and a fresh `npm ci` from the frontend lockfile. Fourteen release tests passed with the trusted private runtime bundle restored. They cover archive boundaries, tamper rejection, environment handling, real cold/warm inference, expected-pattern/risk availability, causal input cutoff, owner isolation, and the distinction between meal confirmation and a later observation-triggered prediction.

Frontend build, source timing, live chart and presentation checks passed. Browser smoke covered the patient workspace, Cardiovascular Twin and Metabolic Twin routes, plus desktop/tablet/mobile patient-shell widths. This is bounded integration evidence; it is not exhaustive browser testing or clinical validation. Reproduction commands are in the README and [manual testing guide](MANUAL_TESTING.md).

## Submission access

Anonymous GitHub access was checked on 17 October 2026 for the public repository and the README already published at that time. New README changes must still be pushed. Presentation, architecture/framework and unlisted YouTube URLs remain pending; their permissions, content and video duration have not been verified. Full inference also requires a private runtime bundle, so public source availability does not establish anonymous full-runtime reproducibility.
