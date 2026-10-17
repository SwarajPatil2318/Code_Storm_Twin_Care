# Local data

Supplied inputs currently live in `../Datasets/` and are ignored by Git.
`raw/` is for immutable future inputs; `interim/` and `processed/` are for derived
records; `audit/` contains local machine-readable audit reports. All records and
artifacts under these directories are ignored. Do not copy records into docs or
notebooks. Never assume `.gitignore` removes files already tracked by Git.

Phase 2 outputs are generated under `processed/{zigong,chiron,perheart,synthea}/`.
They contain private source records, canonical tables, quality/provenance, split
assignments and checksums. Rebuild with `python3 -m src.data.build --dataset all`;
validate with `python3 -m src.data.validate`. See `docs/DATA_PIPELINE.md`.

Phase 3 experiments live under `results/phase3/` (ignored, private), including model serialization, metrics, patient predictions, ShadowResidual records and trajectory plots. `results/phase3_reproduction/` repeats training only. Exploratory/pre-seal runs retain validation-only experiments; they have no clinical final-test evaluation. Only reviewed aggregate documentation goes in `docs/`.

Phase 3.5 extracts only selected eICU/MIMIC files to `raw/phase3_5/`, preserving the
original archive. Private `processed/eicu_demo/` contains raw numeric/QC arrays,
context provenance, events, risk sets, isolated predictors/labels, patient splits
and hash manifests; no final event predictor. Fresh repeat-build destinations and
`reproducibility/phase3_source/` are also ignored. Five public Phase 3.5 documents
contain reviewed aggregate counts and field-level contracts only.
