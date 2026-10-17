"""Render reviewed aggregate preprocessing summaries, never patient-level records."""
import argparse
import json
from pathlib import Path

from .io import dumps
from .schema import DATASETS
from .validate import validate_dataset


def render(reports, validations):
    lines=['# Phase 2 preprocessing report','',
           'Generated from the canonical `report.json` files by `python3 -m src.data.report`. '
           'Source adapters were run on all supplied files. No model was fitted and no performance numbers are reported. '
           'Patient IDs, records and split assignments remain in ignored local outputs.','',
           '## Canonical cohort and observation counts','',
           '| Dataset | Patients | Source samples retained | Canonical observations | Usable observations | Nonmissing quarantined observations | Duplicate source samples | Temporal feature rows |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        r=reports[dataset];counts=r['source_counts']
        source=(counts['clinical_rows_retained'] if dataset=='zigong' else counts['daily_rows_retained'] if dataset=='chiron'
                else counts['medical_rows_retained'] if dataset=='perheart' else counts['Observation'])
        lines.append(f'| {dataset} | {r["patients_retained"]:,} | {source:,} | {r["canonical_observations"]:,} | {r["usable_observations"]:,} | {r["quarantined_observations"]:,} | {counts.get("duplicate_medical_rows",0):,} | {r["temporal_feature_rows"]:,} |')
    lines+=['','Zigong source samples are clinical rows; its six-column predictor matrix has one row per retained patient. '
            'Zigong physiological context is untimed and not converted into a fabricated observation series. '
            'PerHeart source samples are medical measurement rows after exact duplicate consolidation; canonical observations still retain duplicate placeholders with no usable value. '
            'Synthea source samples in this table are Observation resources; component extraction increases canonical events. '
            'Quarantine counts exclude source-missing cells, duplicate placeholders and non-observation clinical/source-feature exclusions.','',
            '## Actual observation quality','',
            '| Dataset | VALID | MISSING | SUSPECT | INVALID | DUPLICATE |','|---|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        state=reports[dataset]['observation_quality']
        lines.append('| '+dataset+' | '+' | '.join(f'{state.get(k,0):,}' for k in ('VALID','MISSING','SUSPECT','INVALID','DUPLICATE'))+' |')
    lines+=['','CHIRON otherwise usable measurements are SUSPECT because their unit conventions remain provisional. '
            'Its 158 impossible SpO₂ values are INVALID; two BP measurements are blocked by one ordering conflict. '
            'Fourteen unparseable `-?` tokens occur in excluded supplied features, outside this observation-state table. '
            'Synthea has three unitless numeric values quarantined. Sixteen potential same-concept/time/unit/value repeats are reported and retained, '
            'without pretending different resources are proven duplicates. Pediatric body weights are exempt from the adult lower screening band.','',
            '## Zigong modelling contract and class balance','',
            'Approved predictors: `gender`, `ageCat`, `admission.ward`, `admission.way`, `visit.times`, `congestive.heart.failure`. '
            'These have semantic index-time support, not verified capture timestamps. All other source clinical fields and drug exposure are excluded. '
            'The benchmark remains **retrospective 28-day readmission benchmark**, with the source clock unconfirmed.','',
            '| Partition | Negative | Positive | Total |','|---|---:|---:|---:|']
    z=reports['zigong']
    for part in ('train','validation','test'):
        counts=z['split_class_counts'][part];negative=counts.get('0',0);positive=counts.get('1',0)
        lines.append(f'| {part} | {negative:,} | {positive:,} | {negative+positive:,} |')
    lines+=['',f'Total recorded positives: {z["class_distribution"].get("1",0)}; negatives: {z["class_distribution"].get("0",0)}. '
            'Discharge conflicts, nonmonotone flags and event-day disagreements remain flagged; no relabeling or outcome-based exclusion is hidden. '
            'The complete per-field machine contract is also exported as [ZIGONG_FEATURE_AVAILABILITY.json](ZIGONG_FEATURE_AVAILABILITY.json).','',
            '## Longitudinal folds and targets','',
            '| Dataset | Fold | Train patients | Validation patients | Test patients |','|---|---:|---:|---:|---:|']
    for dataset in ('chiron','perheart'):
        for row in reports[dataset]['split_counts']:
            lines.append(f'| {dataset} | {row["fold"]} | {row["train"]} | {row["validation"]} | {row["test"]} |')
    lines+=['','Seed 2026; every patient is outer test exactly once. All windows from a patient inherit the same assignment within each fold. '
            'Synthea has an integration-only manifest, with no clinical evaluation split.','',
            'CHIRON Feel remains a separate ordinal report (not HF deterioration): '+
            ', '.join(f'`{label}`: {count}' for label,count in sorted(reports['chiron']['class_distribution'].items()))+'. '
            'PerHeart questionnaire recent-HF responses remain unadjudicated and are not supervised targets.','']
    lines += ['PRO worsening is Feel 1/2 versus 3/4/5, before next-day/window eligibility. Test-day counts vary considerably despite equal patient counts:', '', '| Fold | Worse test days | Nonworse test days | Unknown test days |', '|---|---:|---:|---:|']
    for fold in reports['chiron']['pro_worsening_counts_by_fold']:
        counts=fold['counts']['test']
        lines.append(f'| {fold["fold"]} | {counts.get("worse",0)} | {counts.get("nonworse",0)} | {counts.get("unknown",0)} |')
    lines += ['', 'These are subjective report counts, not HF-event counts; small positive counts limit fold-level estimates.', '']
    for dataset in DATASETS:
        r=reports[dataset]
        lines += [f'## {dataset} coverage and missingness','',
                  'Temporal coverage: '+dumps(r['temporal_coverage'])+'. '+
                  ('CHIRON endpoints are source-relative days, without calendar dates.' if dataset=='chiron' else 'No timestamp is invented for unknown capture times.'),'',
                  'Available canonical concepts/features: '+', '.join('`'+concept+'`' for concept in r['available_features'])+'.','',
                  f'Source-missing assertions across context/outcome/source-feature/observation tables: {r["missingness"]["source_missing_assertions"]:,}. '
                  'This is not a scheduled monitoring-time denominator.','']
        if r['missingness']['by_concept']:
            lines+=['| Concept | Source-missing observations | Unusable observations | Patients without nonmissing values | Patients without usable values |',
                    '|---|---:|---:|---:|---:|']
            for concept,counts in r['concept_quality'].items():
                missing=r['missingness']['by_concept'][concept]
                lines.append(f'| {concept} | {100*missing["source_missing_fraction"]:.2f}% | {100*missing["unusable_fraction"]:.2f}% | {missing["patients_without_modality"]} | {missing["patients_without_usable_modality"]} |')
            lines.append('')
        lines+=['QC reason counts (may overlap):','', '| Reason | Assertions / issues |','|---|---:|']
        for reason,count in r['quality_reasons'].items():
            lines.append(f'| {reason} | {count:,} |')
        lines.append('')
    refs=reports['synthea']['fhir_references']
    lines+=['## Validation and remaining scientific limits','',
            f'FHIR references audited: {refs["total"]:,}; conditional identifier references resolved: {refs["conditional_identifier_resolved"]:,}/{refs["conditional_identifier"]:,}; '
            f'unresolved: {refs.get("unresolved",0):,}.','',
            'Canonical validation results: '+', '.join(f'{d}: {validations[d]["status"]}' for d in DATASETS)+'. '
            'Checks include artifact hashes, schemas, provenance, patient identity, predictor/outcome separation, split isolation, quality masks, duplicates and full causal feature recomputation.','',
            'The test suite covers adapters, malformed schemas/IDs, outcome injection, raw preservation, temporal perturbations/delayed availability, '
            'incompatible units, duplicate/conflict handling, ambiguous/contained/conditional FHIR references, deterministic rebuilds, publication locks and failed-build rollback.','',
            'Unresolved: Zigong endpoint clock/censoring and dynamic feature availability; CHIRON provisional units, extra patient ID and sensor errors; '
            'PerHeart absent wrist inputs and absent confirmed deterioration positives; synthetic FHIR clinical limitations. '
            'Live receipt latency, a paired outcome-labelled EHR/wearable cohort, clinical TDI alerts and invasive hemodynamics remain unavailable.','',
            'See [DATA_PIPELINE.md](DATA_PIPELINE.md) for exact commands, artifacts, policies and Phase 3 recommendations.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/processed'))
    parser.add_argument('--output',type=Path,default=Path('docs/PREPROCESSING_REPORT.md'))
    args=parser.parse_args()
    reports={};validations={}
    for dataset in DATASETS:
        validations[dataset]=validate_dataset(args.root/dataset)
        reports[dataset]=json.loads((args.root/dataset/'report.json').read_text())
    contract=json.loads((args.root/'zigong/feature_availability.json').read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render(reports,validations)+'\n')
    # Contract contains feature names/reasons only, no records or identities.
    (args.output.parent/'ZIGONG_FEATURE_AVAILABILITY.json').write_text(json.dumps(contract,indent=2)+'\n')
    print('Aggregate preprocessing report:',args.output)

if __name__=='__main__':main()
