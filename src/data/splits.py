"""Patient-level split assignment independent of input row order."""
import hashlib
import math
from collections import Counter


def rank(patient, seed):
    return hashlib.sha256(f'{seed}:{patient}'.encode()).hexdigest()


def stratified_split(labels, seed=2026, fractions=(0.6, 0.2, 0.2)):
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError('Three positive split fractions must sum to one')
    assignments = {}
    for label in sorted(set(labels.values())):
        ids = sorted((p for p in labels if labels[p] == label), key=lambda p: rank(p, seed))
        exact = [len(ids) * x for x in fractions]
        counts = [math.floor(x) for x in exact]
        for index in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[:len(ids) - sum(counts)]:
            counts[index] += 1
        offset = 0
        for partition, count in zip(('train', 'validation', 'test'), counts):
            for patient in ids[offset:offset + count]:
                assignments[patient] = partition
            offset += count
    manifest = {'strategy': 'stratified_patient_holdout', 'seed': seed,
                'fractions': fractions, 'assignments': dict(sorted(assignments.items())),
                'class_counts': {part: dict(sorted(Counter(str(labels[p]) for p, value in assignments.items() if value == part).items()))
                                 for part in ('train', 'validation', 'test')}}
    validate_splits(manifest, set(labels))
    return manifest


def grouped_cv(patients, seed=2026, folds=5):
    ids = sorted(set(patients), key=lambda p: rank(p, seed))
    if len(ids) < folds or folds < 3:
        raise ValueError('Grouped cross-validation needs at least three folds and enough patients')
    membership = {p: index % folds for index, p in enumerate(ids)}
    evaluations = []
    for index in range(folds):
        validation_fold = (index + 1) % folds
        assignments = {p: 'test' if f == index else 'validation' if f == validation_fold else 'train'
                       for p, f in sorted(membership.items())}
        evaluations.append({'fold': index, 'assignments': assignments})
    result = {'strategy': 'grouped_patient_rotation', 'seed': seed, 'n_splits': folds,
              'membership': dict(sorted(membership.items())), 'folds': evaluations,
              'note': 'Each patient is outer test once; next fold is validation; remaining folds train. No outcome balancing.'}
    validate_splits(result, set(ids))
    return result


def validate_splits(manifest, patients):
    if manifest.get('strategy') == 'integration_only_no_clinical_evaluation':
        if manifest.get('assignments') or set(manifest.get('patients', [])) != patients:
            raise ValueError('Invalid integration-only patient manifest')
        return
    partitions = [manifest] if 'assignments' in manifest else manifest.get('folds', [])
    if not partitions:
        if manifest.get('strategy') == 'integration_only_no_clinical_evaluation':
            return
        raise ValueError('Split manifest has no evaluation partitions')
    for partition in partitions:
        assignments = partition['assignments']
        if set(assignments) != patients or any(v not in {'train', 'validation', 'test'} for v in assignments.values()):
            raise ValueError('Split does not partition the retained patients')
        groups = [{p for p, v in assignments.items() if v == name} for name in ('train', 'validation', 'test')]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Patient appears in multiple partitions')
    if 'folds' in manifest:
        tests = Counter(p for f in manifest['folds'] for p, v in f['assignments'].items() if v == 'test')
        if set(tests) != patients or any(count != 1 for count in tests.values()):
            raise ValueError('Every patient must be tested in exactly one outer fold')
