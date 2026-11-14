"""Offline speaker controls with a separate calibration and audit cohort."""
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from .voice_metrics import speaker_cosine, voiced_audio


ROLES = ('target_reference', 'genuine_development', 'genuine_audit',
         'negative_reference', 'negative_audit', 'generated')


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def preprocessing_recipe(vad_sha256):
    return dict(sample_rate=16000, vad_sha256=vad_sha256, frame_samples=512,
                threshold=.5, padding_frames=1, minimum_selected_seconds=1.,
                rms_target=.1, peak_limit=.99,
                implementation_sha256=file_hash(Path(__file__).with_name('voice_metrics.py')))


def validate_preprocessing(spec):
    actual = preprocessing_recipe(spec['vad']['sha256'])
    if spec.get('preprocessing') != actual or spec.get('preprocessing_sha256') != fingerprint(actual):
        raise ValueError('preprocessing declaration differs from implementation')


def validate_cohort(cohort, *, root):
    if cohort.get('version') != 1 or not cohort.get('target_speaker'):
        raise ValueError('invalid cohort version or target speaker')
    identities, sources, hashes = set(), set(), set()
    counts = dict.fromkeys(ROLES, 0)
    for row in cohort['records']:
        if row['role'] not in ROLES or not row['id'] or not row['source_id']:
            raise ValueError('invalid cohort identity or role')
        if row['id'] in identities or row['source_id'] in sources or row['wav_sha256'] in hashes:
            raise ValueError('duplicate recording or source across cohort roles')
        if file_hash(Path(root) / row['path']) != row['wav_sha256']:
            raise ValueError('cohort audio hash mismatch')
        target = row['role'] in ROLES[:3]
        if target and (row['speaker_id'] != cohort['target_speaker']
                       or row['original_split'] not in ('validation', 'reference')):
            raise ValueError('target calibration must be nontraining development/reference speech')
        if row['role'].startswith('negative') and row['speaker_id'] == cohort['target_speaker']:
            raise ValueError('negative control belongs to target speaker')
        identities.add(row['id']); sources.add(row['source_id']); hashes.add(row['wav_sha256'])
        counts[row['role']] += 1
    if any(counts[role] < 3 for role in ROLES[:-1]):
        raise ValueError('at least three separate recordings per control role are required')


def _vector(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or not np.any(value):
        raise ValueError('invalid speaker embedding')
    # Scale first to avoid overflow when checking/normalizing finite vectors.
    value = value / np.max(np.abs(value))
    return value / np.linalg.norm(value)


def extract_cache(cohort, spec, *, root, vad, encoder):
    """Call a supplied encoder on identical VAD-selected 16 kHz PCM."""
    validate_cohort(cohort, root=root)
    rows = []
    for item in cohort['records']:
        row = dict(id=item['id'], wav_sha256=item['wav_sha256'])
        start = perf_counter()
        try:
            audio, rate = sf.read(Path(root) / item['path'], dtype='float32')
            selected, qc = voiced_audio(audio, int(rate), vad)
            qc['pcm_sha256'] = sha256(selected.tobytes()).hexdigest()
            repeats = [_vector(encoder(selected.copy())).tolist()
                       for _ in range(3 if item.get('repeat') else 1)]
            row.update(status='measured', embedding=repeats[0], repeats=repeats,
                       preprocessing=qc)
        except ValueError as error:
            row.update(status='unknown', reason=str(error))
        except (OSError, sf.LibsndfileError):
            row.update(status='unknown', reason='audio_read_failed')
        row['seconds'] = perf_counter() - start
        rows.append(row)
    return dict(version=1, spec=deepcopy(spec), cohort_sha256=fingerprint(cohort),
                preprocessing_sha256=spec['preprocessing_sha256'], records=rows,
                voice_similarity_confirmed=False)


def validate_cache(cache, cohort, spec, *, root):
    validate_cohort(cohort, root=root)
    if (cache.get('version') != 1 or cache.get('cohort_sha256') != fingerprint(cohort)
            or cache.get('spec') != spec
            or cache.get('preprocessing_sha256') != spec['preprocessing_sha256']):
        raise ValueError('stale model, preprocessing, or cohort cache')
    expected = {row['id']: row for row in cohort['records']}
    records = cache['records']
    if len(records) != len(expected) or {row['id'] for row in records} != set(expected):
        raise ValueError('missing or duplicated cache rows')
    dimension = None
    for row in records:
        if row['wav_sha256'] != expected[row['id']]['wav_sha256']:
            raise ValueError('cache audio hash mismatch')
        if row['status'] == 'unknown' and row.get('reason'):
            continue
        if row['status'] != 'measured':
            raise ValueError('invalid cache status')
        vector = _vector(row['embedding'])
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError('inconsistent embedding dimensions')
        for repeat in row['repeats']:
            if len(_vector(repeat)) != dimension:
                raise ValueError('invalid repeat embedding')
        if len(row['repeats']) < (3 if expected[row['id']].get('repeat') else 1):
            raise ValueError('repeatability observations missing')


def assess_controls(cohort, caches, *, root):
    """Fit on references/development controls only, then score untouched audit rows.

    A midpoint threshold tests speaker separability on this small panel. It is
    not a perceptual similarity cutoff or a probability of voice identity.
    """
    names = [cache['spec']['encoder'] for cache in caches]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('two independent encoders are required')
    metadata = {row['id']: row for row in cohort['records']}
    encoders = []
    by_id = {key: [] for key in metadata}
    for cache in caches:
        validate_cache(cache, cohort, cache['spec'], root=root)
        vectors = {row['id']: _vector(row['embedding']) for row in cache['records']
                   if row['status'] == 'measured'}
        refs = [key for key in metadata if metadata[key]['role'] == 'target_reference']
        calibration = [key for key in metadata if metadata[key]['role'] in
                       ('target_reference', 'genuine_development', 'negative_reference')]
        reasons, positive, negative = [], [], []
        scores = {}
        if any(key not in vectors for key in calibration):
            reasons.append('calibration_incomplete')
        else:
            for key, vector in vectors.items():
                scores[key] = float(np.mean([speaker_cosine(vector, vectors[ref])
                                             for ref in refs if ref != key]))
            positive = [scores[key] for key in calibration if not metadata[key]['role'].startswith('negative')]
            negative = [scores[key] for key in calibration if metadata[key]['role'].startswith('negative')]
            if min(positive) <= max(negative):
                reasons.append('calibration_overlap')
        threshold = (min(positive) + max(negative)) / 2 if positive and not reasons else None
        for key in metadata:
            value = scores.get(key)
            prediction = ('target_domain' if value >= threshold else 'non_target_domain') if (
                threshold is not None and value is not None) else 'unknown'
            by_id[key].append(dict(encoder=cache['spec']['encoder'], score=value, prediction=prediction))
        encoders.append(dict(encoder=cache['spec']['encoder'], revision=cache['spec']['revision'],
                             threshold=threshold, positive_range=[min(positive), max(positive)] if positive else None,
                             negative_range=[min(negative), max(negative)] if negative else None,
                             reasons=reasons))
    rows, correct, unknown, total = [], 0, 0, 0
    audit_by_class = {key: dict(correct=0, unknown=0, incorrect=0)
                      for key in ('genuine', 'negative')}
    for key, item in metadata.items():
        predictions = {row['prediction'] for row in by_id[key]}
        decision = next(iter(predictions)) if len(predictions) == 1 else 'unknown'
        expected = 'target_domain' if item['role'] == 'genuine_audit' else 'non_target_domain'
        if item['role'] in ('genuine_audit', 'negative_audit'):
            total += 1; correct += decision == expected; unknown += decision == 'unknown'
            category = 'genuine' if item['role'] == 'genuine_audit' else 'negative'
            outcome = 'correct' if decision == expected else 'unknown' if decision == 'unknown' else 'incorrect'
            audit_by_class[category][outcome] += 1
        rows.append(dict(id=key, role=item['role'], decision=decision, encoders=by_id[key]))
    # Full-coverage separation and selective screening are different claims.
    # Abstentions remain in the denominator. Screening needs at least three
    # correct audit examples of EACH class, no wrong consensus, and complete
    # nonoverlapping calibration in every encoder. This is a small-panel gate,
    # not a statistical guarantee on unseen speakers or generated voices.
    screening = (not any(e['reasons'] for e in encoders) and total-correct-unknown == 0
                 and all(group['correct'] >= 3 for group in audit_by_class.values()))
    return dict(version=1, cohort_sha256=fingerprint(cohort), encoders=encoders, records=rows,
                audit=dict(expected=total, correct=correct, unknown=unknown, incorrect=total-correct-unknown),
                audit_by_class=audit_by_class,
                screening_status=('supported_with_abstentions' if unknown else 'supported') if screening else 'unsupported',
                status='controls_separated' if total and correct == total and not any(e['reasons'] for e in encoders)
                else 'controls_failed', voice_similarity_confirmed=False,
                scope='Small real-speaker control panel only; generated-voice likeness is not established')


def build_panels(cohort, caches, *, baseline_id, candidate_id, root):
    """Bridge validated extraction caches to the existing replay evaluator."""
    controls = assess_controls(cohort, caches, root=root)
    if controls['screening_status'] == 'unsupported':
        raise ValueError('speaker controls failed; automatic candidate selection is unavailable')
    if len({cache['preprocessing_sha256'] for cache in caches}) != 1:
        raise ValueError('incompatible preprocessing across encoders')
    metadata = {row['id']: row for row in cohort['records']}
    panels = []
    for cache in caches:
        rows = {row['id']: row for row in cache['records']}
        def reference(item):
            return {key: item[key] for key in ('path', 'wav_sha256', 'source_id', 'speaker_id')} | {
                'embedding': rows[item['id']]['embedding']}
        def observation(key):
            row = rows[key]
            if row['status'] != 'measured':
                raise ValueError('candidate or baseline embedding unavailable')
            return dict(wav_sha256=row['wav_sha256'], embedding=row['embedding'])
        base = observation(baseline_id)
        candidate = observation(candidate_id)
        if len(rows[baseline_id]['repeats']) < 3:
            raise ValueError('baseline repeatability measurements missing')
        panels.append(dict(encoder=cache['spec']['encoder'], revision=cache['spec']['revision'],
                           preprocessing_sha256=cache['preprocessing_sha256'],
                           target_speaker=cohort['target_speaker'],
                           control_audit=dict(controls['audit'], screening_status=controls['screening_status'],
                                              cohort_sha256=controls['cohort_sha256']),
                           references=[reference(item) for item in metadata.values() if
                                       item['role'] in ('target_reference', 'negative_reference')],
                           genuine_controls=[reference(item) for item in metadata.values() if
                                             item['role'] == 'genuine_development'],
                           baseline=base, candidate=candidate,
                           baseline_repeats=[dict(wav_sha256=base['wav_sha256'], embedding=value)
                                             for value in rows[baseline_id]['repeats']]))
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; use a new cache path')
    cohort = json.loads(args.cohort.read_text(encoding='utf-8'))
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    validate_cohort(cohort, root=args.cohort.parent)
    validate_preprocessing(spec)
    from .speaker_encoders import load_cpu_encoder, load_vad
    encoder = load_cpu_encoder(spec)
    vad = load_vad(spec['vad'])
    cache = extract_cache(cohort, spec, root=args.cohort.parent, vad=vad, encoder=encoder)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(cache, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'measured': sum(row['status'] == 'measured' for row in cache['records']),
                      'unknown': sum(row['status'] == 'unknown' for row in cache['records'])}))


if __name__ == '__main__':
    main()
