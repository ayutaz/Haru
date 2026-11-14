"""Tests use toy embeddings; actual model evidence is kept outside Git."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cohort(tmp_path):
    rows = []
    roles = ('target_reference', 'genuine_development', 'genuine_audit',
             'negative_reference', 'negative_audit', 'generated')
    for group, role in enumerate(roles):
        for index in range(3 if role != 'generated' else 1):
            name = f'{role}-{index}'
            path = tmp_path / f'{name}.wav'
            sf.write(path, np.full(24000, .02 + len(rows) * .01), 24000, subtype='FLOAT')
            rows.append(dict(id=name, path=path.name, wav_sha256=digest(path),
                             source_id=name, speaker_id='target' if group < 3 else 'other',
                             role=role, original_split='validation', repeat=role == 'generated'))
    return dict(version=1, target_speaker='target', records=rows)


@pytest.fixture
def spec():
    return dict(encoder='test/first', revision='a' * 40, preprocessing_sha256='b' * 64)


class Vad:
    def reset_states(self):
        pass

    def __call__(self, samples):
        return 1.


def extract(cohort, spec, root, encoder=None):
    from haru_training.speaker_calibration import extract_cache
    return extract_cache(cohort, spec, root=root, vad=Vad(),
                         encoder=encoder or (lambda pcm: np.array([1., .2])))


def test_cache_binds_model_preprocessing_cohort_and_audio(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import validate_cache
    cache = extract(cohort, spec, tmp_path)
    assert validate_cache(cache, cohort, spec, root=tmp_path) is None
    assert len(cache['records']) == len(cohort['records'])
    assert cache['voice_similarity_confirmed'] is False
    last = cache['records'][-1]
    assert len(last['repeats']) == 3
    assert last['status'] == 'measured'
    assert len(last['embedding']) == 2
    assert cache['preprocessing_sha256'] == spec['preprocessing_sha256']


@pytest.mark.parametrize('mutation', ['audio', 'model', 'role', 'missing', 'vector'])
def test_stale_or_malformed_caches_are_rejected(cohort, spec, tmp_path, mutation):
    from haru_training.speaker_calibration import validate_cache
    cache = extract(cohort, spec, tmp_path)
    if mutation == 'audio':
        (tmp_path / cohort['records'][0]['path']).write_bytes(b'changed')
    elif mutation == 'model':
        spec['revision'] = 'c' * 40
    elif mutation == 'role':
        cohort['records'][0]['role'] = 'genuine_audit'
    elif mutation == 'missing':
        cache['records'].pop()
    else:
        cache['records'][0]['embedding'] = [float('nan'), 1.]
    with pytest.raises(ValueError):
        validate_cache(cache, cohort, spec, root=tmp_path)


@pytest.mark.parametrize('mutation', ['duplicate_source', 'duplicate_wave', 'train', 'test', 'speaker'])
def test_cohort_leakage_fails_before_encoder_is_called(cohort, spec, tmp_path, mutation):
    if mutation == 'duplicate_source':
        cohort['records'][3]['source_id'] = cohort['records'][0]['source_id']
    elif mutation == 'duplicate_wave':
        cohort['records'][3].update(path=cohort['records'][0]['path'],
                                    wav_sha256=cohort['records'][0]['wav_sha256'])
    elif mutation in ('train', 'test'):
        cohort['records'][3]['original_split'] = mutation
    else:
        cohort['records'][3]['speaker_id'] = 'other'
    calls = []
    with pytest.raises(ValueError):
        extract(cohort, spec, tmp_path, lambda pcm: calls.append(pcm))
    assert calls == []


def test_invalid_encoder_output_is_unknown_and_stays_in_denominator(cohort, spec, tmp_path):
    cache = extract(cohort, spec, tmp_path, lambda pcm: np.array([0., 0.]))
    assert len(cache['records']) == len(cohort['records'])
    assert all(row['status'] == 'unknown' for row in cache['records'])
    assert all('embedding' not in row for row in cache['records'])


def calibrated_caches(cohort, spec, root):
    cache = extract(cohort, spec, root)
    lookup = {row['id']: row for row in cohort['records']}
    for row in cache['records']:
        role = lookup[row['id']]['role']
        angle = (.1 * int(row['id'].rsplit('-', 1)[1]) if role == 'target_reference'
                 else .3 if role.startswith('genuine') else 2.)
        row['embedding'] = [float(np.cos(angle)), float(np.sin(angle))]
        if role == 'generated':
            row['embedding'] = [.5, .8]
    second = deepcopy(cache)
    second['spec']['encoder'] = 'test/second'
    return [cache, second]


def test_audit_controls_do_not_set_the_calibration_threshold(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import assess_controls
    caches = calibrated_caches(cohort, spec, tmp_path)
    report = assess_controls(cohort, caches, root=tmp_path)
    assert report['status'] == 'controls_separated'
    assert report['audit']['expected'] == 6
    assert report['audit']['correct'] == 6
    assert report['voice_similarity_confirmed'] is False
    threshold = report['encoders'][0]['threshold']
    for cache in caches:
        for row in cache['records']:
            if row['id'].startswith('genuine_audit'):
                row['embedding'] = [-1., 0.]
    changed = assess_controls(cohort, caches, root=tmp_path)
    assert changed['encoders'][0]['threshold'] == threshold
    assert changed['status'] == 'controls_failed'
    assert changed['audit']['correct'] == 3


def test_overlapping_calibration_and_encoder_disagreement_are_not_success(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import assess_controls
    caches = calibrated_caches(cohort, spec, tmp_path)
    for row in caches[1]['records']:
        if row['id'].startswith('genuine_audit'):
            row['embedding'] = [-1., 0.]
    report = assess_controls(cohort, caches, root=tmp_path)
    assert report['status'] == 'controls_failed'
    assert report['audit']['unknown'] == 3
    for row in caches[0]['records']:
        if row['id'].startswith('negative_reference'):
            row['embedding'] = [1., 0.]
    report = assess_controls(cohort, caches, root=tmp_path)
    assert report['status'] == 'controls_failed'
    assert 'calibration_overlap' in report['encoders'][0]['reasons']


def test_duplicate_encoder_is_not_independent(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import assess_controls
    cache = calibrated_caches(cohort, spec, tmp_path)[0]
    with pytest.raises(ValueError, match='independent'):
        assess_controls(cohort, [cache, deepcopy(cache)], root=tmp_path)


def test_modified_weights_fail_before_any_model_loader(tmp_path):
    from haru_training.speaker_encoders import verify_model_files
    path = tmp_path / 'weights.bin'
    path.write_bytes(b'expected')
    spec = {'directory': str(tmp_path), 'files': {'weights.bin': digest(path)}}
    assert verify_model_files(spec) == tmp_path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        verify_model_files(spec)


def test_declared_preprocessing_must_match_implemented_recipe():
    from haru_training.speaker_calibration import fingerprint, preprocessing_recipe, validate_preprocessing
    recipe = preprocessing_recipe('a' * 64)
    spec = dict(vad={'sha256': 'a' * 64}, preprocessing=recipe,
                preprocessing_sha256=fingerprint(recipe))
    validate_preprocessing(spec)
    spec['preprocessing']['rms_target'] = .5
    spec['preprocessing_sha256'] = fingerprint(spec['preprocessing'])
    with pytest.raises(ValueError, match='preprocessing'):
        validate_preprocessing(spec)


def test_control_caches_feed_existing_evaluator_with_bound_repeats(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import build_panels
    caches = calibrated_caches(cohort, spec, tmp_path)
    panels = build_panels(cohort, caches, baseline_id='generated-0',
                          candidate_id='genuine_audit-0', root=tmp_path)
    assert len(panels) == 2
    assert len(panels[0]['references']) == 6
    assert len(panels[0]['genuine_controls']) == 3
    assert len(panels[0]['baseline_repeats']) == 3
    assert all(row['wav_sha256'] == cohort['records'][-1]['wav_sha256']
               for row in panels[0]['baseline_repeats'])
    caches[1]['records'][0]['status'] = 'unknown'
    caches[1]['records'][0]['reason'] = 'insufficient speech'
    with pytest.raises(ValueError, match='controls'):
        build_panels(cohort, caches, baseline_id='generated-0',
                     candidate_id='genuine_audit-0', root=tmp_path)


def test_unreadable_audio_is_unknown_without_disappearing(cohort, spec, tmp_path):
    path = tmp_path / cohort['records'][0]['path']
    path.write_bytes(b'not a WAV')
    cohort['records'][0]['wav_sha256'] = digest(path)
    cache = extract(cohort, spec, tmp_path)
    assert len(cache['records']) == len(cohort['records'])
    assert cache['records'][0]['status'] == 'unknown'
    assert cache['records'][1]['status'] == 'measured'


def test_unknown_audit_stays_in_its_denominator(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import assess_controls
    caches = calibrated_caches(cohort, spec, tmp_path)
    row = next(r for r in caches[0]['records'] if r['id'] == 'genuine_audit-0')
    row.update(status='unknown', reason='insufficient speech')
    report = assess_controls(cohort, caches, root=tmp_path)
    assert report['audit'] == dict(expected=6, correct=5, unknown=1, incorrect=0)
    assert report['status'] == 'controls_failed'


def test_selective_screening_retains_abstentions_but_blocks_wrong_consensus(cohort, spec, tmp_path):
    from haru_training.speaker_calibration import assess_controls, build_panels
    for role in ('genuine_audit', 'negative_audit'):
        template = next(r for r in cohort['records'] if r['role'] == role)
        for index in range(3, 6):
            name = f'{role}-{index}'
            path = tmp_path / (name + '.wav')
            sf.write(path, np.full(24000, .5 + len(cohort['records']) * .01), 24000, subtype='FLOAT')
            cohort['records'].append(dict(template, id=name, path=path.name,
                                          source_id=name, wav_sha256=digest(path)))
    caches = calibrated_caches(cohort, spec, tmp_path)
    row = next(r for r in caches[0]['records'] if r['id'] == 'negative_audit-5')
    row['embedding'] = [1., 0.]
    report = assess_controls(cohort, caches, root=tmp_path)
    assert report['status'] == 'controls_failed'  # Full-coverage separation still failed.
    assert report['audit'] == dict(expected=12, correct=11, unknown=1, incorrect=0)
    assert report['screening_status'] == 'supported_with_abstentions'
    panels = build_panels(cohort, caches, baseline_id='generated-0',
                          candidate_id='genuine_audit-0', root=tmp_path)
    assert panels[0]['control_audit']['unknown'] == 1
    row = next(r for r in caches[1]['records'] if r['id'] == 'negative_audit-5')
    row['embedding'] = [1., 0.]
    report = assess_controls(cohort, caches, root=tmp_path)
    assert report['screening_status'] == 'unsupported'
    assert report['audit']['incorrect'] == 1
    with pytest.raises(ValueError, match='controls'):
        build_panels(cohort, caches, baseline_id='generated-0',
                     candidate_id='genuine_audit-0', root=tmp_path)
