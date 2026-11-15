from copy import deepcopy
import json

import numpy as np
import pytest
import soundfile as sf

from haru_training.direct_speaker_evaluation import evaluate_direct_manifest
from haru_training.evaluation import evaluate_manifest
from test_evaluation import replay, reseal, dump, speaker_panels, digest, TEXT


@pytest.fixture
def manifest(tmp_path):
    changed = '週末は近所の公園でゆっくり過ごしてみましょう'
    m = dict(version=1, kind='haru-direct-head-evaluation', expected_cases=['one'], cases=[dict(
        id='one', baseline=replay(tmp_path, 'base'),
        candidate=replay(tmp_path, 'adapted', text=changed, recognized=changed))])
    for name in ['base', 'adapted']:
        p = tmp_path/name/'response.json'
        d = json.loads(p.read_text(encoding='utf-8'))
        d['feedback_mode'] = 'generated_audio'
        d['provenance'].update(reference_sha256='b'*64, base_revision='c'*40)
        dump(p, d); reseal(p.parent)
    m['cases'][0]['speaker_panels'] = speaker_panels(m, tmp_path)
    return m


def test_normal_feedback_can_change_words_without_faking_a_same_sentence_comparison(manifest, tmp_path):
    report = evaluate_direct_manifest(manifest, root=tmp_path)
    row = report['cases'][0]
    assert row['content']['same_generated_text'] is False
    assert row['content']['baseline']['character_error_rate'] == 0
    assert row['content']['candidate']['character_error_rate'] == 0
    assert 'different_generated_text' not in row['reasons']
    assert not report['voice_similarity_confirmed']
    # Preserve the stricter legacy codec-pair contract.
    assert 'different_generated_text' in evaluate_manifest(manifest, root=tmp_path)['cases'][0]['reasons']


def test_each_asr_reference_must_match_its_own_generated_output(manifest, tmp_path):
    manifest['cases'][0]['candidate']['asr']['generated_text'] = TEXT
    report = evaluate_direct_manifest(manifest, root=tmp_path)
    assert report['decision'] == 'hold'
    assert 'asr_reference_mismatch' in report['cases'][0]['reasons']


def test_numeric_speaker_gain_cannot_hide_bad_candidate_articulation(manifest, tmp_path):
    case = manifest['cases'][0]
    case['candidate']['asr']['recognized_text'] = 'あいう'
    report = evaluate_direct_manifest(manifest, root=tmp_path)
    assert report['decision'] == 'hold'
    assert 'candidate_asr_risk' in report['cases'][0]['reasons']
    assert report['cases'][0]['content']['candidate']['character_error_rate'] > .5


@pytest.mark.parametrize('field,value', [('feedback_mode', 'separate_frozen_head'),
    ('reference_sha256', 'd'*64), ('input_sha256', 'e'*64)])
def test_changed_conversation_conditions_are_not_compared(manifest, tmp_path, field, value):
    p = tmp_path/'adapted/response.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    if field == 'feedback_mode': d[field] = value
    else: d['provenance'][field] = value
    dump(p, d); reseal(p.parent)
    report = evaluate_direct_manifest(manifest, root=tmp_path)
    assert report['decision'] == 'hold'
    assert 'unpaired_direct_conditions' in report['cases'][0]['reasons']


def test_missing_cases_and_tampered_audio_remain_unknown(manifest, tmp_path):
    manifest['expected_cases'].append('missing')
    (tmp_path/'adapted/response.wav').write_bytes(b'changed')
    report = evaluate_direct_manifest(manifest, root=tmp_path)
    assert report['expected_count'] == 2 and report['evaluated_count'] == 1
    assert [r['decision'] for r in report['cases']] == ['hold', 'hold']


def test_silent_candidate_rejected_before_missing_speaker_cache(manifest, tmp_path):
    sf.write(tmp_path/'adapted/response.wav', np.zeros(24000), 24000, subtype='FLOAT')
    reseal(tmp_path/'adapted')
    manifest['cases'][0].pop('speaker_panels')
    report = evaluate_direct_manifest(manifest, root=tmp_path)
    assert report['decision'] == 'reject'
    assert 'silent_audio' in report['cases'][0]['reasons']


def test_an_exact_human_negative_cannot_become_a_voice_success_from_better_scores(manifest, tmp_path):
    report = evaluate_direct_manifest(manifest, root=tmp_path,
        known_negative_wav_sha256=[digest(tmp_path/'adapted/response.wav')])
    assert report['decision'] == 'hold'
    assert 'known_human_voice_negative' in report['cases'][0]['reasons']
    assert not report['voice_similarity_confirmed']
