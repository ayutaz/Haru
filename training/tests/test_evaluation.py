"""Offline triage tests. Synthetic embeddings test decisions, not voice quality."""
from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf


TEXT = '今日は近くのお店でお昼ご飯を食べましょう'
ASR_SPEC = dict(model='test-asr', revision='fixed', settings={'language': 'ja', 'beam_size': 5})


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def replay(root, name, *, text=TEXT, recognized=TEXT, signal=None):
    folder = root / name
    folder.mkdir()
    if signal is None:
        signal = .1 * np.sin(np.arange(24000) * (.1 if name == 'base' else .11))
    sf.write(folder / 'response.wav', signal, 24000, subtype='FLOAT')
    metadata = dict(text=text, config={'seed': 7, 'temperature': .7},
                    sample_rate=24000, system_prompt='会話してください',
                    provenance={'input_sha256': 'a' * 64})
    dump(folder / 'response.json', metadata)
    reseal(folder)
    cache = dict(ASR_SPEC, generated_text=text, recognized_text=recognized,
                 wav_sha256=digest(folder / 'response.wav'), character_error_rate=0.)
    return dict(directory=name, asr=cache)


def reseal(folder):
    dump(folder / 'manifest.json', {'version': 1, 'files': {
        name: digest(folder / name) for name in ('response.wav', 'response.json')}})


@pytest.fixture
def manifest(tmp_path):
    return {'version': 1, 'expected_cases': ['question-1'], 'cases': [{
        'id': 'question-1', 'baseline': replay(tmp_path, 'base'),
        'candidate': replay(tmp_path, 'adapted')} ]}


def evaluate(manifest, root):
    from haru_training.evaluation import evaluate_manifest
    return evaluate_manifest(manifest, root=root)


def test_same_asr_and_high_raw_cosine_do_not_establish_voice_improvement(manifest, tmp_path):
    manifest['cases'][0]['mean_cosine'] = .99
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert report['cases'][0]['content']['status'] == 'no_detected_regression'
    assert 'speaker_evidence_missing' in report['cases'][0]['reasons']
    assert report['voice_similarity_confirmed'] is False


def test_asr_regression_cannot_be_hidden_by_claimed_cer_or_cosine(manifest, tmp_path):
    case = manifest['cases'][0]
    case['candidate']['asr']['recognized_text'] = '意味の異なる発話です'
    case['candidate']['asr']['character_error_rate'] = 0
    case['mean_cosine'] = .99
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'reject'
    assert report['cases'][0]['content']['candidate_cer'] > .5
    assert 'asr_regression' in report['cases'][0]['reasons']


@pytest.mark.parametrize('signal,reason', [
    (np.zeros(24000), 'silent_audio'),
    (np.full(24000, np.nan), 'nonfinite_audio'),
    (np.tile([1., -1.], 12000), 'clipped_audio'),
])
def test_waveform_failures_are_rejected(manifest, tmp_path, signal, reason):
    folder = tmp_path / 'adapted'
    sf.write(folder / 'response.wav', signal, 24000, subtype='FLOAT')
    reseal(folder)
    manifest['cases'][0]['candidate']['asr']['wav_sha256'] = digest(folder / 'response.wav')
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'reject'
    assert reason in report['cases'][0]['reasons']


def test_tampered_audio_is_not_evaluated(manifest, tmp_path):
    (tmp_path / 'adapted' / 'response.wav').write_bytes(b'changed')
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert 'artifact_hash_mismatch' in report['cases'][0]['reasons']


def test_cached_asr_must_belong_to_actual_waveform(manifest, tmp_path):
    manifest['cases'][0]['candidate']['asr']['wav_sha256'] = 'b' * 64
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert 'asr_audio_mismatch' in report['cases'][0]['reasons']


@pytest.mark.parametrize('mutation,reason', [
    ('model', 'asr_spec_mismatch'), ('text', 'different_generated_text'),
    ('input', 'unpaired_conditions'), ('config', 'unpaired_conditions'),
])
def test_incomparable_pairs_are_held(manifest, tmp_path, mutation, reason):
    case = manifest['cases'][0]
    if mutation == 'model':
        case['candidate']['asr']['revision'] = 'other'
    else:
        path = tmp_path / 'adapted' / 'response.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if mutation == 'text':
            data['text'] = '別の文章です'
            case['candidate']['asr'].update(generated_text=data['text'], recognized_text=data['text'])
        elif mutation == 'input':
            data['provenance']['input_sha256'] = 'c' * 64
        else:
            data['config']['seed'] = 8
        dump(path, data)
        reseal(path.parent)
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert reason in report['cases'][0]['reasons']


def test_short_text_and_bad_baseline_are_unknown_not_a_pass(manifest, tmp_path):
    for side in ('baseline', 'candidate'):
        manifest['cases'][0][side]['asr']['recognized_text'] = ''
    report = evaluate(manifest, tmp_path)
    assert 'baseline_asr_risk' in report['cases'][0]['reasons']
    assert report['decision'] == 'hold'


def speaker_panels(manifest, root):
    """Deliberately separated toy vectors, never a claimed quality calibration."""
    panels = []
    for encoder in range(2):
        refs = []
        for index, angle in enumerate((-.2, 0., .2, 1.8, 2., 2.2)):
            path = root / f'ref-{encoder}-{index}.wav'
            sf.write(path, np.full(24, .01 * (1 + encoder * 6 + index)), 24000, subtype='FLOAT')
            refs.append(dict(path=path.name, wav_sha256=digest(path),
                             source_id=path.stem, speaker_id='target' if index < 3 else f'other-{index}',
                             embedding=[np.cos(angle), np.sin(angle)]))
        panels.append(dict(encoder=f'encoder-{encoder}', revision='fixed',
                           preprocessing_sha256='e' * 64, target_speaker='target',
                           references=refs,
                           baseline=dict(wav_sha256=manifest['cases'][0]['baseline']['asr']['wav_sha256'],
                                         embedding=[np.cos(.8), np.sin(.8)]),
                           candidate=dict(wav_sha256=manifest['cases'][0]['candidate']['asr']['wav_sha256'],
                                          embedding=[np.cos(.15), np.sin(.15)]),
                           baseline_repeats=[dict(
                               wav_sha256=manifest['cases'][0]['baseline']['asr']['wav_sha256'],
                               embedding=[np.cos(angle), np.sin(angle)])
                               for angle in (.8, .801, .799)]))
    return panels


def test_calibrated_cross_encoder_gain_is_only_a_listening_candidate(manifest, tmp_path):
    case = manifest['cases'][0]
    case['speaker_panels'] = speaker_panels(manifest, tmp_path)
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'candidate_for_listening'
    assert report['cases'][0]['speaker']['status'] == 'metric_gain'
    assert report['voice_similarity_confirmed'] is False
    assert report['policy']['calibration_status'] == 'provisional_not_perceptually_validated'


@pytest.mark.parametrize('mutation,reason', [
    ('overlap', 'speaker_calibration_overlap'),
    ('leak', 'speaker_reference_leakage'),
    ('one_encoder', 'independent_encoders_missing'),
    ('disagree', 'speaker_encoder_disagreement'),
    ('noise', 'speaker_gain_within_noise'),
    ('missing_repeat', 'speaker_repeatability_missing'),
    ('ceiling', 'speaker_outside_calibration'),
])
def test_speaker_failure_modes_do_not_become_success(manifest, tmp_path, mutation, reason):
    panels = speaker_panels(manifest, tmp_path)
    manifest['cases'][0]['speaker_panels'] = panels
    if mutation == 'overlap':
        for ref in panels[0]['references'][3:]:
            ref['embedding'] = panels[0]['references'][0]['embedding']
    elif mutation == 'leak':
        ref = panels[0]['references'][0]
        ref.update(path='adapted/response.wav', wav_sha256=panels[0]['candidate']['wav_sha256'])
    elif mutation == 'one_encoder':
        panels.pop()
    elif mutation == 'disagree':
        panels[1]['candidate']['embedding'] = [np.cos(1.), np.sin(1.)]
    elif mutation == 'noise':
        for panel in panels:
            panel['baseline_repeats'][0]['embedding'] = [np.cos(1.6), np.sin(1.6)]
    elif mutation == 'missing_repeat':
        panels[0].pop('baseline_repeats')
    else:
        panels[0]['candidate']['embedding'] = [1., 0.]
    report = evaluate(manifest, tmp_path)
    assert report['decision'] != 'candidate_for_listening'
    assert reason in report['cases'][0]['reasons']


def test_no_gain_is_automatically_identified(manifest, tmp_path):
    panels = speaker_panels(manifest, tmp_path)
    for panel in panels:
        panel['candidate']['embedding'] = panel['baseline']['embedding']
    manifest['cases'][0]['speaker_panels'] = panels
    assert evaluate(manifest, tmp_path)['decision'] == 'no_metric_gain'


def test_missing_cases_and_single_failed_case_cannot_hide_in_average(manifest, tmp_path):
    manifest['expected_cases'].append('question-2')
    report = evaluate(manifest, tmp_path)
    assert report['expected_count'] == 2
    assert report['evaluated_count'] == 1
    assert report['counts']['hold'] == 2
    assert report['cases'][1]['reasons'] == ['missing_case']
    manifest['cases'][0]['candidate']['asr']['recognized_text'] = ''
    assert evaluate(manifest, tmp_path)['decision'] == 'reject'


def test_duplicate_or_unplanned_cases_are_manifest_errors(manifest, tmp_path):
    manifest['cases'].append(deepcopy(manifest['cases'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        evaluate(manifest, tmp_path)
    manifest['cases'][1]['id'] = 'unplanned'
    with pytest.raises(ValueError, match='unplanned'):
        evaluate(manifest, tmp_path)


def test_cli_writes_strict_json_and_exits_nonzero_for_rejection(manifest, tmp_path):
    manifest['cases'][0]['candidate']['asr']['recognized_text'] = ''
    source, output = tmp_path / 'evaluation.json', tmp_path / 'report.json'
    dump(source, manifest)
    result = subprocess.run([sys.executable, '-m', 'haru_training.evaluation', str(source),
                             '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['decision'] == 'reject'
    assert report['manifest_sha256'] == digest(source)
    assert report['voice_similarity_confirmed'] is False


def test_invalid_speaker_cache_cannot_mask_a_known_content_regression(manifest, tmp_path):
    panels = speaker_panels(manifest, tmp_path)
    panels[0]['baseline']['embedding'] = [0., 0.]
    case = manifest['cases'][0]
    case['speaker_panels'] = panels
    case['candidate']['asr']['recognized_text'] = ''
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'reject'
    assert 'asr_regression' in report['cases'][0]['reasons']
    assert 'invalid_speaker_evidence' in report['cases'][0]['reasons']


@pytest.mark.parametrize('mutation,reason', [
    ('few_references', 'speaker_references_insufficient'),
    ('duplicate_encoder', 'independent_encoders_missing'),
    ('wrong_audio', 'speaker_audio_mismatch'),
    ('repeat_audio', 'speaker_repeatability_audio_mismatch'),
    ('changed_reference', 'speaker_reference_hash_mismatch'),
    ('source_overlap', 'speaker_reference_leakage'),
])
def test_speaker_provenance_is_required(manifest, tmp_path, mutation, reason):
    panels = speaker_panels(manifest, tmp_path)
    manifest['cases'][0]['speaker_panels'] = panels
    if mutation == 'few_references':
        panels[0]['references'].pop()
    elif mutation == 'duplicate_encoder':
        panels[1]['encoder'] = panels[0]['encoder']
    elif mutation == 'wrong_audio':
        panels[0]['candidate']['wav_sha256'] = 'b' * 64
    elif mutation == 'repeat_audio':
        panels[0]['baseline_repeats'][0]['wav_sha256'] = 'b' * 64
    elif mutation == 'source_overlap':
        panels[0]['references'][1]['source_id'] = panels[0]['references'][0]['source_id']
    else:
        panels[0]['references'][0]['wav_sha256'] = 'b' * 64
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert reason in report['cases'][0]['reasons']


def test_unrecorded_asr_settings_do_not_prevent_diagnostic_rejection(manifest, tmp_path):
    case = manifest['cases'][0]
    for side in ('baseline', 'candidate'):
        case[side]['asr'].pop('settings')
    case['candidate']['asr']['recognized_text'] = ''
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'reject'
    assert 'asr_settings_unrecorded' in report['cases'][0]['reasons']


def test_very_short_asr_reference_stays_unknown(tmp_path):
    manifest = dict(version=1, expected_cases=['short'], cases=[dict(
        id='short', baseline=replay(tmp_path, 'base', text='はい', recognized='はい'),
        candidate=replay(tmp_path, 'adapted', text='はい', recognized=''))])
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert 'asr_reference_too_short' in report['cases'][0]['reasons']


def test_one_failed_case_blocks_a_batch_of_metric_gains(manifest, tmp_path):
    case = manifest['cases'][0]
    case['speaker_panels'] = speaker_panels(manifest, tmp_path)
    bad = deepcopy(case)
    bad['id'] = 'bad'
    bad['candidate']['asr']['recognized_text'] = ''
    manifest['expected_cases'].append('bad')
    manifest['cases'].append(bad)
    report = evaluate(manifest, tmp_path)
    assert report['counts']['candidate_for_listening'] == 1
    assert report['counts']['reject'] == 1
    assert report['decision'] == 'reject'


def test_declared_different_sample_rates_do_not_form_a_controlled_pair(manifest, tmp_path):
    folder = tmp_path / 'adapted'
    audio, _ = sf.read(folder / 'response.wav')
    sf.write(folder / 'response.wav', audio, 16000, subtype='FLOAT')
    metadata = json.loads((folder / 'response.json').read_text(encoding='utf-8'))
    metadata['sample_rate'] = 16000
    dump(folder / 'response.json', metadata)
    reseal(folder)
    manifest['cases'][0]['candidate']['asr']['wav_sha256'] = digest(folder / 'response.wav')
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert 'unpaired_audio_format' in report['cases'][0]['reasons']


def test_unpinned_asr_is_diagnostic_only_and_prompted_asr_is_not_a_pass(manifest, tmp_path):
    case = manifest['cases'][0]
    case['candidate']['asr'].pop('revision')
    report = evaluate(manifest, tmp_path)
    assert report['cases'][0]['content']['candidate_cer'] == 0
    assert report['decision'] == 'hold'
    case['candidate']['asr']['revision'] = 'fixed'
    for side in ('baseline', 'candidate'):
        case[side]['asr']['settings'] = dict(ASR_SPEC['settings'], initial_prompt=TEXT)
    case['speaker_panels'] = speaker_panels(manifest, tmp_path)
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert 'asr_prompt_leakage' in report['cases'][0]['reasons']


@pytest.mark.parametrize('panels', [[None], {'encoder': 'not-a-list'}])
def test_malformed_speaker_panel_does_not_hide_content_rejection(manifest, tmp_path, panels):
    case = manifest['cases'][0]
    case['candidate']['asr']['recognized_text'] = ''
    case['speaker_panels'] = panels
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'reject'
    assert 'invalid_speaker_evidence' in report['cases'][0]['reasons']


def test_three_added_errors_at_policy_boundary_are_not_lost_to_roundoff(tmp_path):
    text = 'あいうえおかきくけこさしすせそたちつてと'
    manifest = dict(version=1, expected_cases=['boundary'], cases=[dict(
        id='boundary', baseline=replay(tmp_path, 'base', text=text, recognized='XXXX' + text[4:]),
        candidate=replay(tmp_path, 'adapted', text=text, recognized='XXXXXXX' + text[7:]))])
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'reject'
    assert report['cases'][0]['content']['added_character_errors'] == 3


def add_genuine_controls(panels, root):
    for index, panel in enumerate(panels):
        controls = []
        for number, angle in enumerate((-.15, .05, .15)):
            path = root / f'genuine-{index}-{number}.wav'
            sf.write(path, np.full(30, .2 + .01 * (index * 3 + number)), 24000, subtype='FLOAT')
            controls.append(dict(path=path.name, wav_sha256=digest(path), source_id=path.stem,
                                 speaker_id='target', embedding=[np.cos(angle), np.sin(angle)]))
        panel['genuine_controls'] = controls


def test_cross_utterance_calibration_does_not_reject_genuine_above_reference_max(manifest, tmp_path):
    panels = speaker_panels(manifest, tmp_path)
    for panel in panels:
        panel['candidate']['embedding'] = [1., 0.]
    add_genuine_controls(panels, tmp_path)
    manifest['cases'][0]['speaker_panels'] = panels
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'candidate_for_listening'
    assert report['voice_similarity_confirmed'] is False
    row = report['cases'][0]['speaker']['encoders'][0]
    assert row['calibration_method'] == 'cross_utterance_controls'
    assert row['outside_observed_genuine_range'] is True
    assert row['candidate_cosine'] == pytest.approx(np.mean([np.cos(.2), 1., np.cos(.2)]))


def test_extra_genuine_controls_cannot_leak_candidate_audio(manifest, tmp_path):
    panels = speaker_panels(manifest, tmp_path)
    add_genuine_controls(panels, tmp_path)
    panels[0]['genuine_controls'][0].update(
        path='adapted/response.wav', wav_sha256=panels[0]['candidate']['wav_sha256'])
    manifest['cases'][0]['speaker_panels'] = panels
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'hold'
    assert 'speaker_reference_leakage' in report['cases'][0]['reasons']


def test_calibrated_gain_uses_encoder_dynamic_range_not_fixed_raw_cosine(manifest, tmp_path):
    panels = speaker_panels(manifest, tmp_path)
    add_genuine_controls(panels, tmp_path)
    for panel in panels:
        for ref in panel['references'][3:]:
            ref['embedding'] = [np.cos(.7), np.sin(.7)]
        panel['baseline']['embedding'] = [np.cos(.22), np.sin(.22)]
        panel['baseline_repeats'] = [dict(wav_sha256=panel['baseline']['wav_sha256'],
                                         embedding=[np.cos(angle), np.sin(angle)])
                                     for angle in (.22, .2201, .2199)]
    manifest['cases'][0]['speaker_panels'] = panels
    report = evaluate(manifest, tmp_path)
    assert report['decision'] == 'candidate_for_listening'
    assert report['cases'][0]['speaker']['encoders'][0]['gain'] < .02
    assert report['cases'][0]['speaker']['encoders'][0]['gain_in_calibration_units'] > .02
