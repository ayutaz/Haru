import json

import numpy as np
import pytest
import torch

from test_trainer import build
from haru_training.trials import condition_config, assess_response, run_measured_trial


def test_condition_matrix_changes_only_intended_update_groups():
    head, lora, projectors = [condition_config(k) for k in ('B', 'C', 'D')]
    assert head.mode == 'head' and not head.train_projectors
    assert lora.mode == 'lm_lora' and not lora.train_projectors
    assert projectors.mode == 'lm_lora' and projectors.train_projectors
    assert head.lora_rank == lora.lora_rank == projectors.lora_rank == 8
    with pytest.raises(ValueError): condition_config('unexpected')


def response(text='今日はいい天気ですね'):
    return {'text': text, 'audio': np.ones(3200, np.float32)*.1}


@pytest.mark.parametrize('result,metrics,reason', [
    (response('。。 um ist。um。 。 。。 。  o 。  O'), (3., 1.), 'punctuation'),
    (response('はいはいはいはいはい'), (3., 1.), 'repetition'),
    (response(), (4.1, 1.), 'text_loss'),
    ({'text': 'はい', 'audio': np.zeros(3200)}, (3., 1.), 'silent'),
    ({'text': 'はい', 'audio': np.array([float('nan')])}, (3., 1.), 'audio'),
])
def test_fixed_stop_rules_identify_known_failures(result, metrics, reason):
    reasons = assess_response({'text_loss': 3., 'audio_loss': 1.},
                              dict(zip(('text_loss', 'audio_loss'), metrics)), result)
    assert any(reason in r for r in reasons)


def test_normal_japanese_punctuation_and_english_are_not_automatically_failures():
    base = {'text_loss': 3., 'audio_loss': 1.}
    for text in ('今日は晴れです。明日はどうでしょう？', 'もももすももももものうち', 'OK、了解しました'):
        assert assess_response(base, base, response(text)) == []


def test_trial_records_step_zero_and_stops_before_extending_bad_weights(tmp_path):
    trainer = build(tmp_path/'trainer')
    calls = []
    def generate(model, destination):
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        calls.append(trainer.state.step)
        destination.mkdir()
        return response('はいはいはいはい' if trainer.state.step == 1 else '今日は晴れです')
    result = run_measured_trial(trainer, tmp_path/'trial', generate)
    assert calls == [0, 1] and result['steps'] == 1
    assert not result['candidate'] and result['stopped_for_quality']
    assert len(result['observations']) == 2
    assert (tmp_path/'trial/result.json').is_file()
    assert not (tmp_path/'trial/step-00000002').exists()
    assert len(result['updates']) == 1
    assert not trainer.failed


def test_successful_trial_matches_uninstrumented_updates_and_preserves_records(tmp_path):
    plain = build(tmp_path/'plain')
    plain.fit()
    measured = build(tmp_path/'measured')
    def generate(model, destination):
        destination.mkdir()
        torch.rand(5)
        return response()
    result = run_measured_trial(measured, tmp_path/'trial', generate)
    assert torch.equal(plain.model.weight, measured.model.weight)
    assert [r['step'] for r in result['observations']] == [0, 1, 2, 3]
    assert result['completed'] and not result['stopped_for_quality']
    assert result['candidate'] is False  # A diagnostic success is not voice quality evidence.
    loaded = json.loads((tmp_path/'trial/result.json').read_text(encoding='utf-8'))
    assert loaded['frozen_parameters_unchanged']
    with pytest.raises(FileExistsError):
        run_measured_trial(build(tmp_path/'new'), tmp_path/'trial', generate)
