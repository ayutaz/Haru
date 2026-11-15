from dataclasses import replace

import pytest
import torch

from haru_training.direct_speaker import direct_spec, run_direct_pilot, validate_membership
from haru_training.data import make_sequence
from test_training_forward import tiny_model


def splits():
    return dict(train=['t0', 't1', 't2', 't3'], validation=['v0', 'v1', 'v2'],
                test=['x0'], reference=['r0'])


@pytest.mark.parametrize('change', [
    dict(train=['t0', 'v0', 't2']), dict(validation=['v0', 'v1', 't2']),
    dict(reference='t0'), dict(train=['t0', 't0', 't2']), dict(train=['t0']),
])
def test_direct_pilot_rejects_evaluation_leakage_or_missing_pilot_data(change):
    args = dict(train=['t0', 't1', 't2'], validation=['v0', 'v1', 'v2'], reference='r0')
    args.update(change)
    with pytest.raises(ValueError):
        validate_membership(splits(), **args)


def records(prefix):
    return [make_sequence(f'{prefix}{i}', [3], torch.ones(1, 64),
                          torch.ones(2, 64) * .1, torch.zeros(2, 64),
                          torch.tensor([4, 0]), 0, 1) for i in range(3)]


def spec(train, val, **kwargs):
    return direct_spec(train, val, split_sha256='a'*64, cache_sha256='b'*64, **kwargs)


def test_spec_is_only_the_bounded_direct_head_output_trial():
    result = spec(records('t'), records('v'))
    assert result.mode == 'head_output'
    assert result.feedback_mode == 'generated_audio'
    assert result.learning_rate == 1e-6
    assert result.max_steps == 100 and result.max_seconds == 600
    with pytest.raises(ValueError):
        spec(records('t'), records('v'), max_steps=101)
    with pytest.raises(ValueError):
        spec(records('t'), records('v'), max_seconds=601)


def test_real_tiny_run_exports_terminal_weights_without_claiming_voice_quality(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = tiny_model()
    train, validation = records('t'), records('v')
    run_spec = spec(train, validation, max_steps=2)
    # A tiny real architecture uses a test base identity; production runner
    # verifies its pinned model before entering this same training function.
    run_spec = replace(run_spec, base_model='test/tiny-haru', base_revision='a'*40)
    seen = []
    def render(current, folder, step):
        assert not current.training and not any(p.requires_grad for p in current.parameters())
        seen.append(step)
        return {'cases': ['one'], 'voice_quality': 'unknown'}
    result = run_direct_pilot(model, train, validation, run_spec, tmp_path/'run', render=render)
    assert seen == [0, 2]
    assert result['completed_updates'] == 2
    assert result['frozen_weights_unchanged']
    assert result['voice_quality'] == 'unknown'
    assert not result['expansion_authorized']
    assert not result['final_test_used']
    assert (tmp_path/'run/inference/step-00000002/manifest.json').is_file()
