from dataclasses import replace

import pytest
import torch

from haru_training.checkpoint import read_checkpoint
from haru_training.direct_speaker import (
    direct_head_spec, direct_spec, run_direct_head_pilot, run_direct_pilot,
)
from test_direct_speaker import records
from test_training_forward import tiny_model


def head_spec(train, validation, **kwargs):
    return direct_head_spec(train, validation, split_sha256='a'*64,
                            cache_sha256='b'*64, **kwargs)


def test_head_condition_changes_only_scope_from_the_output_condition():
    train, validation = records('t'), records('v')
    base = direct_spec(train, validation, split_sha256='a'*64, cache_sha256='b'*64)
    trial = head_spec(train, validation)
    assert trial == replace(base, mode='head')
    assert trial.optimizer_precision == 'fp32_master'
    with pytest.raises(ValueError):
        head_spec(train, validation, max_steps=101)


@pytest.mark.parametrize('change', [
    dict(mode='head_output'), dict(learning_rate=1e-5), dict(seed=7),
    dict(optimizer_precision='native'), dict(train_projectors=True),
    dict(feedback_mode='separate_frozen_head'), dict(max_steps=101),
    dict(max_seconds=601), dict(condition_dropout=0), dict(guidance_max=1),
    dict(weight_decay=.01), dict(warmup_steps=1), dict(micro_batch_size=2),
    dict(accumulation_steps=2),
    dict(preservation_weight=1.),
])
def test_head_trial_rejects_unregistered_changes_before_creating_outputs(tmp_path, change):
    train, validation = records('t'), records('v')
    trial = replace(head_spec(train, validation), **change)
    with pytest.raises(ValueError):
        run_direct_head_pilot(tiny_model(), train, validation, trial, tmp_path/'run',
                              render=lambda *args: {})
    assert not (tmp_path/'run').exists()


def test_legacy_output_runner_does_not_accept_the_new_head_condition(tmp_path):
    train, validation = records('t'), records('v')
    with pytest.raises(ValueError):
        run_direct_pilot(tiny_model(), train, validation, head_spec(train, validation),
                         tmp_path/'run', render=lambda *args: {})
    assert not (tmp_path/'run').exists()


def test_native_head_trial_updates_conditioning_and_internal_layers_and_exports(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = tiny_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train, validation = records('t'), records('v')
    trial = replace(head_spec(train, validation, max_steps=2),
                    base_model='test/tiny-haru', base_revision='a'*40)
    seen = []
    def render(current, folder, step):
        assert not current.training and not any(p.requires_grad for p in current.parameters())
        seen.append(step)
        return dict(voice_quality='unknown')
    result = run_direct_head_pilot(model, train, validation, trial, tmp_path/'run', render=render)
    assert result['complete'] and result['completed_updates'] == 2
    assert result['frozen_weights_unchanged'] and result['adaptation_scope'] == 'head'
    assert result['voice_quality'] == 'unknown' and not result['expansion_authorized']
    assert result['feedback_mode'] == 'generated_audio' and not result['final_test_used']
    assert seen == [0, 2]
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert 'head.cond_proj.weight' in changed
    assert any(n.startswith('head.layers.') for n in changed)
    assert all(n.startswith('head.') for n in changed)
    exported = read_checkpoint(tmp_path/'run/inference/step-00000002')
    assert set(exported['weights']) == {n for n, p in model.named_parameters() if p.requires_grad}
