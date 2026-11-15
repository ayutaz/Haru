from copy import deepcopy

import pytest
import torch

from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.model import training_losses
from haru_training.direct_speaker import NativeSpeakerLoss, native_teacher_hidden
from test_training_forward import batch, tiny_model


def model():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    result = tiny_model()
    configure_trainable(result, TrainingConfig(mode='head_output'))
    return result


def test_known_past_uses_native_prompt_prefill_then_single_frames_without_padding(monkeypatch):
    current = model(); values = batch(); calls = []; embedded = []
    forward = current.lm.forward
    def observe(*args, **kwargs):
        calls.append(kwargs['inputs_embeds'].shape[1])
        assert kwargs['use_cache'] is True
        return forward(*args, **kwargs)
    monkeypatch.setattr(current.lm, 'forward', observe)
    original_embed = current.embed
    def embed(*args, **kwargs):
        embedded.append(args[0].shape[1])
        return original_embed(*args, **kwargs)
    monkeypatch.setattr(current, 'embed', embed)
    hidden = native_teacher_hidden(current, values)
    assert calls == [4, 1, 1, 3, 1]
    assert embedded == calls
    assert hidden.shape == (2, values['tokens'].shape[1], current.config.hidden_size)
    assert not hidden.requires_grad
    assert torch.count_nonzero(hidden[1, 4:]) == 0


def test_native_cached_loss_reuses_only_identical_past_and_still_updates_the_head(monkeypatch):
    current = model(); values = batch(); objective = NativeSpeakerLoss()
    calls = []
    original = current.forward
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(current, 'forward', count)
    first = objective(current, values, seed=7)
    count_before = len(calls)
    second = objective(current, values, seed=8)
    assert len(calls) == count_before
    assert not second['text_loss'].requires_grad
    second['audio_loss'].backward()
    assert current.head.final.proj.weight.grad.abs().sum() > 0
    assert not any(p.grad is not None for p in current.lm.parameters())
    changed = deepcopy(values); changed['tokens'][0, 0] = 17
    objective(current, changed, seed=7)
    assert len(calls) > count_before


def test_condition_cache_rejects_in_place_changes_to_its_frozen_backbone():
    current = model(); objective = NativeSpeakerLoss(); values = batch()
    objective(current, values, seed=7)
    with torch.no_grad():next(current.lm.parameters()).add_(.1)
    with pytest.raises(ValueError, match='changed'):
        objective(current, values, seed=7)


def test_recorded_future_target_does_not_enter_native_conditions():
    current = model(); values = batch()
    before = native_teacher_hidden(current, values)
    values['audio_targets'][0, -1] += 100
    after = native_teacher_hidden(current, values)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_frozen_native_conditions_cannot_be_used_to_skip_a_trainable_lm():
    current = model(); values = batch()
    hidden = native_teacher_hidden(current, values)
    next(current.lm.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):native_teacher_hidden(current, values)
    with pytest.raises(ValueError):training_losses(current, values, hidden=hidden)
