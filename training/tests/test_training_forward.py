import json
from pathlib import Path

import pytest
import torch

from haru.config import HaruConfig
from haru.models.haru import HaruModel
from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.data import collate_sequences, make_sequence
from haru_training.model import training_losses


def tiny_model():
    config = json.loads((Path(__file__).parents[2] / 'haru_mlx/tests/parity/tiny_config.json').read_text())
    config['lm']['vocab_size'] = 128
    config['pad_token_id'], config['eop_token_id'] = 0, 1
    config['lm']['eos_token_id'] = 2
    model = HaruModel(HaruConfig.from_dict(config))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.normal_(0, .03)
            if name.endswith(('gamma', 'norm.weight')):
                parameter.add_(1)
    return model


def batch():
    return collate_sequences([make_sequence('first', [3, 4], torch.randn(2, 64),
        torch.randn(3, 64), torch.randn(3, 64), torch.tensor([5, 6, 0]), 0, 1),
        make_sequence('second', [3], torch.randn(2, 64), torch.randn(2, 64),
                      torch.randn(2, 64), torch.tensor([7, 0]), 0, 1)], 0)


@pytest.mark.parametrize('mode', ['head', 'lm_lora'])
def test_losses_reach_selected_modules_and_frozen_weights_do_not_change(mode):
    torch.manual_seed(4)
    model = tiny_model()
    summary = configure_trainable(model, TrainingConfig(mode=mode, train_projectors=mode == 'lm_lora'))
    assert summary['trainable'] > 0
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    data = batch()
    result = training_losses(model, data, seed=12)
    assert all(torch.isfinite(result[k]) for k in ('loss', 'text_loss', 'audio_loss'))
    result['loss'].backward()
    gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    assert any(n.startswith('head.') and g.abs().sum() > 0 for n, g in gradients.items())
    if mode == 'lm_lora':
        assert any('lora_' in n and g.abs().sum() > 0 for n, g in gradients.items())
        assert any('audio_proj' in n and g.abs().sum() > 0 for n, g in gradients.items())
    assert not any(n.startswith('codec.') for n in gradients)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed and all(dict(model.named_parameters())[n].requires_grad for n in changed)
    assert not model.codec.training


def test_padding_cannot_change_valid_hidden_states_and_sampling_is_seeded():
    torch.manual_seed(4)
    model = tiny_model()
    configure_trainable(model, TrainingConfig())
    data = batch()
    first = training_losses(model, data, seed=7)
    changed = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
    changed['model_latents'][~data['attention_mask']] = 100
    changed['user_latents'][~data['attention_mask']] = -100
    second = training_losses(model, changed, seed=7)
    torch.testing.assert_close(first['loss'], second['loss'])
    torch.testing.assert_close(first['hidden'][data['attention_mask']], second['hidden'][data['attention_mask']])


def test_future_frame_cannot_change_earlier_hidden_state():
    torch.manual_seed(4)
    model = tiny_model()
    configure_trainable(model, TrainingConfig())
    data = batch()
    before = training_losses(model, data, seed=7)['hidden']
    data['model_latents'][0, -1] += 100
    after = training_losses(model, data, seed=7)['hidden']
    torch.testing.assert_close(before[0, :-1], after[0, :-1])


def test_configured_guidance_range_is_used_by_the_real_forward(monkeypatch):
    import haru_training.model as module
    seen=[]
    original=module.meanflow_loss
    def checked(field,data,noise,time,interval,guidance,*args,**kwargs):
        seen.append(guidance.detach())
        return original(field,data,noise,time,interval,guidance,*args,**kwargs)
    monkeypatch.setattr(module,'meanflow_loss',checked)
    model=tiny_model();configure_trainable(model,TrainingConfig())
    training_losses(model,batch(),seed=7,guidance_max=1.)
    assert seen and all((v==1).all() for v in seen)
