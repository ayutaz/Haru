import json
import random

import numpy as np
import pytest
import torch

from test_trainer import build
from haru_training.diagnostics import StepDiagnostics, preserved_inference, diagnose
from haru_training.trainer import capture_rng


def test_optimizer_diagnostics_measure_actual_gradient_and_update(tmp_path):
    trainer = build(tmp_path)
    before = trainer.model.weight.detach().clone()
    with StepDiagnostics(trainer) as diagnostics:
        trainer.fit(until_step=1)
    row = diagnostics.updates[0]
    assert row['step'] == 1
    stats = row['components']['weight']
    assert stats['parameters'] == 1 and stats['gradient_parameters'] == 1
    assert stats['gradient_l2'] > 0
    assert stats['update_l2'] == pytest.approx(float(abs(trainer.model.weight.detach()-before)))
    assert 'frozen' not in row['components']
    assert len(trainer.optimizer._optimizer_step_pre_hooks) == 0
    assert len(trainer.optimizer._optimizer_step_post_hooks) == 0


def test_observation_restores_rng_modes_even_on_failure(tmp_path):
    trainer = build(tmp_path)
    trainer.model.train()
    rng = capture_rng(trainer.model)
    with pytest.raises(RuntimeError, match='observer'):
        with preserved_inference(trainer.model):
            assert not trainer.model.training and not torch.is_grad_enabled()
            assert not any(p.requires_grad for p in trainer.model.parameters())
            random.random(); np.random.random(); torch.rand(2)
            raise RuntimeError('observer')
    after = capture_rng(trainer.model)
    assert trainer.model.training
    assert trainer.model.weight.requires_grad and not trainer.model.frozen.requires_grad
    for key in ('python', 'numpy', 'cuda'):
        assert after[key] == rng[key]
    assert torch.equal(after['torch'], rng['torch'])


def test_step_zero_and_each_update_observed_without_changing_training(tmp_path):
    plain = build(tmp_path/'plain')
    plain.fit()
    observed = build(tmp_path/'observed')
    steps = []
    def observer(trainer, metrics):
        steps.append(trainer.state.step)
        random.random(); np.random.random(); torch.rand(3)
        assert set(metrics) >= {'text_loss', 'audio_loss', 'text_count', 'audio_count'}
    diagnose(observed, observer)
    assert steps == [0, 1, 2, 3]
    assert observed.state == plain.state
    assert torch.equal(observed.model.weight, plain.model.weight)
    events = [json.loads(s) for s in (observed.output/'events.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events if e['event'] == 'diagnostic_update'] == [1, 2, 3]


def test_resume_observes_current_boundary_without_repeating_updates(tmp_path):
    trainer = build(tmp_path/'first')
    trainer.fit(until_step=1)
    checkpoint = trainer.save()
    resumed = build(tmp_path/'resumed')
    resumed.resume(checkpoint)
    steps = []
    diagnose(resumed, lambda trainer, metrics: steps.append(trainer.state.step))
    assert steps == [1, 2, 3]
    assert resumed.state.step == 3


def test_separate_loss_gradients_measure_both_signals_without_changing_update(tmp_path):
    plain = build(tmp_path/'plain')
    measured = build(tmp_path/'measured')
    plain.fit(until_step=1)
    with StepDiagnostics(measured, loss_gradients=True) as diagnostics:
        measured.fit(until_step=1)
    assert torch.equal(measured.model.weight, plain.model.weight)
    row = diagnostics.loss_gradients[0]
    assert row['step'] == 1
    assert row['seed'] == measured.spec.seed
    assert row['text_count'] >= 0 and row['audio_count'] > 0
    assert set(row['losses']) == {'text_loss', 'audio_loss'}
    assert row['losses']['audio_loss']['weight']['l2'] > 0
    assert row['losses']['text_loss']['weight']['l2'] >= 0


def test_random_observer_cannot_change_stochastic_training_or_next_rng(tmp_path):
    plain = build(tmp_path/'plain')
    observed = build(tmp_path/'observed')
    original = plain.loss_fn
    def stochastic(model, batch, *, seed):
        result = original(model, batch, seed=seed)
        result['audio_loss'] = result['audio_loss'] * (torch.rand(()) + random.random() + np.random.random())
        return result
    plain.loss_fn = observed.loss_fn = stochastic
    initial = capture_rng(plain.model)
    from haru_training.trainer import restore_rng
    plain.fit()
    expected_rng = capture_rng(plain.model)
    restore_rng(initial)
    def observer(trainer, metrics):
        random.random(); np.random.random(); torch.rand(9)
    diagnose(observed, observer)
    assert torch.equal(plain.model.weight, observed.model.weight)
    assert torch.equal(expected_rng['torch'], capture_rng(observed.model)['torch'])


@pytest.mark.parametrize('mode', ['head', 'lm_lora'])
def test_real_tiny_haru_loss_gradient_measurement_preserves_update(tmp_path, mode):
    from test_phase4_haru import tiny_trainer
    plain = tiny_trainer(tmp_path/'plain', mode)
    plain.fit(until_step=1)
    measured = tiny_trainer(tmp_path/'measured', mode)
    with StepDiagnostics(measured, loss_gradients=True) as diagnostics:
        measured.fit(until_step=1)
    for (name, value), (other, result) in zip(plain.model.named_parameters(), measured.model.named_parameters()):
        assert name == other and torch.equal(value, result)
    assert diagnostics.loss_gradients
    for row in diagnostics.loss_gradients:
        assert row['losses']['audio_loss']['head']['l2'] > 0
        if mode == 'lm_lora':
            assert row['losses']['text_loss']['lm']['l2'] > 0


def test_zero_update_lora_preserves_original_logits_and_head_output():
    from test_training_forward import tiny_model, batch
    from haru_training.contracts import TrainingConfig, configure_trainable
    torch.set_num_threads(1)
    torch.manual_seed(57)
    model = tiny_model()
    data = batch()
    def probe():
        with preserved_inference(model):
            keys = ('tokens', 'model_latents', 'user_latents', 'model_mask', 'user_mask')
            embeddings = model.embed(*(data[k] for k in keys))
            hidden = model.lm(inputs_embeds=embeddings, attention_mask=data['attention_mask'],
                              use_cache=False, return_dict=True).last_hidden_state
            noise = torch.zeros(2, model.config.latent_dim)
            return model.logits(hidden), model.head.sample(hidden[:, -1], noise, 1.5)
    before = probe()
    configure_trainable(model, TrainingConfig(mode='lm_lora'))
    after = probe()
    for original, wrapped in zip(before, after):
        assert torch.equal(original, wrapped)


def test_failed_quality_gate_stops_at_the_observed_boundary(tmp_path):
    trainer = build(tmp_path)
    steps = []
    def observer(trainer, metrics):
        steps.append(trainer.state.step)
        return trainer.state.step < 1
    updates = diagnose(trainer, observer)
    assert steps == [0, 1] and trainer.state.step == 1
    assert len(updates) == 1 and not trainer.failed
