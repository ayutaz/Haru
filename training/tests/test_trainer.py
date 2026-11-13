from dataclasses import replace
import hashlib
import json

import pytest
import torch

from haru_training.data import collate_sequences, make_sequence
from haru_training.trainer import RunSpec, Trainer, dataset_fingerprint


def records(prefix='train'):
    return [make_sequence(f'{prefix}-{i}', [3], torch.ones(1, 1),
                torch.arange(1, n+1).float()[:, None]/10,
                torch.zeros(n, 1), torch.full((n,), 4), 0, 1)
            for i, n in enumerate([1, 3, 2])]


def spec(train, val, **kwargs):
    return RunSpec(base_model='test/base', base_revision='a'*40,
        dataset_sha256=dataset_fingerprint(train), validation_sha256=dataset_fingerprint(val),
        split_sha256='b'*64, cache_sha256='c'*64, max_steps=3,
        **kwargs)


class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)


def objective(model, batch, *, seed):
    audio = batch['audio_targets'][batch['audio_loss_mask']].double()
    targets = batch['text_targets'] != -100
    text = batch['audio_targets'][targets].double()
    return {'audio_loss': (model.weight-audio).square().mean(),
            'text_loss': (model.weight-2*text).square().mean() if targets.any() else model.weight*0}


def collate(rows):
    batch = collate_sequences(rows, 0)
    # One recording has no transcript, but it still has audio supervision.
    for index, row in enumerate(rows):
        if row.record_id.endswith('-1'):
            batch['text_targets'][index] = -100
    return batch


@pytest.mark.parametrize('kwargs', [dict(max_steps=0), dict(accumulation_steps=0),
    dict(micro_batch_size=0), dict(learning_rate=float('nan')), dict(base_revision='latest'),
    dict(dataset_sha256='unknown'), dict(warmup_steps=-1), dict(checkpoint_interval=-1),
    dict(max_seconds=7201), dict(seed=True)])
def test_run_spec_rejects_ambiguous_or_invalid_contract(kwargs):
    train, val = records(), records('val')
    valid = spec(train, val)
    with pytest.raises(ValueError):
        replace(valid, **kwargs)


def test_dataset_identity_detects_changed_content_and_order():
    first = records()
    before = dataset_fingerprint(first)
    assert before == dataset_fingerprint(records())
    assert before != dataset_fingerprint(list(reversed(first)))
    first[0].model_latents[-1] += .1
    assert before != dataset_fingerprint(first)


def test_trainer_rejects_wrong_dataset_and_split_leakage_before_update(tmp_path):
    train, val = records(), records('val')
    contract = spec(train, val)
    with pytest.raises(ValueError):
        Trainer(Regression(), list(reversed(train)), val, contract, tmp_path,
                collate_fn=collate, loss_fn=objective)
    with pytest.raises(ValueError):
        Trainer(Regression(), train, train, spec(train, train), tmp_path,
                collate_fn=collate, loss_fn=objective)


def build(tmp_path, **kwargs):
    train, val = records(), records('val')
    return Trainer(Regression(), train, val, spec(train, val, **kwargs), tmp_path,
                   collate_fn=collate, loss_fn=objective)


def test_loop_updates_only_selected_weights_and_logs_validation_at_step_limit(tmp_path):
    trainer = build(tmp_path, learning_rate=.01, warmup_steps=2)
    before = trainer.model.weight.detach().clone()
    trainer.fit()
    assert trainer.state.step == 3 and trainer.state.micro_step == 3
    assert trainer.state.epoch == 1 and trainer.state.cursor == 0
    assert trainer.model.weight < before
    assert trainer.model.frozen == 2 and trainer.model.frozen.grad is None
    assert trainer.scheduler.last_epoch == 3
    after = trainer.model.weight.detach().clone()
    trainer.fit()
    torch.testing.assert_close(after, trainer.model.weight, atol=0, rtol=0)
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events if e['event']=='train'] == [1,2,3]
    assert len([e for e in events if e['event']=='validation']) == 3
    assert all(e['text_loss'] >= 0 and e['audio_loss'] >= 0
               for e in events if e['event'] in ('train','validation'))


def test_unequal_transcript_and_audio_counts_accumulate_like_one_effective_batch(tmp_path):
    full = build(tmp_path/'full', micro_batch_size=3, accumulation_steps=1)
    small = build(tmp_path/'small', micro_batch_size=1, accumulation_steps=3)
    full.fit(until_step=1)
    small.fit(until_step=1)
    torch.testing.assert_close(full.model.weight, small.model.weight, atol=1e-12, rtol=1e-12)
    for name in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(full.optimizer.state[full.model.weight][name],
                                   small.optimizer.state[small.model.weight][name], atol=1e-12, rtol=1e-12)


def test_last_partial_accumulation_window_is_updated_without_repeating_records(tmp_path):
    trainer = build(tmp_path, accumulation_steps=2)
    trainer.fit(until_step=2)
    assert trainer.state.micro_step == 3 and trainer.state.epoch == 1
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    ids = [item for event in events if event['event']=='train' for item in event['record_ids']]
    assert len(ids)==3 and set(ids)=={r.record_id for r in records()}


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_nonfinite_loss_stops_without_update_and_requires_explicit_recovery(tmp_path,bad):
    trainer = build(tmp_path)
    before = trainer.model.weight.detach().clone()
    trainer.loss_fn=lambda *a,**k: {'text_loss':trainer.model.weight*bad,'audio_loss':trainer.model.weight*0}
    with pytest.raises(FloatingPointError): trainer.fit()
    assert trainer.state.step==0 and trainer.model.weight.grad is None
    torch.testing.assert_close(trainer.model.weight,before,atol=0,rtol=0)
    assert json.loads((tmp_path/'failure.json').read_text())['step']==0
    with pytest.raises(RuntimeError,match='resume'): trainer.fit()


def test_timeout_returns_at_a_boundary_without_an_extra_update(tmp_path,monkeypatch):
    clock=[10.]
    monkeypatch.setattr('haru_training.trainer.time.monotonic',lambda:clock[0])
    trainer=build(tmp_path,max_seconds=1)
    clock[0]=11.
    trainer.fit()
    assert trainer.state.step==0
    assert json.loads((tmp_path/'events.jsonl').read_text().splitlines()[-1])['reason']=='time_limit'


def test_validation_preserves_all_rng_and_individual_module_modes(tmp_path):
    import random
    import numpy as np
    from haru_training.trainer import capture_rng
    trainer=build(tmp_path)
    trainer.model.eval()
    def noisy(*args,**kwargs):
        random.random();np.random.rand();torch.rand(1)
        return objective(*args,**kwargs)
    trainer.loss_fn=noisy
    before=capture_rng(trainer.model)
    trainer.validate()
    after=capture_rng(trainer.model)
    assert before['python']==after['python'] and before['numpy']==after['numpy']
    torch.testing.assert_close(before['torch'],after['torch'],atol=0,rtol=0)
    assert not trainer.model.training


def test_finite_loss_with_nonfinite_gradient_cannot_update(tmp_path):
    trainer=build(tmp_path)
    before=trainer.model.weight.detach().clone()
    trainer.loss_fn=lambda *a,**k:{'text_loss':torch.sqrt(trainer.model.weight*0),
                                  'audio_loss':trainer.model.weight*0}
    with pytest.raises(FloatingPointError,match='gradient'):trainer.fit()
    torch.testing.assert_close(before,trainer.model.weight,atol=0,rtol=0)


def test_default_haru_forward_receives_dropout_guidance_and_pad_contract(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import haru_training.trainer as module
    seen=[]
    def forward(model,batch,*,seed,condition_dropout,guidance_max):
        seen.append((condition_dropout,guidance_max))
        return objective(model,batch,seed=seed)
    monkeypatch.setattr(module,'training_losses',forward)
    train,val=records(),records('val')
    model=Regression();model.config=SimpleNamespace(pad_token_id=7)
    trainer=Trainer(model,train,val,spec(train,val,condition_dropout=.2,guidance_max=1.),tmp_path,
                    loss_fn=forward)
    batch=trainer.collate(train)
    assert (batch['tokens'][~batch['attention_mask']]==7).all()
    trainer.fit(until_step=1)
    assert seen and all(pair==(.2,1.) for pair in seen)


@pytest.mark.parametrize('kwargs',[{'lora_rank':1.5},{'learning_rate':True},
    {'train_projectors':'yes'},{'schema_version':True}])
def test_config_contract_has_strict_scalar_types(tmp_path,kwargs):
    train,val=records(),records('val')
    with pytest.raises(ValueError):replace(spec(train,val),**kwargs)


def test_mutated_dataset_is_rejected_before_an_update_or_save(tmp_path):
    trainer=build(tmp_path)
    trainer.train[0].model_latents[-1]+=10
    before=trainer.model.weight.detach().clone()
    with pytest.raises(ValueError):trainer.fit()
    with pytest.raises(ValueError):trainer.save()
    torch.testing.assert_close(trainer.model.weight,before,atol=0,rtol=0)
