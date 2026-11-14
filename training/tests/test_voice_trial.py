import json
from dataclasses import replace

import pytest
import torch

from test_adaptation import trainer_at


def test_voice_trial_finishes_epoch_despite_degraded_conversation(tmp_path, monkeypatch):
    import haru_training.voice_trial as runner
    trainer = trainer_at(tmp_path/'trainer')
    exported = []
    monkeypatch.setattr(runner, 'export_inference', lambda current, path: exported.append(current.state.step))
    def generate(model, path):
        assert not model.training and not model.weight.requires_grad
        path.mkdir()
        torch.rand(7)
        return {'probe': {'text': 'baseline' if trainer.state.step == 0 else '。。。。。。'}}
    result = runner.run_voice_trial(trainer, tmp_path/'trial', generate)
    assert result['completed_epoch'] and result['steps'] == 3
    assert not result['voice_improvement_confirmed']
    assert exported == [3]
    assert result['observations'][-1]['texts']['probe'] == '。。。。。。'
    used = [i for line in (trainer.output/'events.jsonl').read_text().splitlines()
            for i in (json.loads(line).get('record_ids', []) if json.loads(line)['event'] == 'train' else [])]
    assert sorted(used) == sorted(row.record_id for row in trainer.train)
    assert trainer.model.weight.requires_grad


def test_voice_trial_preserves_training_rng_and_weights(tmp_path, monkeypatch):
    import haru_training.voice_trial as runner
    trainer, plain = trainer_at(tmp_path/'trainer'), trainer_at(tmp_path/'plain')
    monkeypatch.setattr(runner, 'export_inference', lambda *args: None)
    def generate(model, path):
        path.mkdir()
        torch.rand(11)
        return {'probe': {'text': 'voice comparison'}}
    torch.manual_seed(91)
    plain.fit()
    expected = torch.get_rng_state().clone()
    torch.manual_seed(91)
    runner.run_voice_trial(trainer, tmp_path/'trial', generate)
    assert torch.equal(torch.get_rng_state(), expected)
    torch.testing.assert_close(trainer.model.weight, plain.model.weight, atol=0, rtol=0)


def test_voice_trial_is_bounded_to_one_fresh_epoch(tmp_path):
    from haru_training.voice_trial import run_voice_trial
    trainer = trainer_at(tmp_path/'trainer')
    trainer.spec = replace(trainer.spec, max_steps=4)
    with pytest.raises(ValueError, match='epoch'):
        run_voice_trial(trainer, tmp_path/'trial', None)
    assert not (tmp_path/'trial').exists()
