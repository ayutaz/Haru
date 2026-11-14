from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_trainer import Regression, records, collate, objective, spec
from haru_training.trainer import Trainer


def trainer_at(path):
    train = records()
    val = records('val')
    return Trainer(Regression(), train, val, spec(train, val, learning_rate=.01), path,
                   collate_fn=collate, loss_fn=objective)


def test_adaptation_observes_complete_epoch_and_exports_with_restored_flags(tmp_path, monkeypatch):
    import haru_training.adaptation as runner
    trainer = trainer_at(tmp_path/'out/trainer')
    calls = []
    def export(current, path):
        assert current.model.weight.requires_grad
        path.mkdir(parents=True)
        calls.append(current.state.step)
    def generate(model, path):
        assert not model.training and not model.weight.requires_grad
        path.mkdir()
        return {'probe': {'audio':np.ones(3200,np.float32),'text':'今日は晴れです'}}
    monkeypatch.setattr(runner,'export_inference',export)
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2)
    assert calls==[0,2,3]
    assert result['completed_epoch'] and result['steps']==3
    assert result['candidate_step'] in (2,3)
    assert not result['voice_quality_evaluated']
    rows=[json.loads(x) for x in (trainer.output/'events.jsonl').read_text().splitlines()]
    used=[i for r in rows if r['event']=='train' for i in r['record_ids']]
    assert sorted(used)==sorted(r.record_id for r in trainer.train)
    assert trainer.state.epoch==1
    assert trainer.model.weight.requires_grad


def test_rejected_response_stops_before_any_update_and_cannot_be_candidate(tmp_path, monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *a:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'。。。。。。。。'}}
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2)
    assert result['steps']==0 and result['candidate_step'] is None
    assert result['stopped_for_quality'] and not result['completed_epoch']


def test_early_observation_schedule_stops_at_first_rejected_update(tmp_path,monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *args:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'今日は晴れです'}}
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],observation_steps=[0,1,3],
        boundary_review=lambda row:{'accepted':row['step']==0,'reason':'first update lost input topic'})
    assert result['steps']==1 and [r['step'] for r in result['observations']]==[0,1]
    assert result['candidate_step'] is None


@pytest.mark.parametrize('schedule',[[0,2],[1,3],[0,1,1,3],[0,True,3],[0,4,3]])
def test_invalid_observation_schedule_fails_before_output(tmp_path,schedule):
    from haru_training.adaptation import run_adaptation
    trainer=trainer_at(tmp_path/'trainer')
    with pytest.raises(ValueError,match='schedule'):
        run_adaptation(trainer,tmp_path/'out',None,speech_ids=['val-0','val-2'],
            dialogue_ids=['val-1'],observation_steps=schedule)
    assert not (tmp_path/'out').exists()


def test_bad_intermediate_response_is_preserved_and_not_extended(tmp_path, monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *a:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),
                         'text':'今日は晴れです' if trainer.state.step==0 else '。。。。。。。。'}}
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2)
    assert result['steps']==2 and result['candidate_step'] is None
    assert [row['step'] for row in result['observations']]==[0,2]
    assert (tmp_path/'out/observation-00000002.json').is_file()
    assert not (tmp_path/'out/step-00000003').exists()


def test_observation_rng_is_not_consumed_by_training(tmp_path, monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    plain=trainer_at(tmp_path/'plain')
    monkeypatch.setattr(runner,'export_inference',lambda *a:None)
    def generate(model,path):
        path.mkdir()
        torch.rand(11)
        return {'probe':{'audio':np.ones(3200),'text':'今日は晴れです'}}
    torch.manual_seed(91)
    plain.fit()
    expected=torch.get_rng_state().clone()
    torch.manual_seed(91)
    runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2)
    assert torch.equal(torch.get_rng_state(),expected)
    torch.testing.assert_close(trainer.model.weight,plain.model.weight,atol=0,rtol=0)


def test_validation_roles_and_epoch_limit_are_checked_before_output(tmp_path):
    from haru_training.adaptation import run_adaptation
    trainer=trainer_at(tmp_path/'trainer')
    for speech,dialogue in ((['val-0'],['val-0']),(['train-0'],['val-1']),([],['val-0'])):
        with pytest.raises(ValueError,match='validation'):
            run_adaptation(trainer,tmp_path/'bad',None,speech_ids=speech,dialogue_ids=dialogue)
    trainer.spec=replace(trainer.spec,max_steps=4)
    with pytest.raises(ValueError,match='epoch'):
        run_adaptation(trainer,tmp_path/'bad',None,speech_ids=['val-0','val-2'],dialogue_ids=['val-1'])
    assert not (tmp_path/'bad').exists()


def test_boundary_review_is_saved_and_rejects_before_next_update(tmp_path, monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    exports=[]
    monkeypatch.setattr(runner,'export_inference',lambda trainer,path:exports.append(trainer.state.step))
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'誕生日に友達と会います'}}
    def review(row):
        step=row['step']
        saved=json.loads((tmp_path/f'out/review-request-{step:08d}.json').read_text(encoding='utf-8'))
        assert saved['boundary_review']['status']=='pending'
        assert exports[-1]==step
        assert (trainer.output/f'checkpoints/step-{step:08d}/manifest.json').is_file()
        assert trainer.state.step==step
        # Reviewers receive a copy, not the authoritative metrics/texts.
        row['texts']['probe']='changed by reviewer'
        return {'accepted':step==0,'reason':'baseline accepted' if step==0 else 'food input became birthday reply'}
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2,boundary_review=review)
    assert result['steps']==2 and result['candidate_step'] is None
    assert result['stopped_for_quality'] and not result['completed_epoch']
    assert result['observations'][-1]['boundary_review']=={
        'status':'rejected','reason':'food input became birthday reply'}
    assert result['observations'][-1]['texts']['probe']=='誕生日に友達と会います'
    assert not (tmp_path/'out/observation-00000003.json').exists()


@pytest.mark.parametrize('decision',[None,True,{}, {'accepted':'yes','reason':'ok'},
    {'accepted':True,'reason':''}, {'accepted':False,'reason':None}])
def test_invalid_boundary_review_fails_closed_without_training(tmp_path, monkeypatch,decision):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *args:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'今日は晴れです'}}
    with pytest.raises(ValueError,match='review'):
        runner.run_adaptation(trainer,tmp_path/'out',generate,
            speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],boundary_review=lambda row:decision)
    assert trainer.state.step==0
    assert json.loads((tmp_path/'out/review-request-00000000.json').read_text(encoding='utf-8'))['boundary_review']['status']=='pending'
    assert not (tmp_path/'out/observation-00000000.json').exists()
    assert not (tmp_path/'out/result.json').exists()


def test_boundary_review_preserves_training_rng_and_flags(tmp_path,monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    plain=trainer_at(tmp_path/'plain')
    monkeypatch.setattr(runner,'export_inference',lambda *args:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'今日は晴れです'}}
    def review(row):
        torch.rand(13)
        return {'accepted':True,'reason':'input and reply reviewed'}
    torch.manual_seed(91)
    plain.fit()
    expected=torch.get_rng_state().clone()
    torch.manual_seed(91)
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2,boundary_review=review)
    assert result['completed_epoch']
    assert torch.equal(torch.get_rng_state(),expected)
    torch.testing.assert_close(trainer.model.weight,plain.model.weight,atol=0,rtol=0)
    assert trainer.model.weight.requires_grad


def test_final_boundary_review_rejection_prevents_candidate(tmp_path,monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *args:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'今日は晴れです'}}
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],interval=2,
        boundary_review=lambda row:{'accepted':row['step']<3,'reason':'reviewed each boundary'})
    assert result['completed_epoch'] and result['stopped_for_quality']
    assert result['candidate_step'] is None


def test_review_acceptance_cannot_override_automatic_rejection(tmp_path,monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *args:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'。。。。。。。。'}}
    result=runner.run_adaptation(trainer,tmp_path/'out',generate,
        speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],
        boundary_review=lambda row:{'accepted':True,'reason':'reviewer missed repetition'})
    assert result['steps']==0 and result['stopped_for_quality']
    assert result['candidate_step'] is None


def test_interrupted_review_keeps_request_without_advancing(tmp_path,monkeypatch):
    import haru_training.adaptation as runner
    trainer=trainer_at(tmp_path/'out/trainer')
    monkeypatch.setattr(runner,'export_inference',lambda *args:None)
    def generate(model,path):
        path.mkdir()
        return {'probe':{'audio':np.ones(3200),'text':'今日は晴れです'}}
    def review(row):
        raise TimeoutError('no review decision arrived')
    with pytest.raises(TimeoutError):
        runner.run_adaptation(trainer,tmp_path/'out',generate,
            speech_ids=['val-0','val-2'],dialogue_ids=['val-1'],boundary_review=review)
    assert trainer.state.step==0 and trainer.model.weight.requires_grad
    assert (tmp_path/'out/review-request-00000000.json').is_file()
    assert not (tmp_path/'out/result.json').exists()
