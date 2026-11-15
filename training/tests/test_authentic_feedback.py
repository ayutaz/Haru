from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_content_preservation import fixture
from test_preserving_pilot import records
from haru_training.content_preservation import differentiable_sample,_feedback_logits


def test_teacher_is_native_authentic_audio_feedback_and_never_samples_an_original_head(monkeypatch):
    from haru_training.authentic_feedback import prepare_authentic_feedback_point
    model,data=fixture()
    with torch.no_grad():model.latent_scale.fill_(2);model.latent_bias.fill_(.3)
    monkeypatch.setattr(model.head,'sample',lambda *a:pytest.fail('Authentic teacher must not sample a head'))
    point=prepare_authentic_feedback_point(model,data,row=0,position=3,seed=7)
    with torch.no_grad():
        raw=data['audio_targets'][:,3].to(model.device,model.dtype)
        embedded=model.embed(point.token,raw[:,None],point.user_latent,point.model_mask,point.user_mask)
        expected=model.logits(model(embedded,deepcopy(point.cache))[:,-1])
    torch.testing.assert_close(point.teacher_logits,expected,atol=0,rtol=0)
    assert not point.teacher_logits.requires_grad


def test_authentic_target_only_changes_loss_teacher_not_student_prediction_or_past():
    from haru_training.authentic_feedback import prepare_authentic_feedback_point,authentic_feedback_loss
    model,data=fixture()
    a=prepare_authentic_feedback_point(model,data,row=0,position=3,seed=7)
    changed=deepcopy(data);changed['audio_targets'][:,3]+=10
    changed['model_latents'][:,4:]+=100
    b=prepare_authentic_feedback_point(model,changed,row=0,position=3,seed=7)
    torch.testing.assert_close(a.hidden,b.hidden,atol=0,rtol=0)
    sa=_feedback_logits(model,a,model.unscale_latents(differentiable_sample(model.head,a.hidden,a.noise)))
    sb=_feedback_logits(model,b,model.unscale_latents(differentiable_sample(model.head,b.hidden,b.noise)))
    torch.testing.assert_close(sa,sb,atol=0,rtol=0)
    assert not torch.equal(a.teacher_logits,b.teacher_logits)
    assert not torch.equal(authentic_feedback_loss(model,a),authentic_feedback_loss(model,b))


def test_authentic_loss_has_real_head_gradient_and_reuses_an_immutable_native_cache():
    from haru_training.authentic_feedback import prepare_authentic_feedback_point,authentic_feedback_loss
    model,data=fixture();data['audio_targets'].requires_grad_()
    point=prepare_authentic_feedback_point(model,data,row=0,position=3,seed=7)
    teacher=point.teacher_logits.clone()
    loss=authentic_feedback_loss(model,point)
    torch.testing.assert_close(loss,authentic_feedback_loss(model,point),atol=0,rtol=0)
    assert loss>0 and torch.isfinite(loss)
    loss.backward()
    assert model.head.final.proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('head.'))
    assert data['audio_targets'].grad is None
    with torch.no_grad():model.head.final.proj.weight.add_(.05)
    assert not torch.equal(loss,authentic_feedback_loss(model,point))
    torch.testing.assert_close(teacher,point.teacher_logits,atol=0,rtol=0)


@pytest.mark.parametrize('invalid',['target','position','backbone'])
def test_authentic_point_rejects_invalid_supervision(invalid):
    from haru_training.authentic_feedback import prepare_authentic_feedback_point
    model,data=fixture();position=3
    if invalid=='target':data['audio_targets'][:,3]=float('nan')
    elif invalid=='position':position=6
    else:next(model.lm.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):prepare_authentic_feedback_point(model,data,row=0,position=position,seed=7)


def test_cached_objective_uses_authentic_targets_without_a_reference_head():
    from haru_training.authentic_feedback import AuthenticFeedbackLoss
    model,data=fixture();objective=AuthenticFeedbackLoss(model)
    assert not hasattr(objective,'reference')
    before=objective(model,data,seed=7)
    assert 'loss' not in before and before['preservation_loss']>0
    assert len(objective.points)==1
    with torch.no_grad():model.head.final.proj.weight.add_(.05)
    after=objective(model,data,seed=7)
    assert len(objective.points)==1 and not torch.equal(before['preservation_loss'],after['preservation_loss'])
    with torch.no_grad():next(model.lm.parameters()).add_(.01)
    with pytest.raises(ValueError,match='changed'):objective(model,data,seed=7)


def test_authentic_trial_exports_the_distinct_recipe_and_real_updates(tmp_path):
    from haru_training.authentic_feedback import authentic_spec,run_authentic_pilot
    from haru_training.checkpoint import read_checkpoint
    from haru_training.inference import apply_inference
    from haru_training.contracts import configure_trainable,TrainingConfig
    from test_phase4_haru import tensor_digest
    model,_=fixture();initial=deepcopy(model);train,val=records('t'),records('v')
    spec=authentic_spec(train,val,preservation_weight=1.,split_sha256='a'*64,cache_sha256='b'*64,max_steps=2)
    spec=replace(spec,base_model='test/tiny-haru',base_revision='a'*40)
    result=run_authentic_pilot(model,train,val,spec,tmp_path,render=lambda *a:None)
    assert result['complete'] and result['frozen_weights_unchanged']
    assert result['preservation_target']=='authentic_audio' and not result['final_test_used']
    artifact=tmp_path/'inference/step-00000002';bundle=read_checkpoint(artifact)
    assert bundle['spec']['preservation_target']=='authentic_audio'
    assert tensor_digest(model.head.named_parameters())!=tensor_digest(initial.head.named_parameters())
    configure_trainable(initial,TrainingConfig(mode='head'))
    apply_inference(initial,artifact,base_model=spec.base_model,base_revision=spec.base_revision)
    assert tensor_digest(initial.named_parameters())==tensor_digest(model.named_parameters())


@pytest.mark.parametrize('legacy',['head','preserving','history'])
def test_legacy_recipes_cannot_silently_accept_an_authentic_target(tmp_path,legacy):
    from haru_training.authentic_feedback import authentic_spec
    from haru_training.direct_speaker import run_direct_head_pilot
    from haru_training.content_preservation import run_preserving_pilot
    from haru_training.generated_history import run_history_pilot
    model,_=fixture();train,val=records('t'),records('v')
    spec=authentic_spec(train,val,preservation_weight=0.,split_sha256='a'*64,cache_sha256='b'*64)
    runner=dict(head=run_direct_head_pilot,preserving=run_preserving_pilot,history=run_history_pilot)[legacy]
    with pytest.raises(ValueError):runner(model,train,val,spec,tmp_path/'run',render=lambda *a:None)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('change',[dict(preservation_weight=.5),dict(learning_rate=1e-5),dict(max_steps=101)])
def test_authentic_runner_rejects_unregistered_settings_before_output(tmp_path,change):
    from haru_training.authentic_feedback import authentic_spec,run_authentic_pilot
    model,_=fixture();train,val=records('t'),records('v')
    spec=authentic_spec(train,val,preservation_weight=0.,split_sha256='a'*64,cache_sha256='b'*64)
    with pytest.raises(ValueError):run_authentic_pilot(model,train,val,replace(spec,**change),tmp_path/'run',render=lambda *a:None)
    assert not (tmp_path/'run').exists()
