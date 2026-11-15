from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.data import collate_sequences, make_sequence
from test_training_forward import tiny_model


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model=tiny_model()
    configure_trainable(model,TrainingConfig(mode='head'))
    data=collate_sequences([make_sequence('train-record',[3,4],torch.randn(2,64),
        torch.randn(5,64),torch.randn(5,64),torch.tensor([5,6,7,8,0]),0,1)],0)
    return model,data


@pytest.mark.parametrize('guidance',[1.,1.5,2.])
def test_training_sample_matches_native_and_keeps_head_gradient(guidance):
    from haru_training.content_preservation import differentiable_sample
    model,_=fixture();hidden=torch.randn(2,model.config.hidden_size);noise=torch.randn(2,64)
    expected=model.head.sample(hidden,noise,guidance)
    actual=differentiable_sample(model.head,hidden,noise,guidance)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.square().mean().backward()
    assert model.head.final.proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.lm.parameters())


def test_content_kl_cannot_be_satisfied_by_preserving_only_silence_logits():
    from haru_training.content_preservation import content_kl
    teacher=torch.tensor([[100.,90.,.2,-.1]])
    student=torch.tensor([[-100.,-90.,.2,-.1]],requires_grad=True)
    loss=content_kl(student,teacher,excluded_ids=[0,1])
    assert float(loss.detach())==pytest.approx(0,abs=1e-7)
    changed=student.detach().clone();changed[0,2]+=1;changed.requires_grad_()
    loss=content_kl(changed,teacher,excluded_ids=[0,1]);loss.backward()
    assert loss>0 and torch.count_nonzero(changed.grad[0,:2])==0
    assert changed.grad[0,2:].abs().sum()>0


def test_native_feedback_loss_backpropagates_only_to_head_and_reuses_immutable_past():
    from haru_training.content_preservation import prepare_feedback_point, feedback_preservation_loss
    model,data=fixture();reference=deepcopy(model.head).eval().requires_grad_(False)
    point=prepare_feedback_point(model,data,row=0,position=3,reference_head=reference,seed=7)
    with torch.no_grad():
        raw=model.unscale_latents(reference.sample(point.hidden,point.noise,point.guidance))
        embeddings=model.embed(point.token,raw[:,None],point.user_latent,point.model_mask,point.user_mask)
        native=model.logits(model(embeddings,deepcopy(point.cache))[:,-1])
    torch.testing.assert_close(point.teacher_logits,native,atol=0,rtol=0)
    before=feedback_preservation_loss(model,point)
    assert abs(float(before.detach()))<1e-6
    with torch.no_grad():model.head.final.proj.weight.add_(.1)
    loss=feedback_preservation_loss(model,point)
    repeated=feedback_preservation_loss(model,point)
    torch.testing.assert_close(loss,repeated,atol=0,rtol=0)
    loss.backward()
    assert loss>0 and model.head.final.proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('head.'))
    assert all(p.grad is None for p in reference.parameters())


def test_feedback_gradient_agrees_with_finite_difference_of_the_native_forward():
    from haru_training.content_preservation import prepare_feedback_point, feedback_preservation_loss, content_kl
    model,data=fixture();reference=deepcopy(model.head).eval().requires_grad_(False)
    point=prepare_feedback_point(model,data,row=0,position=3,reference_head=reference,seed=7)
    with torch.no_grad():model.head.final.proj.weight.add_(.1)
    loss=feedback_preservation_loss(model,point);loss.backward()
    parameter=model.head.final.proj.weight
    index=int(parameter.grad.abs().flatten().argmax());analytical=float(parameter.grad.flatten()[index])
    def native_value():
        with torch.no_grad():
            raw=model.unscale_latents(model.head.sample(point.hidden,point.noise,point.guidance))
            hidden=model(model.embed(point.token,raw[:,None],point.user_latent,point.model_mask,point.user_mask),deepcopy(point.cache))[:,-1]
            return float(content_kl(model.logits(hidden),point.teacher_logits,excluded_ids=[0,1,2]))
    with torch.no_grad():parameter.flatten()[index]+=1e-3
    plus=native_value()
    with torch.no_grad():parameter.flatten()[index]-=2e-3
    minus=native_value()
    assert analytical==pytest.approx((plus-minus)/2e-3,rel=.03,abs=2e-5)


def test_future_recorded_audio_and_labels_do_not_enter_the_feedback_comparison():
    from haru_training.content_preservation import prepare_feedback_point
    model,data=fixture();reference=deepcopy(model.head).eval().requires_grad_(False)
    a=prepare_feedback_point(model,data,row=0,position=3,reference_head=reference,seed=7)
    changed=deepcopy(data);changed['model_latents'][:,4:]+=100;changed['audio_targets']+=200
    changed['text_targets'][:,4:]=9
    b=prepare_feedback_point(model,changed,row=0,position=3,reference_head=reference,seed=7)
    torch.testing.assert_close(a.hidden,b.hidden,atol=0,rtol=0)
    torch.testing.assert_close(a.teacher_logits,b.teacher_logits,atol=0,rtol=0)


def test_padding_only_positions_and_mutable_teachers_are_rejected():
    from haru_training.content_preservation import prepare_feedback_point
    model,data=fixture();reference=deepcopy(model.head).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        prepare_feedback_point(model,data,row=0,position=6,reference_head=reference,seed=7)
    with pytest.raises(ValueError):
        prepare_feedback_point(model,data,row=0,position=3,reference_head=model.head,seed=7)


def test_combined_objective_has_separate_measured_preservation_and_caches_frozen_prefixes(monkeypatch):
    from haru_training.content_preservation import ContentPreservingLoss
    model,data=fixture();objective=ContentPreservingLoss(model,points_per_record=2)
    result=objective(model,data,seed=7)
    assert 'preservation_loss' in result and result['preservation_loss'].ndim==0
    assert len(objective.points)==1
    with torch.no_grad():model.head.final.proj.weight.add_(.05)
    result=objective(model,data,seed=7)
    assert result['preservation_loss']>0
    result['preservation_loss'].backward()
    assert model.head.final.proj.weight.grad.abs().sum()>0
    with torch.no_grad():next(model.lm.parameters()).add_(.01)
    with pytest.raises(ValueError,match='changed'):objective(model,data,seed=7)


def test_trainer_uses_preservation_in_the_update_and_reports_it_separately(tmp_path):
    from test_trainer import Regression,records,spec
    from haru_training.trainer import Trainer
    train,val=records(),records('val')
    run=replace(spec(train,val),max_steps=1,preservation_weight=.5)
    model=Regression();before=model.weight.detach().clone()
    def loss(current,batch,*,seed):
        return dict(text_loss=current.weight*0,audio_loss=current.weight*0,
                    preservation_loss=(current.weight-3).square())
    trainer=Trainer(model,train,val,run,tmp_path,loss_fn=loss)
    trainer.fit()
    assert model.weight>before and model.frozen.grad is None
    metrics=trainer.validate()
    assert metrics['audio_loss']==0 and metrics['preservation_loss']>0


def test_enabled_preservation_cannot_silently_omit_the_loss(tmp_path):
    from test_trainer import Regression,records,spec,objective
    from haru_training.trainer import Trainer
    train,val=records(),records('val')
    run=replace(spec(train,val),max_steps=1,preservation_weight=1.)
    model=Regression();before=model.weight.detach().clone()
    trainer=Trainer(model,train,val,run,tmp_path,loss_fn=objective)
    with pytest.raises(ValueError,match='preservation'):trainer.fit()
    assert torch.equal(model.weight,before)


@pytest.mark.parametrize('weight',[-1,float('nan'),True])
def test_preservation_weight_rejects_invalid_values(weight):
    from test_trainer import records,spec
    with pytest.raises(ValueError,match='preservation'):
        replace(spec(records(),records('val')),preservation_weight=weight)


def test_coefficient_calibration_balances_training_gradients_without_updating_weights():
    from haru_training.content_preservation import calibrate_weight
    parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
    before=parameter.detach().clone()
    audio=parameter.square().sum()
    preservation=(parameter*torch.tensor([.2,.3])).square().sum()
    result=calibrate_weight(audio,preservation,[parameter])
    assert result['weight']*result['preservation_gradient_norm']/result['audio_gradient_norm']==pytest.approx(.1)
    assert not result['clipped']
    assert torch.equal(parameter,before) and parameter.grad is None
    with pytest.raises(ValueError,match='gradient'):
        calibrate_weight(parameter.square().sum(),parameter.sum()*0,[parameter])
