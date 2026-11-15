from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from haru_training.contracts import TrainingConfig,configure_trainable
from haru_training.data import collate_sequences
from haru_training.direct_speaker import NativeSpeakerLoss
from test_training_forward import tiny_model,batch
from test_preserving_pilot import records


def fixture():
    torch.set_num_threads(1);torch.manual_seed(31)
    model=tiny_model();configure_trainable(model,TrainingConfig(mode='head'))
    with torch.no_grad():model.latent_scale.fill_(2);model.latent_bias.fill_(.3)
    return model,batch()


def test_generated_history_matches_independent_native_frame_loop_with_fixed_noise():
    from haru_training.generated_history import generated_history_hidden
    model,data=fixture();data={k:v[:1] if torch.is_tensor(v) else v[:1] for k,v in data.items()}
    noises=torch.randn(1,data['tokens'].shape[1],64)
    actual=generated_history_hidden(model,data,seed=7,noises=noises)
    keys=('tokens','model_latents','user_latents','model_mask','user_mask')
    prefix=int(torch.nonzero(data['audio_loss_mask'][0])[0])+1
    with torch.no_grad():
        cache=model.new_cache()
        expected=[model(model.embed(*(data[k][:,:prefix] for k in keys)),cache)]
        for position in range(prefix,data['tokens'].shape[1]):
            raw=model.unscale_latents(model.head.sample(expected[-1][:,-1],noises[:,position-1],1.5))
            embedding=model.embed(data['tokens'][:,position:position+1],raw[:,None],
                data['user_latents'][:,position:position+1],data['model_mask'][:,position:position+1],
                data['user_mask'][:,position:position+1])
            expected.append(model(embedding,cache))
    torch.testing.assert_close(actual,torch.cat(expected,1),atol=0,rtol=0)
    assert not actual.requires_grad


def test_recorded_continuation_and_future_targets_are_never_used_as_audio_history():
    from haru_training.generated_history import generated_history_hidden
    model,data=fixture();before=generated_history_hidden(model,data,seed=7)
    changed=deepcopy(data)
    for row in range(len(data['tokens'])):
        prefix=int(torch.nonzero(data['audio_loss_mask'][row])[0])+1
        changed['model_latents'][row,prefix:]+=100
    changed['audio_targets']+=200;changed['text_targets'].fill_(99)
    after=generated_history_hidden(model,changed,seed=7)
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    changed=deepcopy(data)
    reference_position=int(torch.nonzero(data['model_mask'][0])[0])
    changed['model_latents'][0,reference_position]+=10
    assert not torch.equal(before[0],generated_history_hidden(model,changed,seed=7)[0])


def test_updated_head_changes_fresh_conditions_and_padding_does_not():
    from haru_training.generated_history import generated_history_hidden
    model,data=fixture();before=generated_history_hidden(model,data,seed=7)
    with torch.no_grad():model.head.final.proj.weight.add_(.1)
    after=generated_history_hidden(model,data,seed=7)
    assert not torch.equal(before,after)
    assert torch.count_nonzero(after[1,4:])==0
    changed=deepcopy(data);changed['model_latents'][~data['attention_mask']]+=100
    changed['user_latents'][~data['attention_mask']]-=100
    torch.testing.assert_close(after,generated_history_hidden(model,changed,seed=7),atol=0,rtol=0)


def test_future_user_frame_cannot_change_earlier_conditions():
    from haru_training.generated_history import generated_history_hidden
    model,data=fixture();before=generated_history_hidden(model,data,seed=7)
    data['user_latents'][0,-1]+=100
    after=generated_history_hidden(model,data,seed=7)
    torch.testing.assert_close(before[0,:-1],after[0,:-1],atol=0,rtol=0)


def test_training_loss_updates_only_head_and_validation_uses_common_recorded_past():
    from haru_training.generated_history import GeneratedHistoryLoss
    model,data=fixture();loss=GeneratedHistoryLoss()
    result=loss(model,data,seed=7);result['audio_loss'].backward()
    assert model.head.final.proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('head.'))
    model.eval()
    actual=loss(model,data,seed=7);expected=NativeSpeakerLoss()(model,data,seed=7)
    torch.testing.assert_close(actual['audio_loss'],expected['audio_loss'],atol=0,rtol=0)
    torch.testing.assert_close(actual['hidden'],expected['hidden'],atol=0,rtol=0)


@pytest.mark.parametrize('invalid',['backbone','mask','noise'])
def test_generated_conditions_reject_invalid_inputs(invalid):
    from haru_training.generated_history import generated_history_hidden
    model,data=fixture();extra={}
    if invalid=='backbone':next(model.lm.parameters()).requires_grad_(True)
    elif invalid=='mask':data['attention_mask'][0,0]=False
    else:extra['noises']=torch.full((2,data['tokens'].shape[1],64),float('nan'))
    with pytest.raises(ValueError):generated_history_hidden(model,data,seed=7,**extra)


def test_generated_trial_exports_real_updates_and_preserves_inference_contract(tmp_path):
    from haru_training.generated_history import history_spec,run_history_pilot
    from haru_training.checkpoint import read_checkpoint
    from haru_training.inference import apply_inference
    from test_phase4_haru import tensor_digest
    model,_=fixture();train,val=records('t'),records('v')
    spec=history_spec(train,val,split_sha256='a'*64,cache_sha256='b'*64,generated=True,max_steps=2)
    spec=replace(spec,base_model='test/tiny-haru',base_revision='a'*40)
    initial=deepcopy(model)
    result=run_history_pilot(model,train,val,spec,tmp_path,render=lambda *a:None)
    assert result['complete'] and result['frozen_weights_unchanged']
    assert result['training_audio_history']=='generated' and not result['final_test_used']
    exported=tmp_path/'inference/step-00000002'
    bundle=read_checkpoint(exported)
    assert bundle['spec']['training_audio_history']=='generated'
    configure_trainable(initial,TrainingConfig(mode='head'))
    apply_inference(initial,exported,base_model='test/tiny-haru',base_revision='a'*40)
    assert tensor_digest(initial.named_parameters())==tensor_digest(model.named_parameters())


def test_legacy_head_runner_cannot_accept_the_new_history_condition(tmp_path):
    from haru_training.generated_history import history_spec
    from haru_training.direct_speaker import run_direct_head_pilot
    train,val=records('t'),records('v')
    spec=history_spec(train,val,split_sha256='a'*64,cache_sha256='b'*64,generated=True)
    with pytest.raises(ValueError):run_direct_head_pilot(tiny_model(),train,val,spec,tmp_path/'run',render=lambda *a:None)
    assert not (tmp_path/'run').exists()
