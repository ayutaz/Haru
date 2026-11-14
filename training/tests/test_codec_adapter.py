from copy import deepcopy

import pytest
import torch

from haru.models.vibevoice_codec import CodecStream,open_stream
from test_training_forward import tiny_model


@pytest.fixture
def model():
    torch.set_num_threads(1);torch.manual_seed(13)
    return tiny_model().eval()


def change(adapter):
    with torch.no_grad():
        adapter.up.weight.fill_(.01)
        adapter.up.bias.fill_(.02)


def test_zero_adapter_invalidates_cached_sequence_and_preserves_global_rng(model):
    from haru_training.codec_adapter import attach_adapter
    codec=model.codec;z=torch.randn(1,3,64)
    original=codec.decode(z)
    old_sequence=codec.decoder.sequence()
    rng=torch.random.get_rng_state().clone()
    adapter=attach_adapter(model,rank=4)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert codec.decoder.sequence() is not old_sequence
    assert torch.equal(codec.decode(z),original)
    selected=[name for name,p in model.named_parameters() if p.requires_grad]
    assert selected and all(name.startswith('codec.decoder.stem.adapter.') for name in selected)
    assert sum(p.numel() for p in adapter.parameters())==2*256*4+4+256


def test_nonzero_adapter_changes_waveform_but_not_encoder_and_can_be_disabled(model):
    from haru_training.codec_adapter import attach_adapter,set_adapter_enabled
    z=torch.randn(1,3,64);wave=torch.randn(1,9600)
    baseline=model.codec.decode(z);encoded=model.codec.encode(wave)
    adapter=attach_adapter(model,rank=4);change(adapter)
    assert not torch.equal(model.codec.decode(z),baseline)
    assert torch.equal(model.codec.encode(wave),encoded)
    set_adapter_enabled(model.codec,False)
    assert torch.equal(model.codec.decode(z),baseline)


@pytest.mark.parametrize('rank',[0,-1,True,257,1.5])
def test_invalid_installation_does_not_mutate_model(model,rank):
    from haru_training.codec_adapter import attach_adapter
    stem=model.codec.decoder.stem
    flags=[p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError):attach_adapter(model,rank=rank)
    assert model.codec.decoder.stem is stem
    assert [p.requires_grad for p in model.parameters()]==flags


def test_duplicate_installation_and_incompatible_feedback_artifacts_fail(model):
    from haru_training.codec_adapter import attach_adapter
    model.inference_feedback_mode='separate_frozen_head'
    with pytest.raises(ValueError,match='feedback'):attach_adapter(model,rank=4)
    del model.inference_feedback_mode
    attach_adapter(model,rank=4)
    with pytest.raises(ValueError):attach_adapter(model,rank=4)


@pytest.mark.parametrize('backend',['auto','triton'])
def test_unsupported_backends_fail_instead_of_ignoring_adapter(model,backend):
    from haru_training.codec_adapter import attach_adapter
    attach_adapter(model,rank=4)
    with pytest.raises(ValueError,match='adapter'):open_stream(model.codec,backend=backend)


def test_streams_created_before_installation_or_enable_changes_are_rejected(model):
    from haru_training.codec_adapter import attach_adapter,set_adapter_enabled
    old=CodecStream(model.codec)
    attach_adapter(model,rank=4)
    with pytest.raises(RuntimeError,match='changed'):old.decode(torch.zeros(1,1,64))
    current=open_stream(model.codec)
    set_adapter_enabled(model.codec,False)
    with pytest.raises(RuntimeError,match='changed'):current.reset()


def test_streaming_causality_state_isolation_and_reset_with_nonzero_adapter(model):
    from haru_training.codec_adapter import attach_adapter
    z=torch.randn(2,4,64)
    base_full=model.codec.decode(z)
    base_stream=CodecStream(model.codec,batch=2)
    base_split=torch.cat([base_stream.decode(z[:,i:i+1]) for i in range(4)],dim=1)
    # Freeze tolerance against the unmodified decoder, before testing adaptation.
    torch.testing.assert_close(base_split,base_full,atol=1e-7,rtol=1e-5)
    adapter=attach_adapter(model,rank=4);change(adapter)
    full=model.codec.decode(z)
    stream=CodecStream(model.codec,batch=2)
    split=torch.cat([stream.decode(z[:,i:i+1]) for i in range(4)],dim=1)
    torch.testing.assert_close(split,full,atol=1e-7,rtol=1e-5)
    changed=z.clone();changed[:,2:]+=10
    assert torch.equal(model.codec.decode(changed)[:,:6400],full[:,:6400])
    left,right=CodecStream(model.codec),CodecStream(model.codec)
    interleaved=[[],[]]
    for i in range(4):
        for row,instance in enumerate((left,right)):
            interleaved[row].append(instance.decode(z[row:row+1,i:i+1]))
    for row in range(2):
        torch.testing.assert_close(torch.cat(interleaved[row],1),full[row:row+1],atol=1e-7,rtol=1e-5)
    stream.reset(rows=torch.tensor([0]))
    next_frame=stream.decode(z[:,:1])
    fresh=CodecStream(model.codec,batch=2).decode(z[:,:1])
    torch.testing.assert_close(next_frame[0],fresh[0],atol=1e-7,rtol=1e-5)
    assert not torch.equal(next_frame[1],fresh[1])


def test_training_decode_has_gradients_through_frozen_decoder_and_updates_adapter_only(model):
    from haru_training.codec_adapter import attach_adapter,decode_training
    adapter=attach_adapter(model,rank=4)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    optimizer=torch.optim.SGD(adapter.parameters(),lr=.1)
    z=torch.randn(1,2,64)
    for step in range(3):
        optimizer.zero_grad()
        wave,_=decode_training(model.codec,z)
        loss=(wave-.2).square().mean();loss.backward()
        assert all(p.grad is None for name,p in model.named_parameters() if not p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.up.parameters())
        if step:assert adapter.down.weight.grad.abs().sum()>0
        optimizer.step()
    changed=[name for name,p in model.named_parameters() if not torch.equal(before[name],p)]
    assert changed and all(name.startswith('codec.decoder.stem.adapter.') for name in changed)


@pytest.mark.parametrize('z',[torch.zeros(1,0,64),torch.zeros(1,1,63),torch.full((1,1,64),float('nan'))])
def test_training_decode_rejects_invalid_latents(model,z):
    from haru_training.codec_adapter import attach_adapter,decode_training
    attach_adapter(model,rank=4)
    with pytest.raises(ValueError):decode_training(model.codec,z)
