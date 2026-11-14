import numpy as np
import pytest
import torch


@pytest.mark.parametrize('samples',[6400,6419,24000])
def test_differentiable_resampling_matches_preparation_scipy_filter(samples):
    from scipy.signal import resample_poly
    from haru_training.codec_content_encoder import resample_24_to_16
    torch.set_num_threads(1)
    rng=np.random.default_rng(4);signal=rng.normal(size=samples).astype('float32')*.1
    wave=torch.tensor(signal,requires_grad=True)
    value=resample_24_to_16(wave)
    np.testing.assert_allclose(value.detach().numpy(),resample_poly(signal,2,3),atol=1e-6,rtol=1e-5)
    value.square().mean().backward()
    assert torch.isfinite(wave.grad).all() and wave.grad.abs().sum()>0


def test_real_tiny_ctc_is_frozen_but_preserves_waveform_gradient_and_clock():
    from transformers import Wav2Vec2Config,Wav2Vec2ForCTC,Wav2Vec2FeatureExtractor
    from scipy.signal import resample_poly
    from haru_training.codec_content_encoder import FrozenCTC
    torch.set_num_threads(1);torch.manual_seed(1)
    config=Wav2Vec2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
        conv_dim=[4]*7,conv_kernel=[10,3,3,3,3,2,2],conv_stride=[5,2,2,2,2,2,2],
        num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2,feat_extract_norm='layer',
        vocab_size=5,pad_token_id=4,mask_time_prob=0.)
    encoder=FrozenCTC(Wav2Vec2ForCTC(config))
    wave=torch.randn(6400,requires_grad=True)*.1;wave.retain_grad()
    logits=encoder(wave)
    assert logits.shape==(13,5) and not encoder.training
    processor=Wav2Vec2FeatureExtractor(sampling_rate=16000,do_normalize=True,return_attention_mask=True)
    batch=processor(resample_poly(wave.detach().numpy(),2,3),sampling_rate=16000,return_tensors='pt')
    with torch.no_grad():expected=encoder.model(**batch).logits[0]
    torch.testing.assert_close(logits,expected,atol=1e-6,rtol=1e-5)
    logits.square().mean().backward()
    assert wave.grad.abs().sum()>0 and all(p.grad is None for p in encoder.parameters())


def test_codec_training_with_real_ctc_class_and_physically_aligned_pair_updates_adapter_only():
    from transformers import Wav2Vec2Config,Wav2Vec2ForCTC
    from haru_training.codec_content_encoder import FrozenCTC
    from haru_training.codec_pair_preparation import prepare_observed_pair
    from haru_training.codec_adapter_training import CodecTrainer,TrainSpec
    from test_codec_adapter_checkpoint import base
    from test_observed_ctc_alignment import observations
    torch.set_num_threads(1)
    model=base()
    config=Wav2Vec2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
        conv_dim=[4]*7,conv_kernel=[10,3,3,3,3,2,2],conv_stride=[5,2,2,2,2,2,2],
        num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2,feat_extract_norm='layer',
        vocab_size=32,pad_token_id=0,mask_time_prob=0.)
    content=FrozenCTC(Wav2Vec2ForCTC(config))
    audio=(.1*np.sin(np.arange(48120)*.04)).astype('float32')
    meta=dict(id='line',group_id='line',split='train',source_speaker='source',target_speaker='target',
        source_sha256='a'*64,target_sha256='b'*64,source_text='今日は晴れです',target_text='今日は晴れです',
        codec_revision='c'*40,ctc_revision='d'*40)
    item=prepare_observed_pair(meta,audio,audio*.8,torch.ones(16,64),observations(),observations(),
        list(range(1,21)),blank=0,splits={'train':['line']})
    trainer=CodecTrainer(model,[item],TrainSpec(base_model='test/tiny-haru',base_revision='c'*40,
        target_speaker='target',max_steps=3,content_weight=.1,transition_strength=2),content=content,rank=4)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    records=trainer.fit()
    changed=[name for name,p in model.named_parameters() if not torch.equal(before[name],p)]
    assert len(records)==3 and changed and all('stem.adapter.' in name for name in changed)
    assert all(np.isfinite(r['content']) and r['content']>0 for r in records)
    assert all(p.grad is None for p in content.parameters())
