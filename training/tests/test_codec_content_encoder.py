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
