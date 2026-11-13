import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from haru_training.corpus import Clip
from haru_training.prepare import encode_audio, build_evaluation_inputs


class SmallCodec:
    device = torch.device('cpu')
    dtype = torch.float32
    config = SimpleNamespace(frame_samples=3200, latent_dim=2)

    def encode(self, signal):
        assert signal.shape[1] % 3200 == 0
        values = signal.reshape(signal.shape[0], -1, 3200).mean(-1)
        return torch.stack([values, -values], dim=-1)

    def sample(self, mean, generator):
        return mean + .01 * torch.randn(mean.shape, generator=generator)


def test_encoding_pads_final_frame_preserves_channels_and_samples_deterministically():
    audio = np.stack([np.ones(3201), -np.ones(3201)], axis=-1).astype('float32')
    value = encode_audio(SmallCodec(), audio, seed=4)
    assert value.shape == (2, 2, 2)
    torch.testing.assert_close(value, encode_audio(SmallCodec(), audio, seed=4))
    assert value[0, 0, 0] > .9 and value[1, 0, 0] < -.9
    assert value[:, 1].abs().max() < .1


def test_evaluation_inputs_use_only_heldout_ids_and_have_verifiable_segments(tmp_path):
    clips = []
    for i in range(10):
        path = tmp_path / f'{i}.wav'
        sf.write(path, np.ones(4800) * (i + 1) / 20, 24000, subtype='FLOAT')
        clips.append(Clip(str(i), path, str(i), 24000, .2, str(i), .2, .1, 0))
    output = tmp_path / 'evaluation'
    result = build_evaluation_inputs(clips, {'test': [str(i) for i in range(2, 10)]}, output)
    assert len(result['scenarios']) == 10
    used = {segment['id'] for case in result['scenarios'] for segment in case['segments']}
    assert used == {str(i) for i in range(2, 10)}
    for case in result['scenarios']:
        wave, rate = sf.read(output / case['wav'])
        assert rate == 24000 and len(wave) == case['samples']
        for segment in case['segments']:
            start = segment['start_sample']
            np.testing.assert_allclose(wave[start:start+4800], (int(segment['id'])+1)/20, atol=1e-7)
    assert json.loads((output/'manifest.json').read_text()) == result


def test_corpus_preparation_verifies_source_and_cache_metadata(tmp_path):
    from dataclasses import replace
    from hashlib import sha256
    from haru_training.prepare import prepare_corpus
    import pytest
    source = tmp_path / 'source.wav'
    sf.write(source, np.ones(4800) * .1, 24000, subtype='FLOAT')
    clip = Clip('one', source, '音声', 24000, .2, sha256(source.read_bytes()).hexdigest(), .1, .1, 0)
    output = tmp_path / 'latents'
    first = prepare_corpus(SmallCodec(), [clip], output, codec_revision='test', seed=5)
    assert first['records'][0]['frames'] == 2
    assert first == prepare_corpus(SmallCodec(), [clip], output, codec_revision='test', seed=5)
    with pytest.raises(ValueError):
        prepare_corpus(SmallCodec(), [clip], output, codec_revision='different', seed=5)
    with pytest.raises(ValueError, match='source'):
        prepare_corpus(SmallCodec(), [replace(clip, sha256='wrong')], output, codec_revision='test', seed=5)


def test_reference_uses_live_prompt_trim_and_partial_encoding_without_tail_padding():
    from haru.audio import prepare_reference
    from haru_training.prepare import encode_reference
    class ReferenceCodec(SmallCodec):
        def encode(self, signal):
            self.input = signal.clone()
            return torch.zeros(signal.shape[0], (signal.shape[1]+3199)//3200, 2)
    audio = np.concatenate([np.zeros(24000), np.ones(12000)*.2, np.zeros(24000)]).astype('float32')
    codec = ReferenceCodec()
    encode_reference(codec, audio[:, None], seed=2)
    expected = prepare_reference(audio)
    assert len(expected) % 3200 != 0
    torch.testing.assert_close(codec.input[0], torch.from_numpy(expected))


def test_user_posterior_sampling_occurs_per_frame_as_in_live_session():
    from haru_training.prepare import encode_user_audio
    class FrameCodec(SmallCodec):
        def __init__(self): self.calls=[]
        def sample(self, mean, generator):
            self.calls.append(mean.shape[1])
            return mean + len(self.calls)
    codec = FrameCodec()
    value = encode_user_audio(codec, np.zeros((6400,1),np.float32), seed=3)
    assert codec.calls == [1,1]
    torch.testing.assert_close(value[0,:,0],torch.tensor([1.,2.]))
