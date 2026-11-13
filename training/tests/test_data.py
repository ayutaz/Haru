import numpy as np
import pytest
import soundfile as sf
import torch

from haru_training.data import (TextChunk, align_text, collate_sequences, make_sequence,
                                read_cache, read_wave, write_cache)


def test_resampling_preserves_channels_and_ceil_length(tmp_path):
    path = tmp_path / 'stereo.wav'
    x = np.sin(np.arange(96001) * .01).astype('float32') * .2
    sf.write(path, np.stack([x, -x], axis=-1), 96000, subtype='FLOAT')
    y = read_wave(path)
    assert y.shape == (24001, 2)
    np.testing.assert_allclose(y[:, 0], -y[:, 1], atol=1e-7)
    assert np.isfinite(y).all()


def test_alignment_has_explicit_eop_and_rejects_impossible_or_overlapping_chunks():
    assert align_text(7, [TextChunk(2, [4, 5])], 0, 1).tolist() == [0, 1, 4, 5, 0, 0, 0]
    assert align_text(3, [TextChunk(0, [4, 5])], 0, 1).tolist() == [4, 5, 0]
    for chunks in ([TextChunk(2, [4, 5])], [TextChunk(0, [4, 5]), TextChunk(2, [6])],
                   [TextChunk(-1, [5])], [TextChunk(0, [0])]):
        with pytest.raises(ValueError):
            align_text(3, chunks, 0, 1)


def sequence(n=3, shift=0):
    return make_sequence('record', [8, 9], torch.full((2, 2), 7.),
                         torch.arange(n * 2).reshape(n, 2).float() + shift,
                         torch.full((n, 2), -3.), torch.tensor([4] * n), 0, 1)


def test_next_frame_shift_masks_reference_and_keeps_silence_supervised():
    batch = collate_sequences([sequence(), sequence(2)], 0)
    assert batch['tokens'].shape == (2, 6)
    assert batch['audio_loss_mask'][0].tolist() == [False, False, False, True, True, True]
    assert batch['audio_loss_mask'][1].tolist() == [False, False, False, True, True, False]
    assert batch['text_targets'][0].tolist() == [-100, -100, -100, 4, 4, 4]
    assert batch['attention_mask'][1].tolist() == [True, True, True, True, True, False]
    assert batch['target_frame_indices'][0].tolist() == [-1, -1, -1, 0, 1, 2]
    torch.testing.assert_close(batch['model_latents'][0, 3], torch.tensor([7., 7.]))
    torch.testing.assert_close(batch['audio_targets'][0, 3], torch.tensor([0., 1.]))
    assert batch['user_mask'][0, 4]  # silent user is a valid stream, not absent/padding


def test_future_final_target_is_never_part_of_inputs():
    original = sequence()
    modified = sequence()
    modified.model_latents[-1] += 100
    a, b = collate_sequences([original], 0), collate_sequences([modified], 0)
    torch.testing.assert_close(a['model_latents'], b['model_latents'])
    assert not torch.equal(a['audio_targets'], b['audio_targets'])


def test_invalid_shapes_and_nonfinite_latents_are_rejected():
    with pytest.raises(ValueError):
        make_sequence('bad', [8], torch.ones(2, 2), torch.ones(3, 2), torch.ones(2, 2),
                      torch.tensor([4, 4, 4]), 0, 1)
    bad = sequence()
    bad.model_latents[0, 0] = float('nan')
    with pytest.raises(ValueError):
        collate_sequences([bad], 0)


def test_cache_checks_source_codec_seed_shape_and_bytes(tmp_path):
    path = tmp_path / 'cache.safetensors'
    key = {'source_sha256': 'abc', 'codec_revision': 'rev', 'seed': 3, 'latent_dim': 2,
           'sample_rate': 24000, 'frame_samples': 3200, 'dtype': 'float32'}
    value = torch.arange(12).reshape(1, 6, 2).float()
    write_cache(path, value, key)
    torch.testing.assert_close(read_cache(path, key), value)
    for changed in ({**key, 'seed': 4}, {**key, 'codec_revision': 'other'}):
        with pytest.raises(ValueError):
            read_cache(path, changed)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_cache(path, key)
