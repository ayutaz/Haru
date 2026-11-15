from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from haru_training.data import collate_sequences
from haru_training.monologue import monologue_sequence
from test_monologue import Codec, assets


class ChunkTokenizer:
    def encode(self, text, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(ids={'前。': [4], '後。': [5]}[text])


def timing(clip):
    return dict(version=1, kind='haru-recorded-text-timing',
                source_sha256=clip.sha256,
                transcript_sha256=sha256(clip.text.encode('utf-8')).hexdigest(),
                duration_seconds=clip.seconds, method='observed-test-phone-boundaries',
                chunks=[dict(text='前。', start_seconds=0.),
                        dict(text='後。', start_seconds=.4)])


def sequence(tmp_path, mutate=None):
    clip, rows = assets(tmp_path)
    clip = replace(clip, text='前。後。', seconds=4 * 3200 / 24000)
    observed = timing(clip)
    if mutate:
        mutate(observed)
    return monologue_sequence(Codec(), ChunkTokenizer(), clip, rows, tmp_path,
                             'reference', [8], 0, 1, timing=observed)


def test_recorded_chunks_keep_a_pause_instead_of_front_loading_the_transcript(tmp_path):
    result = sequence(tmp_path)
    assert result.tokens.tolist() == [8, 0, 0, 1, 4, 0, 1, 5]
    batch = collate_sequences([result], 0)
    assert batch['tokens'][0, -1] == 1  # EOP is past input, not the next token
    assert batch['text_targets'][0, -1] == 5
    assert batch['target_frame_indices'][0, -1] == 3
    torch.testing.assert_close(batch['audio_targets'][0, -1], torch.full((2,), 2.))


@pytest.mark.parametrize('mutate', [
    lambda t: t.update(source_sha256='other'),
    lambda t: t.update(transcript_sha256='0' * 64),
    lambda t: t.update(duration_seconds=1.),
    lambda t: t.update(method=''),
    lambda t: t.update(version=2),
    lambda t: t['chunks'][0].update(text='other'),
    lambda t: t['chunks'][0].update(start_seconds=float('nan')),
    lambda t: t['chunks'][0].update(start_seconds=-.1),
    lambda t: t['chunks'][1].update(start_seconds=1.),
    lambda t: t['chunks'][1].update(start_seconds=.1),  # no room for its EOP
    lambda t: t.update(chunks=list(reversed(t['chunks']))),
])
def test_unknown_changed_or_crowded_timing_is_rejected_not_replaced_with_a_guess(tmp_path, mutate):
    with pytest.raises(ValueError):
        sequence(tmp_path, mutate)


def test_recorded_data_target_cannot_leak_into_its_own_prediction_input(tmp_path):
    first = sequence(tmp_path)
    second = deepcopy(first)
    second.model_latents[-1] += 100
    left = collate_sequences([first], 0)
    right = collate_sequences([second], 0)
    torch.testing.assert_close(left['model_latents'], right['model_latents'])
    assert not torch.equal(left['audio_targets'], right['audio_targets'])
