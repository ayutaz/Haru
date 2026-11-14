from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf
import torch

from haru_training.dialogue import inspect_dialogue, dialogue_sequence, validate_cohort


def fixture(tmp_path):
    samples = np.zeros((48000, 2), np.float32)
    samples[2400:19200, 0] = .1
    samples[14400:38400, 1] = .2  # Genuine overlap is retained.
    path = tmp_path/'dialog.wav'
    sf.write(path, samples, 24000, subtype='FLOAT')
    row = {'id': 'dialog-1', 'wav': 'dialog.wav', 'sha256': sha256(path.read_bytes()).hexdigest(),
        'channels': {'model': 0, 'user': 1}, 'split': 'train',
        'source_ids': ['old-dialogue-1'],
        'turns': [{'speaker': 'model', 'start_sec': .1, 'end_sec': .8, 'text': 'hello'},
                  {'speaker': 'user', 'start_sec': .6, 'end_sec': 1.6, 'text': 'yes'}],
        'model_chunks': [{'start_frame': 1, 'tokens': [3, 4]}]}
    return row, samples


def test_stereo_timing_channel_and_overlap_are_preserved(tmp_path):
    row, samples = fixture(tmp_path)
    result = inspect_dialogue(row, tmp_path)
    assert result['samples'] == 48000 and result['sample_rate'] == 24000
    assert result['overlap_seconds'] == pytest.approx(.2)
    assert result['audio_sha256']
    model = torch.ones(15, 2)
    user = torch.full((15, 2), 2.)
    sequence = dialogue_sequence(row, model, user, [7], torch.zeros(2, 2), 0, 1)
    assert torch.equal(sequence.model_latents[-15:], model)
    assert torch.equal(sequence.user_latents[-15:], user)
    assert sequence.tokens[-14:-12].tolist() == [3, 4]
    assert sequence.supervise[-15:].all()


@pytest.mark.parametrize('change', [
    lambda r: r['channels'].update(user=0),
    lambda r: r['turns'][0].update(end_sec=3),
    lambda r: r['turns'][0].update(start_sec=float('nan')),
    lambda r: r['turns'][0].update(text=''),
    lambda r: r.update(sha256='0'*64),
    lambda r: r.update(wav='../outside.wav'),
    lambda r: r.update(source_ids=[]),
])
def test_reject_invalid_source_or_timeline(tmp_path, change):
    row, _ = fixture(tmp_path)
    change(row)
    with pytest.raises(ValueError):
        inspect_dialogue(row, tmp_path)


def test_mono_or_silent_speaker_cannot_be_claimed_as_duplex(tmp_path):
    row, samples = fixture(tmp_path)
    sf.write(tmp_path/'dialog.wav', samples[:, 0], 24000, subtype='FLOAT')
    row['sha256'] = sha256((tmp_path/'dialog.wav').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='stereo'):
        inspect_dialogue(row, tmp_path)
    samples[:, 1] = 0
    sf.write(tmp_path/'dialog.wav', samples, 24000, subtype='FLOAT')
    row['sha256'] = sha256((tmp_path/'dialog.wav').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='silent'):
        inspect_dialogue(row, tmp_path)


def test_duplicate_audio_or_source_and_heldout_lineage_are_rejected(tmp_path):
    row, _ = fixture(tmp_path)
    assert validate_cohort([row], tmp_path, protected_source_ids={'test-1'})
    second = deepcopy(row)
    second.update(id='dialog-2', split='validation')
    with pytest.raises(ValueError, match='duplicate|leak'):
        validate_cohort([row, second], tmp_path, protected_source_ids=set())
    with pytest.raises(ValueError, match='held.out|leak'):
        validate_cohort([row], tmp_path, protected_source_ids={'old-dialogue-1'})


def test_chunk_cannot_extend_past_model_turn_or_overlap_another_chunk(tmp_path):
    row, _ = fixture(tmp_path)
    row['model_chunks'] = [{'start_frame': 5, 'tokens': [3, 4]}]
    with pytest.raises(ValueError, match='turn'):
        dialogue_sequence(row, torch.ones(15, 2), torch.ones(15, 2), [7], torch.zeros(2, 2), 0, 1)
