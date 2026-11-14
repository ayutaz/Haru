from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf
import torch

from test_dialogue import fixture
from test_monologue import Tokenizer
from test_prepare import SmallCodec
from haru_training.adaptation_data import VoiceReference, prepare_dialogue_sequence, assemble_dataset


class Codec(SmallCodec):
    def sample(self, means, generator):
        return means


def assets(tmp_path):
    row, samples = fixture(tmp_path)
    row.update(model_speaker_id='speaker-A', tokenizer_revision='fixed')
    row['model_chunks'][0]['text'] = 'hello'
    reference = VoiceReference('speaker-A', 'separate-reference', 'a'*64,
                               'fixed', torch.ones(23, 2))
    return row, samples, reference


def test_real_stereo_channels_keep_sync_overlap_and_separate_voice_reference(tmp_path):
    row, samples, reference = assets(tmp_path)
    result = prepare_dialogue_sequence(Codec(), Tokenizer([3, 4]), row, tmp_path,
                                      reference, [7], 0, 1, codec_revision='fixed')
    sequence = result.sequence
    expected = samples.T.reshape(2, 15, 3200).mean(-1)
    torch.testing.assert_close(sequence.model_latents[-15:, 0], torch.from_numpy(expected[0]))
    torch.testing.assert_close(sequence.user_latents[-15:, 0], torch.from_numpy(expected[1]))
    assert sequence.record_id == 'dialogue/dialog-1'
    assert sequence.supervise.tolist() == [False]*24+[True]*15
    assert result.provenance['reference_source_id'] == 'separate-reference'
    assert result.provenance['overlap_seconds'] == pytest.approx(.2)
    assert result.provenance['model_encoding'] != result.provenance['user_encoding']
    # Channel roles are explicit: swapping files and mappings must preserve data.
    sf.write(tmp_path/'dialog.wav', samples[:, ::-1], 24000, subtype='FLOAT')
    row['sha256'] = sha256((tmp_path/'dialog.wav').read_bytes()).hexdigest()
    row['channels'] = {'model': 1, 'user': 0}
    swapped = prepare_dialogue_sequence(Codec(), Tokenizer([3, 4]), row, tmp_path,
                                      reference, [7], 0, 1, codec_revision='fixed')
    assert torch.equal(swapped.sequence.model_latents, sequence.model_latents)
    assert torch.equal(swapped.sequence.user_latents, sequence.user_latents)


@pytest.mark.parametrize('change', [
    lambda r, v: (r, replace(v, speaker_id='other-voice')),
    lambda r, v: (r, replace(v, source_id='old-dialogue-1')),
    lambda r, v: (r, replace(v, codec_revision='other')),
    lambda r, v: (r, replace(v, latents=torch.zeros(22, 2))),
    lambda r, v: ({**r, 'tokenizer_revision': 'other'}, v),
])
def test_voice_reference_or_tokenizer_mismatch_is_rejected_before_encoding(tmp_path, change):
    row, _, reference = assets(tmp_path)
    row, reference = change(row, reference)
    class NeverEncode(Codec):
        def encode(self, signal): pytest.fail('must reject before codec work')
    with pytest.raises(ValueError):
        prepare_dialogue_sequence(NeverEncode(), Tokenizer([3, 4]), row, tmp_path,
                                 reference, [7], 0, 1, codec_revision='fixed')


def test_chunk_text_must_match_actual_tokenizer_before_any_encoding(tmp_path):
    row, _, reference = assets(tmp_path)
    class NeverEncode(Codec):
        def encode(self, signal): pytest.fail('invalid tokenization must not reach codec')
    with pytest.raises(ValueError, match='token'):
        prepare_dialogue_sequence(NeverEncode(), Tokenizer([9]), row, tmp_path,
                                 reference, [7], 0, 1, codec_revision='fixed')


def speech_data():
    from haru_training.data import make_sequence
    splits = {'train': [f't{i}' for i in range(80)], 'validation': [f'v{i}' for i in range(10)],
              'test': [f'test{i}' for i in range(8)], 'reference': ['ref0', 'ref1']}
    def make(ids):
        return [make_sequence(i, [7], torch.ones(23, 2), torch.ones(3, 2)*.2,
                             torch.zeros(3, 2), torch.tensor([3, 4, 0]), 0, 1) for i in ids]
    return splits, make(splits['train']), make(splits['validation'])


def test_mixture_includes_all_80_speech_10_validation_and_both_dialogue_splits(tmp_path):
    row, _, reference = assets(tmp_path)
    first = prepare_dialogue_sequence(Codec(), Tokenizer([3, 4]), row, tmp_path,
                                     reference, [7], 0, 1, codec_revision='fixed')
    other = replace(first, sequence=replace(first.sequence, record_id='dialogue/validation'),
        provenance={**first.provenance, 'split': 'validation', 'source_ids': ['other-dialogue'],
                    'source_sha256': 'b'*64, 'audio_sha256': 'c'*64})
    splits, train, val = speech_data()
    mixture = assemble_dataset(splits, train, val, [first, other])
    assert len(mixture.train) == 81 and len(mixture.validation) == 11
    assert set(splits['train']) <= {r.record_id for r in mixture.train}
    assert set(splits['validation']) <= {r.record_id for r in mixture.validation}
    assert mixture.provenance['speech_train_records'] == 80
    assert mixture.provenance['dialogue_train_frames'] == 15
    assert mixture.provenance['speech_train_frames'] == 240
    assert mixture.steps_per_epoch(micro_batch_size=1, accumulation_steps=2) == 41
    with pytest.raises(ValueError, match='80|coverage'):
        assemble_dataset(splits, train[:3], val, [first, other])
    with pytest.raises(ValueError, match='dialogue'):
        assemble_dataset(splits, train, val, [])
    leaked = replace(other, provenance={**other.provenance, 'source_ids': ['test0']})
    with pytest.raises(ValueError, match='leak'):
        assemble_dataset(splits, train, val, [first, leaked])


def test_one_training_epoch_consumes_every_selected_record_without_validation_leakage(tmp_path):
    from haru_training.trainer import RunSpec, Trainer, dataset_fingerprint
    from test_trainer import Regression, collate, objective
    row, _, reference = assets(tmp_path)
    first = prepare_dialogue_sequence(Codec(), Tokenizer([3, 4]), row, tmp_path,
                                     reference, [7], 0, 1, codec_revision='fixed')
    other = replace(first, sequence=replace(first.sequence, record_id='dialogue/validation'),
        provenance={**first.provenance, 'split': 'validation', 'source_ids': ['other-dialogue'],
                    'source_sha256': 'b'*64, 'audio_sha256': 'c'*64})
    splits, train, val = speech_data()
    mixture = assemble_dataset(splits, train, val, [first, other])
    spec = RunSpec(base_model='test/base', base_revision='a'*40,
        dataset_sha256=dataset_fingerprint(mixture.train), validation_sha256=dataset_fingerprint(mixture.validation),
        split_sha256='b'*64, cache_sha256='c'*64,
        max_steps=mixture.steps_per_epoch(micro_batch_size=1, accumulation_steps=2),
        accumulation_steps=2, learning_rate=1e-5)
    trainer = Trainer(Regression(), mixture.train, mixture.validation, spec, tmp_path/'trainer',
                      collate_fn=collate, loss_fn=objective)
    trainer.fit()
    import json
    events = [json.loads(s) for s in (trainer.output/'events.jsonl').read_text().splitlines()]
    seen = [i for event in events if event['event'] == 'train' for i in event['record_ids']]
    assert len(seen) == len(set(seen)) == 81
    assert set(seen) == {r.record_id for r in mixture.train}
    assert not set(seen) & {r.record_id for r in mixture.validation}
    assert trainer.state.epoch == 1 and trainer.state.cursor == 0


def test_speech_loader_builds_all_80_10_from_authenticated_caches(tmp_path):
    from haru_training.adaptation_data import prepare_speech_sequences
    from haru_training.corpus import Clip
    from haru_training.data import write_cache
    splits, _, _ = speech_data()
    clips, rows = {}, {}
    for record_id in [i for group in splits.values() for i in group]:
        key = {'latent_dim': 2, 'source_sha256': sha256(record_id.encode()).hexdigest(), 'codec_revision': 'fixed'}
        path = tmp_path/f'{record_id}.safetensors'
        write_cache(path, torch.ones(1, 30, 2), key)
        rows[record_id] = {'id': record_id, 'cache': path.name, 'key': key}
        clips[record_id] = Clip(record_id, tmp_path/'not-needed.wav', 'hello', 24000, 4.,
                               key['source_sha256'], .2, .1, 0.)
        if record_id in splits['reference']:
            reference_key = {**key, 'role': 'reference'}
            name = f'{record_id}.reference.safetensors'
            write_cache(tmp_path/name, torch.ones(1, 23, 2), reference_key)
            rows[record_id].update(reference_cache=name, reference_key=reference_key)
    train, validation = prepare_speech_sequences(Codec(), Tokenizer([3, 4]), clips, rows, tmp_path,
        splits, 'ref0', [7], 0, 1)
    assert [r.record_id for r in train] == splits['train']
    assert [r.record_id for r in validation] == splits['validation']
    assert len(train) == 80 and len(validation) == 10
    assert not set(splits['test']+splits['reference']) & {r.record_id for r in train+validation}


def test_decoder_uses_the_same_wave_bytes_that_were_hashed(tmp_path, monkeypatch):
    from pathlib import Path
    row, samples, reference = assets(tmp_path)
    original_read = Path.read_bytes
    calls = []
    def read(path):
        payload = original_read(path)
        if path.name == 'dialog.wav':
            calls.append(path)
            sf.write(path, np.ones_like(samples)*.9, 24000, subtype='FLOAT')
        return payload
    monkeypatch.setattr(Path, 'read_bytes', read)
    result = prepare_dialogue_sequence(Codec(), Tokenizer([3, 4]), row, tmp_path,
                                      reference, [7], 0, 1, codec_revision='fixed')
    expected = samples[:, 0].reshape(15, 3200).mean(-1)
    torch.testing.assert_close(result.sequence.model_latents[-15:, 0], torch.from_numpy(expected))
    assert len(calls) == 1
