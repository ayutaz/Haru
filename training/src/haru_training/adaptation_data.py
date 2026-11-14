"""Speaker-consistent dialogue teachers and complete read-speech/replay mixtures."""

from dataclasses import dataclass
from hashlib import sha256
import math
import re

import torch

from .dialogue import dialogue_sequence, load_dialogue, validate_text_chunks
from .corpus import validate_splits
from .data import StreamSequence, read_cache
from .monologue import monologue_sequence
from .prepare import encode_audio, encode_user_audio


@dataclass(frozen=True)
class VoiceReference:
    speaker_id: str
    source_id: str
    source_sha256: str
    codec_revision: str
    latents: torch.Tensor


@dataclass(frozen=True)
class PreparedDialogue:
    sequence: StreamSequence
    provenance: dict


def prepare_dialogue_sequence(codec, tokenizer, row, root, reference: VoiceReference,
                              system_ids, pad_id, eop_id, *, codec_revision, seed=20261114):
    """Preserve synchronous PCM and use each speaker's own reviewed reference.

    The caller authenticates reference cache/provenance and quality before this
    function. Speaker labels are checked for consistency, not inferred from PCM.
    Actual codec processing must be invoked on the authorized vast.ai GPU only.
    """
    if (not reference.speaker_id or row.get('model_speaker_id') != reference.speaker_id
            or not reference.source_id or reference.source_id in row.get('source_ids', [])
            or not re.fullmatch('[a-f0-9]{64}', reference.source_sha256)
            or reference.codec_revision != codec_revision
            or row.get('tokenizer_revision') != codec_revision
            or reference.latents.ndim != 2 or len(reference.latents) < math.ceil(3*7.5)
            or reference.latents.shape[1] != codec.config.latent_dim
            or not torch.isfinite(reference.latents).all()
            or codec.config.frame_samples != 3200):
        raise ValueError('speaker reference, codec/tokenizer revision, or reference duration is invalid')
    audio, checked = load_dialogue(row, root)
    validate_text_chunks(row, checked['frames'], pad_id, eop_id)
    for chunk in row['model_chunks']:
        text = chunk.get('text')
        if (not isinstance(text, str) or not text.strip()
                or tokenizer.encode(text, add_special_tokens=False).ids != chunk['tokens']):
            raise ValueError('chunk text does not match the pinned tokenizer tokens')
    item_seed = seed + int(sha256(row['id'].encode()).hexdigest()[:8], 16)
    model_channel, user_channel = row['channels']['model'], row['channels']['user']
    model = encode_audio(codec, audio[:, model_channel:model_channel+1], seed=item_seed)[0]
    user = encode_user_audio(codec, audio[:, user_channel:user_channel+1], seed=item_seed+1)[0]
    if model.shape[0] != checked['frames'] or user.shape != model.shape:
        raise ValueError('codec returned misaligned dialogue streams')
    sequence = dialogue_sequence(row, model, user, system_ids, reference.latents, pad_id, eop_id)
    sequence.record_id = 'dialogue/'+row['id']
    provenance = {**checked, 'split': row['split'], 'source_ids': list(row['source_ids']),
        'model_speaker_id': row['model_speaker_id'], 'reference_speaker_id': reference.speaker_id,
        'reference_source_id': reference.source_id, 'reference_source_sha256': reference.source_sha256,
        'codec_revision': codec_revision, 'tokenizer_revision': row['tokenizer_revision'],
        'seed': item_seed, 'model_encoding': 'whole-padded-v1', 'user_encoding': 'framewise-user-v1'}
    return PreparedDialogue(sequence, provenance)


@dataclass(frozen=True)
class MixedDataset:
    train: list[StreamSequence]
    validation: list[StreamSequence]
    provenance: dict

    def steps_per_epoch(self, *, micro_batch_size, accumulation_steps):
        if any(type(n) is not int or n <= 0 for n in (micro_batch_size, accumulation_steps)):
            raise ValueError('batch and accumulation sizes must be positive integers')
        return math.ceil(len(self.train)/(micro_batch_size*accumulation_steps))


def prepare_speech_sequences(codec, tokenizer, clips, rows, cache_dir, splits,
                             reference_id, system_ids, pad_id, eop_id):
    """Load the complete fixed speech train/validation partitions, never test."""
    from pathlib import Path
    validate_splits(splits, list(clips.values()))
    if ({k: len(v) for k, v in splits.items()} != {'train': 80, 'validation': 10, 'test': 8, 'reference': 2}
            or set(rows) != set(clips) or reference_id not in splits['reference']):
        raise ValueError('speech data must preserve all 80/10/8/2 sources and a held-out reference')
    ref = rows[reference_id]
    filename = Path(ref['reference_cache'])
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError('reference cache must be a direct relative filename')
    reference = read_cache(Path(cache_dir)/filename, ref['reference_key'])
    if reference.shape[0] != 1 or reference.shape[1] < math.ceil(3*7.5):
        raise ValueError('reference cache must preserve at least three seconds of mono speech')
    def make(role):
        return [monologue_sequence(codec, tokenizer, clips[i], rows, cache_dir, reference_id,
                                  system_ids, pad_id, eop_id) for i in splits[role]]
    return make('train'), make('validation')


def assemble_dataset(splits, speech_train, speech_validation, dialogues):
    """Include all train80/validation10 and disjoint dialogue train/validation.

    Every selected recording occurs once per epoch. This is record-level mixing,
    not an assertion of equal duration; actual supervised-frame totals are saved.
    """
    required = {'train': 80, 'validation': 10, 'test': 8, 'reference': 2}
    if set(splits) != set(required) or any(len(splits[k]) != n for k, n in required.items()):
        raise ValueError('speech coverage must remain 80/10/8/2')
    all_ids = [i for ids in splits.values() for i in ids]
    if len(set(all_ids)) != 100:
        raise ValueError('speech split leakage or duplicate source IDs')
    for records, role in ((speech_train, 'train'), (speech_validation, 'validation')):
        ids = [r.record_id for r in records]
        if len(ids) != len(set(ids)) or set(ids) != set(splits[role]):
            raise ValueError('speech coverage must include all 80 train and 10 validation records')
    by_split = {'train': [], 'validation': []}
    seen_ids, seen_sources, seen_audio, reference_ids = set(all_ids), set(), set(), set()
    for prepared in dialogues:
        row, sequence = prepared.provenance, prepared.sequence
        sources = set(row['source_ids'])
        if (row.get('split') not in by_split or not sources
                or sequence.record_id in seen_ids or sources & (set(all_ids) | seen_sources)
                or row['audio_sha256'] in seen_audio
                or row['model_speaker_id'] != row['reference_speaker_id']):
            raise ValueError('dialogue duplicate, speaker mismatch, or source leakage')
        seen_ids.add(sequence.record_id); seen_sources.update(sources)
        seen_audio.add(row['audio_sha256']); reference_ids.add(row['reference_source_id'])
        by_split[row['split']].append(sequence)
    if reference_ids & seen_sources:
        raise ValueError('dialogue reference source leaks into supervised teachers')
    if any(not rows for rows in by_split.values()):
        raise ValueError('separate dialogue training and validation records are required')
    train_by_id = {r.record_id: r for r in speech_train}
    val_by_id = {r.record_id: r for r in speech_validation}
    train = [train_by_id[i] for i in splits['train']]+by_split['train']
    validation = [val_by_id[i] for i in splits['validation']]+by_split['validation']
    def frames(rows):
        return sum(int(row.supervise.sum()) for row in rows)
    provenance = {'speech_train_records': len(speech_train), 'speech_validation_records': len(speech_validation),
        'dialogue_train_records': len(by_split['train']), 'dialogue_validation_records': len(by_split['validation']),
        'speech_train_frames': frames(speech_train), 'dialogue_train_frames': frames(by_split['train']),
        'speech_validation_frames': frames(speech_validation),
        'dialogue_validation_frames': frames(by_split['validation']),
        'mixing': 'each selected recording once per epoch; not equal duration',
        'dialogues': [d.provenance for d in dialogues]}
    return MixedDataset(train, validation, provenance)
