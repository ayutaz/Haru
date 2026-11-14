"""Explicitly timed stereo dialogue ingestion; technical QC is not listening QC."""

from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch

from .data import TextChunk, align_text, make_sequence


def _timeline(row, seconds):
    if (not isinstance(row.get('id'), str) or not row['id']
            or row.get('split') not in {'train', 'validation'}
            or not isinstance(row.get('source_ids'), list) or not row['source_ids']
            or any(not isinstance(s, str) or not s for s in row['source_ids'])
            or len(row['source_ids']) != len(set(row['source_ids']))):
        raise ValueError('dialogue requires identity, split and source lineage')
    channels = row.get('channels')
    if (not isinstance(channels, dict) or set(channels) != {'model', 'user'}
            or any(type(v) is not int for v in channels.values())
            or set(channels.values()) != {0, 1}):
        raise ValueError('model/user must identify separate stereo channels')
    turns = row.get('turns')
    if not isinstance(turns, list) or not turns:
        raise ValueError('timed dialogue turns are required')
    by_speaker = {'model': [], 'user': []}
    for turn in turns:
        start, end = turn.get('start_sec'), turn.get('end_sec')
        if (turn.get('speaker') not in by_speaker
                or type(start) not in (float, int) or type(end) not in (float, int)
                or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end <= seconds
                or not isinstance(turn.get('text'), str) or not turn['text'].strip()):
            raise ValueError('invalid turn times, speaker or transcript')
        by_speaker[turn['speaker']].append((start, end))
    for spans in by_speaker.values():
        spans.sort()
        if not spans or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('both speakers need nonoverlapping turns on their own channel')
    return by_speaker


def load_dialogue(row, root):
    """Decode authenticated bytes with channel roles and timing, without GPU.

    Usage permission and perceptual voice/pronunciation quality remain external
    review gates; this function never certifies those from manifest assertions.
    """
    root = Path(root).resolve()
    raw = row.get('wav')
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError('dialogue audio must use a relative local path')
    path = (root/raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError('audio path escapes dialogue root')
    if not re.fullmatch('[a-f0-9]{64}', row.get('sha256', '')):
        raise ValueError('source requires a SHA-256')
    if not path.is_file():
        raise ValueError('missing source audio')
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != row['sha256']:
        raise ValueError('missing or changed source audio')
    audio, rate = sf.read(BytesIO(payload), dtype='float32', always_2d=True)
    if (rate != 24000 or audio.shape[1] != 2 or not len(audio)
            or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1):
        raise ValueError('dialogue requires finite nonempty 24 kHz stereo PCM within [-1,1]')
    spans = _timeline(row, len(audio)/rate)
    for speaker, turns in spans.items():
        channel = row['channels'][speaker]
        for start, end in turns:
            segment = audio[round(start*rate):round(end*rate), channel]
            if not len(segment) or not np.any(segment):
                raise ValueError('annotated speaker turn is silent')
    overlap = sum(max(0., min(a1, b1)-max(a0, b0))
                  for a0, a1 in spans['model'] for b0, b1 in spans['user'])
    return audio, {'id': row['id'], 'sample_rate': rate, 'samples': len(audio),
        'frames': math.ceil(len(audio)/3200), 'overlap_seconds': overlap,
        'audio_sha256': sha256(np.ascontiguousarray(audio).tobytes()).hexdigest(),
        'source_sha256': row['sha256'], 'technical_qc': 'passed',
        'perceptual_quality': 'unverified', 'usage_permission': 'unverified'}


def inspect_dialogue(row, root):
    return load_dialogue(row, root)[1]


def validate_cohort(rows, root, *, protected_source_ids):
    if not rows:
        raise ValueError('dialogue cohort is empty')
    ids, audio, sources = set(), set(), set()
    results = []
    for row in rows:
        result = inspect_dialogue(row, root)
        lineage = set(row['source_ids'])
        if (row['id'] in ids or result['audio_sha256'] in audio or lineage & sources
                or lineage & set(protected_source_ids)):
            raise ValueError('duplicate dialogue or source lineage/held-out leakage')
        ids.add(row['id']); audio.add(result['audio_sha256']); sources.update(lineage)
        results.append(result)
    return results


def validate_text_chunks(row, frames, pad_id, eop_id):
    """Check timing and collisions without loading a codec or producing teachers."""
    spans = _timeline(row, frames/7.5)
    chunks = []
    for chunk in row.get('model_chunks', []):
        start, tokens = chunk['start_frame'], chunk['tokens']
        if (type(start) is not int or not isinstance(tokens, list)
                or any(type(t) is not int for t in tokens)
                or not any(a <= start/7.5 and (start+len(tokens))/7.5 <= b
                           for a, b in spans['model'])):
            raise ValueError('text chunk extends beyond its model speaker turn')
        chunks.append(TextChunk(start, tokens))
    if not chunks:
        raise ValueError('explicit timed model text chunks are required')
    return align_text(frames, chunks, pad_id, eop_id)


def dialogue_sequence(row, model_latents, user_latents, system_ids, reference, pad_id, eop_id):
    """Use separately encoded synchronous channels and explicit timed text chunks."""
    if model_latents.ndim != 2 or user_latents.shape != model_latents.shape:
        raise ValueError('dialogue channels require matching latent frames')
    tokens = validate_text_chunks(row, len(model_latents), pad_id, eop_id)
    return make_sequence(row['id'], system_ids, reference, model_latents, user_latents,
                         tokens, pad_id, eop_id)
