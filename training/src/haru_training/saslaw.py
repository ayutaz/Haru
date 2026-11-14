"""Convert synchronized SaSLaW speech recordings, retaining source annotation."""

import math
from pathlib import Path
import re

import numpy as np
from scipy.signal import resample_poly

from .dialogue import validate_text_chunks


def read_annotations(speaker_root, dialogue_id):
    if not re.fullmatch(r'dialogue\d{3}', dialogue_id):
        raise ValueError('invalid source dialogue ID')
    root = Path(speaker_root)
    spans = [line.split() for line in (root/'seg_info'/f'{dialogue_id}.txt').read_text(encoding='utf-8').splitlines()
             if line.strip()]
    names = {p.name for p in (root/'transcript_segment').glob(dialogue_id+'-seg*.txt')}
    expected = {f'{dialogue_id}-seg{i:02d}.txt' for i in range(len(spans))}
    if not spans or names != expected:
        raise ValueError('source transcript coverage is incomplete; do not fill or skip turns')
    turns = []
    for index, span in enumerate(spans):
        if len(span) != 2:
            raise ValueError('invalid source turn interval')
        start, end = map(float, span)
        text = (root/'transcript_segment'/f'{dialogue_id}-seg{index:02d}.txt').read_text(encoding='utf-8').strip()
        if not text:
            raise ValueError('source transcript is empty')
        turns.append({'start': start, 'end': end, 'text': text, 'source_segment': index})
    return turns


def _rasterize(turns, duration, frames):
    result = []
    previous_end = 0.
    for source in turns:
        start, end, text = source['start'], source['end'], source['text']
        if (not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration
                or start < previous_end or not isinstance(text, str) or not text.strip()):
            raise ValueError('invalid source turn, order or transcript')
        previous_end = end
        first, last = math.floor(start*7.5), min(frames, math.ceil(end*7.5))
        if result and first <= result[-1]['last_frame']:
            # Adjacent intervals can share one 133 ms frame. Preserve both texts
            # and source intervals instead of creating colliding EOP/chunk slots.
            result[-1]['text'] += text
            result[-1]['last_frame'] = last
            result[-1]['source_end_sec'] = end
            result[-1]['source_segments'].append(dict(source))
        else:
            result.append({'first_frame': first, 'last_frame': last, 'text': text,
                'source_start_sec': start, 'source_end_sec': end, 'source_segments': [dict(source)]})
    if not result:
        raise ValueError('both source speakers require timed turns')
    return result


def convert_pair(model_audio, user_audio, sample_rate, annotations, tokenizer, *, dialogue_id,
                 model_speaker, user_speaker, split, revision, pad_id, eop_id):
    """Map VAD turn times to Haru frames; this is not word-level alignment.

    revision identifies the pinned Haru tokenizer, not the dataset revision.
    Dataset revision and original-file hashes are attached by the acquisition
    manifest. Missing/misaligned annotations are rejected, never fabricated.
    """
    model_audio, user_audio = np.asarray(model_audio), np.asarray(user_audio)
    if (model_audio.ndim != 1 or user_audio.shape != model_audio.shape or not model_audio.size
            or type(sample_rate) is not int or sample_rate <= 0):
        raise ValueError('source recordings must be synchronous, nonempty mono streams')
    if (not re.fullmatch(r'dialogue\d{3}', dialogue_id) or model_speaker == user_speaker
            or set(annotations) != {model_speaker, user_speaker}):
        raise ValueError('source identity and speaker channels are inconsistent')
    audio = np.stack([model_audio, user_audio], axis=1).astype(np.float32)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError('source audio must be finite PCM within [-1, 1]')
    duration = len(audio)/sample_rate
    divisor = math.gcd(sample_rate, 24000)
    if sample_rate != 24000:
        audio = resample_poly(audio, 24000//divisor, sample_rate//divisor, axis=0)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    tail = (-len(audio)) % 3200
    if tail:
        audio = np.pad(audio, ((0, tail), (0, 0)))
    gain = min(1., .999/max(float(np.max(np.abs(audio))), .999))
    audio *= gain  # Shared attenuation only if resampling overshoots the PCM range.
    frames = math.ceil(len(audio)/3200)
    source_id = f'SaSLaW/{model_speaker}-{user_speaker}/{dialogue_id}'
    row = {'id': f'SaSLaW-{model_speaker}-{user_speaker}-{dialogue_id}', 'split': split,
           'source_ids': [source_id], 'channels': {'model': 0, 'user': 1},
           'model_speaker_id': 'SaSLaW/'+model_speaker, 'tokenizer_revision': revision,
           'source_sample_rate': sample_rate, 'resample_gain': gain,
           'tail_padding_samples': tail,
           'turns': [], 'model_chunks': [], 'timing': 'turn intervals rasterized at 7.5 Hz, not word alignment'}
    for role, speaker in (('model', model_speaker), ('user', user_speaker)):
        for turn in _rasterize(annotations[speaker], duration, frames):
            first = turn.pop('first_frame')
            last = turn.pop('last_frame')
            row['turns'].append({**turn, 'speaker': role, 'start_sec': first/7.5, 'end_sec': last/7.5})
            if role == 'model':
                tokens = tokenizer.encode(turn['text'], add_special_tokens=False).ids
                row['model_chunks'].append({'start_frame': first, 'tokens': tokens, 'text': turn['text']})
    row['turns'].sort(key=lambda turn: turn['start_sec'])
    validate_text_chunks(row, frames, pad_id, eop_id)
    return audio, row
