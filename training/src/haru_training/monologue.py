"""Explicit read-speech continuation examples, never relabeled as duplex dialogue."""

from pathlib import Path
from hashlib import sha256
import math
import numpy as np

from .data import TextChunk, align_text, make_sequence, read_cache
from .prepare import encode_user_audio


def recorded_text_chunks(clip, tokenizer, frames, timing, *, frame_samples=3200, sample_rate=24000):
    """Place authenticated, observed text groups on the recording's clock.

    These are estimated group onsets, not phoneme truth or a claim about the
    unpublished pretraining recipe. Crowded groups fail instead of moving the
    text or dropping tokens to make a training example fit.
    """
    try:
        duration = timing['duration_seconds']
        if (timing['version'] != 1 or timing['kind'] != 'haru-recorded-text-timing'
                or timing['source_sha256'] != clip.sha256
                or timing['transcript_sha256'] != sha256(clip.text.encode('utf-8')).hexdigest()
                or not isinstance(timing['method'], str) or not timing['method'].strip()
                or type(duration) not in (int, float) or not math.isfinite(duration)
                or abs(duration - clip.seconds) > 1 / sample_rate
                or frames != math.ceil(duration * sample_rate / frame_samples - 1e-8)
                or not timing['chunks']
                or ''.join(c['text'] for c in timing['chunks']) != clip.text):
            raise ValueError('recorded timing does not match the source and transcript')
        chunks = []
        previous = -1.
        for row in timing['chunks']:
            seconds = row['start_seconds']
            if (not isinstance(row['text'], str) or not row['text']
                    or type(seconds) not in (int, float) or not math.isfinite(seconds)
                    or not 0 <= seconds < duration or seconds <= previous):
                raise ValueError('recorded chunk clock must be finite and strictly increasing')
            tokens = tokenizer.encode(row['text'], add_special_tokens=False).ids
            chunks.append(TextChunk(math.floor(seconds * sample_rate / frame_samples), tokens))
            previous = seconds
        return chunks
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('recorded text timing is incomplete') from error


def monologue_sequence(codec, tokenizer, clip, rows: dict, cache_dir: Path,
                        reference_id: str, system_ids: list[int], pad_id: int, eop_id: int,
                        *, seed: int = 20261114, timing: dict | None = None):
    if clip.id == reference_id:
        raise ValueError('reference recording must be excluded from supervised targets')
    row, ref = rows[clip.id], rows[reference_id]
    if 'reference_cache' not in ref or 'reference_key' not in ref:
        raise ValueError('reference must be prepared with the live-prompt protocol')
    if row['key']['source_sha256'] != clip.sha256:
        raise ValueError('cached target does not match source inventory')
    if ref['reference_key']['source_sha256'] != ref['key']['source_sha256']:
        raise ValueError('reference cache does not match source inventory')
    if row['key']['codec_revision'] != ref['reference_key']['codec_revision']:
        raise ValueError('target and reference codec revisions differ')
    for name in (row['cache'], ref['reference_cache']):
        filename = Path(name)
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError('cache must be a direct relative filename')
    target = read_cache(Path(cache_dir)/row['cache'], row['key'])
    reference = read_cache(Path(cache_dir)/ref['reference_cache'], ref['reference_key'])
    if target.shape[0] != 1 or reference.shape[0] != 1:
        raise ValueError('monologue data needs mono target and reference')
    count = target.shape[1]
    chunks = (recorded_text_chunks(clip, tokenizer, count, timing,
                                  frame_samples=codec.config.frame_samples)
              if timing is not None else
              [TextChunk(0, tokenizer.encode(clip.text, add_special_tokens=False).ids)])
    tokens = align_text(count, chunks, pad_id, eop_id)
    # Real codec silence has a nonzero latent and stochastic acoustic sampling.
    silence = encode_user_audio(codec, np.zeros((count * codec.config.frame_samples, 1), np.float32), seed=seed)[0]
    return make_sequence(clip.id, system_ids, reference[0], target[0], silence, tokens, pad_id, eop_id)
