"""Explicit read-speech continuation examples, never relabeled as duplex dialogue."""

from pathlib import Path
import numpy as np

from .data import TextChunk, align_text, make_sequence, read_cache
from .prepare import encode_user_audio


def monologue_sequence(codec, tokenizer, clip, rows: dict, cache_dir: Path,
                        reference_id: str, system_ids: list[int], pad_id: int, eop_id: int,
                        *, seed: int = 20261114):
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
    text_ids = tokenizer.encode(clip.text, add_special_tokens=False).ids
    tokens = align_text(count, [TextChunk(0, text_ids)], pad_id, eop_id)
    # Real codec silence has a nonzero latent and stochastic acoustic sampling.
    silence = encode_user_audio(codec, np.zeros((count * codec.config.frame_samples, 1), np.float32), seed=seed)[0]
    return make_sequence(clip.id, system_ids, reference[0], target[0], silence, tokens, pad_id, eop_id)
