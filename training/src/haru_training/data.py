"""Waveforms, authenticated latent caches and causal training batches."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def read_wave(path: Path, sample_rate: int = 24000) -> np.ndarray:
    audio, source = sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('audio must be finite and nonempty')
    divisor = math.gcd(source, sample_rate)
    if source != sample_rate:
        audio = resample_poly(audio, sample_rate // divisor, source // divisor, axis=0)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _fingerprint(key):
    return sha256(json.dumps(key, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _tensor_hash(value):
    return sha256(value.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def write_cache(path: Path, latents: torch.Tensor, key: dict) -> None:
    value = latents.detach().cpu().float().contiguous()
    if value.ndim != 3 or value.shape[-1] != key['latent_dim'] or not torch.isfinite(value).all():
        raise ValueError('invalid cache tensor')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    save_file({'latents': value}, str(temporary), metadata={
        'fingerprint': _fingerprint(key), 'tensor_sha256': _tensor_hash(value)})
    temporary.replace(path)


def read_cache(path: Path, expected_key: dict) -> torch.Tensor:
    try:
        with safe_open(str(path), framework='pt') as cache:
            metadata = cache.metadata()
            if metadata.get('fingerprint') != _fingerprint(expected_key):
                raise ValueError('cache configuration mismatch')
            value = cache.get_tensor('latents')
        if (value.ndim != 3 or value.shape[-1] != expected_key['latent_dim']
                or not torch.isfinite(value).all() or _tensor_hash(value) != metadata['tensor_sha256']):
            raise ValueError('cache tensor corrupt or shape mismatch')
        return value
    except Exception as error:
        raise ValueError('latent cache unavailable, corrupt, or configuration mismatch') from error


@dataclass(frozen=True)
class TextChunk:
    start_frame: int
    tokens: list[int]


def align_text(frames: int, chunks: list[TextChunk], pad_id: int, eop_id: int) -> torch.Tensor:
    if frames <= 0 or pad_id == eop_id:
        raise ValueError('invalid frame count or special tokens')
    result = torch.full((frames,), pad_id, dtype=torch.long)
    occupied = set()
    for chunk in chunks:
        start, tokens = chunk.start_frame, chunk.tokens
        positions = list(range(max(0, start - 1), start + len(tokens)))
        if (start < 0 or not tokens or start + len(tokens) > frames or occupied.intersection(positions)
                or any(t in (pad_id, eop_id) or t < 0 for t in tokens)):
            raise ValueError('text chunks overlap, contain special tokens, or exceed available frames')
        if start:
            result[start - 1] = eop_id
        result[start:start + len(tokens)] = torch.tensor(tokens)
        occupied.update(positions)
    return result


@dataclass
class StreamSequence:
    record_id: str
    tokens: torch.Tensor
    model_latents: torch.Tensor
    user_latents: torch.Tensor
    model_mask: torch.Tensor
    user_mask: torch.Tensor
    supervise: torch.Tensor
    frame_indices: torch.Tensor


def make_sequence(record_id: str, system_ids: list[int], reference: torch.Tensor,
                  model: torch.Tensor, user: torch.Tensor, tokens: torch.Tensor,
                  pad_id: int, eop_id: int) -> StreamSequence:
    if (model.ndim != 2 or user.shape != model.shape or tokens.shape != (len(model),)
            or reference.ndim != 2 or reference.shape[1] != model.shape[1]
            or len(reference) == 0 or len(model) == 0):
        raise ValueError('streams need aligned nonempty frames, reference and matching latent dimensions')
    k, r, t = len(system_ids), len(reference), len(model)
    total, dim = k + r + t, model.shape[1]
    ids = torch.tensor(system_ids + [pad_id] * (r - 1) + [eop_id] + tokens.tolist(), dtype=torch.long)
    model_stream = torch.cat([torch.zeros(k, dim), reference.float().cpu(), model.float().cpu()])
    user_stream = torch.cat([torch.zeros(k + r, dim), user.float().cpu()])
    positions = torch.arange(total)
    result = StreamSequence(record_id, ids, model_stream, user_stream, positions >= k,
                            positions >= k + r, positions >= k + r,
                            torch.cat([torch.full((k+r,), -1), torch.arange(t)]))
    _validate(result)
    return result


def _validate(sequence: StreamSequence):
    n = len(sequence.tokens)
    if (n < 2 or sequence.model_latents.ndim != 2
            or sequence.model_latents.shape[0] != n or sequence.user_latents.shape != sequence.model_latents.shape
            or any(value.shape != (n,) for value in (sequence.model_mask, sequence.user_mask,
                                                    sequence.supervise, sequence.frame_indices))
            or not torch.isfinite(sequence.model_latents).all() or not torch.isfinite(sequence.user_latents).all()):
        raise ValueError('invalid or nonfinite stream sequence')


def collate_sequences(sequences: list[StreamSequence], pad_id: int) -> dict:
    if not sequences:
        raise ValueError('batch is empty')
    for sequence in sequences:
        _validate(sequence)
    batch, width = len(sequences), max(len(s.tokens) - 1 for s in sequences)
    dim = sequences[0].model_latents.shape[-1]
    result = {'tokens': torch.full((batch, width), pad_id, dtype=torch.long),
              'text_targets': torch.full((batch, width), -100, dtype=torch.long),
              'target_frame_indices': torch.full((batch, width), -1, dtype=torch.long),
              'record_ids': [s.record_id for s in sequences]}
    for key in ('model_latents', 'user_latents', 'audio_targets'):
        result[key] = torch.zeros(batch, width, dim)
    for key in ('model_mask', 'user_mask', 'attention_mask', 'audio_loss_mask'):
        result[key] = torch.zeros(batch, width, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        n = len(sequence.tokens) - 1
        if sequence.model_latents.shape[1] != dim:
            raise ValueError('latent dimensions differ within batch')
        result['tokens'][row, :n] = sequence.tokens[:-1]
        for key in ('model_latents', 'user_latents', 'model_mask', 'user_mask'):
            result[key][row, :n] = getattr(sequence, key)[:-1]
        active = sequence.supervise[1:]
        result['text_targets'][row, :n] = torch.where(active, sequence.tokens[1:], -100)
        result['audio_targets'][row, :n] = sequence.model_latents[1:]
        result['audio_loss_mask'][row, :n] = active
        result['attention_mask'][row, :n] = True
        result['target_frame_indices'][row, :n] = sequence.frame_indices[1:]
    return result
