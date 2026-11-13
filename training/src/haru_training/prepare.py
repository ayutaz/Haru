"""Deterministic data preparation; GPU use is always an explicit caller action."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .data import read_cache, read_wave, write_cache


@torch.no_grad()
def encode_audio(codec, audio: np.ndarray, *, seed: int) -> torch.Tensor:
    if audio.ndim != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('expected finite samples-by-channels audio')
    frame = codec.config.frame_samples
    signal = torch.from_numpy(np.ascontiguousarray(audio.T)).to(device=codec.device, dtype=codec.dtype)
    signal = F.pad(signal, (0, (-signal.shape[-1]) % frame))
    means = codec.encode(signal)
    generator = torch.Generator(device=codec.device).manual_seed(seed)
    latents = codec.sample(means, generator).float().cpu()
    if not torch.isfinite(latents).all():
        raise FloatingPointError('codec returned nonfinite latents')
    return latents


@torch.no_grad()
def encode_reference(codec, audio: np.ndarray, *, seed: int) -> torch.Tensor:
    from haru.audio import prepare_reference
    if audio.ndim != 2 or audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError('reference must be finite mono speech')
    prepared = prepare_reference(audio[:, 0])
    if len(prepared) == 0:
        raise ValueError('reference contains no speech')
    signal = torch.from_numpy(prepared.copy()).to(codec.device, codec.dtype)[None]
    means = codec.encode(signal)  # live prompt handles partial hops, without whole-frame zero extension
    generator = torch.Generator(device=codec.device).manual_seed(seed)
    return codec.sample(means, generator).float().cpu()


@torch.no_grad()
def encode_user_audio(codec, audio: np.ndarray, *, seed: int) -> torch.Tensor:
    if audio.ndim != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('user audio must be finite and nonempty')
    frame = codec.config.frame_samples
    signal = torch.from_numpy(np.ascontiguousarray(audio.T)).to(codec.device, codec.dtype)
    signal = F.pad(signal, (0, (-signal.shape[-1]) % frame))
    means = codec.encode(signal)
    generator = torch.Generator(device=codec.device).manual_seed(seed)
    # DuplexSession.step draws a separate posterior scale and noise for every frame.
    values = [codec.sample(means[:, i:i+1], generator) for i in range(means.shape[1])]
    result = torch.cat(values, dim=1).float().cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError('codec returned nonfinite user latents')
    return result


def prepare_corpus(codec, clips, output: Path, *, codec_revision: str, seed: int = 20261114,
                   reference_ids: tuple[str, ...] | list[str] = ()) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not set(reference_ids) <= {c.id for c in clips}:
        raise ValueError('reference recording is missing from source inventory')
    records = []
    for clip in sorted(clips, key=lambda c: c.id):
        if sha256(clip.path.read_bytes()).hexdigest() != clip.sha256:
            raise ValueError('source recording changed after inventory')
        item_seed = seed + int(sha256(clip.id.encode()).hexdigest()[:8], 16)
        key = {'source_sha256': clip.sha256, 'codec_revision': codec_revision, 'seed': item_seed,
               'sample_rate': 24000, 'frame_samples': codec.config.frame_samples,
               'latent_dim': codec.config.latent_dim, 'dtype': str(codec.dtype),
               'representation': 'raw_sampled_latents', 'algorithm': 'whole-padded-v1',
               'latent_std': getattr(codec.config, 'latent_std', None)}
        path = output / f'{clip.id}.safetensors'
        if path.exists():
            value = read_cache(path, key)
        else:
            value = encode_audio(codec, read_wave(clip.path), seed=item_seed)
            write_cache(path, value, key)
        row = {'id': clip.id, 'cache': path.name, 'frames': value.shape[1],
                        'channels': value.shape[0], 'key': key,
                        'cache_sha256': sha256(path.read_bytes()).hexdigest()}
        if clip.id in reference_ids:
            from haru.audio import prepare_reference
            audio = read_wave(clip.path)
            if len(prepare_reference(audio[:, 0])) < 3 * 24000:
                raise ValueError('reference must contain at least 3 seconds after live-prompt preparation')
            ref_key = {**key, 'algorithm': 'prompt-trim-direct-v1', 'role': 'reference'}
            ref_path = output / f'{clip.id}.reference.safetensors'
            if ref_path.exists():
                ref_value = read_cache(ref_path, ref_key)
            else:
                ref_value = encode_reference(codec, audio, seed=item_seed)
                write_cache(ref_path, ref_value, ref_key)
            row.update(reference_cache=ref_path.name, reference_key=ref_key,
                       reference_frames=ref_value.shape[1],
                       reference_cache_sha256=sha256(ref_path.read_bytes()).hexdigest())
        records.append(row)
    manifest = {'codec_revision': codec_revision, 'records': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def build_evaluation_inputs(clips, splits: dict, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = sorted(splits['test'])
    if len(ids) < 3:
        raise ValueError('evaluation needs at least three held-out source recordings')
    by_id = {c.id: c for c in clips}
    # Indices select held-out recordings, never training/reference speech.
    specs = [
        ('short_statement', [0], .5, .3, 5),
        ('long_statement', [1, 2, 3], .5, .2, 5),
        ('two_statements', [4, 5], .5, .2, 5),
        ('long_pause', [6, 7], .5, 5, 5),
        ('delayed_start', [2], 3, .3, 5),
        ('sustained_speech', [3, 4, 5], .5, .1, 5),
        ('interruption_attempt_3s', [6], 3, .3, 5),
        ('interruption_attempt_7s', [7], 7, .3, 5),
        ('silent_input', [], 0, 0, 20),
        ('silence_after_statement', [2], .5, .3, 30),
    ]
    scenarios = []
    for name, selections, delay, gap, tail in specs:
        pieces = [np.zeros(round(delay * 24000), dtype=np.float32)]
        segments, cursor = [], len(pieces[0])
        for index in selections:
            source_id = ids[index % len(ids)]
            clip = by_id[source_id]
            speech = read_wave(clip.path)[:, 0]
            segments.append({'id': source_id, 'source_sha256': clip.sha256,
                             'start_sample': cursor, 'end_sample': cursor + len(speech)})
            pieces += [speech, np.zeros(round(gap * 24000), dtype=np.float32)]
            cursor += len(speech) + len(pieces[-1])
        pieces.append(np.zeros(round(tail * 24000), dtype=np.float32))
        audio = np.concatenate(pieces)
        path = output / f'{name}.wav'
        sf.write(path, audio, 24000, subtype='FLOAT')
        scenarios.append({'id': name, 'wav': path.name, 'sha256': sha256(path.read_bytes()).hexdigest(),
                          'samples': len(audio), 'segments': segments,
                          'note': 'interruption is attempted; actual overlap must be measured from the model output'
                                  if name.startswith('interruption') else ''})
    result = {'sample_rate': 24000, 'seeds': [7, 23, 61], 'scenarios': scenarios,
              'input_kind': 'held-out read speech with controlled timing; not spontaneous question-answer recordings'}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
