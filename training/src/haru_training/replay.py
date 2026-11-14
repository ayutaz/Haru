"""Record the existing Haru conversation API with explicit, reproducible inputs."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

from haru import DuplexSession


def replay(model, user, *, config, reference=None, system_prompt=''):
    """Replay mono 24 kHz PCM with the same frame alignment as haru-inference.

    Output frame t is generated before consuming input frame t. The final input
    frame is therefore not consumed, matching the original offline inference.
    This is an API/waveform check, not an assessment of the response quality.
    """
    user = np.asarray(user, dtype=np.float32)
    if user.ndim != 1 or not len(user) or not np.isfinite(user).all():
        raise ValueError('replay requires finite, nonempty mono PCM')
    if config.seed is None or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('replay requires an explicit seed and an eval, frozen model')
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float32)
        if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
            raise ValueError('reference requires finite nonempty mono PCM')
    size = model.config.frame_samples
    count = (len(user)+size-1)//size
    padded = np.pad(user, (0, count*size-len(user)))
    config = replace(config, max_frames=count)
    session = DuplexSession(model, system_prompt, reference, config)
    frames = [session.start()]
    for index in range(count-1):
        frames.append(session.step(padded[index*size:(index+1)*size]))
    for index, frame in enumerate(frames):
        if (frame.audio.shape != (size,) or not np.isfinite(frame.audio).all()
                or frame.index != index or frame.seconds < 0):
            raise ValueError('invalid response frame')
    return {'audio': np.concatenate([f.audio for f in frames]).astype(np.float32),
            'sample_rate': model.config.sample_rate, 'frame_samples': size,
            'input_samples': len(user), 'padded_samples': len(padded)-len(user),
            'text': ''.join(f.text for f in frames).strip(),
            'frames': [{k: v for k, v in asdict(f).items() if k != 'audio'} for f in frames],
            'config': asdict(config), 'system_prompt': system_prompt}


def save_replay(destination, result, *, provenance=None):
    """Save float WAV and frame metadata without overwriting earlier runs."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('replay output already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.parent/f'.pending-{uuid.uuid4().hex}'
    pending.mkdir()
    sf.write(pending/'response.wav', result['audio'], result['sample_rate'], subtype='FLOAT')
    metadata = {k: v for k, v in result.items() if k != 'audio'}
    metadata['provenance'] = provenance or {}
    (pending/'response.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                   allow_nan=False), encoding='utf-8')
    hashes = {name: hashlib.sha256((pending/name).read_bytes()).hexdigest()
              for name in ('response.wav', 'response.json')}
    (pending/'manifest.json').write_text(json.dumps({'version': 1, 'files': hashes}), encoding='utf-8')
    for name in (*hashes, 'manifest.json'):
        with (pending/name).open('r+b') as stream:
            os.fsync(stream.fileno())
    os.replace(pending, destination)
    return destination
