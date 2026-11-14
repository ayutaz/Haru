"""Predeclare inference reload tolerances from a baseline repeated unchanged."""

import numpy as np


def _differences(first, second):
    errors = {}
    for key in ('audio', 'logits', 'latent'):
        a, b = np.asarray(first[key], dtype=np.float64), np.asarray(second[key], dtype=np.float64)
        if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('comparison requires matching, finite nonempty outputs')
        errors[key] = float(np.max(np.abs(a-b)))
    errors['tokens'] = list(first['tokens']) == list(second['tokens'])
    return errors


def measure_repeatability(first, second):
    errors = _differences(first, second)
    if not errors['tokens']:
        raise ValueError('baseline tokens are not reproducible; investigate before adaptation')
    return {**{k: errors[k] for k in ('audio', 'logits', 'latent')}, 'tokens': 'exact'}


def compare_outputs(first, second, policy):
    if (set(policy) != {'audio', 'logits', 'latent', 'tokens'} or policy['tokens'] != 'exact'
            or any(not np.isfinite(policy[k]) or policy[k] < 0 for k in ('audio', 'logits', 'latent'))):
        raise ValueError('invalid repeatability policy')
    errors = _differences(first, second)
    return {'passed': errors['tokens'] and all(errors[k] <= policy[k] for k in ('audio', 'logits', 'latent')),
            'max_absolute_errors': {k: errors[k] for k in ('audio', 'logits', 'latent')},
            'tokens_equal': errors['tokens'], 'policy': dict(policy)}


def compare_waveforms(first, second):
    """Compare decoded samples exactly, retaining file hashes separately.

    WAV metadata such as the PEAK timestamp can change even when its waveform
    does not. A rate/shape mismatch or nonfinite/non-mono signal is rejected;
    every finite sample must still match with zero numerical tolerance.
    """
    from hashlib import sha256
    import io

    import soundfile as sf

    a, first_rate = sf.read(io.BytesIO(first), dtype='float64')
    b, second_rate = sf.read(io.BytesIO(second), dtype='float64')
    if (a.ndim != 1 or b.shape != a.shape or not a.size or first_rate != second_rate
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('waveform comparison requires matching-rate finite nonempty mono samples')
    def pcm_hash(value):
        canonical = value.astype('<f8', copy=True)
        canonical[canonical == 0] = 0.  # Signed zero has the same amplitude.
        return sha256(str(first_rate).encode() + b'\0' + canonical.tobytes()).hexdigest()
    return {'passed': bool(np.array_equal(a, b)), 'max_absolute_error': float(np.max(np.abs(a-b))),
            'sample_rate': first_rate, 'samples': len(a),
            'file_bytes_equal': first == second,
            'first_file_sha256': sha256(first).hexdigest(), 'second_file_sha256': sha256(second).hexdigest(),
            'first_pcm_sha256': pcm_hash(a), 'second_pcm_sha256': pcm_hash(b)}
