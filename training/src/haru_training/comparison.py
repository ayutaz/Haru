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
