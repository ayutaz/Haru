import numpy as np
import pytest

from haru_training.comparison import measure_repeatability, compare_outputs


def output(value=0.):
    return {'audio': np.array([0., value], np.float32),
            'logits': np.array([1., 2.], np.float32),
            'latent': np.array([3.], np.float32), 'tokens': [3, 4]}


def test_repeatability_sets_each_tolerance_before_candidate_comparison():
    a, b = output(), output(.125)
    policy = measure_repeatability(a, b)
    assert policy == {'audio': .125, 'logits': 0., 'latent': 0., 'tokens': 'exact'}
    assert compare_outputs(a, b, policy)['passed']
    assert not compare_outputs(a, output(.25), policy)['passed']
    assert policy['audio'] == .125


def test_logit_latent_and_token_errors_cannot_hide_behind_audio_tolerance():
    policy = measure_repeatability(output(), output())
    for key in ('audio', 'logits', 'latent', 'tokens'):
        changed = output()
        changed[key][-1] += 1
        assert not compare_outputs(output(), changed, policy)['passed']
    changed = output()
    changed['audio'] = np.ones(3)
    with pytest.raises(ValueError): compare_outputs(output(), changed, policy)
    changed = output()
    changed['logits'][0] = float('nan')
    with pytest.raises(ValueError): measure_repeatability(output(), changed)


def test_nondeterministic_baseline_tokens_require_investigation():
    changed = output()
    changed['tokens'][0] = 5
    with pytest.raises(ValueError): measure_repeatability(output(), changed)
