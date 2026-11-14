from copy import deepcopy
from hashlib import sha256
import json

import pytest


def specification(tmp_path):
    files = {}
    for name in ('dataset', 'membership', 'reference', 'question0', 'question1', 'question2'):
        p = tmp_path / (name + '.bin')
        p.write_bytes(name.encode())
        files[name] = dict(path=p.name, sha256=sha256(p.read_bytes()).hexdigest())
    return dict(version=1, base_model='higurashi-ai/Haru-4B', base_revision='a'*40,
        dataset_sha256='b'*64, target_speaker='tsukuyomi', seed=7, rank=8,
        max_steps=30, max_seconds=600, learning_rate=.001,
        system_prompt='日本語で短く答えてください', dataset=files['dataset'],
        membership=files['membership'], reference=dict(files['reference'],
        group_id='reserved', split='reference', speaker='tsukuyomi'),
        cases=[dict(files[f'question{i}'], id=f'dev-{i}', text='質問') for i in range(3)])


def test_trial_validates_all_frozen_files_without_loading_models(tmp_path):
    from haru_training.codec_adapter_trial import validate_trial
    spec = specification(tmp_path)
    assert validate_trial(spec, root=tmp_path) == spec
    (tmp_path/'question1.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        validate_trial(spec, root=tmp_path)


@pytest.mark.parametrize('field,value', [('max_steps',31), ('max_seconds',601),
    ('max_steps',True), ('learning_rate',float('nan')), ('rank',16), ('seed',None)])
def test_trial_rejects_expanded_or_invalid_bounds(tmp_path, field, value):
    from haru_training.codec_adapter_trial import validate_trial
    spec = specification(tmp_path); spec[field] = value
    with pytest.raises(ValueError): validate_trial(spec, root=tmp_path)


def test_trial_keeps_three_distinct_questions_and_untrained_reference(tmp_path):
    from haru_training.codec_adapter_trial import validate_trial
    original = specification(tmp_path)
    for mode in ('missing', 'duplicate', 'reference', 'speaker'):
        spec = deepcopy(original)
        if mode == 'missing': spec['cases'].pop()
        if mode == 'duplicate': spec['cases'][1]['id'] = spec['cases'][0]['id']
        if mode == 'reference': spec['reference']['split'] = 'train'
        if mode == 'speaker': spec['reference']['speaker'] = 'other'
        with pytest.raises(ValueError): validate_trial(spec, root=tmp_path)


def test_cpu_or_wrong_provider_cannot_start_paid_trial(tmp_path, monkeypatch):
    from haru_training.codec_adapter_trial import require_vast_cuda
    monkeypatch.delenv('HARU_COMPUTE_PROVIDER', raising=False)
    with pytest.raises(RuntimeError): require_vast_cuda()


def test_frozen_conversation_trace_compares_every_frame_and_rng():
    from haru_training.codec_adapter_trial import compare_traces
    trace = dict(tokens=[1,2], feedback=['a','b'], rng=['c','d'],
                 global_cpu='e', global_cuda='f', head_calls=2)
    assert compare_traces(trace, deepcopy(trace))
    for key in trace:
        changed = deepcopy(trace); changed[key] = None
        assert not compare_traces(trace, changed)


def test_capture_records_real_tiny_session_latents_without_extra_head_calls(monkeypatch):
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from test_training_forward import tiny_model
    from haru_training.codec_adapter_trial import capture, compare_traces
    from haru_training.codec_adapter import attach_adapter
    torch.set_num_threads(1); torch.manual_seed(13)
    model = tiny_model().eval().requires_grad_(False)
    model.tokenizer = Tokenizer(WordLevel({f'token{i}':i for i in range(128)}, unk_token='token3'))
    spec = dict(seed=7, system_prompt='test')
    user = np.zeros(4*3200, dtype=np.float32)
    def forbidden_cuda(*args, **kwargs):
        raise AssertionError('CPU trace must not initialize the local CUDA runtime')
    monkeypatch.setattr(torch.cuda, 'get_rng_state', forbidden_cuda)
    first, latent, trace = capture(model, user, None, spec)
    adapter = attach_adapter(model, rank=4)
    with torch.no_grad(): adapter.up.bias.fill_(.02)
    model.requires_grad_(False)
    other, other_latent, other_trace = capture(model, user, None, spec)
    assert compare_traces(trace, other_trace)
    assert trace['head_calls'] == 4
    assert latent.shape == (1,4,64) and torch.equal(latent, other_latent)
    assert not np.array_equal(first['audio'], other['audio'])
