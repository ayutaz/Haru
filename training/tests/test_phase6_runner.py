from hashlib import sha256
import json
from pathlib import Path

import pytest

from test_dialogue import fixture
from haru_training.phase6 import preflight, predecessor


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def assets(tmp_path):
    row, _ = fixture(tmp_path)
    write(tmp_path/'dialogue.json', {'records': [row]})
    review = {'manifest': 'dialogue.json',
              'manifest_sha256': sha256((tmp_path/'dialogue.json').read_bytes()).hexdigest(),
              'quality_status': 'reviewed_acceptable_for_replay',
              'source_and_usage_notes': 'Fixture only, no claim about real voices or usage.',
              'voice_and_pronunciation_notes': 'Artificial test PCM, not a real quality review.',
              'reference_policy': 'Use the model-channel speaker reference, not target-voice substitution.',
              'source_split_checked': True}
    write(tmp_path/'review.json', review)
    write(tmp_path/'splits.json', {'splits': {'train': ['target-train'],
        'validation': ['target-val'], 'test': ['target-test'], 'reference': ['target-ref']}})
    return review


def test_review_requires_real_files_and_rejects_unknown_quality_before_gpu(tmp_path):
    review = assets(tmp_path)
    result = preflight(tmp_path, tmp_path/'review.json')
    assert result['dialogues'] == 1 and result['manual_review_recorded'] is True
    review['quality_status'] = 'unknown'
    write(tmp_path/'review.json', review)
    with pytest.raises(ValueError, match='review'):
        preflight(tmp_path, tmp_path/'review.json')


def test_review_cannot_override_changed_audio_or_heldout_source_lineage(tmp_path):
    assets(tmp_path)
    data = json.loads((tmp_path/'dialogue.json').read_text(encoding='utf-8'))
    data['records'][0]['source_ids'] = ['target-test']
    write(tmp_path/'dialogue.json', data)
    with pytest.raises(ValueError, match='manifest'):
        preflight(tmp_path, tmp_path/'review.json')
    review = json.loads((tmp_path/'review.json').read_text(encoding='utf-8'))
    review['manifest_sha256'] = sha256((tmp_path/'dialogue.json').read_bytes()).hexdigest()
    write(tmp_path/'review.json', review)
    with pytest.raises(ValueError, match='leak'):
        preflight(tmp_path, tmp_path/'review.json')


def test_review_does_not_replace_explicit_timed_model_chunks(tmp_path):
    review = assets(tmp_path)
    data = json.loads((tmp_path/'dialogue.json').read_text(encoding='utf-8'))
    del data['records'][0]['model_chunks']
    write(tmp_path/'dialogue.json', data)
    review['manifest_sha256'] = sha256((tmp_path/'dialogue.json').read_bytes()).hexdigest()
    write(tmp_path/'review.json', review)
    with pytest.raises(ValueError, match='chunk'):
        preflight(tmp_path, tmp_path/'review.json')


def test_each_stage_requires_the_previous_result_and_D_requires_stability(tmp_path):
    with pytest.raises(ValueError): predecessor(tmp_path, 'B')
    (tmp_path/'A').mkdir()
    write(tmp_path/'A/result.json', {'zero_update_parity': True})
    predecessor(tmp_path, 'B')
    with pytest.raises(ValueError): predecessor(tmp_path, 'C')
    (tmp_path/'B').mkdir()
    write(tmp_path/'B/result.json', {'steps': 1, 'completed': False, 'stopped_for_quality': True})
    with pytest.raises(ValueError, match='incomplete'): predecessor(tmp_path, 'C')
    write(tmp_path/'B/training.json', {'condition': 'B', 'steps': 1})
    predecessor(tmp_path, 'C')  # An alternate update group can still be diagnosed.
    (tmp_path/'C').mkdir()
    write(tmp_path/'C/result.json', {'steps': 1, 'completed': False, 'stopped_for_quality': True})
    write(tmp_path/'C/training.json', {'condition': 'C', 'steps': 1})
    with pytest.raises(ValueError, match='stable'): predecessor(tmp_path, 'D')
    write(tmp_path/'C/result.json', {'steps': 3, 'completed': True, 'stopped_for_quality': False})
    write(tmp_path/'C/training.json', {'condition': 'C', 'steps': 3})
    predecessor(tmp_path, 'D')


def test_local_gpu_is_rejected_before_loading_weights(monkeypatch):
    from haru_training.phase6 import main
    monkeypatch.delenv('HARU_COMPUTE_PROVIDER', raising=False)
    monkeypatch.setattr('sys.argv', ['phase6', 'A', '--inputs', 'none', '--output', 'none', '--review', 'none'])
    with pytest.raises(RuntimeError, match='vast.ai'):
        main()


def test_condition_execution_reloads_base_and_writes_all_boundaries(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file
    from test_trainer import Regression, records, collate, objective
    from haru_training.trainer import Trainer
    import haru_training.phase6 as runner
    models, configs = [], []
    def load():
        model = Regression()
        models.append(model)
        return model
    train, val = records(), records('val')
    monkeypatch.setattr(runner, '_model', load)
    monkeypatch.setattr(runner, 'configure_trainable', lambda model, config: configs.append(config))
    monkeypatch.setattr(runner, 'preflight', lambda *args: {'review_sha256': 'a'*64})
    monkeypatch.setattr(runner, '_contract', lambda *args: {'seed': 20261114,
        'inputs': {'splits.json': 'b'*64, 'latents/manifest.json': 'c'*64}})
    monkeypatch.setattr(runner, '_sequences', lambda *args: (train, val, {'train': ['x'], 'validation': ['y']}))
    monkeypatch.setattr(runner, 'Trainer', lambda model, train, val, spec, path:
        Trainer(model, train, val, spec, path, collate_fn=collate, loss_fn=objective))
    def outputs(model, inputs):
        number = float(model.weight.detach())
        arrays = {'audio': np.full(3200, number, np.float32),
                  'logits': np.array([[number]], np.float32),
                  'latent': np.array([[number]], np.float32), 'tokens': np.array([3], np.int64)}
        return {'audio': arrays['audio'], 'text': '今日は晴れです'}, arrays
    def record(path, result, arrays, inputs):
        path.mkdir()
        save_file(arrays, str(path/'probe.safetensors'))
    def export(trainer, path):
        assert trainer.model.weight.requires_grad
        path.mkdir()
        return path
    monkeypatch.setattr(runner, '_outputs', outputs)
    monkeypatch.setattr(runner, '_record', record)
    monkeypatch.setattr(runner, 'export_inference', export)
    monkeypatch.setattr(runner.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(runner.torch.cuda, 'max_memory_allocated', lambda: 0)
    for condition in ('A', 'B', 'C', 'D'):
        result = runner.execute(condition, tmp_path, tmp_path/'out', tmp_path/'unused-review')
        if condition != 'A':
            assert result['steps'] == 3 and result['completed']
            assert [row['step'] for row in result['observations']] == [0, 1, 2, 3]
    assert len(models) == 4 and len({id(m) for m in models}) == 4
    assert configs[1].mode == 'head' and not configs[1].train_projectors
    assert configs[2].mode == 'lm_lora' and not configs[2].train_projectors
    assert configs[3].train_projectors


def test_changed_contract_is_rejected_before_the_model_loader(tmp_path, monkeypatch):
    import haru_training.phase6 as runner
    (tmp_path/'A').mkdir()
    write(tmp_path/'A/result.json', {'zero_update_parity': True})
    write(tmp_path/'A/contract.json', {'seed': 7})
    monkeypatch.setattr(runner, 'preflight', lambda *args: {})
    monkeypatch.setattr(runner, '_contract', lambda *args: {'seed': 23})
    monkeypatch.setattr(runner, '_model', lambda: pytest.fail('must reject before loading a GPU model'))
    with pytest.raises(ValueError, match='conditions changed'):
        runner.execute('B', tmp_path, tmp_path, tmp_path/'review')


def test_source_identity_includes_the_runtime_haru_implementation(tmp_path, monkeypatch):
    import haru
    import haru_training.phase6 as runner
    package = tmp_path/'haru'
    package.mkdir()
    (package/'__init__.py').write_text('', encoding='utf-8')
    module = package/'model.py'
    module.write_text('version = 1', encoding='utf-8')
    monkeypatch.setattr(haru, '__file__', str(package/'__init__.py'))
    before = runner.source_fingerprint()
    assert 'haru/model.py' in before
    module.write_text('version = 2', encoding='utf-8')
    assert runner.source_fingerprint()['haru/model.py'] != before['haru/model.py']
    assert all(not Path(p).is_absolute() for p in before)
