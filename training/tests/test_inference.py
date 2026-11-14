"""Inference artifacts contain selected weights, never optimizer/RNG state."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from haru_training.checkpoint import read_checkpoint, write_checkpoint
from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.inference import export_inference, apply_inference, load_inference
from test_phase4_haru import tiny_trainer, tensor_digest
from test_training_forward import tiny_model


def base():
    torch.manual_seed(13)
    return tiny_model()


def configured(mode):
    model = base()
    configure_trainable(model, TrainingConfig(mode=mode, train_projectors=mode == 'lm_lora'))
    return model


def probe(model):
    model.eval()
    with torch.no_grad():
        hidden = model(model.embed(torch.tensor([[3, 4, 5]])), model.new_cache())
        noise = torch.ones(1, 64)
        return model.logits(hidden).detach(), model.head.sample(hidden[:, -1], noise, 1.5).detach()


@pytest.mark.parametrize('mode', ['head', 'lm_lora', 'head_output'])
def test_export_and_apply_reproduce_inference_and_fresh_base_restores_all_parts(tmp_path, mode):
    trainer = tiny_trainer(tmp_path/'train', mode)
    original = tensor_digest(base().named_parameters())
    trainer.fit()
    path = export_inference(trainer, tmp_path/'inference')
    bundle = read_checkpoint(path)
    assert set(bundle) == {'kind', 'version', 'spec', 'config_sha256', 'frozen_sha256',
                           'buffers_sha256', 'weights', 'step'}
    assert bundle['kind'] == 'haru-inference' and bundle['step'] == 3
    assert set(bundle['weights']) == {n for n, p in trainer.model.named_parameters() if p.requires_grad}
    loaded = configured(mode)
    apply_inference(loaded, path, base_model='test/tiny-haru', base_revision='a'*40)
    expected = probe(trainer.model)
    for a, b in zip(probe(loaded), expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert tensor_digest(base().named_parameters()) == original
    assert tensor_digest(base().head.named_parameters()) != tensor_digest(loaded.head.named_parameters())
    if mode == 'lm_lora':
        assert tensor_digest(base().model_audio_proj.named_parameters()) != tensor_digest(loaded.model_audio_proj.named_parameters())
    with pytest.raises(FileExistsError): export_inference(trainer, path)


@pytest.mark.parametrize('bad', ['base', 'revision', 'frozen', 'buffer', 'config', 'mode',
                               'missing', 'extra', 'shape', 'dtype', 'nan', 'corrupt'])
def test_rejected_artifacts_do_not_mutate_model(tmp_path, bad):
    trainer = tiny_trainer(tmp_path/'train', 'head')
    path = export_inference(trainer, tmp_path/'valid')
    model = configured('head')
    identity = {'base_model': 'test/tiny-haru', 'base_revision': 'a'*40}
    if bad in ('base', 'revision'):
        identity['base_model' if bad == 'base' else 'base_revision'] = 'wrong/base' if bad == 'base' else 'f'*40
    elif bad == 'frozen':
        next(model.codec.parameters()).data.add_(1)
    elif bad == 'buffer': model.latent_scale.add_(1)
    elif bad == 'config': model.config = replace(model.config, pad_token_id=7)
    elif bad == 'mode': model = configured('lm_lora')
    elif bad == 'corrupt': (path/'tensors.safetensors').write_bytes(b'broken')
    else:
        bundle = read_checkpoint(path)
        key = next(iter(bundle['weights']))
        if bad == 'missing': del bundle['weights'][key]
        if bad == 'extra': bundle['weights']['codec.bad'] = torch.ones(1)
        if bad == 'shape': bundle['weights'][key] = torch.ones(1)
        if bad == 'dtype': bundle['weights'][key] = bundle['weights'][key].double()
        if bad == 'nan': bundle['weights'][key].fill_(float('nan'))
        path = write_checkpoint(tmp_path/'invalid', bundle)
    before = tensor_digest(list(model.named_parameters())+list(model.named_buffers()))
    modes = {n: m.training for n, m in model.named_modules()}
    with pytest.raises(ValueError): apply_inference(model, path, **identity)
    assert before == tensor_digest(list(model.named_parameters())+list(model.named_buffers()))
    assert modes == {n: m.training for n, m in model.named_modules()}


@pytest.mark.parametrize('mode', ['head', 'lm_lora', 'head_output'])
def test_artifact_loads_in_a_fresh_process_without_optimizer_or_training_records(tmp_path, mode):
    trainer = tiny_trainer(tmp_path/'train', mode)
    trainer.fit()
    path = export_inference(trainer, tmp_path/'inference')
    expected = tensor_digest(trainer.model.named_parameters())
    script = '''
import sys
from test_inference import configured
from test_phase4_haru import tensor_digest
from haru_training.inference import apply_inference
m=configured(sys.argv[2])
apply_inference(m,sys.argv[1],base_model='test/tiny-haru',base_revision='a'*40)
assert tensor_digest(m.named_parameters())==sys.argv[3]
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.resolve())+os.pathsep+env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-c', script, str(path), mode, expected], env=env, check=True)


def test_loader_builds_the_pinned_base_and_preserves_original_tokenizer(tmp_path, monkeypatch):
    import haru.models.haru
    trainer = tiny_trainer(tmp_path/'train', 'lm_lora')
    trainer.fit()
    path = export_inference(trainer, tmp_path/'inference')
    seen = []
    token = object()
    def factory(source, **kwargs):
        seen.append((source, kwargs))
        model = base()
        model.tokenizer = token
        return model
    monkeypatch.setattr(haru.models.haru.HaruModel, 'from_pretrained', factory)
    loaded = load_inference(path, device='cpu', dtype=torch.float32)
    assert seen == [('test/tiny-haru', {'revision': 'a'*40, 'device': 'cpu', 'dtype': torch.float32})]
    assert loaded.tokenizer is token
    for a, b in zip(probe(loaded), probe(trainer.model)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_export_refuses_failed_midstep_or_changed_frozen_models(tmp_path):
    trainer = tiny_trainer(tmp_path/'train', 'head')
    trainer.failed = True
    with pytest.raises(RuntimeError): export_inference(trainer, tmp_path/'failed')
    trainer.failed = False
    trainer._in_step = True
    with pytest.raises(RuntimeError): export_inference(trainer, tmp_path/'midstep')
    trainer._in_step = False
    next(trainer.model.codec.parameters()).data.add_(1)
    with pytest.raises(ValueError): export_inference(trainer, tmp_path/'frozen')


@pytest.mark.parametrize('mode',['head','head_output'])
def test_feedback_loader_keeps_original_head_separate_from_applied_voice_weights(tmp_path,monkeypatch,mode):
    import haru.models.haru
    from haru_training.inference import load_with_frozen_feedback
    trainer=tiny_trainer(tmp_path/'train',mode);trainer.fit()
    path=export_inference(trainer,tmp_path/'artifact')
    original=tensor_digest(base().head.named_parameters())
    monkeypatch.setattr(haru.models.haru.HaruModel,'from_pretrained',lambda *a,**k:base())
    model,feedback=load_with_frozen_feedback(path,device='cpu',dtype=torch.float32)
    assert tensor_digest(feedback.named_parameters())==original
    assert tensor_digest(model.head.named_parameters())==tensor_digest(trainer.model.head.named_parameters())
    assert tensor_digest(model.head.named_parameters())!=original
    assert not feedback.training and not any(p.requires_grad for p in feedback.parameters())
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(feedback.parameters(),model.head.parameters()))


@pytest.mark.parametrize('change',['lm_lora','projectors'])
def test_feedback_loader_rejects_changes_to_lm_or_projectors_before_loading_base(tmp_path,monkeypatch,change):
    import haru.models.haru
    from haru_training.inference import load_with_frozen_feedback
    trainer=tiny_trainer(tmp_path/'train','lm_lora' if change=='lm_lora' else 'head')
    path=export_inference(trainer,tmp_path/'artifact')
    if change=='projectors':
        bundle=read_checkpoint(path);bundle['spec']['train_projectors']=True
        path=write_checkpoint(tmp_path/'projectors',bundle)
    def forbidden(*args,**kwargs): raise AssertionError('must reject before base download')
    monkeypatch.setattr(haru.models.haru.HaruModel,'from_pretrained',forbidden)
    with pytest.raises(ValueError,match='feedback'):
        load_with_frozen_feedback(path,device='cpu',dtype=torch.float32)
