from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_trainer import build, objective


def test_checkpoint_resumes_weights_optimizer_scheduler_order_and_rng_in_a_new_process(tmp_path):
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    uninterrupted = build(tmp_path/'full', learning_rate=.01, warmup_steps=2)
    interrupted = build(tmp_path/'split', learning_rate=.01, warmup_steps=2)
    uninterrupted.fit()
    interrupted.fit(until_step=1)
    checkpoint = interrupted.save()
    script = '''
import sys,random,numpy as np,torch
from pathlib import Path
from test_trainer import build
random.seed(999);np.random.seed(999);torch.manual_seed(999)
trainer=build(Path(sys.argv[2]),learning_rate=.01,warmup_steps=2)
trainer.resume(Path(sys.argv[1]))
trainer.fit()
trainer.save()
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.resolve())+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-c',script,str(checkpoint),str(tmp_path/'child')],check=True,env=env)
    resumed = build(tmp_path/'loaded', learning_rate=.01, warmup_steps=2)
    resumed.resume(tmp_path/'child/checkpoints/step-00000003')
    assert asdict(resumed.state)==asdict(uninterrupted.state)
    torch.testing.assert_close(resumed.model.weight,uninterrupted.model.weight,atol=0,rtol=0)
    assert resumed.scheduler.state_dict()==uninterrupted.scheduler.state_dict()
    for key in ('exp_avg','exp_avg_sq','step'):
        torch.testing.assert_close(resumed.optimizer.state[resumed.model.weight][key],
            uninterrupted.optimizer.state[uninterrupted.model.weight][key],atol=0,rtol=0)


def test_rng_and_frozen_parameters_are_preserved(tmp_path):
    trainer=build(tmp_path/'first')
    trainer.fit(until_step=1)
    random.seed(3);np.random.seed(4);torch.manual_seed(5)
    path=trainer.save()
    expected=(random.random(),np.random.rand(),torch.rand(3))
    loaded=build(tmp_path/'loaded')
    loaded.model.weight.data.fill_(-8)
    loaded.resume(path)
    actual=(random.random(),np.random.rand(),torch.rand(3))
    assert actual[:2]==expected[:2]
    torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)
    assert loaded.model.frozen==2 and loaded.model.frozen.grad is None
    with pytest.raises(FileExistsError): trainer.save()


@pytest.mark.parametrize('field,value',[('base_revision','d'*40),('cache_sha256','e'*64),
    ('split_sha256','f'*64),('learning_rate',.5),('accumulation_steps',2)])
def test_incompatible_contract_is_rejected_without_mutating_model(tmp_path,field,value):
    trainer=build(tmp_path/'first');trainer.fit(until_step=1);path=trainer.save()
    loaded=build(tmp_path/'loaded');loaded.spec=replace(loaded.spec,**{field:value})
    before=loaded.model.weight.detach().clone()
    with pytest.raises(ValueError): loaded.resume(path)
    torch.testing.assert_close(before,loaded.model.weight,atol=0,rtol=0)
    assert loaded.state.step==0


@pytest.mark.parametrize('filename',['tensors.safetensors','metadata.json','manifest.json'])
def test_corrupt_or_partial_checkpoint_is_rejected_before_mutation(tmp_path,filename):
    trainer=build(tmp_path/'first');trainer.fit(until_step=1);path=trainer.save()
    (path/filename).write_bytes(b'broken')
    loaded=build(tmp_path/'loaded')
    before=loaded.model.weight.detach().clone()
    with pytest.raises(ValueError): loaded.resume(path)
    torch.testing.assert_close(before,loaded.model.weight,atol=0,rtol=0)


def test_wrong_frozen_base_or_trainable_signature_is_rejected(tmp_path):
    trainer=build(tmp_path/'first');path=trainer.save()
    loaded=build(tmp_path/'loaded')
    loaded.model.frozen.data.add_(1)
    with pytest.raises(ValueError): loaded.resume(path)
    loaded=build(tmp_path/'other')
    loaded.model.frozen.requires_grad_(True)
    with pytest.raises(ValueError): loaded.resume(path)


def test_interrupted_write_leaves_no_complete_checkpoint(tmp_path,monkeypatch):
    trainer=build(tmp_path)
    def crash(*args,**kwargs): raise OSError('simulated rename failure')
    monkeypatch.setattr('haru_training.checkpoint.os.replace',crash)
    with pytest.raises(OSError): trainer.save()
    destination=tmp_path/'checkpoints/step-00000000'
    assert not destination.exists()
    pending=list(destination.parent.glob('.pending-*'))
    assert len(pending)==1
    with pytest.raises(ValueError): trainer.resume(pending[0])


def test_failure_can_only_recover_by_loading_last_complete_boundary(tmp_path):
    trainer=build(tmp_path)
    trainer.fit(until_step=1);path=trainer.save()
    trainer.loss_fn=lambda *a,**k:{'text_loss':trainer.model.weight*float('nan'),
                                  'audio_loss':trainer.model.weight*0}
    with pytest.raises(FloatingPointError):trainer.fit()
    trainer.loss_fn=objective
    trainer.resume(path)
    trainer.fit()
    assert trainer.state.step==3 and not trainer.failed
    assert (tmp_path/'failure.json').exists()


def test_checkpoint_cannot_be_saved_mid_accumulation(tmp_path):
    trainer=build(tmp_path)
    def saving(*args,**kwargs):
        trainer.save()
        return objective(*args,**kwargs)
    trainer.loss_fn=saving
    with pytest.raises(RuntimeError,match='boundary'):trainer.fit()


def test_scheduled_checkpoints_are_complete_and_do_not_overwrite(tmp_path):
    trainer=build(tmp_path,checkpoint_interval=2)
    trainer.fit()
    assert sorted(p.name for p in (tmp_path/'checkpoints').iterdir())==['step-00000002','step-00000003']
    loaded=build(tmp_path/'loaded',checkpoint_interval=2)
    loaded.resume(tmp_path/'checkpoints/step-00000002')
    loaded.fit()
    torch.testing.assert_close(trainer.model.weight,loaded.model.weight,atol=0,rtol=0)


@pytest.mark.parametrize('bad',['shape','nonfinite'])
def test_semantically_invalid_optimizer_state_is_rejected_before_mutation(tmp_path,bad):
    from haru_training.checkpoint import read_checkpoint,write_checkpoint
    trainer=build(tmp_path/'first');trainer.fit(until_step=1)
    bundle=read_checkpoint(trainer.save())
    state=next(iter(bundle['optimizer']['state'].values()))
    state['exp_avg']=torch.ones(2,dtype=torch.float64) if bad=='shape' else torch.tensor(float('nan'),dtype=torch.float64)
    path=write_checkpoint(tmp_path/'invalid',bundle)
    loaded=build(tmp_path/'loaded')
    before=loaded.model.weight.detach().clone()
    with pytest.raises(ValueError):loaded.resume(path)
    torch.testing.assert_close(before,loaded.model.weight,atol=0,rtol=0)
