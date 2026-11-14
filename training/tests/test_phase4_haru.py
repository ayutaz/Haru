from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.data import make_sequence
from haru_training.trainer import RunSpec, Trainer, dataset_fingerprint
from test_training_forward import tiny_model


def tensor_digest(values):
    digest=hashlib.sha256()
    for name,value in values:
        digest.update(name.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def tiny_trainer(output,mode):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model=tiny_model()
    configure_trainable(model,TrainingConfig(mode=mode,train_projectors=mode=='lm_lora'))
    def make(prefix):
        return [make_sequence(f'{prefix}-{i}',[3,4],torch.randn(2,64),torch.randn(n,64),
                 torch.randn(n,64),torch.tensor([5]*n),0,1) for i,n in enumerate([2,3,4])]
    train,val=make('train'),make('val')
    spec=RunSpec(base_model='test/tiny-haru',base_revision='a'*40,
        dataset_sha256=dataset_fingerprint(train),validation_sha256=dataset_fingerprint(val),
        split_sha256='b'*64,cache_sha256='c'*64,max_steps=3,accumulation_steps=2,
        learning_rate=.001,warmup_steps=2,mode=mode,train_projectors=mode=='lm_lora')
    return Trainer(model,train,val,spec,output)


@pytest.mark.parametrize('mode',['head','lm_lora','head_output'])
def test_tiny_haru_forward_and_adam_resume_exactly_in_another_process(tmp_path,mode):
    full=tiny_trainer(tmp_path/'full',mode)
    frozen=full.frozen_digest
    full.fit()
    split=tiny_trainer(tmp_path/'split',mode)
    split.fit(until_step=1);path=split.save()
    child=tmp_path/'child'
    script='''
import sys
from pathlib import Path
from test_phase4_haru import tiny_trainer
t=tiny_trainer(Path(sys.argv[2]),sys.argv[3]);t.resume(Path(sys.argv[1]));t.fit();t.save()
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).parent.resolve())+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-c',script,str(path),str(child),mode],env=env,check=True)
    loaded=tiny_trainer(tmp_path/'loaded',mode);loaded.resume(child/'checkpoints/step-00000003')
    assert asdict(loaded.state)==asdict(full.state)
    assert tensor_digest(loaded.model.named_parameters())==tensor_digest(full.model.named_parameters())
    assert tensor_digest((n,p) for n,p in loaded.model.named_parameters() if not p.requires_grad)==frozen
    for (name,p),(other,q) in zip(full.model.named_parameters(),loaded.model.named_parameters()):
        assert name==other
        if p in full.optimizer.state:
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(full.optimizer.state[p][key],loaded.optimizer.state[q][key],atol=0,rtol=0)
