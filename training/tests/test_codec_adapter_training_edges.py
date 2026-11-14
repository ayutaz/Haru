from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from haru_training.checkpoint import read_checkpoint
from haru_training.codec_adapter import get_adapter
from haru_training.codec_adapter_data import make_pair
from haru_training.codec_adapter_training import CodecTrainer,TrainSpec
from test_codec_adapter_checkpoint import base
from test_codec_adapter_data import pair
from test_codec_adapter_training import trainer


def varied_trainer():
    model=base();template=pair();ids=[f'line-{i}' for i in range(3)];items=[]
    for index,name in enumerate(ids):
        meta=dict(template.metadata,id=name,group_id=name,
            source_sha256=f'{index+1:064x}',target_sha256=f'{index+101:064x}')
        items.append(make_pair(meta,template.latents+index*.2,template.spectrum*(1+index*.1),
            template.posteriors,splits={'train':ids}))
    return CodecTrainer(model,items,TrainSpec(base_model='test/tiny-haru',base_revision='c'*40,
        target_speaker='target',max_steps=5,learning_rate=.01),rank=4)


@pytest.mark.parametrize('failure',['gradient','update'])
def test_nonfinite_gradient_or_update_never_advances_or_leaves_changed_weights(tmp_path,monkeypatch,failure):
    state=trainer();adapter=get_adapter(state.model.codec);before=deepcopy(adapter.state_dict())
    if failure=='gradient':
        adapter.up.weight.register_hook(lambda value:torch.full_like(value,float('nan')))
    else:
        original=state.optimizer.step
        def step():
            original()
            with torch.no_grad():adapter.up.bias.fill_(float('nan'))
        monkeypatch.setattr(state.optimizer,'step',step)
    with pytest.raises(ValueError,match='finite'):state.fit(until_step=1)
    assert state.step==0 and state.failed
    for key,value in before.items():assert torch.equal(value,adapter.state_dict()[key])
    with pytest.raises(ValueError):state.save(tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_partial_epoch_resume_in_fresh_process_preserves_data_order_and_update(tmp_path):
    full=varied_trainer();history=full.fit();cpu_rng=torch.random.get_rng_state().clone()
    first=varied_trainer();first.fit(until_step=1)
    assert first.cursor==1 and len(first.order)==3
    first.save(tmp_path/'partial')
    script='''
from pathlib import Path
import sys,torch
from test_codec_adapter_training_edges import varied_trainer
from haru_training.checkpoint import write_checkpoint
from haru_training.codec_adapter import get_adapter
p=Path(sys.argv[1]);t=varied_trainer();torch.rand(19);t.restore(p/'partial');history=t.fit()
write_checkpoint(p/'resumed',dict(weights=get_adapter(t.model.codec).state_dict(),
    optimizer=t.optimizer.state_dict(),ids=[r['id'] for r in history],order=t.order,cursor=t.cursor,
    sampler_rng=t.generator.get_state(),cpu_rng=torch.random.get_rng_state()))
'''
    root=Path(__file__).parents[2]
    env={**os.environ,'PYTHONPATH':os.pathsep.join(str(root/p) for p in ('training/tests','training/src','haru'))}
    result=subprocess.run([sys.executable,'-X','utf8','-c',script,str(tmp_path)],env=env,
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    resumed=read_checkpoint(tmp_path/'resumed')
    assert resumed['ids']==[r['id'] for r in history[1:]]
    assert resumed['order']==full.order and resumed['cursor']==full.cursor==2
    assert torch.equal(resumed['sampler_rng'],full.generator.get_state())
    assert torch.equal(resumed['cpu_rng'],cpu_rng)
    for key,value in get_adapter(full.model.codec).state_dict().items():assert torch.equal(value,resumed['weights'][key])
    for index,state in full.optimizer.state_dict()['state'].items():
        for key,value in state.items():assert torch.equal(value,resumed['optimizer']['state'][index][key])
