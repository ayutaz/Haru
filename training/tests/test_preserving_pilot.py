from dataclasses import replace

import pytest
import torch

from haru_training.data import make_sequence
from haru_training.checkpoint import read_checkpoint
from test_training_forward import tiny_model


def records(prefix):
    return [make_sequence(f'{prefix}{i}',[3],torch.ones(1,64),torch.ones(4,64)*.1,
        torch.zeros(4,64),torch.tensor([4,5,6,0]),0,1) for i in range(3)]


def test_content_preserving_pilot_updates_exports_and_records_the_new_loss(tmp_path):
    from haru_training.content_preservation import preserving_spec,run_preserving_pilot
    torch.set_num_threads(1);torch.manual_seed(3)
    model=tiny_model();train,val=records('t'),records('v')
    spec=preserving_spec(train,val,split_sha256='a'*64,cache_sha256='b'*64,
                         preservation_weight=1.,max_steps=2)
    spec=replace(spec,base_model='test/tiny-haru',base_revision='a'*40)
    seen=[]
    result=run_preserving_pilot(model,train,val,spec,tmp_path,render=lambda m,p,s:seen.append(s))
    assert result['complete'] and result['completed_updates']==2
    assert result['frozen_weights_unchanged'] and result['preservation_weight']==1
    assert result['voice_quality']=='unknown' and not result['final_test_used']
    assert seen==[0,2]
    assert not (tmp_path/'trainer/checkpoints').exists()
    exported=read_checkpoint(tmp_path/'inference/step-00000002')
    assert exported['spec']['preservation_weight']==1.
    assert all(n.startswith('head.') for n in exported['weights'])
    assert result['observations'][-1]['metrics']['preservation_loss']>=-1e-6


@pytest.mark.parametrize('change',[dict(preservation_weight=.5),dict(learning_rate=1e-5),
    dict(mode='head_output'),dict(max_steps=101),dict(feedback_mode='separate_frozen_head')])
def test_preserving_trial_rejects_unregistered_recipe_before_outputs(tmp_path,change):
    from haru_training.content_preservation import preserving_spec,run_preserving_pilot
    train,val=records('t'),records('v')
    spec=preserving_spec(train,val,split_sha256='a'*64,cache_sha256='b'*64,preservation_weight=0.)
    spec=replace(spec,**change)
    with pytest.raises(ValueError):run_preserving_pilot(tiny_model(),train,val,spec,tmp_path/'run',render=lambda *a:None)
    assert not (tmp_path/'run').exists()
