from copy import deepcopy

import pytest
import torch


def test_master_updates_accumulate_below_bfloat16_resolution():
    from haru_training.optim import MasterAdamW
    parameter=torch.nn.Parameter(torch.tensor(.1,dtype=torch.bfloat16))
    initial=parameter.detach().float().clone()
    optimizer=MasterAdamW([parameter],lr=1e-6,weight_decay=0)
    for _ in range(512):
        parameter.grad=torch.ones_like(parameter)
        optimizer.step()
    master=optimizer.state[parameter]['master_param']
    assert master.dtype==torch.float32
    assert float(initial-master)==pytest.approx(512e-6,abs=1e-6)
    assert parameter.dtype==torch.bfloat16 and parameter<initial
    assert torch.equal(parameter,master.to(parameter.dtype))
    assert optimizer.state[parameter]['exp_avg'].dtype==torch.float32


def test_master_adamw_matches_float32_reference_and_calls_hooks_once():
    from haru_training.optim import MasterAdamW
    parameter=torch.nn.Parameter(torch.tensor([.1,-.2],dtype=torch.bfloat16))
    reference=torch.nn.Parameter(parameter.detach().float())
    optimizer=MasterAdamW([parameter],lr=1e-3,weight_decay=.1)
    plain=torch.optim.AdamW([reference],lr=1e-3,weight_decay=.1)
    calls=[]
    optimizer.register_step_pre_hook(lambda *args:calls.append('before'))
    optimizer.register_step_post_hook(lambda *args:calls.append('after'))
    for number in range(4):
        grad=torch.tensor([1.,number/4],dtype=parameter.dtype)
        parameter.grad=grad
        reference.grad=grad.float()
        optimizer.step();plain.step()
        torch.testing.assert_close(optimizer.state[parameter]['master_param'],reference,atol=0,rtol=0)
    assert calls==['before','after']*4


def test_master_state_roundtrip_keeps_fractional_updates_and_scheduler():
    from haru_training.optim import MasterAdamW
    parameter=torch.nn.Parameter(torch.tensor(.1,dtype=torch.bfloat16))
    optimizer=MasterAdamW([parameter],lr=1e-6,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1,(step+1)/4))
    for _ in range(3):
        parameter.grad=torch.ones_like(parameter)
        optimizer.step();scheduler.step()
    copied=torch.nn.Parameter(parameter.detach().clone())
    resumed=MasterAdamW([copied],lr=1e-6,weight_decay=0)
    resumed_scheduler=torch.optim.lr_scheduler.LambdaLR(resumed,lambda step:min(1,(step+1)/4))
    resumed.load_state_dict(deepcopy(optimizer.state_dict()))
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    for _ in range(3):
        parameter.grad=torch.ones_like(parameter);copied.grad=torch.ones_like(copied)
        optimizer.step();scheduler.step()
        resumed.step();resumed_scheduler.step()
    assert torch.equal(parameter,copied)
    for key in ('master_param','exp_avg','exp_avg_sq','step'):
        torch.testing.assert_close(optimizer.state[parameter][key],resumed.state[copied][key],atol=0,rtol=0)


def test_master_optimizer_skips_missing_gradients_and_rejects_closures():
    from haru_training.optim import MasterAdamW
    first=torch.nn.Parameter(torch.tensor(.1,dtype=torch.bfloat16))
    second=torch.nn.Parameter(torch.tensor(.2,dtype=torch.bfloat16))
    optimizer=MasterAdamW([first,second],lr=1e-3)
    first.grad=torch.ones_like(first)
    before=second.detach().clone()
    optimizer.step()
    assert torch.equal(second,before) and not optimizer.state[second]
    with pytest.raises(ValueError,match='closure'):
        optimizer.step(lambda:None)
