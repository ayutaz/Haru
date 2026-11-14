"""FP32 AdamW master weights with unchanged model/forward parameter dtypes."""

from copy import deepcopy

import torch


class MasterAdamW(torch.optim.Optimizer):
    """Accumulate sub-BF16 updates without changing the inference weight format.

    The public parameter groups refer to model parameters for scheduler/hooks.
    PyTorch AdamW operates on FP32 copies. Checkpoints include those copies and
    FP32 moments, so resume preserves updates not yet visible in BF16 weights.
    """

    def __init__(self,params,lr=1e-3,weight_decay=.01):
        super().__init__(params,{'lr':lr,'weight_decay':weight_decay})
        self._masters={p:torch.nn.Parameter(p.detach().float().clone())
                       for group in self.param_groups for p in group['params']}
        if any(p.dtype not in (torch.float32,torch.float16,torch.bfloat16) for p in self._masters):
            raise ValueError('master AdamW requires float32, float16 or bfloat16 parameters')
        self._inner=torch.optim.AdamW([
            {**group,'params':[self._masters[p] for p in group['params']]}
            for group in self.param_groups],lr=lr,weight_decay=weight_decay)
        for group,inner in zip(self.param_groups,self._inner.param_groups):
            group.update({key:value for key,value in inner.items() if key!='params'})

    def _sync_state(self):
        self.state.clear()
        for parameter,master in self._masters.items():
            if self._inner.state[master]:
                self.state[parameter]={**self._inner.state[master],'master_param':master.detach()}

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:
            raise ValueError('master AdamW does not support closures; supply computed gradients')
        for group,inner in zip(self.param_groups,self._inner.param_groups):
            inner.update({key:value for key,value in group.items() if key!='params'})
        for parameter,master in self._masters.items():
            if not self._inner.state[master]:
                master.copy_(parameter)
            master.grad=parameter.grad.detach().float() if parameter.grad is not None else None
        self._inner.step()
        for parameter,master in self._masters.items():
            if master.grad is not None:
                parameter.copy_(master)
            master.grad=None
        self._sync_state()

    def load_state_dict(self,state_dict):
        saved=deepcopy(state_dict)
        masters={key:state.pop('master_param') for key,state in saved['state'].items()}
        self._inner.load_state_dict(saved)
        for group,inner,ids in zip(self.param_groups,self._inner.param_groups,state_dict['param_groups']):
            group.update({key:value for key,value in inner.items() if key!='params'})
            for parameter,index in zip(group['params'],ids['params']):
                if index not in masters:
                    continue
                value=masters[index]
                if (value.shape!=parameter.shape or value.dtype!=torch.float32
                        or not torch.isfinite(value).all()):
                    raise ValueError('invalid FP32 master weight')
                with torch.no_grad():
                    self._masters[parameter].copy_(value)
        self._sync_state()
