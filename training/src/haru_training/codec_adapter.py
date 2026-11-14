"""Pointwise decoder adaptation, leaving raw latents and feedback untouched."""
import torch
from torch import nn
import torch.nn.functional as F


class DecoderAdapter(nn.Module):
    def __init__(self,channels,rank,*,device,dtype):
        super().__init__()
        # Do not consume a conversation's CPU/CUDA random stream at installation.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.down=nn.Conv1d(channels,rank,1,device='cpu')
            self.up=nn.Conv1d(rank,channels,1,device='cpu')
            nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.to(device=device,dtype=dtype)
        self.rank=rank;self.enabled=True

    def forward(self,x):
        return x+self.up(F.silu(self.down(x))) if self.enabled else x


class AdapterStem(nn.Module):
    def __init__(self,base,adapter):
        super().__init__();self.base=base;self.adapter=adapter

    def state_shape(self):
        return self.base.state_shape()

    def forward(self,x,state):
        value,state=self.base(x,state)
        return self.adapter(value),state


def attach_adapter(model,*,rank=8):
    codec=model.codec
    if getattr(model,'inference_feedback_mode',None) is not None:
        raise ValueError('codec adaptation requires a fresh base, not a head/feedback artifact')
    if getattr(codec,'_codec_adapter_attached',False):
        raise ValueError('codec adapter already installed')
    stem=codec.decoder.stem
    channels=stem.weight.shape[0]
    if (type(rank) is not int or not 1<=rank<=channels
            or codec.config.latent_dim!=64 or codec.config.frame_samples!=3200):
        raise ValueError('invalid codec adapter rank or codec dimensions')
    adapter=DecoderAdapter(channels,rank,device=stem.weight.device,dtype=stem.weight.dtype)
    wrapper=AdapterStem(stem,adapter)
    # All validation/allocation precedes mutation of the shared model.
    model.requires_grad_(False)
    codec.decoder.stem=wrapper
    codec.decoder._sequence=None
    codec._codec_adapter_attached=True
    codec._codec_adapter_generation=getattr(codec,'_codec_adapter_generation',0)+1
    model.eval();adapter.requires_grad_(True)
    return adapter


def get_adapter(codec):
    if not getattr(codec,'_codec_adapter_attached',False):
        raise ValueError('codec adapter is not installed')
    return codec.decoder.stem.adapter


def set_adapter_enabled(codec,enabled):
    if type(enabled) is not bool:raise ValueError('adapter enabled must be boolean')
    adapter=get_adapter(codec)
    if adapter.enabled!=enabled:
        adapter.enabled=enabled
        codec._codec_adapter_generation+=1


def decode_training(codec,latents,state=None):
    adapter=get_adapter(codec)
    if (not adapter.enabled or latents.ndim!=3 or min(latents.shape[:2])<1 or latents.shape[-1]!=64
            or not latents.is_floating_point() or not torch.isfinite(latents).all()):
        raise ValueError('training needs enabled adaptation and finite raw codec latents')
    value=latents.to(codec.device,codec.dtype)
    state=codec.decoder.init_state(len(value)) if state is None else state
    return codec.decoder(value,state)
