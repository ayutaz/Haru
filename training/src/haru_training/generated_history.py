"""Authentic next-audio targets conditioned on the current head's own history.

Text remains teacher-forced; this is not a full free-running dialogue rollout.
History is detached and recomputed after every update, with no frozen old head
or external voice teacher. Validation retains the shared recorded-past protocol.
"""
from dataclasses import replace

import torch

from .direct_speaker import _conditioning_tensors,NativeSpeakerLoss,direct_head_spec
from .model import training_losses
from .content_preservation import _execute_head_pilot


@torch.no_grad()
def generated_history_hidden(model,batch,*,seed,guidance=1.5,noises=None):
    if (not 1<=guidance<=2 or any(p.requires_grad for p in _conditioning_tensors(model))
            or any(m.training for m in (model.lm,model.model_audio_proj,model.user_audio_proj))):
        raise ValueError('generated conditions require a frozen eval backbone and native guidance')
    batch_size,width=batch['tokens'].shape
    dim=model.config.latent_dim
    if noises is not None:
        if noises.shape!=(batch_size,width,dim) or not torch.isfinite(noises).all():
            raise ValueError('invalid supplied history noise')
        noises=noises.to(device=model.device,dtype=model.dtype)
    hidden=torch.zeros(batch_size,width,model.config.hidden_size,device=model.device,dtype=model.dtype)
    keys=('tokens','model_latents','user_latents','model_mask','user_mask')
    for row in range(batch_size):
        mask=batch['attention_mask'][row];length=int(mask.sum())
        active=torch.nonzero(batch['audio_loss_mask'][row,:length]).flatten()
        if not length or not mask[:length].all() or mask[length:].any() or not len(active):
            raise ValueError('generated conditions need contiguous valid supervised past')
        prefix=int(active[0])+1
        values={k:batch[k][row:row+1,:length].to(model.device) for k in keys}
        noise=(noises[row:row+1,:length] if noises is not None else
            torch.randn(1,length,dim,device=model.device,dtype=model.dtype,
                generator=torch.Generator(device=model.device).manual_seed(seed+row)))
        cache=model.new_cache();parts=[]
        for start in range(0,prefix,512):
            end=min(prefix,start+512)
            parts.append(model(model.embed(*(values[k][:,start:end] for k in keys)),cache))
        for position in range(prefix,length):
            raw=model.unscale_latents(model.head.sample(parts[-1][:,-1],noise[:,position-1],guidance))
            embedding=model.embed(values['tokens'][:,position:position+1],raw[:,None],
                values['user_latents'][:,position:position+1],values['model_mask'][:,position:position+1],
                values['user_mask'][:,position:position+1])
            parts.append(model(embedding,cache))
        hidden[row:row+1,:length]=torch.cat(parts,dim=1)
    return hidden


class GeneratedHistoryLoss:
    def __init__(self,*,condition_dropout=.1,guidance_max=2.):
        self.condition_dropout,self.guidance_max=condition_dropout,guidance_max
        self.validation=NativeSpeakerLoss(condition_dropout=condition_dropout,guidance_max=guidance_max)

    def __call__(self,model,batch,*,seed):
        if not model.training:
            return self.validation(model,batch,seed=seed)
        # Keep rollout randomness separate from MeanFlow supervision noise.
        hidden=generated_history_hidden(model,batch,seed=seed^0x48415255)
        return training_losses(model,batch,seed=seed,hidden=hidden,
            condition_dropout=self.condition_dropout,guidance_max=self.guidance_max)


def history_spec(train,validation,*,generated,**kwargs):
    if type(generated) is not bool:raise ValueError('history selection must be explicit')
    return replace(direct_head_spec(train,validation,**kwargs),
        training_audio_history='generated' if generated else 'recorded')


def run_history_pilot(model,train,validation,spec,destination,*,render):
    expected=replace(history_spec(train,validation,generated=spec.training_audio_history=='generated',
        split_sha256=spec.split_sha256,cache_sha256=spec.cache_sha256,max_steps=spec.max_steps,max_seconds=spec.max_seconds),
        base_model=spec.base_model,base_revision=spec.base_revision)
    if spec!=expected:raise ValueError('not the registered audio-history condition')
    factory=(lambda _:GeneratedHistoryLoss()) if spec.training_audio_history=='generated' else lambda _:NativeSpeakerLoss()
    return _execute_head_pilot(model,train,validation,spec,destination,render=render,objective_factory=factory)
