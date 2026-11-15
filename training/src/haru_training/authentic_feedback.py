"""Native sampler feedback supervised by authentic audio, not an old voice head.

The recorded next latent is used only for the detached loss teacher. Student
conditions contain recorded past, with the current sampled audio as feedback.
Text remains teacher-forced; one-frame consistency is not dialogue quality.
"""
from dataclasses import replace
from types import SimpleNamespace

import torch

from .content_preservation import (_identity,_positions,_prepare_feedback_context,
    _feedback_logits,feedback_preservation_loss,preserving_spec,_execute_head_pilot)
from .direct_speaker import NativeSpeakerLoss
from .trainer import dataset_fingerprint


@torch.no_grad()
def prepare_authentic_feedback_point(model,batch,*,row,position,seed,guidance=1.5):
    point=_prepare_feedback_context(model,batch,row=row,position=position,seed=seed,guidance=guidance)
    raw=batch['audio_targets'][row:row+1,position].to(device=model.device,dtype=model.dtype)
    if raw.shape!=(1,model.config.latent_dim) or not torch.isfinite(raw).all():
        raise ValueError('invalid authentic feedback target')
    # This branch does not update the cached past or sample any voice teacher.
    point.teacher_logits=_feedback_logits(model,point,raw).detach()
    return point


authentic_feedback_loss=feedback_preservation_loss


class AuthenticFeedbackLoss:
    def __init__(self,model,*,points_per_record=3,seed=20261114):
        if type(points_per_record) is not int or not 1<=points_per_record<=3:
            raise ValueError('authentic feedback supports one to three fixed positions')
        self.identity=_identity(model)
        self.audio=NativeSpeakerLoss()
        self.points={}
        self.points_per_record,self.seed=points_per_record,seed

    def __call__(self,model,batch,*,seed):
        if _identity(model)!=self.identity:
            raise ValueError('frozen authentic-feedback backbone changed')
        result=self.audio(model,batch,seed=seed)
        fingerprint=dataset_fingerprint([SimpleNamespace(**batch)])
        losses=[]
        for row in range(len(batch['tokens'])):
            eligible=_positions(model,batch,row)
            if not eligible:raise ValueError('record has no lexical feedback positions')
            indices=torch.linspace(0,len(eligible)-1,min(len(eligible),self.points_per_record)).round().long().tolist()
            position=eligible[indices[seed%len(indices)]]
            key=(fingerprint,row,position)
            if key not in self.points:
                self.points[key]=prepare_authentic_feedback_point(model,batch,row=row,position=position,seed=self.seed+position)
            losses.append(authentic_feedback_loss(model,self.points[key]))
        result['preservation_loss']=torch.stack(losses).mean()
        result.pop('loss')
        return result


def authentic_spec(train,validation,*,preservation_weight,**kwargs):
    return replace(preserving_spec(train,validation,preservation_weight=preservation_weight,**kwargs),
        preservation_target='authentic_audio')


def run_authentic_pilot(model,train,validation,spec,destination,*,render,registered_weight=1.):
    expected=replace(authentic_spec(train,validation,preservation_weight=spec.preservation_weight,
        split_sha256=spec.split_sha256,cache_sha256=spec.cache_sha256,max_steps=spec.max_steps,max_seconds=spec.max_seconds),
        base_model=spec.base_model,base_revision=spec.base_revision)
    if (not 0<registered_weight<=1000 or spec!=expected
            or spec.preservation_weight not in (0,registered_weight)):
        raise ValueError('not the registered authentic-feedback condition')
    factory=AuthenticFeedbackLoss if spec.preservation_weight else lambda _:NativeSpeakerLoss()
    return _execute_head_pilot(model,train,validation,spec,destination,render=render,objective_factory=factory)
