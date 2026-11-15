"""One native feedback step, preserving lexical predictions of the pinned base.

The original head is a training-only reference. Live inference keeps the adapted
audio as feedback. Recorded future model audio never enters this comparison.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
from pathlib import Path
import json
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .direct_speaker import NativeSpeakerLoss, _conditioning_tensors,direct_head_spec
from .trainer import dataset_fingerprint,Trainer
from .contracts import TrainingConfig,configure_trainable
from .diagnostics import preserved_inference
from .inference import export_inference


def differentiable_sample(head, hidden, noise, guidance=1.5):
    """Same Euler calculation as MeanFlowHead.sample, retaining autograd."""
    if not 1 <= guidance <= 2:
        raise ValueError('unsupported preservation guidance')
    omega=torch.full((noise.shape[0],),guidance,device=noise.device)
    condition=head.cond_proj(hidden.to(noise.dtype))
    z,interval=noise,1./head.config.num_steps
    for index in range(head.config.num_steps):
        z=z+interval*head.velocity(z,condition,index*interval,interval,omega)
    return z


def content_kl(student_logits, teacher_logits, *, excluded_ids):
    """KL(reference || student) conditional on lexical rather than PAD/EOP/EOS."""
    if (student_logits.shape!=teacher_logits.shape or student_logits.ndim!=2
            or not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all()):
        raise ValueError('preservation requires matching finite logits')
    keep=torch.ones(student_logits.shape[-1],device=student_logits.device,dtype=torch.bool)
    keep[list(excluded_ids)]=False
    if int(keep.sum())<2:
        raise ValueError('preservation needs at least two lexical tokens')
    student=F.log_softmax(student_logits[:,keep].float(),dim=-1)
    teacher=F.log_softmax(teacher_logits[:,keep].detach().float(),dim=-1)
    return F.kl_div(student,teacher,reduction='batchmean',log_target=True)


def _identity(model):
    tensors=_conditioning_tensors(model)
    if any(p.requires_grad for p in tensors) or any(module.training for module in
            (model.lm,model.model_audio_proj,model.user_audio_proj)):
        raise ValueError('preservation requires an eval frozen backbone')
    return id(model),tuple((id(p),p._version) for p in tensors)


def _excluded(model):
    return sorted({i for i in (model.config.pad_token_id,model.config.eop_token_id,model.config.eos_token_id) if i is not None})


def _positions(model,batch,row):
    mask=batch['attention_mask'][row]
    length=int(mask.sum())
    if not length or not mask[:length].all() or mask[length:].any():
        raise ValueError('preservation requires contiguous valid past')
    labels=batch['text_targets'][row]
    return [p for p in range(length-1) if batch['audio_loss_mask'][row,p]
            and int(labels[p+1])>=0 and int(labels[p+1]) not in _excluded(model)]


@dataclass
class FeedbackPoint:
    identity: tuple
    hidden: torch.Tensor
    cache: object
    token: torch.Tensor
    user_latent: torch.Tensor
    model_mask: torch.Tensor
    user_mask: torch.Tensor
    noise: torch.Tensor
    teacher_logits: torch.Tensor
    guidance: float


class _SnapshotStateCache:
    """Single-use cache: native updates cannot overwrite saved autograd inputs.

    Linear-attention getters expose buffers subsequently updated in-place by
    Transformers. Snapshot only those reads, rather than cloning LM weights.
    This cache is discarded after one frame, never used for live inference.
    """
    def __init__(self,cache):
        self._cache=deepcopy(cache)

    def __getattr__(self,name):
        return getattr(self._cache,name)

    def get_recurrent_state(self,*args,**kwargs):
        state=self._cache.get_recurrent_state(*args,**kwargs)
        return None if state is None else state.clone()

    def get_conv_state(self,*args,**kwargs):
        state=self._cache.get_conv_state(*args,**kwargs)
        return None if state is None else state.clone()


def _feedback_logits(model,point,raw):
    embeddings=model.embed(point.token,raw[:,None],point.user_latent,point.model_mask,point.user_mask)
    hidden=model(embeddings,_SnapshotStateCache(point.cache))[:,-1]
    return model.logits(hidden)


@torch.no_grad()
def _prepare_feedback_context(model,batch,*,row,position,seed,guidance=1.5):
    identity=_identity(model)
    if not 1<=guidance<=2 or position not in _positions(model,batch,row):
        raise ValueError('invalid lexical feedback position or guidance')
    keys=('tokens','model_latents','user_latents','model_mask','user_mask')
    values={k:batch[k][row:row+1].to(model.device) for k in keys}
    first=int(torch.nonzero(batch['audio_loss_mask'][row])[0])+1
    cache=model.new_cache()
    def advance(start,end):
        return model(model.embed(*(values[k][:,start:end] for k in keys)),cache)
    for start in range(0,first,512):hidden=advance(start,min(first,start+512))[:,-1]
    for index in range(first,position+1):hidden=advance(index,index+1)[:,-1]
    generator=torch.Generator(device=model.device).manual_seed(seed)
    noise=torch.randn(1,model.config.latent_dim,device=model.device,dtype=model.dtype,generator=generator)
    return FeedbackPoint(identity,hidden.detach(),cache,values['tokens'][:,position+1:position+2],
        values['user_latents'][:,position+1:position+2],values['model_mask'][:,position+1:position+2],
        values['user_mask'][:,position+1:position+2],noise,None,guidance)


@torch.no_grad()
def prepare_feedback_point(model,batch,*,row,position,reference_head,seed,guidance=1.5):
    if (reference_head is model.head or reference_head.training
            or any(p.requires_grad for p in reference_head.parameters())
            or reference_head.config!=model.head.config):
        raise ValueError('invalid reference head')
    point=_prepare_feedback_context(model,batch,row=row,position=position,seed=seed,guidance=guidance)
    raw=model.unscale_latents(reference_head.sample(point.hidden,point.noise,guidance))
    point.teacher_logits=_feedback_logits(model,point,raw).detach()
    return point


def feedback_preservation_loss(model,point):
    if _identity(model)!=point.identity:
        raise ValueError('frozen preservation backbone changed')
    raw=model.unscale_latents(differentiable_sample(model.head,point.hidden,point.noise,point.guidance))
    return content_kl(_feedback_logits(model,point,raw),point.teacher_logits,excluded_ids=_excluded(model))


class ContentPreservingLoss:
    """Authentic audio loss plus separately reported one-point-per-record KL.

    At most three evenly spaced lexical positions per recording are cached;
    the supplied seed chooses one. These are recorded-past diagnostics, not a
    claim that a single step guarantees a free-running conversation.
    """
    def __init__(self,model,*,points_per_record=3,seed=20261114):
        if type(points_per_record) is not int or not 1<=points_per_record<=3:
            raise ValueError('preservation supports one to three fixed positions')
        self.identity=_identity(model)
        self.reference=deepcopy(model.head).eval().requires_grad_(False)
        self.reference_versions=tuple(p._version for p in self.reference.parameters())
        self.audio=NativeSpeakerLoss()
        self.points={}
        self.points_per_record,self.seed=points_per_record,seed

    def __call__(self,model,batch,*,seed):
        if (_identity(model)!=self.identity
                or tuple(p._version for p in self.reference.parameters())!=self.reference_versions):
            raise ValueError('frozen preservation reference or backbone changed')
        result=self.audio(model,batch,seed=seed)
        fingerprint=dataset_fingerprint([SimpleNamespace(**batch)])
        losses=[]
        for row in range(len(batch['tokens'])):
            eligible=_positions(model,batch,row)
            if not eligible:raise ValueError('record has no lexical preservation positions')
            indices=torch.linspace(0,len(eligible)-1,min(len(eligible),self.points_per_record)).round().long().tolist()
            position=eligible[indices[seed%len(indices)]]
            key=(fingerprint,row,position)
            if key not in self.points:
                self.points[key]=prepare_feedback_point(model,batch,row=row,position=position,
                    reference_head=self.reference,seed=self.seed+position)
            losses.append(feedback_preservation_loss(model,self.points[key]))
        result['preservation_loss']=torch.stack(losses).mean()
        # Trainer applies the explicit RunSpec coefficient. Avoid exposing the
        # old supervision-only total as if it included the new component.
        result.pop('loss')
        return result


def preserving_spec(train,validation,*,preservation_weight,**kwargs):
    if (type(preservation_weight) not in (int,float) or not math.isfinite(preservation_weight)
            or not 0<=preservation_weight<=1000):
        raise ValueError('preservation coefficient is outside the bounded recipe')
    return replace(direct_head_spec(train,validation,**kwargs),preservation_weight=preservation_weight)


def calibrate_weight(audio_loss,preservation_loss,parameters):
    """One training-only calibration: preservation gradient norm is 10% of audio.

    Called on the completed control before looking at development comparisons.
    This measures gradients without an optimizer step or populating .grad.
    """
    parameters=list(parameters)
    def norm(loss,retain):
        gradients=torch.autograd.grad(loss,parameters,retain_graph=retain,allow_unused=True)
        return math.sqrt(sum(float(g.detach().float().square().sum()) for g in gradients if g is not None))
    audio=norm(audio_loss,True)
    preservation=norm(preservation_loss,False)
    if not all(math.isfinite(x) and x>0 for x in (audio,preservation)):
        raise ValueError('calibration requires finite nonzero gradients')
    raw=.1*audio/preservation
    weight=min(1000.,max(.001,raw))
    return dict(weight=weight,unclipped_weight=raw,clipped=weight!=raw,
                audio_gradient_norm=audio,preservation_gradient_norm=preservation,
                target_gradient_ratio=.1)


def run_preserving_pilot(model,train,validation,spec,destination,*,render,registered_weight=1.):
    """A paired, bounded condition; exports inference only, never auto-adopts.

    The caller registers the nonzero coefficient (or its training-only
    calibration rule) before execution. Zero selects the shared control recipe.
    """
    expected=replace(direct_head_spec(train,validation,split_sha256=spec.split_sha256,
        cache_sha256=spec.cache_sha256,max_steps=spec.max_steps,max_seconds=spec.max_seconds),
        base_model=spec.base_model,base_revision=spec.base_revision,preservation_weight=spec.preservation_weight)
    if (not 0<registered_weight<=1000 or spec!=expected
            or spec.preservation_weight not in (0,registered_weight)):
        raise ValueError('not the registered content-preservation condition')
    factory=ContentPreservingLoss if spec.preservation_weight else lambda _:NativeSpeakerLoss()
    return _execute_head_pilot(model,train,validation,spec,destination,render=render,objective_factory=factory)


def _execute_head_pilot(model,train,validation,spec,destination,*,render,objective_factory):
    """Shared execution after each public runner validates its own recipe."""
    root=Path(destination)
    if root.exists() and any(root.iterdir()):raise FileExistsError('pilot output already exists')
    root.mkdir(parents=True,exist_ok=True)
    configure_trainable(model,TrainingConfig(mode='head'))
    objective=objective_factory(model)
    trainer=Trainer(model,train,validation,spec,root/'trainer',loss_fn=objective)
    observations=[]
    for intended in sorted({0,min(30,spec.max_steps),spec.max_steps}):
        if intended:trainer.fit(until_step=intended)
        actual=trainer.state.step
        if not observations or observations[-1]['step']!=actual:
            metrics=trainer.validate()
            export_inference(trainer,root/'inference'/f'step-{actual:08d}')
            rendered=None
            if intended in (0,spec.max_steps):
                with preserved_inference(model):rendered=render(model,root/f'step-{actual:08d}',actual)
            observations.append(dict(step=actual,metrics=metrics,rendered=rendered))
        if actual!=intended:break
    result=dict(complete=trainer.state.step==spec.max_steps,completed_updates=trainer.state.step,
        frozen_weights_unchanged=trainer._frozen_digest()==trainer.frozen_digest,
        observations=observations,preservation_weight=spec.preservation_weight,
        training_audio_history=spec.training_audio_history,
        preservation_target=spec.preservation_target,
        feedback_mode='generated_audio',voice_quality='unknown',final_test_used=False,
        optimizer_checkpoints_exported=False,expansion_authorized=False)
    (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result
