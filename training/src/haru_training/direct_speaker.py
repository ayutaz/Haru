"""One bounded target-corpus head trial, with the normal audio feedback path.

No downloads, rentals or synthetic voice teachers are created here. Callers
authenticate the corpus, timing and model before invoking this training loop.
"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import TrainingConfig, configure_trainable
from .diagnostics import preserved_inference
from .inference import export_inference
from .trainer import RunSpec, Trainer, dataset_fingerprint
from .model import training_losses


def _conditioning_tensors(model):
    return [p for module in (model.lm, model.model_audio_proj, model.user_audio_proj)
            for p in module.parameters()] + [model.latent_bias, model.latent_scale]


@torch.no_grad()
def native_teacher_hidden(model, batch):
    """Use native prompt prefill, then one recorded past frame per LM call.

    BF16 gated-recurrent kernels can give different features for a full-sequence
    forward. Only frozen conditioning is supported; the recorded future targets
    never enter these inputs. Each padded row starts its own conversation cache.
    """
    if any(p.requires_grad for p in _conditioning_tensors(model)) or model.lm.training:
        raise ValueError('native conditions require a frozen eval backbone')
    keys = ('tokens', 'model_latents', 'user_latents', 'model_mask', 'user_mask')
    hidden = torch.zeros((*batch['tokens'].shape, model.config.hidden_size),
                         device=model.device, dtype=model.dtype)
    for row in range(len(batch['tokens'])):
        mask = batch['attention_mask'][row]
        length = int(mask.sum())
        if length < 1 or not mask[:length].all() or mask[length:].any():
            raise ValueError('native conditions need contiguous unpadded past frames')
        active = torch.nonzero(batch['audio_loss_mask'][row, :length]).flatten()
        if not len(active):
            raise ValueError('native conditions need a supervised continuation')
        prefix = int(active[0]) + 1
        values = {k: batch[k][row:row+1, :length].to(model.device) for k in keys}
        def embedding(start, end):
            return model.embed(*(values[k][:, start:end] for k in keys))
        cache = model.new_cache()
        # This is the same prefill split used by DuplexSession.start().
        parts = []
        for start in range(0, prefix, 512):
            parts.append(model(embedding(start, min(prefix, start+512)), cache))
        for position in range(prefix, length):
            parts.append(model(embedding(position, position+1), cache))
        hidden[row:row+1, :length] = torch.cat(parts, dim=1)
    return hidden


class NativeSpeakerLoss:
    """Cache native frozen conditions for identical authenticated batches only."""
    def __init__(self, *, condition_dropout=.1, guidance_max=2.):
        self.condition_dropout, self.guidance_max = condition_dropout, guidance_max
        self.cache = {}
        self.identity = None

    def hidden_for(self, model, batch):
        tensors = _conditioning_tensors(model)
        if any(p.requires_grad for p in tensors) or model.lm.training:
            raise ValueError('native conditions require a frozen eval backbone')
        identity = (id(model), tuple((id(p), p._version) for p in tensors))
        if self.identity is None:
            self.identity = identity
        elif self.identity != identity:
            raise ValueError('frozen conditioning weights changed; discard the feature cache')
        key = dataset_fingerprint([SimpleNamespace(**batch)])
        if key not in self.cache:
            self.cache[key] = native_teacher_hidden(model, batch)
        return self.cache[key]

    def __call__(self, model, batch, *, seed):
        return training_losses(model, batch, seed=seed, hidden=self.hidden_for(model, batch),
                               condition_dropout=self.condition_dropout, guidance_max=self.guidance_max)


def validate_membership(splits, *, train, validation, reference):
    try:
        groups = [list(splits[name]) for name in ('train', 'validation', 'test', 'reference')]
        all_ids = [i for group in groups for i in group]
        if (len(all_ids) != len(set(all_ids))
                or len(train) != 3 or len(set(train)) != 3
                or len(validation) != 3 or len(set(validation)) != 3
                or not set(train) <= set(splits['train'])
                or not set(validation) <= set(splits['validation'])
                or reference not in splits['reference']):
            raise ValueError('pilot requires three train and three held-out corpus recordings')
    except (KeyError, TypeError) as error:
        raise ValueError('incomplete corpus split membership') from error


def direct_spec(train, validation, *, split_sha256, cache_sha256,
                max_steps=100, max_seconds=600):
    if (type(max_steps) is not int or not 1 <= max_steps <= 100
            or type(max_seconds) is not int or not 0 < max_seconds <= 600):
        raise ValueError('direct pilot exceeds the fixed update or time limit')
    if len(train) != 3 or len(validation) != 3:
        raise ValueError('direct pilot requires exactly three train and validation records')
    return RunSpec(base_model='higurashi-ai/Haru-4B',
                   base_revision='8d6e83bcd3701a76fc27a764965665da1df7da78',
                   mode='head_output', optimizer_precision='fp32_master',
                   feedback_mode='generated_audio', learning_rate=1e-6,
                   seed=20261114, max_steps=max_steps, max_seconds=max_seconds,
                   validation_interval=max_steps, checkpoint_interval=0,
                   dataset_sha256=dataset_fingerprint(train),
                   validation_sha256=dataset_fingerprint(validation),
                   split_sha256=split_sha256, cache_sha256=cache_sha256)


def run_direct_pilot(model, train, validation, spec, destination, *, render):
    """Save the base, diagnostic step 30 and terminal step; never choose by score."""
    if (spec.mode != 'head_output' or spec.feedback_mode != 'generated_audio'
            or spec.train_projectors or spec.max_steps > 100 or spec.max_seconds > 600
            or spec.learning_rate != 1e-6 or len(train) != 3 or len(validation) != 3):
        raise ValueError('not the registered bounded direct speaker pilot')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (root/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    summary = configure_trainable(model, TrainingConfig(mode='head_output'))
    objective = NativeSpeakerLoss(condition_dropout=spec.condition_dropout, guidance_max=spec.guidance_max)
    trainer = Trainer(model, train, validation, spec, root/'trainer', loss_fn=objective)
    save('spec.json', asdict(spec))
    save('trainable.json', summary)
    observations = []
    milestones = sorted({0, min(30, spec.max_steps), spec.max_steps})
    for intended in milestones:
        if intended:
            trainer.fit(until_step=intended)
        actual = trainer.state.step
        if not observations or observations[-1]['step'] != actual:
            metrics = trainer.validate()
            checkpoint = trainer.save()
            export_inference(trainer, root/'inference'/f'step-{actual:08d}')
            with preserved_inference(model):
                rendered = render(model, root/f'step-{actual:08d}', actual)
            row = dict(step=actual, metrics=metrics, checkpoint=str(checkpoint.relative_to(root)),
                       rendered=rendered, voice_quality='unknown')
            save(f'observation-{actual:08d}.json', row)
            observations.append(row)
        if actual != intended:
            break
    result = dict(completed_updates=trainer.state.step,
                  complete=trainer.state.step == spec.max_steps,
                  frozen_weights_unchanged=trainer._frozen_digest() == trainer.frozen_digest,
                  observations=observations, voice_quality='unknown',
                  expansion_authorized=False, final_test_used=False,
                  feedback_mode='generated_audio',
                  conditioning='native_prefill_then_recorded_single_frames_cached',
                  conclusion='Await independent audio evaluation and human listening; not an adoption')
    save('result.json', result)
    return result
