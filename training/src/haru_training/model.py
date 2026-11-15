"""Differentiable training forward, separate from live inference/cache state."""

import torch

from .losses import masked_text_loss, meanflow_loss, sample_flow_times


class HeadField:
    def __init__(self, head):
        self.head = head

    def __call__(self, z, condition, time, interval, guidance):
        dtype = self.head.latent_proj.weight.dtype
        projected = self.head.cond_proj(condition.to(dtype))
        return self.head.velocity(z.to(dtype), projected, time, interval, guidance).float()


def training_losses(model, batch: dict, *, seed: int = 0, condition_dropout: float = .1,
                    guidance_max: float = 2., hidden: torch.Tensor | None = None) -> dict:
    if not 0 <= condition_dropout < 1 or not 1 <= guidance_max <= 2:
        raise ValueError('condition dropout or guidance is outside its supported range')
    device = model.device
    values = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    audio_mask = values['audio_loss_mask'] & values['attention_mask']
    if not audio_mask.any():
        raise ValueError('batch has no supervised audio frames')
    if hidden is None:
        embeddings = model.embed(values['tokens'], values['model_latents'], values['user_latents'],
                                 values['model_mask'], values['user_mask'])
        hidden = model.lm(inputs_embeds=embeddings, attention_mask=values['attention_mask'],
                          use_cache=False, return_dict=True).last_hidden_state
    else:
        if (any(p.requires_grad for module in (model.lm, model.model_audio_proj, model.user_audio_proj)
                for p in module.parameters()) or hidden.requires_grad
                or hidden.shape != (*values['tokens'].shape, model.config.hidden_size)
                or not torch.isfinite(hidden).all()):
            raise ValueError('cached conditions require a frozen backbone and matching finite hidden states')
        hidden = hidden.to(device)
    text_loss = masked_text_loss(model.logits(hidden), values['text_targets'])
    conditions = hidden[audio_mask]
    targets = model.scale_latents(values['audio_targets'][audio_mask])
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(targets.shape, device=device, dtype=torch.float32, generator=generator)
    time, interval = sample_flow_times(len(targets), generator, device)
    guidance = 1 + (guidance_max-1)*torch.rand(len(targets), device=device, generator=generator)
    dropped = torch.rand(len(targets), device=device, generator=generator) < condition_dropout
    flow = meanflow_loss(HeadField(model.head), targets, noise, time, interval, guidance, conditions,
                         drop_mask=dropped)
    return {'loss': text_loss + flow.loss, 'text_loss': text_loss,
            'audio_loss': flow.loss, 'hidden': hidden}
