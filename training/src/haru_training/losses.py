"""iMF boundary-velocity objective in Haru's noise->data time convention.

Reference: https://arxiv.org/abs/2512.02012 (Algorithm 1, flexible CFG).
This is an explicit adaptation recipe, not the unreleased Haru training code.
Haru's forward interval gives V = u - h * sg(J_z u v + d_s u - d_h u).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowLoss:
    loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    jvp: torch.Tensor


def masked_text_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError('text logits and targets do not align')
    active = targets != -100
    selected = logits[active].float()
    if not torch.isfinite(selected).all():
        raise FloatingPointError('nonfinite supervised text logits')
    if not active.any():
        return selected.sum()
    return F.cross_entropy(selected, targets[active])


def sample_flow_times(count: int, generator: torch.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError('flow batch must not be empty')
    pair = torch.rand(2, count, generator=generator, device=device)
    start, end = pair.min(dim=0).values, pair.max(dim=0).values
    interval = end - start
    interval[:count // 2] = 0
    return start, interval


def meanflow_loss(field, data: torch.Tensor, noise: torch.Tensor, time: torch.Tensor,
                  interval: torch.Tensor, guidance: torch.Tensor, condition: torch.Tensor,
                  drop_mask: torch.Tensor | None = None) -> FlowLoss:
    count = data.shape[0]
    if (data.ndim != 2 or noise.shape != data.shape or condition.shape[0] != count or count == 0
            or any(v.shape != (count,) for v in (time, interval, guidance))):
        raise ValueError('invalid flow input shapes')
    if not all(torch.isfinite(v).all() for v in (data, noise, time, interval, guidance, condition)):
        raise FloatingPointError('nonfinite flow input')
    if ((time < 0).any() or (interval < 0).any() or (time + interval > 1 + 1e-6).any()
            or (guidance < 1).any() or (guidance > 2).any()):
        raise ValueError('flow time/interval/guidance is outside the trained range')
    if drop_mask is None:
        drop_mask = torch.zeros(count, dtype=torch.bool, device=data.device)
    if drop_mask.shape != (count,) or drop_mask.dtype != torch.bool:
        raise ValueError('condition dropout needs one boolean per sample')
    condition = torch.where(drop_mask[:, None], torch.zeros_like(condition), condition)
    z = noise * (1 - time[:, None]) + data * time[:, None]
    with torch.no_grad():
        vc = field(z, condition, time, torch.zeros_like(interval), guidance)
        vu = field(z, torch.zeros_like(condition), time, torch.zeros_like(interval), guidance)
        target = data - noise + (1 - 1 / guidance[:, None]) * (vc - vu)
        target = torch.where(drop_mask[:, None], data - noise, target).detach()

    def average(position, start, duration):
        return field(position, condition, start, duration, guidance)

    u, derivative = torch.func.jvp(average, (z, time, interval),
                                   (vc.detach(), torch.ones_like(time), -torch.ones_like(interval)))
    prediction = u - interval[:, None] * derivative.detach()
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError('nonfinite meanflow prediction or target')
    return FlowLoss((prediction - target).square().mean(), prediction, target, derivative.detach())
