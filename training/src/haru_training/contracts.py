"""Validated training contracts; no model download or compute allocation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = 'head'
    lora_rank: int = 8
    train_projectors: bool = False
    guidance_max: float = 2.0
    condition_dropout: float = .1
    max_seconds: int = 7200
    max_dollars: float = 100.0

    def __post_init__(self):
        if self.mode not in {'head', 'lm_lora'}:
            raise ValueError('mode must be head or lm_lora')
        if isinstance(self.lora_rank, bool) or self.lora_rank <= 0:
            raise ValueError('lora_rank must be positive')
        if not 1 <= self.guidance_max <= 2 or not 0 <= self.condition_dropout < 1:
            raise ValueError('invalid guidance or dropout')
        if not 0 < self.max_seconds <= 7200 or not 0 < self.max_dollars <= 100:
            raise ValueError('execution limits must be positive and within authorization')


def configure_trainable(model, config: TrainingConfig) -> dict:
    for name in ('head', 'codec', 'lm', 'model_audio_proj', 'user_audio_proj'):
        if not hasattr(model, name):
            raise ValueError(f'model is missing required module: {name}')
    model.requires_grad_(False)
    if config.mode == 'lm_lora':
        from peft import LoraConfig, get_peft_model
        model.lm = get_peft_model(model.lm, LoraConfig(r=config.lora_rank,
            lora_alpha=2 * config.lora_rank, lora_dropout=0, bias='none', target_modules='all-linear'))
    model.head.requires_grad_(True)
    if config.train_projectors:
        model.model_audio_proj.requires_grad_(True)
        model.user_audio_proj.requires_grad_(True)
    model.train()
    model.codec.eval()
    if hasattr(model.lm.config, 'use_cache'):
        model.lm.config.use_cache = False
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {'names': names, 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total': sum(p.numel() for p in model.parameters())}
