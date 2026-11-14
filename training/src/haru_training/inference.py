"""Selected-weight inference artifacts on top of an immutable Haru base.

The checkpoint container supplies atomic writes and file hashes. Its payload is
distinct from a Trainer resume checkpoint: no optimizer, RNG, or data cursor.
Use load_inference to construct a fresh model, or apply_inference on a freshly
configured model. Restore the base by loading its pinned original weights;
disabling LoRA alone does not restore the trained head or projectors.
"""

from dataclasses import asdict
import hashlib
import json

import torch

from .checkpoint import read_checkpoint, write_checkpoint
from .contracts import TrainingConfig, configure_trainable
from .trainer import RunSpec, _tensor_bytes


def _digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _config_digest(model):
    return hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _frozen_digest(model):
    # Same contract as Trainer._frozen_digest, including PEFT names.
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _selected(model, spec):
    peft = getattr(model.lm, 'peft_config', {})
    if spec.mode == 'lm_lora':
        if set(peft) != {'default'}:
            raise ValueError('expected exactly one default LoRA adapter')
        config = peft['default']
        if (config.r != spec.lora_rank or config.lora_alpha != 2*spec.lora_rank
                or config.lora_dropout != 0 or config.bias != 'none'
                or config.use_dora or config.use_rslora or config.modules_to_save):
            raise ValueError('LoRA configuration differs from inference contract')
    elif peft:
        raise ValueError('head-only artifact requires an unwrapped LM')
    head_prefix='head.final.proj.' if spec.mode=='head_output' else 'head.'
    expected = {name for name, _ in model.named_parameters()
                if name.startswith(head_prefix) or
                (spec.train_projectors and name.startswith(('model_audio_proj.', 'user_audio_proj.'))) or
                (spec.mode == 'lm_lora' and name.startswith('lm.') and '.lora_' in name)}
    selected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not expected or set(selected) != expected:
        raise ValueError('configured trainable parameters differ from inference contract')
    return selected


def _read(path):
    bundle = read_checkpoint(path)
    required = {'kind', 'version', 'spec', 'config_sha256', 'frozen_sha256',
                'buffers_sha256', 'weights', 'step'}
    if (not isinstance(bundle, dict) or set(bundle) != required
            or bundle['kind'] != 'haru-inference' or bundle['version'] != 1
            or type(bundle['step']) is not int or bundle['step'] < 0):
        raise ValueError('not a supported Haru inference artifact')
    try:
        spec = RunSpec(**bundle['spec'])
    except (ValueError, TypeError) as error:
        raise ValueError('invalid inference specification') from error
    if bundle['step'] > spec.max_steps or not isinstance(bundle['weights'], dict):
        raise ValueError('invalid inference step or weight mapping')
    return bundle, spec


def export_inference(trainer, destination):
    """Export at a successful optimizer boundary; preserve the resume artifact."""
    if trainer.failed or trainer._in_step:
        raise RuntimeError('inference export requires a successful optimizer-step boundary')
    trainer._check_data()
    selected = _selected(trainer.model, trainer.spec)
    frozen = _frozen_digest(trainer.model)
    if frozen != trainer.frozen_digest:
        raise ValueError('frozen base weights changed; inference export rejected')
    if any(not torch.isfinite(value).all() for value in selected.values()):
        raise ValueError('nonfinite inference weights')
    return write_checkpoint(destination, {'kind': 'haru-inference', 'version': 1,
        'spec': asdict(trainer.spec), 'config_sha256': _config_digest(trainer.model),
        'frozen_sha256': frozen, 'buffers_sha256': _digest(trainer.model.named_buffers()),
        'weights': selected, 'step': trainer.state.step})


def _apply(model, bundle, spec, base_model, base_revision):
    if spec.base_model != base_model or spec.base_revision != base_revision:
        raise ValueError('inference base model or revision mismatch')
    selected = _selected(model, spec)
    if (_config_digest(model) != bundle['config_sha256']
            or _frozen_digest(model) != bundle['frozen_sha256']
            or _digest(model.named_buffers()) != bundle['buffers_sha256']):
        raise ValueError('base configuration, frozen weights, or buffers mismatch')
    weights = bundle['weights']
    if set(weights) != set(selected):
        raise ValueError('inference tensor names mismatch')
    # Validate every tensor before copying any weight or changing model modes.
    for name, value in weights.items():
        target = selected[name]
        if (not torch.is_tensor(value) or value.shape != target.shape
                or value.dtype != target.dtype or not torch.isfinite(value).all()):
            raise ValueError('inference tensor shape, dtype, or finiteness mismatch')
    with torch.no_grad():
        for name, value in weights.items():
            selected[name].copy_(value)
    return model.eval().requires_grad_(False)


def apply_inference(model, path, *, base_model, base_revision):
    """Apply to a fresh configure_trainable model; reject mismatches before copy."""
    bundle, spec = _read(path)
    return _apply(model, bundle, spec, base_model, base_revision)


def load_inference(path, *, device, dtype=None, base_directory=None):
    """Construct a fresh pinned base and apply its selected-weight artifact.

    base_directory may point to an offline base snapshot. Frozen parameter and
    buffer hashes still have to match. dtype must match the artifact's weights.
    No training records or optimizer are required for inference.
    """
    from haru import HaruModel

    bundle, spec = _read(path)
    model = HaruModel.from_pretrained(base_directory or spec.base_model,
        revision=spec.base_revision, device=device, dtype=dtype)
    configure_trainable(model, TrainingConfig(mode=spec.mode, lora_rank=spec.lora_rank,
                                             train_projectors=spec.train_projectors))
    return _apply(model, bundle, spec, spec.base_model, spec.base_revision)


def load_with_frozen_feedback(path, *, device, dtype=None, base_directory=None):
    """Return an adapted model and a separate original head for replay feedback.

    Only head adaptation is supported: changes to the LM or either projector
    would defeat preservation of the base internal stream. This extra head
    increases inference memory and compute; it does not prove audible quality.
    Pass the returned head explicitly to replay(..., feedback_head=head).
    """
    from copy import deepcopy
    from haru import HaruModel

    bundle,spec=_read(path)
    if spec.mode not in ('head','head_output') or spec.train_projectors:
        raise ValueError('frozen feedback requires unchanged LM and audio projectors')
    model=HaruModel.from_pretrained(base_directory or spec.base_model,
        revision=spec.base_revision,device=device,dtype=dtype)
    feedback=deepcopy(model.head).eval().requires_grad_(False)
    configure_trainable(model,TrainingConfig(mode=spec.mode))
    return _apply(model,bundle,spec,spec.base_model,spec.base_revision),feedback
