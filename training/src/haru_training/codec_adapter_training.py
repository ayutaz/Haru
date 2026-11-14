"""Bounded paired decoder training; no LM/head updates or latent feedback edits."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
import time

import torch

from .checkpoint import read_checkpoint, write_checkpoint
from .codec_adapter import attach_adapter, decode_training, get_adapter
from .codec_adapter_inference import adapter_bundle, codec_digest, _validate
from .optim import MasterAdamW


@dataclass(frozen=True)
class TrainSpec:
    base_model: str
    base_revision: str
    target_speaker: str
    max_steps: int = 100
    learning_rate: float = .001
    content_weight: float = 0.
    transition_strength: float = 0.
    seed: int = 42
    max_seconds: float = 600.

    def __post_init__(self):
        if (not re.fullmatch(r'[\w.-]+/[\w.-]+', self.base_model)
                or not re.fullmatch('[a-f0-9]{40}', self.base_revision)
                or not self.target_speaker or type(self.max_steps) is not int or self.max_steps < 1
                or type(self.seed) is not int or self.seed < 0
                or not all(math.isfinite(v) for v in (self.learning_rate, self.content_weight,
                                                     self.transition_strength, self.max_seconds))
                or self.learning_rate <= 0 or self.content_weight < 0
                or self.transition_strength < 0 or self.max_seconds <= 0):
            raise ValueError('invalid bounded codec training specification')


def transition_weights(posteriors, strength):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('invalid transition strength')
    change = torch.cat([posteriors.new_zeros(1),
                        .5 * (posteriors[1:] - posteriors[:-1]).abs().sum(-1)])
    weights = 1 + strength * change
    return weights / weights.mean()


def paired_loss(wave, pair, *, content=None, content_weight=0., transition_strength=0.):
    """STFT and CTC KL on the cached source clock, excluding padding and warmup."""
    m = pair.metadata
    if wave.ndim != 1 or len(wave) < m['valid_samples'] or not torch.isfinite(wave).all():
        raise ValueError('decoded waveform must be finite and cover the valid interval')
    audio = wave[:m['valid_samples']].float()
    spectrum = torch.stft(audio, n_fft=512, hop_length=m['spectral_hop'], win_length=512,
        window=torch.hann_window(512, device=audio.device), center=True, return_complex=True).abs().log1p().T
    teacher = pair.spectrum.to(audio.device)
    start = math.ceil(m['warmup_samples']/m['spectral_hop'])
    if spectrum.shape != teacher.shape or start >= len(spectrum):
        raise ValueError('spectral target shape or warmup interval differs')
    spectral = (spectrum[start:] - teacher[start:]).abs().mean()
    kl = audio.new_zeros(())
    if content_weight:
        if content is None: raise ValueError('content loss needs a frozen content encoder')
        if any(p.requires_grad for p in content.parameters()) or content.training:
            raise ValueError('content encoder must be frozen and in eval mode')
        logits = content(audio).float()
        target = pair.posteriors.to(audio.device)
        if logits.shape != target.shape or not torch.isfinite(logits).all():
            raise ValueError('content logits must be finite and match the cached CTC clock')
        # RF=400 and stride=320 at 16 kHz; indices are not phoneme boundaries.
        times = (torch.arange(len(logits),device=audio.device)*320+200)/16000
        mask = (times >= m['warmup_samples']/24000) & (times < len(audio)/24000)
        if not mask.any(): raise ValueError('no valid content frames after warmup')
        target = target[mask]
        values = torch.nn.functional.kl_div(logits[mask].log_softmax(-1), target, reduction='none').sum(-1)
        kl = (transition_weights(target, transition_strength) * values).mean()
    loss = spectral + content_weight * kl
    if not torch.isfinite(loss): raise ValueError('training loss is not finite')
    return dict(loss=loss, spectral=spectral, content=kl)


def _data_digest(pairs):
    digest = sha256()
    for item in pairs:
        digest.update(json.dumps(item.metadata, sort_keys=True, separators=(',', ':')).encode())
        for value in (item.latents, item.spectrum, item.posteriors):
            value = value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_digest(module):
    digest = sha256()
    if module is not None:
        for key, value in module.state_dict().items():
            value = value.detach().cpu().contiguous()
            digest.update(key.encode()); digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CodecTrainer:
    """One complete utterance per optimizer step; save only at utterance boundaries."""
    def __init__(self, model, pairs, spec, *, content=None, rank=8):
        self.spec = spec
        self.pairs = tuple(pairs)
        if not self.pairs or any(p.metadata['split'] != 'train'
            or p.metadata['target_speaker'] != spec.target_speaker
            or p.metadata['codec_revision'] != spec.base_revision for p in self.pairs):
            raise ValueError('training needs nonempty matching train-only pairs')
        if len({p.metadata['id'] for p in self.pairs}) != len(self.pairs):
            raise ValueError('duplicate training groups')
        if spec.content_weight and content is None:
            raise ValueError('content encoder is required')
        self.model = model
        self.content = None if content is None else content.eval().requires_grad_(False)
        self.data_sha256 = _data_digest(self.pairs)
        self.content_sha256 = _module_digest(self.content)
        adapter = attach_adapter(model, rank=rank)
        self.optimizer = MasterAdamW(adapter.parameters(), lr=spec.learning_rate)
        self.generator = torch.Generator(device='cpu').manual_seed(spec.seed)
        self.order = torch.randperm(len(self.pairs), generator=self.generator).tolist()
        self.cursor = self.step = 0
        self.elapsed = 0.
        self.failed = False
        self._versions = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        self._versions += [(b, b._version) for b in model.buffers()]
        if self.content is not None:
            self._versions += [(p, p._version) for p in self.content.parameters()]
            self._versions += [(b, b._version) for b in self.content.buffers()]

    def _identity(self):
        return dict(base_model=self.spec.base_model, base_revision=self.spec.base_revision,
                    target_speaker=self.spec.target_speaker, dataset_sha256=self.data_sha256)

    def _check(self):
        if self.failed: raise ValueError('failed training state cannot be continued or saved')
        if any(p._version != v for p,v in self._versions):
            self.failed = True
            raise ValueError('frozen weights or buffers changed')

    def fit(self, *, until_step=None):
        self._check()
        limit = self.spec.max_steps if until_step is None else until_step
        if type(limit) is not int or not self.step <= limit <= self.spec.max_steps:
            raise ValueError('invalid step boundary')
        adapter = get_adapter(self.model.codec)
        history = []
        started = time.monotonic()
        try:
            while self.step < limit:
                if self.elapsed + time.monotonic()-started >= self.spec.max_seconds:
                    break
                self._check()
                item = self.pairs[self.order[self.cursor]]
                self.optimizer.zero_grad(set_to_none=True)
                wave, _ = decode_training(self.model.codec, item.latents[None])
                losses = paired_loss(wave[0], item, content=self.content,
                    content_weight=self.spec.content_weight, transition_strength=self.spec.transition_strength)
                losses['loss'].backward()
                gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise ValueError('adapter gradients are not finite')
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                before = deepcopy(adapter.state_dict())
                optimizer_before = deepcopy(self.optimizer.state_dict())
                try:
                    self.optimizer.step()
                    if not all(torch.isfinite(p).all() for p in adapter.parameters()):
                        raise ValueError('adapter update is not finite')
                except Exception:
                    adapter.load_state_dict(before)
                    self.optimizer.load_state_dict(optimizer_before)
                    raise
                self._check()
                self.step += 1; self.cursor += 1
                if self.cursor == len(self.order):
                    self.order = torch.randperm(len(self.pairs), generator=self.generator).tolist()
                    self.cursor = 0
                history.append(dict(step=self.step, id=item.metadata['id'],
                    **{key:float(value.detach()) for key,value in losses.items()}))
        except Exception:
            self.failed = True
            raise
        finally:
            self.elapsed += time.monotonic()-started
        return history

    def save(self, path):
        self._check()
        return write_checkpoint(path, dict(kind='haru-codec-trainer', version=1,
            spec=asdict(self.spec), data_sha256=self.data_sha256, content_sha256=self.content_sha256,
            adapter=adapter_bundle(self.model, **self._identity()), optimizer=self.optimizer.state_dict(),
            step=self.step, order=self.order, cursor=self.cursor, sampler_rng=self.generator.get_state(),
            rng=torch.random.get_rng_state(), elapsed=self.elapsed,
            cuda_rng=torch.cuda.get_rng_state(self.model.codec.device) if self.model.codec.device.type=='cuda' else None))

    def restore(self, path):
        self._check()
        bundle = read_checkpoint(path)
        try:
            if (bundle['kind'] != 'haru-codec-trainer' or bundle['version'] != 1
                    or bundle['spec'] != asdict(self.spec) or bundle['data_sha256'] != self.data_sha256
                    or bundle['content_sha256'] != self.content_sha256
                    or sorted(bundle['order']) != list(range(len(self.pairs)))
                    or type(bundle['step']) is not int or not 0 <= bundle['step'] <= self.spec.max_steps
                    or type(bundle['cursor']) is not int or not 0 <= bundle['cursor'] < len(self.pairs)
                    or not math.isfinite(bundle['elapsed']) or bundle['elapsed'] < 0):
                raise ValueError('training specification, data, or progress differs')
            artifact = bundle['adapter']; _validate(artifact)
            current = adapter_bundle(self.model, **self._identity())
            if any(artifact[k] != current[k] for k in ('spec','codec_sha256','config_sha256')):
                raise ValueError('resume codec or adapter identity differs')
            # Validate all small states before copying any model weights.
            generator = torch.Generator().set_state(bundle['sampler_rng'])
            torch.Generator().set_state(bundle['rng'])
            if self.model.codec.device.type == 'cuda':
                state = bundle['cuda_rng']
                expected = torch.cuda.get_rng_state(self.model.codec.device)
                if not torch.is_tensor(state) or state.dtype != expected.dtype or state.shape != expected.shape:
                    raise ValueError('invalid CUDA RNG state')
            elif bundle['cuda_rng'] is not None:
                raise ValueError('resume device differs')
            adapter = get_adapter(self.model.codec)
            optimizer = MasterAdamW(adapter.parameters(), lr=self.spec.learning_rate)
            optimizer.load_state_dict(bundle['optimizer'])
            staged = {k:v.to(self.model.codec.device) for k,v in artifact['weights'].items()}
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError('invalid or incompatible codec training checkpoint') from error
        adapter.load_state_dict(staged, strict=True)
        self.optimizer = optimizer; self.generator = generator
        self.step = bundle['step']; self.order = bundle['order']; self.cursor = bundle['cursor']
        self.elapsed = bundle['elapsed']
        torch.random.set_rng_state(bundle['rng'])
        if bundle['cuda_rng'] is not None: torch.cuda.set_rng_state(bundle['cuda_rng'],self.model.codec.device)
