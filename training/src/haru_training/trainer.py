"""Explicit training execution; importing this module performs no I/O or allocation."""

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import math
import json
from pathlib import Path
import random
import re
import time

import numpy as np

import torch

from .contracts import TrainingConfig, train_selected_modules
from .checkpoint import read_checkpoint, write_checkpoint
from .data import collate_sequences
from .model import training_losses


@dataclass(frozen=True)
class RunSpec:
    base_model: str
    base_revision: str
    dataset_sha256: str
    validation_sha256: str
    split_sha256: str
    cache_sha256: str
    max_steps: int
    micro_batch_size: int = 1
    accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_steps: int = 0
    validation_interval: int = 1
    checkpoint_interval: int = 0
    seed: int = 20261114
    max_seconds: int = 7200
    mode: str = 'head'
    lora_rank: int = 8
    train_projectors: bool = False
    condition_dropout: float = .1
    guidance_max: float = 2.
    schema_version: int = 1
    optimizer_precision: str = 'native'
    feedback_mode: str = 'generated_audio'

    def __post_init__(self):
        if (self.feedback_mode not in ('generated_audio','separate_frozen_head')
                or (self.feedback_mode=='separate_frozen_head'
                    and (self.mode not in ('head','head_output') or self.train_projectors))):
            raise ValueError('invalid feedback mode or changes to the preserved conversation path')
        if self.optimizer_precision not in ('native','fp32_master'):
            raise ValueError('invalid optimizer precision')
        positive = ('max_steps', 'micro_batch_size', 'accumulation_steps',
                    'validation_interval', 'max_seconds')
        nonnegative = ('warmup_steps', 'checkpoint_interval', 'seed')
        for name in positive + nonnegative:
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError(f'invalid integer setting: {name}')
        if self.max_seconds > 7200 or self.seed >= 2**32 or self.warmup_steps > self.max_steps:
            raise ValueError('execution or random-seed limit exceeded')
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError('invalid optimizer settings')
        if (type(self.learning_rate) is bool or type(self.weight_decay) is bool
                or type(self.train_projectors) is not bool or type(self.lora_rank) is not int):
            raise ValueError('invalid optimizer or adaptation setting types')
        if (type(self.schema_version) is not int or self.schema_version != 1
                or not re.fullmatch(r'[\w.-]+/[\w.-]+', self.base_model)):
            raise ValueError('unsupported schema or ambiguous base model')
        if not re.fullmatch('[a-f0-9]{40}', self.base_revision):
            raise ValueError('base model requires an immutable revision')
        for name in ('dataset_sha256', 'validation_sha256', 'split_sha256', 'cache_sha256'):
            if not re.fullmatch('[a-f0-9]{64}', getattr(self, name)):
                raise ValueError(f'invalid identity: {name}')
        TrainingConfig(mode=self.mode, lora_rank=self.lora_rank,
                       train_projectors=self.train_projectors, guidance_max=self.guidance_max,
                       condition_dropout=self.condition_dropout, max_seconds=self.max_seconds)


def _tensor_bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def dataset_fingerprint(records):
    digest = hashlib.sha256()
    for record in records:
        for name, value in vars(record).items():
            digest.update(name.encode())
            if torch.is_tensor(value):
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(_tensor_bytes(value))
            else:
                digest.update(str(value).encode())
            digest.update(b'\0')
    return digest.hexdigest()


def _optimizer(parameters,spec):
    if spec.optimizer_precision=='fp32_master':
        from .optim import MasterAdamW
        return MasterAdamW(parameters,lr=spec.learning_rate,weight_decay=spec.weight_decay)
    return torch.optim.AdamW(parameters,lr=spec.learning_rate,weight_decay=spec.weight_decay)


class Trainer:
    def __init__(self, model, train, validation, spec: RunSpec, output: Path,
                 *, collate_fn=None, loss_fn=training_losses):
        self.model, self.spec = model, spec
        self.train, self.validation = list(train), list(validation)
        train_ids = [r.record_id for r in self.train]
        val_ids = [r.record_id for r in self.validation]
        if (not train_ids or not val_ids or len(set(train_ids)) != len(train_ids)
                or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids)):
            raise ValueError('training and validation require unique, disjoint record IDs')
        if (dataset_fingerprint(self.train) != spec.dataset_sha256
                or dataset_fingerprint(self.validation) != spec.validation_sha256):
            raise ValueError('dataset content or order differs from run specification')
        self.output = Path(output)
        pad = getattr(getattr(model, 'config', None), 'pad_token_id', 0)
        self.collate = collate_fn or (lambda rows: collate_sequences(rows, pad))
        self.loss_fn = (partial(loss_fn, condition_dropout=spec.condition_dropout,
                               guidance_max=spec.guidance_max)
                       if loss_fn is training_losses else loss_fn)
        selected = [p for p in model.parameters() if p.requires_grad]
        if not selected:
            raise ValueError('model has no trainable parameters')
        self.optimizer = _optimizer(selected,spec)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda step: min(1., (step+1)/max(1, spec.warmup_steps)))
        self.state = Progress()
        self.failed = False
        self.started = time.monotonic()
        self._in_step = False
        self.frozen_digest = self._frozen_digest()

    def _signature(self):
        return {name: {'shape':list(p.shape), 'dtype':str(p.dtype), 'trainable':p.requires_grad,
                       'device':p.device.type}
                for name,p in self.model.named_parameters()}

    def _check_data(self):
        if (dataset_fingerprint(self.train) != self.spec.dataset_sha256
                or dataset_fingerprint(self.validation) != self.spec.validation_sha256):
            raise ValueError('dataset was changed after run initialization')

    def _frozen_digest(self):
        digest = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                digest.update(name.encode())
                digest.update(_tensor_bytes(parameter))
        return digest.hexdigest()

    def save(self):
        if self._in_step or self.failed:
            raise RuntimeError('checkpoint requires a successful optimizer-step boundary')
        self._check_data()
        if self._frozen_digest() != self.frozen_digest:
            raise ValueError('frozen base weights changed; checkpoint rejected')
        bundle = {'version':1, 'spec':asdict(self.spec), 'signature':self._signature(),
            'frozen_sha256':self.frozen_digest, 'progress':asdict(self.state), 'order':self._order(),
            'model':{n:p for n,p in self.model.named_parameters() if p.requires_grad},
            'buffers':dict(self.model.named_buffers()), 'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(), 'rng':capture_rng(self.model),
            'torch_version':str(torch.__version__)}
        return write_checkpoint(self.output/'checkpoints'/f'step-{self.state.step:08d}', bundle)

    def resume(self, path):
        if self._in_step: raise RuntimeError('resume requires an optimizer-step boundary')
        self._check_data()
        bundle = read_checkpoint(path)
        if (bundle.get('version') != 1 or asdict(RunSpec(**bundle.get('spec',{}))) != asdict(self.spec)
                or bundle.get('signature') != self._signature()
                or bundle.get('frozen_sha256') != self._frozen_digest()
                or bundle.get('torch_version') != str(torch.__version__)):
            raise ValueError('checkpoint base, configuration, environment, or update targets differ')
        progress = Progress(**bundle['progress'])
        width = self.spec.micro_batch_size*self.spec.accumulation_steps
        per_epoch = math.ceil(len(self.train)/width)
        epoch, remainder = divmod(progress.step, per_epoch)
        if (any(type(v) is not int or v < 0 for v in asdict(progress).values())
                or progress.step > self.spec.max_steps or progress.epoch != epoch
                or progress.cursor != remainder*width
                or progress.micro_step != epoch*math.ceil(len(self.train)/self.spec.micro_batch_size)
                    +remainder*self.spec.accumulation_steps):
            raise ValueError('checkpoint progress is inconsistent with its dataset')
        order = list(range(len(self.train)))
        random.Random(self.spec.seed+progress.epoch).shuffle(order)
        if bundle['order'] != order: raise ValueError('checkpoint data order is inconsistent')
        selected = {n:p for n,p in self.model.named_parameters() if p.requires_grad}
        buffers = dict(self.model.named_buffers())
        for expected, saved in ((selected,bundle['model']), (buffers,bundle['buffers'])):
            if expected.keys() != saved.keys(): raise ValueError('checkpoint tensor names differ')
            for name, value in saved.items():
                if (value.shape != expected[name].shape or value.dtype != expected[name].dtype
                        or not torch.isfinite(value).all()):
                    raise ValueError('checkpoint tensor shape, dtype or values are invalid')
        validate_rng(bundle['rng'], self.model)
        # Validate optimizer/scheduler before copying any model weights.
        optimizer = _optimizer(list(selected.values()),self.spec)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step:min(1.,(step+1)/max(1,self.spec.warmup_steps)))
        optimizer.load_state_dict(bundle['optimizer'])
        scheduler.load_state_dict(bundle['scheduler'])
        for parameter, state in optimizer.state.items():
            expected_keys={'step','exp_avg','exp_avg_sq'}
            if self.spec.optimizer_precision=='fp32_master': expected_keys.add('master_param')
            if set(state) != expected_keys:
                raise ValueError('checkpoint optimizer state is unsupported')
            for key in ('exp_avg','exp_avg_sq'):
                value = state[key]
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError('checkpoint optimizer moments are invalid')
            step = state['step']
            if step.ndim != 0 or not torch.isfinite(step) or not 0 < float(step) <= progress.step:
                raise ValueError('checkpoint optimizer step is invalid')
        if self.spec.optimizer_precision=='fp32_master':
            for name,parameter in selected.items():
                state=optimizer.state.get(parameter)
                if state and not torch.equal(state['master_param'].to(dtype=parameter.dtype).cpu(),bundle['model'][name]):
                    raise ValueError('checkpoint master weight differs from saved model weight')
        if scheduler.last_epoch != progress.step:
            raise ValueError('checkpoint scheduler step differs from training progress')
        with torch.no_grad():
            for expected, saved in ((selected,bundle['model']), (buffers,bundle['buffers'])):
                for name,value in saved.items(): expected[name].copy_(value)
        self.optimizer, self.scheduler, self.state = optimizer, scheduler, progress
        self.failed = False
        self.frozen_digest = bundle['frozen_sha256']
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(bundle['rng'])
        self.started = time.monotonic()
        self._event('resumed')

    def _event(self, event, **values):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': event, 'step': self.state.step, **values},
                                    allow_nan=False)+'\n')

    def _order(self):
        order = list(range(len(self.train)))
        random.Random(self.spec.seed+self.state.epoch).shuffle(order)
        return order

    @staticmethod
    def _counts(batch):
        return int((batch['text_targets'] != -100).sum()), int((batch['audio_loss_mask'] & batch['attention_mask']).sum())

    @staticmethod
    def _check(losses):
        for key in ('text_loss', 'audio_loss'):
            if losses[key].ndim != 0 or not torch.isfinite(losses[key]):
                raise FloatingPointError('nonfinite or nonscalar supervised loss')

    def _training_mode(self):
        train_selected_modules(self.model)

    def validate(self, *, record_ids=None):
        records = list(enumerate(self.validation))
        if record_ids is not None:
            ids = list(record_ids)
            known = {r.record_id for r in self.validation}
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= known:
                raise ValueError('validation subset must contain unique held-out validation IDs')
            records = [(index, row) for index, row in records if row.record_id in set(ids)]
        modes = {module: module.training for module in self.model.modules()}
        rng = capture_rng(self.model)
        totals = [0., 0.]
        counts = [0, 0]
        try:
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(records), self.spec.micro_batch_size):
                    group = records[start:start+self.spec.micro_batch_size]
                    batch = self.collate([row for _, row in group])
                    number = self._counts(batch)
                    losses = self.loss_fn(self.model, batch, seed=self.spec.seed+group[0][0])
                    self._check(losses)
                    for i, key in enumerate(('text_loss', 'audio_loss')):
                        totals[i] += float(losses[key])*number[i]
                        counts[i] += number[i]
        finally:
            for module, mode in modes.items(): module.training = mode
            restore_rng(rng)
        metrics = dict(zip(('text_loss', 'audio_loss'),
                           (total/max(1, n) for total, n in zip(totals, counts))))
        metrics.update(text_count=counts[0], audio_count=counts[1])
        self._event('validation', **metrics,
                    **({'record_ids': [row.record_id for _, row in records]} if record_ids is not None else {}))
        return metrics

    def fit(self, *, until_step=None):
        if self.failed:
            raise RuntimeError('training failed; explicitly resume a valid checkpoint')
        self._check_data()
        limit = self.spec.max_steps if until_step is None else until_step
        if type(limit) is not int or not self.state.step <= limit <= self.spec.max_steps:
            raise ValueError('requested step is outside the current run')
        self._training_mode()
        try:
            while self.state.step < limit:
                if time.monotonic()-self.started >= self.spec.max_seconds:
                    self._event('stopped', reason='time_limit')
                    break
                order = self._order()
                width = self.spec.micro_batch_size*self.spec.accumulation_steps
                selected = order[self.state.cursor:self.state.cursor+width]
                batches = [self.collate([self.train[i] for i in selected[j:j+self.spec.micro_batch_size]])
                           for j in range(0, len(selected), self.spec.micro_batch_size)]
                counts = [self._counts(batch) for batch in batches]
                text_count, audio_count = (sum(c[i] for c in counts) for i in (0,1))
                if not audio_count:
                    raise ValueError('accumulation window has no supervised audio')
                self.optimizer.zero_grad(set_to_none=True)
                self._in_step = True
                totals = [0., 0.]
                for index, (batch, number) in enumerate(zip(batches, counts)):
                    losses = self.loss_fn(self.model, batch, seed=self.spec.seed+self.state.micro_step+index)
                    self._check(losses)
                    loss = (losses['text_loss']*number[0]/max(1,text_count)
                            +losses['audio_loss']*number[1]/audio_count)
                    loss.backward()
                    for i, key in enumerate(('text_loss','audio_loss')):
                        totals[i] += float(losses[key].detach())*number[i]
                selected_parameters = [p for p in self.model.parameters() if p.requires_grad]
                if not any(p.grad is not None for p in selected_parameters):
                    raise RuntimeError('supervision did not reach selected parameters')
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in selected_parameters):
                    raise FloatingPointError('nonfinite gradient; update was rejected')
                self.optimizer.step()
                if any(not torch.isfinite(p).all() for p in selected_parameters):
                    raise FloatingPointError('optimizer produced nonfinite parameters; recover last checkpoint')
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._in_step = False
                self.state.step += 1
                self.state.micro_step += len(batches)
                self.state.cursor += len(selected)
                if self.state.cursor == len(order):
                    self.state.epoch += 1
                    self.state.cursor = 0
                self._event('train', record_ids=[self.train[i].record_id for i in selected],
                    text_loss=totals[0]/max(1,text_count), audio_loss=totals[1]/audio_count,
                    learning_rate=self.optimizer.param_groups[0]['lr'])
                if self.state.step % self.spec.validation_interval == 0:
                    self.validate()
                if self.spec.checkpoint_interval and (self.state.step % self.spec.checkpoint_interval == 0
                                                       or self.state.step == self.spec.max_steps):
                    self.save()
        except Exception as error:
            self.failed = True
            self._in_step = False
            self.optimizer.zero_grad(set_to_none=True)
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output/'failure.json').write_text(json.dumps({'step': self.state.step,
                'error_type': type(error).__name__, 'recovery': 'explicit resume from last complete checkpoint'}))
            raise
        return self.state


@dataclass
class Progress:
    step: int = 0
    epoch: int = 0
    cursor: int = 0
    micro_step: int = 0


def capture_rng(model):
    numpy = np.random.get_state()
    cuda = any(p.device.type == 'cuda' for p in model.parameters())
    return {'python': random.getstate(), 'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if cuda else []}


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def validate_rng(state, model):
    try:
        random.Random().setstate(state['python'])
        numpy = state['numpy']
        np.random.RandomState().set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
        torch.Generator().set_state(state['torch'])
        cuda = any(p.device.type == 'cuda' for p in model.parameters())
        if len(state['cuda']) != (torch.cuda.device_count() if cuda else 0):
            raise ValueError('checkpoint CUDA RNG topology differs')
        for i,value in enumerate(state['cuda']): torch.Generator(device=f'cuda:{i}').set_state(value)
    except Exception as error:
        raise ValueError('checkpoint RNG state is invalid') from error
