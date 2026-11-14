"""Opt-in measurements around the production trainer; never allocate compute."""

from contextlib import contextmanager
import math

import torch

from .trainer import capture_rng, restore_rng


@contextmanager
def preserved_inference(model):
    """Observation must not consume training RNG or change nested module modes."""
    modes = {module: module.training for module in model.modules()}
    selected = {parameter: parameter.requires_grad for parameter in model.parameters()}
    rng = capture_rng(model)
    try:
        model.eval()
        model.requires_grad_(False)  # Existing replay API requires frozen weights.
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode
        for parameter, requires_grad in selected.items():
            parameter.requires_grad_(requires_grad)
        restore_rng(rng)


class StepDiagnostics:
    """Measure accumulated gradients and actual AdamW deltas, not LR estimates.

    Snapshots contain only selected parameters, on CPU. Frozen base/codec tensors
    are excluded. Hooks are removed on exit, including a failed optimizer step.
    """
    def __init__(self, trainer, *, loss_gradients=False):
        self.trainer = trainer
        self.updates = []
        self.handles = []
        self.before = {}
        self.components = {}
        self.measure_loss_gradients = loss_gradients
        self.loss_gradients = []
        self.original_loss = None

    def __enter__(self):
        if self.handles:
            raise RuntimeError('diagnostic hooks already attached')
        self.handles = [self.trainer.optimizer.register_step_pre_hook(self._before),
                        self.trainer.optimizer.register_step_post_hook(self._after)]
        if self.measure_loss_gradients:
            self.original_loss = self.trainer.loss_fn
            self.trainer.loss_fn = self._loss
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.before.clear()
        if self.original_loss is not None:
            self.trainer.loss_fn = self.original_loss
            self.original_loss = None

    def _loss(self, model, batch, *, seed):
        losses = self.original_loss(model, batch, seed=seed)
        if not torch.is_grad_enabled():
            return losses
        selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        measurements = {}
        for key in ('text_loss', 'audio_loss'):
            gradients = (torch.autograd.grad(losses[key], [p for _, p in selected],
                         retain_graph=True, allow_unused=True) if losses[key].requires_grad
                         else [None]*len(selected))
            components = {}
            for (name, parameter), grad in zip(selected, gradients):
                stats = components.setdefault(name.split('.')[0], {'squared': 0., 'parameters': 0})
                if grad is not None:
                    stats['parameters'] += parameter.numel()
                    stats['squared'] += float(grad.detach().double().square().sum())
            measurements[key] = {name: {'l2': math.sqrt(stats['squared']),
                'gradient_parameters': stats['parameters']} for name, stats in components.items()}
            del gradients
        text_count, audio_count = self.trainer._counts(batch)
        row = {'step': self.trainer.state.step+1, 'seed': seed,
            'record_ids': batch['record_ids'], 'text_count': text_count, 'audio_count': audio_count,
            'losses': measurements, 'scope': 'unweighted microbatch loss gradients'}
        self.trainer._event('diagnostic_loss_gradient', **row)
        self.loss_gradients.append(row)
        return losses

    def _before(self, optimizer, args, kwargs):
        self.before, self.components = {}, {}
        for name, parameter in self.trainer.model.named_parameters():
            if not parameter.requires_grad:
                continue
            component = name.split('.')[0]
            stats = self.components.setdefault(component, {'parameters': 0,
                'gradient_parameters': 0, 'gradient_squared': 0., 'update_squared': 0.})
            stats['parameters'] += parameter.numel()
            self.before[name] = parameter.detach().cpu().clone()
            if parameter.grad is not None:
                stats['gradient_parameters'] += parameter.numel()
                stats['gradient_squared'] += float(parameter.grad.detach().double().square().sum())

    def _after(self, optimizer, args, kwargs):
        for name, parameter in self.trainer.model.named_parameters():
            if name in self.before:
                delta = parameter.detach().cpu().double()-self.before[name].double()
                self.components[name.split('.')[0]]['update_squared'] += float(delta.square().sum())
        components = {name: {'parameters': stats['parameters'],
            'gradient_parameters': stats['gradient_parameters'],
            'gradient_l2': math.sqrt(stats['gradient_squared']),
            'update_l2': math.sqrt(stats['update_squared'])}
            for name, stats in self.components.items()}
        row = {'step': self.trainer.state.step+1, 'components': components}
        self.trainer._event('diagnostic_update', **row)
        self.updates.append(row)
        self.before.clear()


def diagnose(trainer, observer, *, loss_gradients=False):
    """Observe the current boundary and every subsequent successful update.

    A resumed trainer begins at its restored step, never reruns earlier updates.
    Validation and user observation both preserve training random state/modes.
    An observer returning False rejects the boundary and stops further updates.
    """
    def observe():
        metrics = trainer.validate()
        with preserved_inference(trainer.model):
            accepted = observer(trainer, metrics)
        if accepted is False:
            trainer._event('diagnostic_stopped', reason='observer_rejected_boundary')
            return False
        return True

    if not observe():
        return []
    with StepDiagnostics(trainer, loss_gradients=loss_gradients) as measurements:
        while trainer.state.step < trainer.spec.max_steps:
            before = trainer.state.step
            trainer.fit(until_step=before+1)
            if trainer.state.step == before:
                break  # Production trainer's deadline stopped the update.
            if not observe():
                break
    return measurements.updates
