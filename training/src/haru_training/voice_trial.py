"""One epoch for a voice comparison, without conversation-quality selection."""
from dataclasses import asdict
import math
from pathlib import Path

from .diagnostics import preserved_inference
from .inference import export_inference
from .trials import _json


def run_voice_trial(trainer, destination, generate):
    """Keep baseline/final responses; numerical and resource limits still apply."""
    width = trainer.spec.micro_batch_size * trainer.spec.accumulation_steps
    if (trainer.state.step != 0 or trainer.failed or trainer.spec.checkpoint_interval
            or trainer.spec.max_steps != math.ceil(len(trainer.train) / width)):
        raise ValueError('voice trial requires a fresh, bounded single epoch')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    _json(root/'spec.json', asdict(trainer.spec))
    observations = []
    for label in ('zero-shot', 'finetuned'):
        if label == 'finetuned':
            trainer.fit()
            trainer.save()
            export_inference(trainer, root/'inference')
        metrics = trainer.validate()
        with preserved_inference(trainer.model):
            responses = generate(trainer.model, root/label)
        row = {'step': trainer.state.step, 'directory': label, 'metrics': metrics,
               'texts': {name: answer['text'] for name, answer in responses.items()}}
        observations.append(row)
        _json(root/(label+'.json'), row)
    result = {'steps': trainer.state.step,
              'completed_epoch': trainer.state.step == trainer.spec.max_steps and trainer.state.epoch == 1,
              'frozen_parameters_unchanged': trainer._frozen_digest() == trainer.frozen_digest,
              'observations': observations, 'voice_improvement_confirmed': False,
              'purpose': 'paired voice comparison; conversation drift is recorded, not a stop criterion'}
    if not result['frozen_parameters_unchanged']:
        raise ValueError('frozen base parameters changed')
    _json(root/'result.json', result)
    return result
