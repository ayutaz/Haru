"""One bounded target-corpus head trial, with the normal audio feedback path.

No downloads, rentals or synthetic voice teachers are created here. Callers
authenticate the corpus, timing and model before invoking this training loop.
"""
from dataclasses import asdict
import json
from pathlib import Path

from .contracts import TrainingConfig, configure_trainable
from .diagnostics import preserved_inference
from .inference import export_inference
from .trainer import RunSpec, Trainer, dataset_fingerprint


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
    trainer = Trainer(model, train, validation, spec, root/'trainer')
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
                  conclusion='Await independent audio evaluation and human listening; not an adoption')
    save('result.json', result)
    return result
