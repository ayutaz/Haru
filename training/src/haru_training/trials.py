"""Measured short comparisons; diagnostic stability never proves voice quality."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import time
import unicodedata

import numpy as np

from .contracts import TrainingConfig
from .diagnostics import diagnose


STOP_POLICY = {'text_loss_increase': 1., 'audio_loss_ratio': 2.,
               'punctuation_fraction': .5, 'minimum_text_characters': 8,
               'consecutive_phrase_repetitions': 4,
               'scope': 'conservative diagnostic stop rules, not a quality acceptance test'}


def condition_config(condition):
    if condition not in {'B', 'C', 'D'}:
        raise ValueError('update condition must be B, C or D')
    return TrainingConfig(mode='head' if condition == 'B' else 'lm_lora',
                          train_projectors=condition == 'D')


def assess_response(initial, current, response):
    """Fixed before GPU comparisons; deliberately does not score speaker identity."""
    reasons = []
    for key in ('text_loss', 'audio_loss'):
        if not all(math.isfinite(float(value[key])) for value in (initial, current)):
            reasons.append(f'nonfinite_{key}')
    if current['text_loss'] > initial['text_loss']+STOP_POLICY['text_loss_increase']:
        reasons.append('text_loss_increase')
    if current['audio_loss'] > max(initial['audio_loss'], 1e-12)*STOP_POLICY['audio_loss_ratio']:
        reasons.append('audio_loss_increase')
    audio = np.asarray(response['audio'])
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        reasons.append('invalid_audio')
    elif not np.any(audio):
        reasons.append('silent_audio')
    text = ''.join(response['text'].split())
    if not text:
        reasons.append('empty_text')
    if len(text) >= STOP_POLICY['minimum_text_characters']:
        punctuation = sum(unicodedata.category(c)[0] in {'P', 'S'} for c in text)
        if punctuation/len(text) >= STOP_POLICY['punctuation_fraction']:
            reasons.append('punctuation_dominated_text')
        if re.search(r'(.{2,16}?)\1{3,}', text):
            reasons.append('consecutive_phrase_repetition')
    return reasons


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run_measured_trial(trainer, destination, generate):
    """Persist step zero and every response before deciding on further updates.

    The caller supplies a fresh trainer and a rendering function which writes
    immutable replay artifacts to its destination and returns text/audio. The
    renderer runs frozen in eval, with training RNG/modes/gradient flags restored.
    This function is also exercised with CPU trainers, without cloud operations.
    """
    if trainer.state.step != 0 or trainer.failed:
        raise ValueError('each comparison requires a fresh, successful trainer')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    _json(root/'spec.json', asdict(trainer.spec))
    _json(root/'stop-policy.json', STOP_POLICY)
    observations = []
    started = time.monotonic()

    def observe(current, metrics):
        output = root/f'step-{current.state.step:08d}'
        response = generate(current.model, output)
        initial = observations[0]['metrics'] if observations else metrics
        reasons = assess_response(initial, metrics, response)
        row = {'step': current.state.step, 'metrics': metrics, 'text': response['text'],
               'rejection_reasons': reasons, 'response_directory': output.name}
        _json(root/f'observation-{current.state.step:08d}.json', row)
        observations.append(row)
        return not reasons

    try:
        updates = diagnose(trainer, observe, loss_gradients=True)
        frozen_unchanged = trainer._frozen_digest() == trainer.frozen_digest
        if not frozen_unchanged:
            raise ValueError('frozen parameters changed during trial')
        result = {'steps': trainer.state.step, 'completed': trainer.state.step == trainer.spec.max_steps,
            'stopped_for_quality': bool(observations[-1]['rejection_reasons']),
            'candidate': False, 'voice_quality_evaluated': False,
            'frozen_parameters_unchanged': frozen_unchanged,
            'observations': observations, 'updates': updates,
            'elapsed_seconds': time.monotonic()-started}
        _json(root/'result.json', result)
    except Exception as error:
        _json(root/'failure.json', {'step': trainer.state.step, 'error_type': type(error).__name__})
        raise
    return result
