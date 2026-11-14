"""Run one fresh Phase 6 condition per process on explicitly supplied vast.ai GPU.

No rentals or remote API calls occur here. Data/quality review and the external
budget/deadline watchdog must be completed by the operator before invocation.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import numpy as np
from safetensors.numpy import load_file
import torch

from .comparison import compare_outputs, measure_repeatability
from .contracts import configure_trainable
from .diagnostics import preserved_inference
from .dialogue import validate_cohort, validate_text_chunks
from .inference import export_inference, _digest
from .pilot import BASE, REVISION, PROMPT, _assets, _json, _model, _outputs, _record, _sequences, _sha
from .trainer import RunSpec, Trainer, dataset_fingerprint
from .trials import condition_config, run_measured_trial


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _relative(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('asset must have a relative local path')
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('asset path escapes its root or does not exist')
    return path


def preflight(inputs, review_path):
    """Authenticate real data and record an explicit prior review, not infer one.

    Notes/status are review evidence supplied by the operator; their truth cannot
    be established by parsing JSON or passing waveform checks. No quality verdict
    is generated automatically from these technical checks.
    """
    inputs, review_path = Path(inputs), Path(review_path)
    review = _read(review_path)
    if (review.get('quality_status') != 'reviewed_acceptable_for_replay'
            or review.get('source_split_checked') is not True
            or any(not isinstance(review.get(key), str) or not review[key].strip()
                   for key in ('source_and_usage_notes', 'voice_and_pronunciation_notes', 'reference_policy'))):
        raise ValueError('dialogue source, usage, quality and reference review is incomplete')
    manifest = _relative(review_path.parent, review.get('manifest'))
    if _sha(manifest) != review.get('manifest_sha256'):
        raise ValueError('dialogue manifest changed after review')
    splits = _read(inputs/'splits.json')['splits']
    protected = set(splits['test']) | set(splits['reference']) | set(splits['validation'])
    records = _read(manifest)['records']
    qc = validate_cohort(records, manifest.parent, protected_source_ids=protected)
    for row, checked in zip(records, qc):
        # Negative sentinels cannot collide with real nonnegative token IDs.
        # Actual model PAD/EOP are checked again when constructing the sequence.
        validate_text_chunks(row, checked['frames'], -1, -2)
    return {'review_sha256': _sha(review_path), 'manifest_sha256': _sha(manifest),
            'dialogues': len(qc), 'manual_review_recorded': True,
            'technical_checks': qc, 'review': review}


def predecessor(output, condition):
    output = Path(output)
    if condition not in {'A', 'B', 'C', 'D'}:
        raise ValueError('unknown condition')
    if condition == 'A':
        return
    previous = {'B': 'A', 'C': 'B', 'D': 'C'}[condition]
    path = output/previous/'result.json'
    if not path.is_file():
        raise ValueError(f'condition {previous} must be inspected before {condition}')
    result = _read(path)
    if condition == 'B' and not result.get('zero_update_parity'):
        raise ValueError('zero-update parity must pass before training')
    if previous in {'B', 'C'}:
        metadata_path = output/previous/'training.json'
        if not metadata_path.is_file():
            raise ValueError('previous condition export/metadata is incomplete')
        metadata = _read(metadata_path)
        if metadata.get('condition') != previous or metadata.get('steps') != result.get('steps'):
            raise ValueError('previous condition completion metadata is inconsistent')
    if condition == 'D' and (not result.get('completed') or result.get('stopped_for_quality')):
        raise ValueError('projector trial requires a stable C result')


def source_fingerprint():
    import haru
    roots = {'haru_training': Path(__file__).parent, 'haru': Path(haru.__file__).parent}
    return {f'{name}/{path.relative_to(root).as_posix()}': _sha(path)
            for name, root in roots.items() for path in sorted(root.rglob('*.py'))}


def _contract(inputs, review):
    _, rows, selected = _assets(inputs)
    # Authenticate cache payloads as well as the manifest used to select them.
    for row in rows.values():
        for file_key, hash_key in (('cache', 'cache_sha256'),
                                   ('reference_cache', 'reference_cache_sha256')):
            if file_key in row:
                if _sha(_relative(inputs/'latents', row[file_key])) != row[hash_key]:
                    raise ValueError('latent cache changed after preparation')
    names = ('splits.json', 'corpus-manifest.json', 'latents/manifest.json',
             'reference.wav', 'validation-input.wav')
    return {'base': BASE, 'revision': REVISION, 'prompt': PROMPT,
        'inputs': {n: _sha(inputs/n) for n in names}, 'selected': selected,
        'review_sha256': review['review_sha256'], 'dialogue_manifest_sha256': review['manifest_sha256'],
        'source_sha256': source_fingerprint(), 'torch_version': str(torch.__version__),
        'seed': 20261114, 'learning_rate': 1e-5, 'max_steps': 3,
        'response_seed': 7, 'codec_backend': 'torch'}


def execute(condition, inputs, output, review_path):
    """Called by guarded CLI; all conditions independently reload pinned base."""
    inputs, output = Path(inputs), Path(output)
    review = preflight(inputs, review_path)
    predecessor(output, condition)
    contract = _contract(inputs, review)
    root = output/condition
    if root.exists():
        raise FileExistsError('condition already exists; preserve the original result')
    if condition != 'A' and _read(output/'A/contract.json') != contract:
        raise ValueError('source, data, review, reference or comparison conditions changed')
    # A fresh process and a fresh loader for each invocation, never prior weights.
    model = _model()
    base_digest = _digest(model.named_parameters())
    if condition == 'A':
        root.mkdir(parents=True, exist_ok=False)
        _json(root/'contract.json', contract)
        _json(root/'dialogue-review.json', review)
        with preserved_inference(model):
            first, a = _outputs(model, inputs)
            repeated, b = _outputs(model, inputs)
        policy = measure_repeatability(a, b)
        _record(root/'base', first, a, inputs)
        _record(root/'repeat', repeated, b, inputs)
        # Fix tolerance from unchanged base before exposing any wrapped result.
        _json(root/'repeatability-policy.json', policy)
        configure_trainable(model, condition_config('C'))
        with preserved_inference(model):
            wrapped, c = _outputs(model, inputs)
        _record(root/'zero-update', wrapped, c, inputs)
        comparison = compare_outputs(a, c, policy)
        result = {'zero_update_parity': comparison['passed'], 'comparison': comparison,
                  'base_parameters_sha256': base_digest, 'candidate': False}
        _json(root/'result.json', result)
        if not comparison['passed']:
            raise ValueError('zero-update LoRA changed inference; investigate before training')
        return result

    baseline = _read(output/'A/result.json')
    if base_digest != baseline['base_parameters_sha256']:
        raise ValueError('fresh model differs from baseline parameters')
    config = condition_config(condition)
    configure_trainable(model, config)
    training, validation, selected = _sequences(model, inputs)
    spec = RunSpec(base_model=BASE, base_revision=REVISION, mode=config.mode,
        train_projectors=config.train_projectors, dataset_sha256=dataset_fingerprint(training),
        validation_sha256=dataset_fingerprint(validation), split_sha256=contract['inputs']['splits.json'],
        cache_sha256=contract['inputs']['latents/manifest.json'], max_steps=3, learning_rate=1e-5,
        warmup_steps=1, seed=contract['seed'])
    trainer = Trainer(model, training, validation, spec, root/'trainer')
    policy = _read(output/'A/repeatability-policy.json')
    base_arrays = load_file(str(output/'A/base/probe.safetensors'))

    def generate(current, destination):
        result, arrays = _outputs(current, inputs)
        _record(destination, result, arrays, inputs)
        if trainer.state.step == 0:
            comparison = compare_outputs(base_arrays, arrays, policy)
            _json(destination/'zero-update-parity.json', comparison)
            if not comparison['passed']:
                raise ValueError('condition step zero differs from original inference')
        return result

    torch.cuda.reset_peak_memory_stats()
    result = run_measured_trial(trainer, root, generate)
    artifact = export_inference(trainer, root/'inference') if trainer.state.step else None
    _json(root/'training.json', {'condition': condition, 'spec': asdict(spec), 'selected': selected,
        'contract': contract, 'dialogue_review': review, 'steps': trainer.state.step,
        'inference': artifact.name if artifact else None,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'diagnostic_only': True, 'candidate': False, 'voice_quality_evaluated': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition', choices=['A', 'B', 'C', 'D'])
    for name in ('inputs', 'output', 'review'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('HARU_COMPUTE_PROVIDER') != 'vast.ai' or not torch.cuda.is_available():
        raise RuntimeError('Phase 6 GPU execution requires explicit vast.ai compute')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261114); random.seed(20261114); np.random.seed(20261114)
    execute(args.condition, args.inputs, args.output, args.review)


if __name__ == '__main__':
    main()
