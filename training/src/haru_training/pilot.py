"""Bounded Phase 5 integration runner; execute each stage in a fresh process.

python -m haru_training.pilot baseline|train|reload|restore --inputs ... --output ...
GPU use requires explicit vast.ai environment. This module never rents compute.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from safetensors.numpy import load_file, save_file
import torch

from haru import DuplexConfig, HaruModel
from haru.audio import prepare_reference
from haru.prompt import system_prompt_ids
from .comparison import measure_repeatability, compare_outputs
from .contracts import TrainingConfig, configure_trainable
from .corpus import Clip
from .data import read_wave, collate_sequences
from .inference import export_inference, load_inference, _digest
from .monologue import monologue_sequence
from .replay import replay, save_replay
from .trainer import Trainer, RunSpec, dataset_fingerprint

BASE = 'higurashi-ai/Haru-4B'
REVISION = '8d6e83bcd3701a76fc27a764965665da1df7da78'
PROMPT = 'あなたはつくよみちゃんです。相手と自然に日本語で話します。'


def select_records(splits, rows):
    ids = [i for group in splits.values() for i in group]
    if (set(splits) != {'train', 'validation', 'test', 'reference'}
            or len(ids) != len(set(ids)) or set(ids) != set(rows)
            or len(splits['train']) < 3 or len(splits['validation']) < 2
            or not splits['reference'] or any(rows[i]['frames'] <= 0 for i in ids)):
        raise ValueError('invalid or overlapping pilot splits/cache')
    ordered = sorted(splits['train'], key=lambda i: (rows[i]['frames'], i))
    return {'train': [ordered[0], ordered[len(ordered)//2], ordered[-1]],
            'validation': splits['validation'][:2], 'reference': splits['reference'][0]}


def _json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assets(inputs):
    manifest = json.loads((inputs/'latents/manifest.json').read_text())
    rows = {row['id']: row for row in manifest['records']}
    splits = json.loads((inputs/'splits.json').read_text())['splits']
    selected = select_records(splits, rows)
    if manifest['codec_revision'] != REVISION:
        raise ValueError('wrong codec revision')
    inventory = json.loads((inputs/'corpus-manifest.json').read_text())['records']
    clips = {row['id']: Clip(**{**row, 'path': inputs/row['path']}) for row in inventory}
    if set(clips) != set(rows):
        raise ValueError('inventory/cache mismatch')
    for row in rows.values():
        if row['key']['source_sha256'] != clips[row['id']].sha256:
            raise ValueError('cache/source identity mismatch')
    return clips, rows, selected


def _model():
    return HaruModel.from_pretrained(BASE, revision=REVISION, device='cuda', dtype=torch.bfloat16)


def _sequences(model, inputs):
    clips, rows, selected = _assets(inputs)
    system = system_prompt_ids(model.tokenizer, PROMPT)
    def make(ids):
        return [monologue_sequence(model.codec, model.tokenizer, clips[i], rows, inputs/'latents',
            selected['reference'], system, model.config.pad_token_id, model.config.eop_token_id) for i in ids]
    return make(selected['train']), make(selected['validation']), selected


def _smoke(model, inputs):
    user = read_wave(inputs/'validation-input.wav')[:, 0]
    ref = prepare_reference(read_wave(inputs/'reference.wav')[:, 0])
    return replay(model, user, reference=ref, system_prompt=PROMPT,
                  config=DuplexConfig(seed=7, speak_first=True, codec_backend='torch'))


@torch.no_grad()
def _probe(model, inputs):
    # Complete validation utterance with real tokens/reference/silence; no labels
    # are used to pick weights. Cache behavior is the original inference path.
    _, validation, _ = _sequences(model, inputs)
    batch = collate_sequences(validation[:1], model.config.pad_token_id)
    keys = ('tokens', 'model_latents', 'user_latents', 'model_mask', 'user_mask')
    hidden = model(model.embed(*(batch[k].to(model.device) for k in keys)), model.new_cache())[:, -1]
    generator = torch.Generator(device=model.device).manual_seed(7)
    noise = torch.randn(1, model.config.latent_dim, dtype=model.dtype, device=model.device, generator=generator)
    return {'logits': model.logits(hidden).float().cpu().numpy(),
            'latent': model.head.sample(hidden, noise, 1.5).float().cpu().numpy()}


def _outputs(model, inputs):
    result = _smoke(model, inputs)
    return result, {**_probe(model, inputs), 'audio': result['audio'],
                    'tokens': np.array([f['token'] for f in result['frames']], np.int64)}


def _record(destination, result, arrays, inputs):
    save_replay(destination, result, provenance={'base_model': BASE, 'revision': REVISION,
        'reference_sha256': _sha(inputs/'reference.wav'), 'input_sha256': _sha(inputs/'validation-input.wav'),
        'split_sha256': _sha(inputs/'splits.json'), 'cache_sha256': _sha(inputs/'latents/manifest.json')})
    save_file(arrays, str(destination/'probe.safetensors'))


def baseline(inputs, output):
    assert not (output/'head').exists() and not (output/'lm_lora').exists()
    model = _model()
    first, a = _outputs(model, inputs)
    second, b = _outputs(model, inputs)
    policy = measure_repeatability(a, b)
    _record(output/'baseline-validation', first, a, inputs)
    _record(output/'baseline-repeat', second, b, inputs)
    _json(output/'repeatability-policy.json', policy)
    fixed = json.loads((inputs/'evaluation-inputs/manifest.json').read_text())
    count = 0
    ref = prepare_reference(read_wave(inputs/'reference.wav')[:, 0])
    for scenario in fixed['scenarios']:
        path = inputs/'evaluation-inputs'/scenario['wav']
        assert _sha(path) == scenario['sha256']
        user = read_wave(path)[:, 0]
        for seed in fixed['seeds']:
            destination = output/'fixed-baseline'/scenario['id']/f'seed-{seed}'
            result = replay(model, user, reference=ref, system_prompt=PROMPT,
                            config=DuplexConfig(seed=seed, speak_first=True, codec_backend='torch'))
            save_replay(destination, result, provenance={'input_sha256': scenario['sha256'],
                'reference_sha256': _sha(inputs/'reference.wav'), 'base_model': BASE, 'revision': REVISION,
                'input_kind': fixed['input_kind'], 'purpose': 'original baseline only; no adaptation tuning'})
            count += 1
            print(json.dumps({'fixed_baseline_completed': count, 'case': scenario['id'], 'seed': seed}), flush=True)
    _json(output/'baseline.json', {'fixed_baseline_outputs': count, 'policy': policy,
        'parameters_sha256': _digest(model.named_parameters()), 'voice_quality_evaluated': False})


def train(inputs, output, mode):
    assert (output/'baseline.json').is_file(), 'baseline and policy must precede training'
    root = output/mode
    root.mkdir(exist_ok=False)
    model = _model()
    configure_trainable(model, TrainingConfig(mode=mode, train_projectors=mode == 'lm_lora'))
    training, validation, selected = _sequences(model, inputs)
    spec = RunSpec(base_model=BASE, base_revision=REVISION, mode=mode,
        train_projectors=mode == 'lm_lora', dataset_sha256=dataset_fingerprint(training),
        validation_sha256=dataset_fingerprint(validation), split_sha256=_sha(inputs/'splits.json'),
        cache_sha256=_sha(inputs/'latents/manifest.json'), max_steps=3, learning_rate=1e-4,
        accumulation_steps=1, warmup_steps=1)
    trainer = Trainer(model, training, validation, spec, root/'trainer')
    before = _digest((n, p) for n, p in model.named_parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit()
    peak = torch.cuda.max_memory_allocated()
    selected_after = _digest((n, p) for n, p in model.named_parameters() if p.requires_grad)
    frozen_after = trainer._frozen_digest()
    artifact = export_inference(trainer, root/'inference')
    data = {'mode': mode, 'steps': trainer.state.step, 'full_utterances': True, 'real_text': True,
        'selected': selected, 'training_frames': [int(r.supervise.sum()) for r in training],
        'training_sequence_lengths': [len(r.tokens) for r in training],
        'selected_before': before, 'selected_after': selected_after, 'frozen_before': trainer.frozen_digest,
        'frozen_after': frozen_after, 'parameters_sha256': _digest(model.named_parameters()),
        'peak_training_vram_bytes': peak, 'training_seconds': time.monotonic()-started,
        'spec': asdict(spec), 'inference': artifact.name, 'voice_quality_evaluated': False}
    assert data['selected_before'] != data['selected_after'] and data['frozen_before'] == frozen_after
    del trainer.optimizer
    model.eval().requires_grad_(False)
    torch.cuda.empty_cache()
    result, arrays = _outputs(model, inputs)
    _record(root/'before-reload', result, arrays, inputs)
    _json(root/'training.json', data)


def reload(inputs, output, mode):
    root = output/mode
    model = load_inference(root/'inference', device='cuda', dtype=torch.bfloat16)
    result, arrays = _outputs(model, inputs)
    _record(root/'reloaded', result, arrays, inputs)
    policy = json.loads((output/'repeatability-policy.json').read_text())
    comparison = compare_outputs(load_file(str(root/'before-reload/probe.safetensors')), arrays, policy)
    trained = json.loads((root/'training.json').read_text())
    digest = _digest(model.named_parameters())
    _json(root/'reloaded.json', {'parameters_sha256': digest, 'comparison': comparison,
                                'fresh_process': True, 'voice_quality_evaluated': False})
    assert trained['parameters_sha256'] == digest and comparison['passed']


def restore(inputs, output):
    model = _model()
    result, arrays = _outputs(model, inputs)
    _record(output/'restored', result, arrays, inputs)
    original = json.loads((output/'baseline.json').read_text())
    comparison = compare_outputs(load_file(str(output/'baseline-validation/probe.safetensors')), arrays,
                                 json.loads((output/'repeatability-policy.json').read_text()))
    summary = {'all_base_parameters_match': _digest(model.named_parameters()) == original['parameters_sha256'],
               'has_lora': bool(getattr(model.lm, 'peft_config', {})), 'comparison': comparison}
    _json(output/'restored.json', summary)
    assert summary['all_base_parameters_match'] and not summary['has_lora'] and comparison['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['baseline', 'train', 'reload', 'restore'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['head', 'lm_lora'])
    args = parser.parse_args()
    if args.phase in ('train', 'reload') and args.mode is None:
        parser.error('train/reload requires --mode')
    if os.environ.get('HARU_COMPUTE_PROVIDER') != 'vast.ai' or not torch.cuda.is_available():
        raise RuntimeError('pilot CUDA execution is restricted to explicit vast.ai compute')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(73); random.seed(73); np.random.seed(73)
    args.output.mkdir(parents=True, exist_ok=True)
    _json(args.output/(args.phase+'-'+(args.mode or 'base')+'-environment.json'), {
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
        'total_vram_bytes': torch.cuda.get_device_properties(0).total_memory,
        'phase': args.phase, 'mode': args.mode, 'started_at': time.time()})
    if args.phase in ('train', 'reload'):
        globals()[args.phase](args.inputs, args.output, args.mode)
    else:
        globals()[args.phase](args.inputs, args.output)


if __name__ == '__main__':
    main()
