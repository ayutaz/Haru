"""Bounded first real-weight trial; preparation is inert and CPU execution is refused."""
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_trial(spec, *, root):
    """Check the predeclared pilot bounds and every input before loading weights."""
    try:
        if (spec['version'] != 1 or spec['base_model'] != 'higurashi-ai/Haru-4B'
                or not re.fullmatch('[a-f0-9]{40}', spec['base_revision'])
                or not re.fullmatch('[a-f0-9]{64}', spec['dataset_sha256'])
                or type(spec['max_steps']) is not int or not 1 <= spec['max_steps'] <= 30
                or not math.isfinite(spec['max_seconds']) or not 0 < spec['max_seconds'] <= 600
                or not math.isfinite(spec['learning_rate']) or spec['learning_rate'] <= 0
                or spec['rank'] != 8 or type(spec['seed']) is not int or spec['seed'] < 0
                or not isinstance(spec['system_prompt'], str) or not spec['system_prompt']
                or not spec['target_speaker']):
            raise ValueError('invalid bounded trial specification')
        cases = spec['cases']; ref = spec['reference']
        if (len(cases) != 3 or len({r['id'] for r in cases}) != 3
                or any(not r['id'].startswith('dev-') or not r['text'] for r in cases)
                or ref['split'] not in ('reference', 'validation') or not ref['group_id']
                or ref['speaker'] != spec['target_speaker']):
            raise ValueError('trial requires three development cases and an untrained reference')
        for row in [spec['dataset'], spec['membership'], ref, *cases]:
            path = (Path(root)/row['path']).resolve()
            if (not path.is_relative_to(Path(root).resolve())
                    or not re.fullmatch('[a-f0-9]{64}', row['sha256'])
                    or file_hash(path) != row['sha256']):
                raise ValueError('frozen input path or hash differs')
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('incomplete frozen trial specification') from error
    return spec


def require_vast_cuda():
    if os.environ.get('HARU_COMPUTE_PROVIDER') != 'vast.ai' or not torch.cuda.is_available():
        raise RuntimeError('real-weight codec trial requires an explicitly provisioned Vast.ai CUDA GPU')


def compare_traces(left, right):
    keys = ('tokens', 'feedback', 'rng', 'global_cpu', 'global_cuda', 'head_calls')
    return all(key in left and key in right and left[key] == right[key] for key in keys)


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def capture(model, user, reference, spec, *, max_frames=None):
    from unittest.mock import patch
    from haru import DuplexConfig, DuplexSession
    size = model.config.frame_samples
    count = (len(user)+size-1)//size
    if max_frames is not None: count = min(count, max_frames)
    padded = np.pad(user, (0, max(0, count*size-len(user))))
    config = DuplexConfig(seed=spec['seed'], max_frames=count, codec_backend='torch', speak_first=False)
    session = DuplexSession(model, spec['system_prompt'], reference, config)
    frames = []; latents = []
    trace = dict(tokens=[], feedback=[], rng=[])
    with patch.object(model.head, 'sample', wraps=model.head.sample) as head:
        for index in range(count):
            frame = session.start() if index == 0 else session.step(padded[(index-1)*size:index*size])
            frames.append(frame); latents.append(session._latent[:, None].clone())
            trace['tokens'].append(frame.token)
            trace['feedback'].append(_tensor_hash(session._latent))
            trace['rng'].append(_tensor_hash(session._generator.get_state()))
        trace['head_calls'] = head.call_count
    trace.update(global_cpu=_tensor_hash(torch.random.get_rng_state()),
                 global_cuda=_tensor_hash(torch.cuda.get_rng_state(model.device))
                 if model.device.type == 'cuda' else None)
    result = dict(audio=np.concatenate([f.audio for f in frames]), sample_rate=24000,
        text=''.join(f.text for f in frames).strip(), config=asdict(config),
        system_prompt=spec['system_prompt'], feedback_mode='unchanged_raw_latent',
        frames=[{k:v for k,v in asdict(f).items() if k != 'audio'} for f in frames])
    return result, torch.cat(latents, dim=1), trace


def run_trial(path, destination):
    from haru.models.haru import HaruModel
    from .codec_adapter import get_adapter, set_adapter_enabled, decode_training
    from .codec_adapter_dataset import load_dataset
    from .codec_adapter_training import CodecTrainer, TrainSpec
    from .codec_adapter_inference import codec_digest, export_adapter
    from .codec_adapter_evaluation import save_decoder_comparison
    from .checkpoint import read_checkpoint
    from .data import read_wave
    from .replay import save_replay
    require_vast_cuda()
    path = Path(path).resolve(); root = path.parent
    spec = validate_trial(json.loads(path.read_text(encoding='utf-8')), root=root)
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    def report(name, value):
        (out/(name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, json.dumps(value, ensure_ascii=False), flush=True)
    data, pairs = load_dataset(root/spec['dataset']['path'])
    if (data['base_revision'] != spec['base_revision'] or data['base_model'] != spec['base_model']
            or data['dataset_sha256'] != spec['dataset_sha256']
            or data['target_speaker'] != spec['target_speaker']
            or data['dataset_input_manifest_sha256'] != spec['membership']['sha256']):
        raise ValueError('prepared dataset and frozen trial disagree')
    membership = json.loads((root/spec['membership']['path']).read_text())
    ref = spec['reference']
    if any(r['group_id'] == ref['group_id'] or ref['sha256'] in [r.get(k) for k in
        ('source_sha256','target_sha256','source_original_sha256','target_original_sha256')]
        for r in membership['records']):
        raise ValueError('reference recording was used in training')
    torch.set_num_threads(4); torch.manual_seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from haru.models.loaders import resolve_checkpoint
    from safetensors import safe_open
    directory = resolve_checkpoint(spec['base_model'], spec['base_revision'])
    if file_hash(directory/'model.safetensors') != data['codec_weights_sha256']:
        raise ValueError('base checkpoint differs from teacher preparation')
    model = HaruModel.from_pretrained(str(directory),
        device='cuda', dtype=torch.bfloat16)
    model.codec.float()
    # Restore the original codec directly, avoiding BF16 -> FP32 round-trip loss.
    with safe_open(str(directory/'model.safetensors'), framework='pt', device='cpu') as saved:
        state = {key.removeprefix('codec.'):saved.get_tensor(key).to('cuda', torch.float32)
                 for key in saved.keys() if key.startswith('codec.')}
    model.codec.load_state_dict(state, strict=True)
    del state
    report('environment', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), codec_dtype=str(model.codec.dtype),
        trial_sha256=file_hash(path), base_revision=spec['base_revision'], dataset_sha256=data['dataset_sha256']))
    frozen_digest = codec_digest(model.codec)
    train = CodecTrainer(model, pairs, TrainSpec(base_model=spec['base_model'],
        base_revision=spec['base_revision'], target_speaker=spec['target_speaker'],
        max_steps=spec['max_steps'], max_seconds=spec['max_seconds'],
        learning_rate=spec['learning_rate'], seed=spec['seed']), rank=spec['rank'])
    adapter = get_adapter(model.codec)
    reference = read_wave(root/ref['path'])[:,0]
    user = read_wave(root/spec['cases'][0]['path'])[:,0]
    model.requires_grad_(False)
    set_adapter_enabled(model.codec, False)
    base, _, left_trace = capture(model, user, reference, spec, max_frames=4)
    set_adapter_enabled(model.codec, True)
    zero, _, zero_trace = capture(model, user, reference, spec, max_frames=4)
    if not np.array_equal(base['audio'], zero['audio']) or not compare_traces(left_trace, zero_trace):
        raise RuntimeError('zero adapter does not reproduce the base')
    with torch.no_grad(): adapter.up.bias.fill_(.02)
    changed, _, changed_trace = capture(model, user, reference, spec, max_frames=4)
    if np.array_equal(base['audio'], changed['audio']) or not compare_traces(left_trace, changed_trace):
        raise RuntimeError('nonzero adapter does not isolate the waveform change')
    with torch.no_grad(): adapter.up.bias.zero_()
    adapter.requires_grad_(True)
    # Measure one actual update, retaining it as step 1 of the same bounded trial.
    before = {k:v.clone() for k,v in adapter.state_dict().items()}
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    history = train.fit(until_step=1)
    torch.cuda.synchronize()
    if not any(not torch.equal(before[k],v) for k,v in adapter.state_dict().items()):
        raise RuntimeError('real adapter weights did not update')
    if codec_digest(model.codec) != frozen_digest: raise RuntimeError('frozen codec changed')
    train.save(out/'step1')
    report('preflight', dict(zero_identity=True, nonzero_wave_change=True, conversation_trace_exact=True,
        frozen_codec_exact=True, real_update=True, step1_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), trainable_parameters=sum(p.numel() for p in adapter.parameters())))
    history += train.fit()
    train.save(out/'checkpoint')
    export_adapter(model, out/'adapter', base_model=spec['base_model'], base_revision=spec['base_revision'],
        target_speaker=spec['target_speaker'], dataset_sha256=data['dataset_sha256'])
    saved = read_checkpoint(out/'adapter')
    weights = {k:v.clone() for k,v in adapter.state_dict().items()}
    # Verify checkpoint readback on the live base without loading a second 4B model.
    from .codec_adapter_inference import _validate
    _validate(saved)
    with torch.no_grad():
        expected,_ = decode_training(model.codec, pairs[0].latents[None])
        for p in adapter.parameters(): p.zero_()
        adapter.load_state_dict(saved['weights'], strict=True)
        restored,_ = decode_training(model.codec, pairs[0].latents[None])
    if not torch.equal(expected,restored): raise RuntimeError('adapter reload changed waveform')
    del expected,restored,weights
    if codec_digest(model.codec) != frozen_digest: raise RuntimeError('frozen codec changed after training')
    report('training', dict(steps=train.step, elapsed=train.elapsed, history=history,
        frozen_codec_exact=True, adapter_reload_exact=True, voice_similarity_confirmed=False))
    model.eval().requires_grad_(False)
    for case in spec['cases']:
        pcm = read_wave(root/case['path'])[:,0]
        set_adapter_enabled(model.codec, False)
        base, latent, trace = capture(model, pcm, reference, spec)
        set_adapter_enabled(model.codec, True)
        candidate, other_latent, other_trace = capture(model, pcm, reference, spec)
        if not compare_traces(trace, other_trace) or not torch.equal(latent, other_latent):
            raise RuntimeError('adapted conversation path differs')
        metadata = dict(representation='raw_codec_sample', input_sha256=case['sha256'],
            base_revision=spec['base_revision'], target_speaker=spec['target_speaker'],
            dataset_input_manifest_sha256=spec['membership']['sha256'], text=base['text'],
            system_prompt=spec['system_prompt'], config=base['config'])
        folder = out/case['id']
        save_decoder_comparison(folder, model, latent, metadata)
        save_replay(folder/'stream-baseline', base, provenance=metadata)
        save_replay(folder/'stream-candidate', candidate, provenance=metadata)
        report(case['id'], dict(text=base['text'], conversation_trace_exact=True,
            frames=len(trace['tokens']), trace=trace, input_sha256=case['sha256']))
    report('complete', dict(execution_complete=True, steps=train.step,
        automatic_evaluation_pending=True, listening_pending=True, voice_similarity_confirmed=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('spec'); parser.add_argument('destination')
    args = parser.parse_args()
    run_trial(args.spec, args.destination)
