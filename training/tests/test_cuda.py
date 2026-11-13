"""Required integration evidence on an explicitly provisioned vast.ai GPU.

Run with -m cuda, HARU_MODEL, HARU_CORPUS_ROOT and HARU_EVIDENCE_DIR set.
No test rents a GPU. Missing CUDA/checkpoint is an error, never a green skip.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from haru import HaruModel
from haru.models.vibevoice_codec import CodecStream
from haru.prompt import system_prompt_ids
from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.corpus import inspect_corpus, split_corpus
from haru_training.data import (TextChunk, align_text, collate_sequences, make_sequence,
                                read_cache, read_wave)
from haru_training.model import training_losses
from haru_training.monologue import monologue_sequence
from haru_training.prepare import encode_audio, encode_reference, encode_user_audio, prepare_corpus

pytestmark = pytest.mark.cuda
REVISION = '8d6e83bcd3701a76fc27a764965665da1df7da78'


def evidence(name, data):
    output = Path(os.environ['HARU_EVIDENCE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{name}.json').write_text(json.dumps(data, indent=2))


@pytest.fixture(scope='module')
def model():
    assert os.environ.get('HARU_COMPUTE_PROVIDER') == 'vast.ai'
    assert torch.cuda.is_available(), 'CUDA is required; CPU fallback is forbidden'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = HaruModel.from_pretrained(os.environ['HARU_MODEL'], device='cuda', dtype=torch.bfloat16,
                                       revision=REVISION)
    evidence('environment', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'total_vram': torch.cuda.get_device_properties(0).total_memory, 'checkpoint_revision': REVISION,
        'parameters': {name: sum(p.numel() for p in getattr(result, name).parameters())
                       for name in ('lm', 'head', 'codec', 'model_audio_proj', 'user_audio_proj')}})
    return result


@pytest.fixture(scope='module')
def corpus():
    clips = inspect_corpus(Path(os.environ['HARU_CORPUS_ROOT']))
    return clips, split_corpus(clips)


def test_actual_codec_whole_clip_matches_streaming_and_caches_real_corpus(model, corpus):
    clips, splits = corpus
    audio = read_wave(clips[0].path)[:5*3200+1]
    # FP32 checks the same causal operation without BF16 rounding amplification.
    codec = model.codec.float()
    signal = F.pad(torch.from_numpy(audio.T.copy()).cuda(), (0, (-len(audio)) % 3200))
    with torch.no_grad():
        whole = codec.encode(signal)
        stream = CodecStream(codec)
        pieces = [stream.encode(signal[:, start:start+3200]) for start in range(0, signal.shape[1], 3200)]
        sequential = torch.cat(pieces, dim=1)
        torch.testing.assert_close(whole, sequential, atol=1e-3, rtol=1e-3)
        sampled_stream = CodecStream(codec)
        generator = torch.Generator(device='cuda').manual_seed(23)
        sampled = [codec.sample(sampled_stream.encode(signal[:, i:i+3200]), generator)
                   for i in range(0, signal.shape[1], 3200)]
        offline_user = encode_user_audio(codec, audio, seed=23)
        torch.testing.assert_close(offline_user, torch.cat(sampled, dim=1).float().cpu(), atol=1e-3, rtol=1e-3)
    error = float((whole-sequential).abs().max())
    codec.to(dtype=torch.bfloat16)
    output = Path(os.environ['HARU_EVIDENCE_DIR']) / 'latents'
    begin = time.monotonic()
    manifest = prepare_corpus(codec, clips, output, codec_revision=REVISION, reference_ids=splits['reference'])
    assert len(manifest['records']) == 100
    assert sum(row['frames'] for row in manifest['records']) > 4000
    assert all(row['channels'] == 1 for row in manifest['records'])
    # A real cache round-trip on the frozen reference recording.
    reference = next(row for row in manifest['records'] if row['id'] == splits['reference'][0])
    value = read_cache(output/reference['reference_cache'], reference['reference_key'])
    assert value.shape[0] == 1 and value.shape[-1] == 64
    original = next(c for c in clips if c.id == reference['id'])
    live_reference = encode_reference(codec, read_wave(original.path), seed=reference['reference_key']['seed'])
    torch.testing.assert_close(value, live_reference)
    evidence('codec-preprocessing', {'recordings': len(manifest['records']),
        'frames': sum(row['frames'] for row in manifest['records']), 'seconds': time.monotonic()-begin,
        'fp32_streaming_max_abs_error': error, 'reference_id': reference['id']})


def parameter_digest(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def test_actual_corpus_becomes_causal_batches_with_real_tokenizer_and_encoded_silence(model, corpus):
    clips, splits = corpus
    cache_dir = Path(os.environ['HARU_EVIDENCE_DIR'])/'latents'
    manifest = json.loads((cache_dir/'manifest.json').read_text())
    rows = {r['id']: r for r in manifest['records']}
    references = set(splits['reference'])
    system = system_prompt_ids(model.tokenizer, 'あなたはつくよみちゃんです。相手と自然に日本語で話します。')
    sequences = [monologue_sequence(model.codec, model.tokenizer, clip, rows, cache_dir,
                 splits['reference'][0], system, model.config.pad_token_id, model.config.eop_token_id)
                 for clip in clips if clip.id not in references]
    assert len(sequences) == 98
    batch = collate_sequences(sequences, model.config.pad_token_id)
    assert set(batch['record_ids']).isdisjoint(references)
    assert torch.isfinite(batch['model_latents']).all() and torch.isfinite(batch['user_latents']).all()
    assert (batch['text_targets'][~batch['attention_mask']] == -100).all()
    assert not batch['audio_loss_mask'][~batch['attention_mask']].any()
    for row, record_id in enumerate(batch['record_ids']):
        active = batch['audio_loss_mask'][row]
        assert int(active.sum()) == rows[record_id]['frames']
        assert batch['target_frame_indices'][row, active].tolist() == list(range(rows[record_id]['frames']))
    evidence('actual-corpus-batches', {'supervised_recordings': len(sequences),
        'reference_recordings': sorted(references), 'batch_shape': list(batch['tokens'].shape),
        'supervised_frames': int(batch['audio_loss_mask'].sum()),
        'excluded_recordings': [], 'task': 'read-speech continuation, not dialogue training'})


@pytest.mark.parametrize('mode', ['head', 'lm_lora'])
def test_actual_haru_gradients_reach_selected_modules_and_preserve_frozen_weights(model, corpus, mode):
    clips, splits = corpus
    output = Path(os.environ['HARU_EVIDENCE_DIR'])/'latents'
    manifest = json.loads((output/'manifest.json').read_text())
    rows = {r['id']: r for r in manifest['records']}
    reference_row = rows[splits['reference'][0]]
    reference = read_cache(output/reference_row['reference_cache'], reference_row['reference_key'])[0, :4]
    examples = []
    for record_id in splits['train'][:2]:
        row = rows[record_id]
        latent = read_cache(output/row['cache'], row['key'])[0, :4]
        # Deliberately short graph for gradient tests, not a complete training utterance.
        text = torch.tensor([10, 11, 0, 0])
        user = encode_user_audio(model.codec, np.zeros((4*3200, 1), np.float32), seed=17)[0]
        examples.append(make_sequence(record_id, [3, 4], reference, latent, user, text,
                                      model.config.pad_token_id, model.config.eop_token_id))
    batch = collate_sequences(examples, model.config.pad_token_id)
    summary = configure_trainable(model, TrainingConfig(mode=mode, train_projectors=mode == 'lm_lora'))
    model.zero_grad(set_to_none=True)
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    before = parameter_digest(frozen)
    selected_before = parameter_digest(selected)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    losses = training_losses(model, batch, seed=23)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    gradients = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(p.grad).all() for n, p in gradients)
    head_norm = sum(float(p.grad.float().norm()) for n, p in gradients if n.startswith('head.'))
    lora_norm = sum(float(p.grad.float().norm()) for n, p in gradients if 'lora_' in n)
    projector_norm = sum(float(p.grad.float().norm()) for n, p in gradients if 'audio_proj' in n)
    assert head_norm > 0
    if mode == 'lm_lora':
        assert lora_norm > 0
        assert projector_norm > 0
    assert not any(n.startswith('codec.') for n, p in gradients)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()  # one disposable test update; no trainer/checkpoint is created
    assert before == parameter_digest(frozen)
    selected_after = parameter_digest(selected)
    assert selected_before != selected_after, 'selected parameters must actually update, including BF16 rounding'
    torch.cuda.synchronize()
    evidence(f'gradients-{mode}', {'loss': float(losses['loss'].detach()),
        'text_loss': float(losses['text_loss'].detach()), 'audio_loss': float(losses['audio_loss'].detach()),
        'trainable_parameters': summary['trainable'], 'head_gradient_norm': head_norm,
        'lora_gradient_norm': lora_norm, 'projector_gradient_norm': projector_norm,
        'frozen_sha256_before_and_after': before,
        'updated_parameters_sha256_before': selected_before, 'updated_parameters_sha256_after': selected_after,
        'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-start,
        'scope': 'single disposable forward/backward/update, not speaker adaptation quality'})
    model.zero_grad(set_to_none=True)
