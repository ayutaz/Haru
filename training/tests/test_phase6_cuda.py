"""Actual Phase 6 controlled trials, only after dialogue review and provisioning.

Run A, then B/C, as fresh subprocesses. These tests never rent compute and never
substitute CPU or a skip for missing CUDA. D is an explicit optional invocation.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.cuda


def run(condition):
    assert os.environ.get('HARU_COMPUTE_PROVIDER') == 'vast.ai'
    assert torch.cuda.is_available(), 'CUDA is mandatory'
    root = Path(os.environ['HARU_PHASE6_OUTPUT'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/f'condition-{condition}.log').open('x', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'haru_training.phase6', condition,
            '--inputs', os.environ['HARU_PHASE6_INPUTS'], '--output', str(root),
            '--review', os.environ['HARU_PHASE6_REVIEW']], stdout=log, stderr=subprocess.STDOUT, check=True)
    return root


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_A_original_repeatability_and_zero_update_lora_session_parity():
    root = run('A')
    result = read(root/'A/result.json')
    assert result['zero_update_parity'] and result['comparison']['passed']
    assert (root/'A/zero-update/response.wav').is_file()
    assert read(root/'A/dialogue-review.json')['dialogues'] > 0
    assert not (root/'B').exists() and not (root/'C').exists()


@pytest.mark.parametrize('condition', ['B', 'C'])
def test_controlled_updates_save_every_response_and_stop_at_rejected_boundary(condition):
    root = run(condition)
    result = read(root/condition/'result.json')
    metadata = read(root/condition/'training.json')
    assert 0 <= result['steps'] <= 3
    assert result['candidate'] is False and not metadata['voice_quality_evaluated']
    assert result['frozen_parameters_unchanged']
    assert metadata['contract'] == read(root/'A/contract.json')
    assert metadata['spec']['learning_rate'] == 1e-5
    assert metadata['spec']['train_projectors'] is False
    assert [row['step'] for row in result['observations']] == list(range(result['steps']+1))
    assert len(result['updates']) == result['steps']
    for row in result['observations']:
        folder = root/condition/row['response_directory']
        assert (folder/'response.wav').is_file() and (folder/'probe.safetensors').is_file()
        assert row['metrics']['text_count'] > 0 and row['metrics']['audio_count'] > 0
    assert read(root/condition/'step-00000000/zero-update-parity.json')['passed']
    if result['stopped_for_quality']:
        assert result['observations'][-1]['rejection_reasons']
    else:
        assert result['steps'] == 3 and result['completed']
    if result['steps']:
        assert (root/condition/'inference/manifest.json').is_file()
    assert metadata['peak_allocated_bytes'] > 0
