"""Phase 5 acceptance: full utterances -> trained weights -> fresh conversation.

Provisioning is explicit and external. Missing CUDA is a failure, never a skip.
Baseline 10 cases x 3 seeds are generated before any adaptation. Candidate
smoke tests use validation only; test speech is never used for tuning.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.cuda


def run(phase, *extra):
    assert os.environ.get('HARU_COMPUTE_PROVIDER') == 'vast.ai'
    assert torch.cuda.is_available(), 'CUDA is mandatory'
    root = Path(os.environ['HARU_EVIDENCE_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/(phase+'-'+('-'.join(extra) or 'base')+'.log')).open('x') as log:
        subprocess.run([sys.executable, '-m', 'haru_training.pilot', phase,
                        '--inputs', os.environ['HARU_PHASE5_INPUTS'], '--output', str(root), *extra],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return root


def test_original_fixed_baselines_and_repeatability_are_saved_before_training():
    root = run('baseline')
    result = json.loads((root/'baseline.json').read_text())
    assert result['fixed_baseline_outputs'] == 30
    assert result['voice_quality_evaluated'] is False
    assert result['policy']['tokens'] == 'exact'
    assert not (root/'head').exists() and not (root/'lm_lora').exists()


@pytest.mark.parametrize('mode', ['head', 'lm_lora'])
def test_full_utterance_training_exports_and_fresh_process_session_reloads(mode):
    root = run('train', '--mode', mode)
    run('reload', '--mode', mode)
    trained = json.loads((root/mode/'training.json').read_text())
    reloaded = json.loads((root/mode/'reloaded.json').read_text())
    assert trained['steps'] == 3
    assert min(trained['training_frames']) > 5
    assert trained['full_utterances'] and trained['real_text']
    assert trained['selected_before'] != trained['selected_after']
    assert trained['frozen_before'] == trained['frozen_after']
    assert reloaded['parameters_sha256'] == trained['parameters_sha256']
    assert reloaded['comparison']['passed']
    assert (root/mode/'reloaded/response.wav').is_file()


def test_fresh_original_base_restores_head_projectors_and_session_output():
    root = run('restore')
    restored = json.loads((root/'restored.json').read_text())
    assert restored['all_base_parameters_match'] and restored['comparison']['passed']
    assert restored['has_lora'] is False
