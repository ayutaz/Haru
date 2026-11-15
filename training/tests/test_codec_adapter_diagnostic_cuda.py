"""Opt-in proof on a dedicated Vast lease; never runs on local CUDA by default."""
import json
import os
from pathlib import Path

import pytest


@pytest.mark.cuda
def test_real_weight_roundtrip_and_fresh_v2_cuda_loader():
    if not os.environ.get('HARU_CODEC_DIAGNOSTIC_SPEC'):
        pytest.skip('explicit diagnostic specification required')
    from haru_training.codec_adapter_diagnostic import run_roundtrip
    destination=Path(os.environ['HARU_CODEC_DIAGNOSTIC_OUTPUT'])
    report=run_roundtrip(os.environ['HARU_CODEC_DIAGNOSTIC_SPEC'],destination)
    assert report['execution_complete'] and report['training_started'] is False
    assert len(report['records'])==12
    assert {r['representation'] for r in report['records']}=={'mean','sample'}
    assert all(r['samples']>3200 and r['rms']>1e-5 for r in report['records'])
    fresh=json.loads((destination/'fresh-cuda.json').read_text(encoding='utf-8'))
    assert fresh['conversation_trace_exact'] and fresh['adapter_tensors_exact']
    assert fresh['frozen_codec_exact'] and fresh['waveform_changed']


@pytest.mark.cuda
def test_real_weight_three_pair_training_after_independent_codec_screening():
    if not os.environ.get('HARU_CODEC_SMALL_TRAIN_GATE'):
        pytest.skip('independent roundtrip gate and explicit small-training job required')
    from haru_training.codec_adapter_diagnostic import run_small_training
    from haru_training.checkpoint import read_checkpoint
    import torch
    import numpy as np
    import soundfile as sf
    destination=Path(os.environ['HARU_CODEC_SMALL_TRAIN_OUTPUT'])
    report=run_small_training(os.environ['HARU_CODEC_DIAGNOSTIC_SPEC'],destination,
        gate_path=os.environ['HARU_CODEC_SMALL_TRAIN_GATE'],roundtrip_report=os.environ['HARU_CODEC_ROUNDTRIP_REPORT'])
    assert report['execution_complete'] and report['frozen_codec_exact']
    assert 1<=report['training_steps']<=300
    assert len(report['training_ids'])==3
    first=read_checkpoint(destination/'step-0/adapter')
    last=read_checkpoint(destination/report['observations'][-1]['path']/'adapter')
    assert any(not torch.equal(first['weights'][k],last['weights'][k]) for k in first['weights'])
    spec=json.loads(Path(os.environ['HARU_CODEC_DIAGNOSTIC_SPEC']).read_text(encoding='utf-8'))
    for case in spec['cases']:
        folder=destination/'step-0'/case['id']
        assert np.array_equal(sf.read(folder/'baseline/response.wav')[0],sf.read(folder/'candidate/response.wav')[0])
