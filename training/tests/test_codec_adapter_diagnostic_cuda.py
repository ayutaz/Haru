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
