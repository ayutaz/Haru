"""Opt-in end-to-end real-weight proof on the owned, bounded Vast.ai lease."""
import json
import os
from pathlib import Path

import pytest


@pytest.mark.cuda
def test_real_weight_bounded_codec_trial():
    if not os.environ.get('HARU_CODEC_TRIAL_SPEC'):
        pytest.skip('explicit frozen trial specification required')
    from haru_training.codec_adapter_trial import run_trial
    destination = Path(os.environ['HARU_CODEC_TRIAL_OUTPUT'])
    run_trial(os.environ['HARU_CODEC_TRIAL_SPEC'], destination)
    preflight = json.loads((destination/'preflight.json').read_text())
    for key in ('zero_identity','nonzero_wave_change','conversation_trace_exact','frozen_codec_exact','real_update'):
        assert preflight[key] is True
    assert preflight['trainable_parameters'] == 34824
    training = json.loads((destination/'training.json').read_text())
    assert 1 <= training['steps'] <= 30 and training['adapter_reload_exact']
    spec = json.loads(Path(os.environ['HARU_CODEC_TRIAL_SPEC']).read_text())
    for case in spec['cases']:
        record = json.loads((destination/(case['id']+'.json')).read_text())
        assert record['conversation_trace_exact'] and record['frames'] > 4
        assert (destination/case['id']/'comparison.json').is_file()
