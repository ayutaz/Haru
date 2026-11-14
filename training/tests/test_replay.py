from dataclasses import asdict
import json

import numpy as np
import pytest
import soundfile as sf
import torch
from tokenizers import Tokenizer, models

from haru import DuplexConfig
from haru_training.replay import replay, save_replay
from test_inference import base


def session_model():
    torch.set_num_threads(1)
    model = base().eval().requires_grad_(False)
    model.tokenizer = Tokenizer(models.WordLevel({f't{i}': i for i in range(128)}, unk_token='t3'))
    return model


def test_real_tiny_duplex_session_produces_seeded_frames_and_preserves_input(tmp_path):
    model = session_model()
    user = np.zeros(6401, np.float32)
    config = DuplexConfig(seed=7, speak_first=True, codec_backend='torch')
    config_before = asdict(config)
    a = replay(model, user, config=config)
    b = replay(model, user, config=config)
    assert asdict(config) == config_before
    assert len(a['frames']) == 3 and len(a['audio']) == 9600
    assert a['audio'].dtype == np.float32 and np.isfinite(a['audio']).all()
    np.testing.assert_array_equal(a['audio'], b['audio'])
    assert [f['token'] for f in a['frames']] == [f['token'] for f in b['frames']]
    assert [f['index'] for f in a['frames']] == [0, 1, 2]
    assert all(f['seconds'] >= 0 for f in a['frames'])
    assert a['input_samples'] == 6401 and a['padded_samples'] == 3199
    assert np.count_nonzero(user) == 0
    assert a['text'] == ''.join(f['text'] for f in a['frames']).strip()
    path = save_replay(tmp_path/'run', a, provenance={'input_sha256': 'a'*64})
    audio, rate = sf.read(path/'response.wav', dtype='float32')
    np.testing.assert_array_equal(audio, a['audio'])
    assert rate == 24000
    meta = json.loads((path/'response.json').read_text())
    assert meta['provenance']['input_sha256'] == 'a'*64
    assert meta['frames'] == a['frames'] and 'audio' not in meta
    assert 'response.wav' in json.loads((path/'manifest.json').read_text())['files']
    with pytest.raises(FileExistsError): save_replay(path, a)


@pytest.mark.parametrize('user', [np.array([], np.float32), np.zeros((3, 2), np.float32),
                                  np.array([float('nan')]), np.array([float('inf')])])
def test_invalid_replay_inputs_are_rejected(user):
    with pytest.raises(ValueError): replay(session_model(), user, config=DuplexConfig(seed=7))


def test_replay_requires_explicit_seed_and_eval_model():
    with pytest.raises(ValueError): replay(session_model(), np.zeros(3200), config=DuplexConfig())
    model = session_model().train()
    with pytest.raises(ValueError): replay(model, np.zeros(3200), config=DuplexConfig(seed=7))


def test_frozen_feedback_preserves_internal_stream_and_tokens_while_output_voice_changes():
    from copy import deepcopy
    from unittest.mock import patch
    model=session_model()
    feedback=deepcopy(model.head).eval().requires_grad_(False)
    user=np.zeros(8*3200,np.float32)
    config=DuplexConfig(seed=23,speak_first=True,codec_backend='torch')
    def capture(**kwargs):
        streams=[]
        original=model.embed
        def embed(*args,**named):
            if len(args)==3:
                streams.append(args[1].detach().clone())
            return original(*args,**named)
        with patch.object(model,'embed',side_effect=embed):
            result=replay(model,user,config=config,**kwargs)
        return result,streams
    baseline,original=capture()
    with torch.no_grad(): model.head.final.proj.weight.mul_(2)
    adapted,changed=capture()
    separated,retained=capture(feedback_head=feedback)
    assert len(original)==len(retained)==7
    for a,b in zip(original,retained): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert any(not torch.equal(a,b) for a,b in zip(original,changed))
    assert [f['token'] for f in separated['frames']]==[f['token'] for f in baseline['frames']]
    assert not np.array_equal(separated['audio'],baseline['audio'])
    assert separated['feedback_mode']=='separate_frozen_head'
    assert baseline['feedback_mode']==adapted['feedback_mode']=='generated_audio'


@pytest.mark.parametrize('invalid',['train','grad','shape'])
def test_replay_rejects_invalid_feedback_before_any_audio(invalid):
    from copy import deepcopy
    from dataclasses import replace
    model=session_model();feedback=deepcopy(model.head)
    if invalid=='train': feedback.train()
    elif invalid=='grad': next(feedback.parameters()).requires_grad_(True)
    else: feedback.config=replace(feedback.config,latent_dim=32)
    with pytest.raises(ValueError,match='feedback'):
        replay(model,np.zeros(3200),config=DuplexConfig(seed=7),feedback_head=feedback)
