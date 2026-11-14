from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from haru import DuplexConfig,DuplexSession
from haru_training.codec_adapter import attach_adapter
from test_training_forward import tiny_model


def test_nonzero_adapter_changes_only_waveform_in_real_multiframe_session():
    torch.set_num_threads(1);torch.manual_seed(13)
    baseline=tiny_model().eval().requires_grad_(False)
    tokenizer=Tokenizer(WordLevel({f'token{i}':i for i in range(128)},unk_token='token3'))
    tokenizer.pre_tokenizer=Whitespace();baseline.tokenizer=tokenizer
    adapted=deepcopy(baseline)
    adapter=attach_adapter(adapted,rank=4)
    with torch.no_grad():adapter.up.weight.fill_(.01);adapter.up.bias.fill_(.02)
    config=DuplexConfig(seed=7,max_frames=4,codec_backend='torch',dry=None)
    first=DuplexSession(baseline,system_prompt='test',config=config)
    second=DuplexSession(adapted,system_prompt='test',config=config)
    user=np.sin(np.arange(3200,dtype=np.float32)*.03)*.1
    changed=[]
    with patch.object(baseline.head,'sample',wraps=baseline.head.sample) as left_head, \
         patch.object(adapted.head,'sample',wraps=adapted.head.sample) as right_head:
        for step in range(4):
            left=first.start() if step==0 else first.step(user)
            right=second.start() if step==0 else second.step(user)
            assert (left.token,left.text)==(right.token,right.text)
            assert torch.equal(first._latent,second._latent)
            assert torch.equal(first._generator.get_state(),second._generator.get_state())
            changed.append(not np.array_equal(left.audio,right.audio))
        assert left_head.call_count==right_head.call_count==4
    assert any(changed)
