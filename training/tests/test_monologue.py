from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from haru_training.corpus import Clip
from haru_training.data import write_cache
from haru_training.monologue import monologue_sequence


class Codec:
    device = torch.device('cpu')
    dtype = torch.float32
    config = SimpleNamespace(frame_samples=3200, latent_dim=2)
    def encode(self, audio):
        return torch.ones(audio.shape[0], audio.shape[-1]//3200, 2) * 5
    def sample(self, means, generator):
        return means


class Tokenizer:
    def __init__(self, tokens): self.tokens = tokens
    def encode(self, text, **kwargs): return SimpleNamespace(ids=self.tokens)


def assets(tmp_path):
    rows = {}
    for name, frames, value in [('target', 4, 2), ('reference', 3, 99)]:
        key = {'latent_dim':2, 'source_sha256':name, 'codec_revision':'fixed'}
        path = tmp_path/f'{name}.safetensors'
        write_cache(path, torch.full((1, frames, 2), float(value)), key)
        rows[name] = {'id':name,'cache':path.name,'key':key}
    key = {'latent_dim':2, 'source_sha256':'reference', 'codec_revision':'fixed', 'role':'reference'}
    write_cache(tmp_path/'reference.prepared.safetensors',torch.full((1,3,2),7.),key)
    rows['reference'].update(reference_cache='reference.prepared.safetensors',reference_key=key)
    clip = Clip('target', Path('unused.wav'), 'テスト', 24000, .5, 'target', .2, .1, 0)
    return clip,rows


def test_real_text_and_cached_voice_form_traceable_monologue_without_zero_latent_silence(tmp_path):
    clip,rows=assets(tmp_path)
    result=monologue_sequence(Codec(),Tokenizer([4,5]),clip,rows,tmp_path,'reference',[8],0,1)
    assert result.record_id == 'target'
    assert result.tokens.tolist()==[8,0,0,1,4,5,0,0]
    torch.testing.assert_close(result.user_latents[-4:],torch.full((4,2),5.))
    torch.testing.assert_close(result.model_latents[-4:],torch.full((4,2),2.))
    torch.testing.assert_close(result.model_latents[1:4],torch.full((3,2),7.))
    assert result.supervise.tolist()==[False]*4+[True]*4


def test_overlong_text_and_reference_leakage_are_not_silently_truncated(tmp_path):
    clip,rows=assets(tmp_path)
    with pytest.raises(ValueError):
        monologue_sequence(Codec(),Tokenizer([4]*5),clip,rows,tmp_path,'reference',[8],0,1)
    with pytest.raises(ValueError):
        monologue_sequence(Codec(),Tokenizer([4]),clip,rows,tmp_path,'target',[8],0,1)


@pytest.mark.parametrize('field', ['codec_revision','source_sha256'])
def test_reference_cache_cannot_disagree_with_its_source_and_target_codec(tmp_path, field):
    clip,rows=assets(tmp_path)
    key=rows['reference']['reference_key']
    key[field]='other'
    write_cache(tmp_path/rows['reference']['reference_cache'],torch.full((1,3,2),7.),key)
    with pytest.raises(ValueError):
        monologue_sequence(Codec(),Tokenizer([4]),clip,rows,tmp_path,'reference',[8],0,1)
