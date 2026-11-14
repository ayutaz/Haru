from copy import deepcopy

import pytest
import torch

from haru_training.checkpoint import read_checkpoint,write_checkpoint
from haru_training.codec_adapter import attach_adapter
from test_training_forward import tiny_model


def base():
    torch.set_num_threads(1);torch.manual_seed(13)
    return tiny_model().eval()


def artifact(tmp_path):
    from haru_training.codec_adapter_inference import export_adapter
    model=base();adapter=attach_adapter(model,rank=4)
    with torch.no_grad():adapter.up.weight.fill_(.01);adapter.up.bias.fill_(.02)
    path=export_adapter(model,tmp_path/'adapter',base_model='test/tiny-haru',base_revision='a'*40,
                        target_speaker='target',dataset_sha256='b'*64)
    return model,path


def test_artifact_contains_adapter_only_and_reproduces_waveform(tmp_path):
    from haru_training.codec_adapter_inference import apply_adapter
    model,path=artifact(tmp_path);bundle=read_checkpoint(path)
    assert bundle['kind']=='haru-codec-adapter'
    assert set(bundle['weights'])=={'down.weight','down.bias','up.weight','up.bias'}
    assert 'optimizer' not in bundle
    target=base()
    apply_adapter(target,path,base_model='test/tiny-haru',base_revision='a'*40)
    z=torch.ones(1,3,64)
    assert torch.equal(model.codec.decode(z),target.codec.decode(z))


@pytest.mark.parametrize('mutation',['kind','revision','config','shape','nonfinite','fixed_codec','dtype'])
def test_invalid_artifact_does_not_partially_modify_model(tmp_path,mutation):
    from haru_training.codec_adapter_inference import apply_adapter
    _,path=artifact(tmp_path);bundle=read_checkpoint(path);target=base()
    if mutation=='kind':bundle['kind']='haru-inference'
    elif mutation=='revision':bundle['spec']['base_revision']='c'*40
    elif mutation=='config':bundle['spec']['codec_config']['base_channels']=8
    elif mutation=='shape':bundle['weights']['up.weight']=torch.zeros(1,1,1)
    elif mutation=='nonfinite':bundle['weights']['up.bias'][0]=float('nan')
    elif mutation=='dtype':bundle['spec']['dtype']='bfloat16'
    else:
        with torch.no_grad():target.codec.decoder.head.bias.add_(.1)
    changed=write_checkpoint(tmp_path/'changed',bundle)
    stem=target.codec.decoder.stem;flags=[p.requires_grad for p in target.parameters()]
    with pytest.raises(ValueError):apply_adapter(target,changed,base_model='test/tiny-haru',base_revision='a'*40)
    assert target.codec.decoder.stem is stem
    assert [p.requires_grad for p in target.parameters()]==flags


def test_loader_reads_kind_before_download_and_loads_pinned_fresh_base(tmp_path,monkeypatch):
    from haru.models.haru import HaruModel
    from haru_training.codec_adapter_inference import load_adapter
    _,path=artifact(tmp_path)
    calls=[]
    def load(repo,**kwargs):calls.append((repo,kwargs));return base()
    monkeypatch.setattr(HaruModel,'from_pretrained',load)
    model=load_adapter(path,device='cpu')
    assert calls==[('test/tiny-haru',dict(revision='a'*40,device='cpu',dtype=torch.float32))]
    assert model.codec._codec_adapter_attached
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    bad=read_checkpoint(path);bad['kind']='haru-inference'
    changed=write_checkpoint(tmp_path/'bad',bad)
    with pytest.raises(ValueError):load_adapter(changed,device='cpu')
    assert len(calls)==1


def test_legacy_head_artifact_marker_cannot_be_combined_with_codec_update():
    model=base();model.inference_feedback_mode='generated_audio'
    with pytest.raises(ValueError,match='feedback'):attach_adapter(model,rank=4)
