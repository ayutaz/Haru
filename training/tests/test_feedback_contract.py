"""Voice artifacts must not silently fall back to the coupled feedback path."""
from dataclasses import replace

import pytest
import torch

from haru import DuplexConfig,DuplexSession
from haru_training.checkpoint import read_checkpoint,write_checkpoint
from haru_training.inference import apply_inference,export_inference,load_inference,load_with_frozen_feedback
from test_inference import base,configured
from test_phase4_haru import tiny_trainer,tensor_digest


@pytest.mark.parametrize('changes',[
    {'feedback_mode':'unknown'},
    {'feedback_mode':True},
    {'feedback_mode':'separate_frozen_head','mode':'lm_lora'},
    {'feedback_mode':'separate_frozen_head','train_projectors':True},
])
def test_feedback_contract_rejects_invalid_or_nonpreserving_training_modes(tmp_path,changes):
    trainer=tiny_trainer(tmp_path/'train','head')
    with pytest.raises(ValueError,match='feedback'):
        replace(trainer.spec,**changes)


def test_feedback_contract_prevents_wrong_loader_and_missing_runtime_head(tmp_path,monkeypatch):
    from haru.models.haru import HaruModel
    trainer=tiny_trainer(tmp_path/'train','head_output')
    trainer.spec=replace(trainer.spec,feedback_mode='separate_frozen_head')
    path=export_inference(trainer,tmp_path/'artifact')
    assert read_checkpoint(path)['spec']['feedback_mode']=='separate_frozen_head'
    def forbidden(*args,**kwargs):raise AssertionError('wrong loader must reject before model download')
    monkeypatch.setattr(HaruModel,'from_pretrained',forbidden)
    with pytest.raises(ValueError,match='feedback'):load_inference(path,device='cpu',dtype=torch.float32)
    target=configured('head_output');before=tensor_digest(target.named_parameters())
    with pytest.raises(ValueError,match='feedback'):
        apply_inference(target,path,base_model='test/tiny-haru',base_revision='a'*40)
    assert tensor_digest(target.named_parameters())==before
    monkeypatch.setattr(HaruModel,'from_pretrained',lambda *a,**k:base())
    model,feedback=load_with_frozen_feedback(path,device='cpu',dtype=torch.float32)
    assert model.inference_feedback_mode=='separate_frozen_head'
    with pytest.raises(ValueError,match='feedback'):
        DuplexSession(model,config=DuplexConfig(seed=7))
    session=DuplexSession(model,config=DuplexConfig(seed=7),feedback_head=feedback)
    assert session.feedback_head is feedback


def test_legacy_feedback_artifact_defaults_to_original_coupled_inference(tmp_path,monkeypatch):
    from haru.models.haru import HaruModel
    trainer=tiny_trainer(tmp_path/'train','head')
    path=export_inference(trainer,tmp_path/'original')
    bundle=read_checkpoint(path);bundle['spec'].pop('feedback_mode',None)
    legacy=write_checkpoint(tmp_path/'legacy',bundle)
    monkeypatch.setattr(HaruModel,'from_pretrained',lambda *a,**k:base())
    model=load_inference(legacy,device='cpu',dtype=torch.float32)
    assert model.inference_feedback_mode=='generated_audio'
