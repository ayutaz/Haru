import pytest
import torch

from haru_training.inference import export_inference,load_with_frozen_feedback
from test_inference import base
from test_phase4_haru import tiny_trainer,tensor_digest


@pytest.mark.parametrize('mode',['head','head_output'])
@pytest.mark.parametrize('strength',[0.,.25,1.])
def test_voice_strength_preserves_base_feedback_and_artifact(tmp_path,monkeypatch,mode,strength):
    from haru import HaruModel
    trainer=tiny_trainer(tmp_path/'train',mode);trainer.fit()
    path=export_inference(trainer,tmp_path/'artifact')
    original=base();original_head=dict(original.head.named_parameters())
    trained_head=dict(trainer.model.head.named_parameters())
    artifact_before=(path/'manifest.json').read_bytes()
    monkeypatch.setattr(HaruModel,'from_pretrained',lambda *a,**k:base())
    model,feedback=load_with_frozen_feedback(path,device='cpu',dtype=torch.float32,
                                          adaptation_strength=strength)
    assert tensor_digest(feedback.named_parameters())==tensor_digest(original.head.named_parameters())
    assert tensor_digest(model.lm.named_parameters())==tensor_digest(original.lm.named_parameters())
    for name,value in model.head.named_parameters():
        expected=torch.lerp(original_head[name],trained_head[name],strength)
        torch.testing.assert_close(value,expected,atol=1e-7,rtol=1e-6)
    assert (path/'manifest.json').read_bytes()==artifact_before
    assert model.voice_adaptation_strength==strength
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('strength',[-.1,1.1,float('nan'),float('inf'),True,'0.25'])
def test_invalid_strength_fails_before_loading_any_artifact(monkeypatch,strength):
    def forbidden(*a,**k):raise AssertionError('must validate strength first')
    monkeypatch.setattr('haru_training.inference._read',forbidden)
    with pytest.raises(ValueError,match='strength'):
        load_with_frozen_feedback('unused',device='cpu',adaptation_strength=strength)
