import torch
import pytest

from haru_training.data import make_sequence


def test_role_prompts_use_official_japanese_card_layout():
    from haru_training.protocol import character_prompt
    target=character_prompt('target')
    dialogue=character_prompt('dialogue')
    assert '【あなたの人物設定】' in target and '【あなたの名前】' in target
    assert 'つくよみちゃん' in target and 'つくよみちゃん' not in dialogue
    assert target!=dialogue
    with pytest.raises(ValueError): character_prompt('unknown')


def test_replacing_system_preserves_all_reference_and_teacher_frames():
    from haru_training.protocol import replace_system
    sequence=make_sequence('train-1',[2,3],torch.ones(3,2),torch.randn(4,2),
                           torch.randn(4,2),torch.tensor([4,5,0,0]),0,1)
    old={key:value.clone() for key,value in vars(sequence).items() if torch.is_tensor(value)}
    result=replace_system(sequence,[6,7,8,9])
    assert result.record_id==sequence.record_id
    assert result.tokens[:4].tolist()==[6,7,8,9]
    for key,value in old.items():
        assert torch.equal(getattr(sequence,key),value)
        assert torch.equal(getattr(result,key)[4:],value[2:]),key
    assert not result.supervise[:4].any() and not result.model_mask[:4].any()
    assert not result.user_mask[:4].any() and (result.frame_indices[:4]==-1).all()
    assert not result.model_latents[:4].any() and not result.user_latents[:4].any()


def test_non_system_prefix_is_not_silently_discarded():
    from haru_training.protocol import replace_system
    sequence=make_sequence('train-1',[2,3],torch.ones(3,2),torch.randn(4,2),
                           torch.randn(4,2),torch.tensor([4,5,0,0]),0,1)
    sequence.supervise[0]=True
    with pytest.raises(ValueError,match='prefix'): replace_system(sequence,[6,7])
