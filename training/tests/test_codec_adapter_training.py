from copy import deepcopy

import pytest
import torch
from torch import nn

from test_codec_adapter_data import pair
from test_codec_adapter_checkpoint import base


class Content(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 5, 3, padding=1)

    def forward(self, wave):
        return torch.nn.functional.adaptive_avg_pool1d(self.conv(wave[None, None]), 20)[0].T


def trainer():
    from haru_training.codec_adapter_training import CodecTrainer, TrainSpec
    model = base()
    torch.manual_seed(9)
    content = Content()
    return CodecTrainer(model, [pair()], TrainSpec(base_model='test/tiny-haru',
        base_revision='c'*40, target_speaker='target', max_steps=3,
        learning_rate=.01, content_weight=.1, transition_strength=2), content=content, rank=4)


def test_transition_weights_uniform_control_and_masked_teacher_prefix():
    from haru_training.codec_adapter_training import transition_weights, paired_loss
    p = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    assert torch.equal(transition_weights(p, 0), torch.ones(4))
    w = transition_weights(p, 2)
    assert w[2] > w[0] and w.mean() == 1
    item = pair(); changed = deepcopy(item)
    changed.spectrum[:20].fill_(30)
    changed.posteriors[:7] = torch.tensor([1., 0., 0., 0., 0.])
    wave = torch.randn(7000, requires_grad=True)
    content = Content().eval().requires_grad_(False)
    original = paired_loss(wave, item, content=content, content_weight=.1, transition_strength=2)
    other = paired_loss(wave, changed, content=content, content_weight=.1, transition_strength=2)
    torch.testing.assert_close(original['loss'], other['loss'], atol=0, rtol=0)
    original['loss'].backward()
    assert wave.grad[:6400].abs().sum() > 0
    assert wave.grad[6400:].count_nonzero() == 0
    assert all(p.grad is None for p in content.parameters())


def test_three_actual_updates_freeze_base_and_resume_exactly(tmp_path):
    from haru_training.codec_adapter import get_adapter
    full = trainer()
    before = {n: p.clone() for n,p in full.model.named_parameters()}
    full.fit()
    changed = [n for n,p in full.model.named_parameters() if not torch.equal(before[n],p)]
    assert changed and all('stem.adapter.' in n for n in changed)
    assert all(p.grad is None for p in full.content.parameters())
    first = trainer(); first.fit(until_step=1); first.save(tmp_path/'step1')
    resumed = trainer(); resumed.restore(tmp_path/'step1'); resumed.fit()
    for key, value in get_adapter(full.model.codec).state_dict().items():
        assert torch.equal(value, get_adapter(resumed.model.codec).state_dict()[key])
    assert full.order == resumed.order and full.cursor == resumed.cursor and full.step == 3
    assert torch.equal(full.generator.get_state(), resumed.generator.get_state())
    for left,right in zip(full.optimizer.state.values(),resumed.optimizer.state.values()):
        for key in left:
            assert torch.equal(left[key],right[key])


def test_nonfinite_loss_is_not_updated_or_saved(tmp_path):
    from haru_training.codec_adapter import get_adapter
    item = trainer()
    before = deepcopy(get_adapter(item.model.codec).state_dict())
    item.content.forward = lambda wave: wave.new_full((20,5), float('nan'))
    with pytest.raises(ValueError, match='finite'): item.fit()
    assert item.step == 0
    for key,value in before.items(): assert torch.equal(value,get_adapter(item.model.codec).state_dict()[key])
    with pytest.raises(ValueError): item.save(tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_resume_rejects_changed_data_or_spec_before_mutation(tmp_path):
    from haru_training.codec_adapter import get_adapter
    from haru_training.checkpoint import read_checkpoint,write_checkpoint
    original=trainer();original.fit(until_step=1);original.save(tmp_path/'checkpoint')
    for mutation in ('data_sha256','spec','optimizer','rng'):
        bundle=read_checkpoint(tmp_path/'checkpoint')
        if mutation=='data_sha256':bundle[mutation]='0'*64
        elif mutation=='spec':bundle[mutation]['learning_rate']=.9
        elif mutation=='optimizer':bundle[mutation]['param_groups'][0]['params'] = []
        else:bundle['rng']=torch.zeros(2,dtype=torch.uint8)
        path=write_checkpoint(tmp_path/mutation,bundle)
        target=trainer();before=deepcopy(get_adapter(target.model.codec).state_dict())
        with pytest.raises(ValueError):target.restore(path)
        assert target.step == 0
        for key,value in before.items():assert torch.equal(value,get_adapter(target.model.codec).state_dict()[key])
