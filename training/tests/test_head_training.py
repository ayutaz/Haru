import pytest
import torch
from haru.config import HeadConfig
from haru.models.meanflow_head import MeanFlowHead


def head():
    return MeanFlowHead(HeadConfig(hidden_size=8, latent_dim=2, num_layers=1, ffn_ratio=2))


def test_tensor_time_preserves_scalar_inference_and_has_derivatives():
    torch.manual_seed(3)
    model = head()
    z, cond = torch.randn(2, 2), torch.randn(2, 8)
    guidance = torch.tensor([1., 2.])
    expected = model.velocity(z, cond, .25, .5, guidance)
    t = torch.tensor([.25, .25], requires_grad=True)
    h = torch.tensor([.5, .5], requires_grad=True)
    actual = model.velocity(z, cond, t, h, guidance)
    torch.testing.assert_close(actual, expected)
    gt, gh = torch.autograd.grad(actual.sum(), (t, h))
    assert torch.isfinite(gt).all() and gt.abs().sum() > 0
    assert torch.isfinite(gh).all() and gh.abs().sum() > 0


def test_time_has_one_value_per_row():
    with pytest.raises(ValueError):
        head().velocity(torch.ones(2, 2), torch.ones(2, 8), torch.ones(3), .1, torch.ones(2))
