import pytest
import torch
from haru_training.losses import meanflow_loss, masked_text_loss, sample_flow_times


class AffineField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def forward(self, z, condition, t, h, guidance):
        return self.a * z + 3 * t[:, None] + 5 * h[:, None] + condition


def arguments():
    return [torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64),
            torch.tensor([.25], dtype=torch.float64), torch.tensor([.5], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)]


def test_improved_meanflow_matches_analytic_field_and_stops_jvp_gradient():
    field = AffineField()
    result = meanflow_loss(field, *arguments())
    # z=.25; u=3.75; v=u(h=0)=1.25; D=a*v+3-5=.5; V=u-h*D=3.5
    assert result.prediction.item() == pytest.approx(3.5)
    assert result.target.item() == 1
    assert result.loss.item() == pytest.approx(6.25)
    result.loss.backward()
    # JVP and target are detached: dL/da=2*(3.5-1)*z=1.25
    assert field.a.grad.item() == pytest.approx(1.25)


def test_guidance_target_and_condition_dropout_have_explicit_semantics():
    args = arguments()
    args[4] = torch.tensor([2.], dtype=torch.float64)
    args[5] = torch.tensor([[4.]], dtype=torch.float64)
    result = meanflow_loss(AffineField(), *args)
    assert result.target.item() == pytest.approx(3.)  # 1 + (1 - 1/2) * (vc-vu=4)
    dropped = meanflow_loss(AffineField(), *args, drop_mask=torch.tensor([True]))
    assert dropped.target.item() == 1
    torch.testing.assert_close(dropped.loss, meanflow_loss(AffineField(), *arguments()).loss)


def test_zero_interval_reduces_to_instantaneous_velocity_regression():
    args = arguments()
    args[3].zero_()
    result = meanflow_loss(AffineField(), *args)
    assert result.prediction.item() == pytest.approx(1.25)
    assert result.loss.item() == pytest.approx(.25 ** 2)


def test_flow_times_include_boundary_and_one_step_intervals_and_are_seeded():
    a = sample_flow_times(8, torch.Generator().manual_seed(2), 'cpu')
    b = sample_flow_times(8, torch.Generator().manual_seed(2), 'cpu')
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    t, h = a
    assert (h[:4] == 0).all() and (h[4:] > 0).all()
    assert (t >= 0).all() and (t + h <= 1).all()


@pytest.mark.parametrize('index,value', [(0, float('nan')), (2, -1.), (3, 1.), (4, 3.)])
def test_bad_flow_inputs_fail_before_backward(index, value):
    args = arguments()
    args[index].fill_(value)
    with pytest.raises((ValueError, FloatingPointError)):
        meanflow_loss(AffineField(), *args)


def test_masked_cross_entropy_has_known_value_and_ignores_prompt_logits():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    targets = torch.tensor([[-100, 1, -100]])
    assert masked_text_loss(logits, targets).item() == pytest.approx(.69314718)
    altered = logits.detach().clone()
    altered[:, 0] = float('nan')
    altered[:, 2] = 100
    torch.testing.assert_close(masked_text_loss(logits, targets), masked_text_loss(altered, targets))
    improved = logits.detach().clone()
    improved[0, 1, 1] = 5
    assert masked_text_loss(improved, targets) < masked_text_loss(logits, targets)
    masked_text_loss(logits, targets).backward()
    assert logits.grad[:, 0].abs().sum() == 0


def test_nonfinite_supervised_logits_are_not_silently_skipped():
    with pytest.raises(FloatingPointError):
        masked_text_loss(torch.full((1, 1, 2), float('nan')), torch.tensor([[0]]))
