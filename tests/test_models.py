import torch

from stable_icnn_physics.models import ICNN, PositiveDefiniteLyapunov, ReHU, build_stable_model


def test_positive_definite_lyapunov_zero_and_nonzero_values():
    torch.manual_seed(0)
    V = PositiveDefiniteLyapunov(ICNN([2, 8, 8, 1], activation=ReHU(0.05)), eps=0.01)
    zero = torch.zeros(1, 2)
    points = torch.randn(16, 2)
    assert torch.allclose(V(zero), torch.zeros(1, 1), atol=1e-6)
    assert torch.all(V(points) > 0)


def test_stable_dynamics_satisfies_lyapunov_decrease_condition():
    torch.manual_seed(0)
    model = build_stable_model(dim=2, hidden=12, lyapunov_hidden=10, alpha=1e-3)
    x = torch.randn(32, 2)
    decrease = model.lyapunov_decrease(x)
    assert torch.max(decrease).item() <= 1e-5

