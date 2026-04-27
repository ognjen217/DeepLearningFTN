import numpy as np
import torch
from torch import nn

from stable_icnn_physics.eval import autoregressive_rollout_model, rollout_error, rollout_system
from stable_icnn_physics.systems import MassSpringDamper


class ExactOscillatorModel(nn.Module):
    def __init__(self, system):
        super().__init__()
        self.system = system

    def forward(self, x):
        pos = x[:, 0]
        vel = x[:, 1]
        acc = -(self.system.stiffness / self.system.mass) * pos - (self.system.damping / self.system.mass) * vel
        return torch.stack([vel, acc], dim=1)


def test_autoregressive_rollout_feeds_predictions_back_each_step():
    system = MassSpringDamper(mass=1.0, damping=0.3, stiffness=1.0)
    model = ExactOscillatorModel(system)
    x0 = np.array([[1.0, 0.0], [-0.5, 0.3]], dtype=np.float32)

    true_traj = rollout_system(system, x0, steps=20, dt=0.02)
    pred_traj = autoregressive_rollout_model(model, x0, steps=20, dt=0.02, device="cpu")
    errors = rollout_error(system, true_traj, pred_traj)

    assert pred_traj.shape == (21, 2, 2)
    assert np.isfinite(pred_traj).all()
    assert np.max(errors) < 1e-8

