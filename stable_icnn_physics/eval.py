from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .systems import PhysicalSystem


def rk4_step_numpy(rhs, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout_system(system: PhysicalSystem, x0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    x = np.asarray(x0, dtype=np.float32)
    traj = np.zeros((steps + 1, *x.shape), dtype=np.float32)
    traj[0] = x
    for i in range(steps):
        x = rk4_step_numpy(system.rhs, x, dt)
        x = system.wrap_state(x)
        traj[i + 1] = x
    return traj


def rollout_model(
    model: nn.Module,
    x0: np.ndarray,
    steps: int,
    dt: float,
    device: str | torch.device | None = None,
    wrap_fn=None,
) -> np.ndarray:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    was_training = model.training
    model.eval()
    model.to(device)

    x_np = np.asarray(x0, dtype=np.float32)
    traj = np.zeros((steps + 1, *x_np.shape), dtype=np.float32)
    traj[0] = x_np
    x = torch.from_numpy(x_np).to(device)

    for i in range(steps):
        x = _rk4_step_torch(model, x, dt).detach()
        x_np = x.cpu().numpy()
        if wrap_fn is not None:
            x_np = wrap_fn(x_np)
            x = torch.from_numpy(x_np.astype(np.float32)).to(device)
        traj[i + 1] = x_np

    model.train(was_training)
    return traj


def rollout_error(system: PhysicalSystem, true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    return system.state_error(true_traj, pred_traj)


def lyapunov_decrease_values(model: nn.Module, x: np.ndarray, device: str | torch.device | None = None) -> np.ndarray:
    if not hasattr(model, "lyapunov_decrease"):
        raise TypeError("model does not expose lyapunov_decrease")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    values = model.lyapunov_decrease(xt)
    return values.detach().cpu().numpy()


def _rk4_step_torch(model: nn.Module, x: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = model(x)
    k2 = model(x + 0.5 * dt * k1)
    k3 = model(x + 0.5 * dt * k2)
    k4 = model(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

