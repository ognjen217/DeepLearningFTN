from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .swe2d import SWE2DConfig, compute_energy, compute_mass, compute_rmse


@dataclass
class RolloutSummary:
    final_rmse: float
    mean_rmse: float
    max_rmse: float
    true_energy_start: float
    true_energy_end: float
    pred_energy_start: float
    pred_energy_end: float
    true_mass_start: float
    true_mass_end: float
    pred_mass_start: float
    pred_mass_end: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def rollout_torch_dynamics(
    dynamics: torch.nn.Module,
    state0: np.ndarray,
    steps: int,
    dt: float,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Autoregressively roll out a derivative model using Euler updates.

    The learned model is trained as ``x_next = x + dt*f(x)``, so this evaluation
    uses the same update rule. Later we can add RK4 for learned continuous-time
    models, but Euler is the fairest first comparison to the training objective.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    was_training = dynamics.training
    dynamics.eval().to(device)

    x_np = np.asarray(state0, dtype=np.float32)
    if x_np.ndim == 3:
        x_np = x_np[None, ...]
    if x_np.ndim != 4 or x_np.shape[1] != 3:
        raise ValueError("state0 must have shape (3, ny, nx) or (batch, 3, ny, nx)")

    traj = np.zeros((steps + 1, *x_np.shape), dtype=np.float32)
    traj[0] = x_np
    x = torch.from_numpy(x_np).to(device)

    for i in range(steps):
        with torch.enable_grad():
            x = (x + float(dt) * dynamics(x)).detach()
        traj[i + 1] = x.cpu().numpy()

    dynamics.train(was_training)
    return traj[:, 0] if state0.ndim == 3 else traj


def summarize_rollout(traj_true: np.ndarray, traj_pred: np.ndarray, cfg: SWE2DConfig) -> RolloutSummary:
    """Compute rollout and conservation diagnostics."""
    rmse = compute_rmse(traj_true, traj_pred)
    true_energy = np.array([compute_energy(s, cfg) for s in traj_true])
    pred_energy = np.array([compute_energy(s, cfg) for s in traj_pred])
    true_mass = np.array([compute_mass(s, cfg) for s in traj_true])
    pred_mass = np.array([compute_mass(s, cfg) for s in traj_pred])

    return RolloutSummary(
        final_rmse=float(rmse[-1]),
        mean_rmse=float(rmse.mean()),
        max_rmse=float(rmse.max()),
        true_energy_start=float(true_energy[0]),
        true_energy_end=float(true_energy[-1]),
        pred_energy_start=float(pred_energy[0]),
        pred_energy_end=float(pred_energy[-1]),
        true_mass_start=float(true_mass[0]),
        true_mass_end=float(true_mass[-1]),
        pred_mass_start=float(pred_mass[0]),
        pred_mass_end=float(pred_mass[-1]),
    )


def energy_projection_stats(
    model: torch.nn.Module,
    trajectory: np.ndarray,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compute projection diagnostics for EnergyProjectedDynamics-like models."""
    if not hasattr(model, "projection_diagnostics"):
        return {}

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    fire_rates = []
    correction_norms = []
    violations = []

    for state in trajectory[:-1]:
        x = torch.from_numpy(np.asarray(state[None], dtype=np.float32)).to(device)
        with torch.enable_grad():
            diag = model.projection_diagnostics(x)
        fire_rates.append(float(diag["fire"].float().mean().cpu()))
        correction_norms.append(float(diag["correction_norm"].mean().cpu()))
        violations.append(float(diag["violation"].mean().cpu()))

    return {
        "projection_fire_rate": float(np.mean(fire_rates)),
        "correction_norm_mean": float(np.mean(correction_norms)),
        "correction_norm_max": float(np.max(correction_norms)),
        "nominal_violation_mean": float(np.mean(violations)),
        "nominal_violation_max": float(np.max(violations)),
    }
