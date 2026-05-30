from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from stable_icnn_physics.eval import autoregressive_rollout_model
from stable_icnn_physics.rollout_augmented import projection_diagnostics

from .system import ThreeBodySystem3D, integrate_rk4_fixed, integrate_solve_ivp


@dataclass
class RolloutMetrics:
    final_rmse: float
    mean_rmse: float
    max_rmse: float
    energy_drift_final: float
    energy_drift_max_abs: float
    momentum_drift_final: float
    momentum_drift_max: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def rmse_per_step(system: ThreeBodySystem3D, true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    return np.sqrt(system.state_error(true_traj, pred_traj)).astype(np.float32)


def summarize_rollout(system: ThreeBodySystem3D, true_traj: np.ndarray, pred_traj: np.ndarray) -> RolloutMetrics:
    rmse = rmse_per_step(system, true_traj, pred_traj)
    energy_true = system.energy(true_traj.reshape(-1, 18)).reshape(true_traj.shape[:-1])
    energy_pred = system.energy(pred_traj.reshape(-1, 18)).reshape(pred_traj.shape[:-1])
    energy_drift = energy_pred - energy_true

    momentum_true = system.momentum(true_traj.reshape(-1, 18)).reshape(*true_traj.shape[:-1], 3)
    momentum_pred = system.momentum(pred_traj.reshape(-1, 18)).reshape(*pred_traj.shape[:-1], 3)
    momentum_drift = np.linalg.norm(momentum_pred - momentum_true, axis=-1)

    return RolloutMetrics(
        final_rmse=float(rmse[-1].mean()),
        mean_rmse=float(rmse.mean()),
        max_rmse=float(rmse.max()),
        energy_drift_final=float(np.mean(energy_drift[-1])),
        energy_drift_max_abs=float(np.max(np.abs(energy_drift))),
        momentum_drift_final=float(np.mean(momentum_drift[-1])),
        momentum_drift_max=float(np.max(momentum_drift)),
    )


def rollout_nn(model: torch.nn.Module, x0: np.ndarray, steps: int, dt: float, device=None) -> np.ndarray:
    return autoregressive_rollout_model(model, x0, steps=steps, dt=dt, device=device, wrap_fn=None)


def benchmark_n_step(
    system: ThreeBodySystem3D,
    x0: np.ndarray,
    steps: int,
    dt: float,
    models: dict[str, torch.nn.Module] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """Compare wall-clock time for solver and neural N-step rollouts."""
    results: dict[str, dict[str, float]] = {}

    def time_call(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - start
        return out, elapsed

    t_eval = np.arange(steps + 1, dtype=np.float32) * float(dt)
    solve_ivp_traj, elapsed = time_call(lambda: integrate_solve_ivp(system, x0, t_eval))
    results["solve_ivp_dop853"] = {"seconds": elapsed, "steps_per_second": steps / elapsed}

    rk4_traj, elapsed = time_call(lambda: integrate_rk4_fixed(system, x0, steps=steps, dt=dt))
    rk4_rmse = float(np.sqrt(system.state_error(solve_ivp_traj, rk4_traj)).mean())
    results["rk4_fixed"] = {"seconds": elapsed, "steps_per_second": steps / elapsed, "rmse_vs_solve_ivp": rk4_rmse}

    if models:
        for name, model in models.items():
            pred, elapsed = time_call(lambda m=model: rollout_nn(m, x0[None], steps=steps, dt=dt, device=device)[:, 0])
            rmse = float(np.sqrt(system.state_error(solve_ivp_traj, pred)).mean())
            results[name] = {
                "seconds": elapsed,
                "steps_per_second": steps / elapsed,
                "rmse_vs_solve_ivp": rmse,
                "speedup_vs_solve_ivp": results["solve_ivp_dop853"]["seconds"] / elapsed,
                "speedup_vs_rk4": results["rk4_fixed"]["seconds"] / elapsed,
            }
            if hasattr(model, "fhat"):
                diag = projection_diagnostics(model, pred[:, None], device=device)
                results[name]["projection_fire_rate"] = diag["projection_fire_rate"]
                results[name]["correction_norm_mean"] = diag["correction_norm_mean"]

    return results
