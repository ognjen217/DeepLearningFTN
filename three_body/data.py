from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .system import ThreeBodySystem3D, integrate_rk4_fixed, integrate_solve_ivp

SolverName = Literal["solve_ivp", "rk4"]


def generate_trajectory_dataset(
    system: ThreeBodySystem3D,
    n_trajs: int,
    steps: int,
    dt: float,
    split: str = "train",
    seed: int = 0,
    solver: SolverName = "rk4",
    solve_ivp_rtol: float = 1e-10,
    solve_ivp_atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate perturbed three-body trajectories."""
    initial_states = system.sample_initial_conditions(n_trajs, split=split, seed=seed)
    time_grid = np.arange(steps + 1, dtype=np.float32) * float(dt)
    trajectories = np.zeros((n_trajs, steps + 1, system.state_dim), dtype=np.float32)

    for i in range(n_trajs):
        if solver == "solve_ivp":
            trajectories[i] = integrate_solve_ivp(
                system,
                initial_states[i],
                time_grid,
                method="DOP853",
                rtol=solve_ivp_rtol,
                atol=solve_ivp_atol,
            )
        elif solver == "rk4":
            trajectories[i] = integrate_rk4_fixed(system, initial_states[i], steps=steps, dt=dt)
        else:
            raise ValueError(f"unknown solver: {solver}")

    return trajectories.astype(np.float32), time_grid.astype(np.float32)


def generate_reference_figure_eight(
    system: ThreeBodySystem3D,
    initial_state: np.ndarray,
    steps: int,
    dt: float,
    solver: SolverName = "solve_ivp",
) -> tuple[np.ndarray, np.ndarray]:
    time_grid = np.arange(steps + 1, dtype=np.float32) * float(dt)
    if solver == "solve_ivp":
        trajectory = integrate_solve_ivp(system, initial_state, time_grid, method="DOP853", rtol=1e-11, atol=1e-13)
    elif solver == "rk4":
        trajectory = integrate_rk4_fixed(system, initial_state, steps=steps, dt=dt)
    else:
        raise ValueError(f"unknown solver: {solver}")
    return trajectory[None].astype(np.float32), time_grid.astype(np.float32)


def trajectory_pairs_to_derivatives(trajectories: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert trajectories to pairs of state and finite-difference derivative."""
    arr = np.asarray(trajectories, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 18:
        raise ValueError("trajectories must have shape (n_trajs, steps + 1, 18)")
    states = arr[:, :-1].reshape(-1, 18)
    next_states = arr[:, 1:].reshape(-1, 18)
    derivatives = (next_states - states) / float(dt)
    return states.astype(np.float32), derivatives.astype(np.float32)


def save_trajectory_dataset(path: str | Path, trajectories: np.ndarray, t: np.ndarray, metadata: dict | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        trajectories=np.asarray(trajectories, dtype=np.float32),
        t=np.asarray(t, dtype=np.float32),
        metadata=np.array(metadata or {}, dtype=object),
    )
    return out


def load_trajectory_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(Path(path), allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    t = data["t"].astype(np.float32)
    metadata = data["metadata"].item() if "metadata" in data else {}
    return trajectories, t, metadata


def dataset_path(root: str | Path, name: str, n_trajs: int, steps: int, dt: float, seed: int, solver: str) -> Path:
    return Path(root) / f"{name}_n{n_trajs}_steps{steps}_dt{dt:g}_seed{seed}_{solver}.npz"
