from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DampedSoftenedThreeBody2D:
    """Damped softened planar three-body system.

    State layout:
        [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]

    This is intentionally a *stable-ish* three-body variant for Lyapunov/rollout
    experiments, not a high-accuracy celestial mechanics simulator. Pairwise
    gravity is softened to avoid singularities, damping removes orbital energy,
    and sampled initial conditions are shifted to zero center of mass.
    """

    gravity: float = 1.0
    damping: float = 0.08
    softening: float = 0.08
    masses: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position_scale: float = 1.0
    velocity_scale: float = 0.9
    perturbation: float = 0.15
    sample_mode: str = "figure_eight_perturbed"

    @property
    def name(self) -> str:
        return "damped_softened_three_body_2d"

    @property
    def state_dim(self) -> int:
        return 12

    def rhs(self, x: np.ndarray) -> np.ndarray:
        xb = _as_batch(x, self.state_dim)
        batch = xb.shape[0]
        pos = xb[:, :6].reshape(batch, 3, 2)
        vel = xb[:, 6:].reshape(batch, 3, 2)
        masses = np.asarray(self.masses, dtype=np.float32)

        acc = np.zeros_like(pos, dtype=np.float32)
        eps2 = float(self.softening) ** 2

        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r = pos[:, j, :] - pos[:, i, :]
                dist2 = np.sum(r * r, axis=1, keepdims=True) + eps2
                inv_dist3 = dist2 ** (-1.5)
                acc[:, i, :] += self.gravity * masses[j] * r * inv_dist3

        acc -= self.damping * vel
        deriv = np.concatenate([vel.reshape(batch, 6), acc.reshape(batch, 6)], axis=1)
        return deriv.astype(np.float32)

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> np.ndarray:
        """Sample states for derivative training.

        For this system, states are generated around physically interesting
        zero-center-of-mass configurations rather than independent uniform boxes.
        """
        return self.sample_initial_conditions(n=n, seed=seed, split=split)

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> np.ndarray:
        rng = np.random.default_rng(_split_seed(seed, split))
        if self.sample_mode == "random":
            pos = rng.normal(0.0, self.position_scale, size=(n, 3, 2)).astype(np.float32)
            vel = rng.normal(0.0, self.velocity_scale, size=(n, 3, 2)).astype(np.float32)
        else:
            base = figure_eight_like_state().astype(np.float32)
            pos0 = base[:6].reshape(1, 3, 2) * self.position_scale
            vel0 = base[6:].reshape(1, 3, 2) * self.velocity_scale
            pos = np.repeat(pos0, n, axis=0)
            vel = np.repeat(vel0, n, axis=0)
            pos += rng.normal(0.0, self.perturbation, size=pos.shape).astype(np.float32)
            vel += rng.normal(0.0, self.perturbation, size=vel.shape).astype(np.float32)

        pos, vel = remove_center_of_mass(pos, vel, self.masses)
        state = np.concatenate([pos.reshape(n, 6), vel.reshape(n, 6)], axis=1)
        return state.astype(np.float32)

    def wrap_state(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def state_error(self, x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        """Squared state error after removing center-of-mass translation."""
        xt = np.asarray(x_true, dtype=np.float32)
        xp = np.asarray(x_pred, dtype=np.float32)
        xt = center_state(xt, self.masses)
        xp = center_state(xp, self.masses)
        return np.sum((xt - xp) ** 2, axis=-1)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "params": {
                "gravity": self.gravity,
                "damping": self.damping,
                "softening": self.softening,
                "masses": list(self.masses),
                "position_scale": self.position_scale,
                "velocity_scale": self.velocity_scale,
                "perturbation": self.perturbation,
                "sample_mode": self.sample_mode,
            },
            "notes": [
                "Damped and softened variant used as a visually interesting ICNN rollout stress test.",
                "Initial conditions are shifted to zero center of mass to remove trivial translation.",
            ],
        }


def figure_eight_like_state() -> np.ndarray:
    """A convenient choreography-like equal-mass three-body initial condition."""
    return np.array(
        [
            -0.97000436,
            0.24308753,
            0.97000436,
            -0.24308753,
            0.0,
            0.0,
            0.466203685,
            0.43236573,
            0.466203685,
            0.43236573,
            -0.93240737,
            -0.86473146,
        ],
        dtype=np.float32,
    )


def remove_center_of_mass(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    masses_arr = np.asarray(masses, dtype=np.float32).reshape(1, 3, 1)
    total_mass = float(np.sum(masses_arr))
    com_pos = np.sum(pos * masses_arr, axis=1, keepdims=True) / total_mass
    com_vel = np.sum(vel * masses_arr, axis=1, keepdims=True) / total_mass
    return (pos - com_pos).astype(np.float32), (vel - com_vel).astype(np.float32)


def center_state(state: np.ndarray, masses: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    original_shape = arr.shape
    flat = arr.reshape(-1, 12)
    pos = flat[:, :6].reshape(-1, 3, 2)
    vel = flat[:, 6:].reshape(-1, 3, 2)
    pos, vel = remove_center_of_mass(pos, vel, masses)
    centered = np.concatenate([pos.reshape(-1, 6), vel.reshape(-1, 6)], axis=1)
    return centered.reshape(original_shape).astype(np.float32)


def _as_batch(x: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != dim:
        raise ValueError(f"Expected last dimension {dim}, got {arr.shape}")
    return arr.reshape(-1, dim)


def _split_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "val": 10_000, "test": 20_000}
    return int(seed) + offsets.get(split, 30_000)
