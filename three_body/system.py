from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ThreeBodyConfig:
    """Configuration for a 3D Newtonian three-body system."""

    gravity: float = 1.0
    masses: tuple[float, float, float] = (1.0, 1.0, 1.0)
    softening: float = 0.0
    damping: float = 0.0
    sample_noise_pos: float = 0.03
    sample_noise_vel: float = 0.03
    sample_z_noise: float = 0.0


class ThreeBodySystem3D:
    """Planar/3D Newtonian three-body dynamics embedded in 18D state space.

    State layout:
        [r1, r2, r3, v1, v2, v3]

    where each ``ri`` and ``vi`` is a 3D vector.  The default preset is the known
    equal-mass figure-eight initial condition embedded in the ``z=0`` plane.
    """

    state_dim: int = 18

    def __init__(self, config: ThreeBodyConfig | None = None):
        self.config = config or ThreeBodyConfig()

    @property
    def name(self) -> str:
        return "three_body_3d"

    def rhs(self, x: Array) -> Array:
        xb = _as_batch(x, self.state_dim)
        batch = xb.shape[0]
        pos = xb[:, :9].reshape(batch, 3, 3)
        vel = xb[:, 9:].reshape(batch, 3, 3)
        masses = np.asarray(self.config.masses, dtype=np.float32)

        acc = np.zeros_like(pos, dtype=np.float32)
        eps2 = float(self.config.softening) ** 2
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r = pos[:, j] - pos[:, i]
                dist2 = np.sum(r * r, axis=1, keepdims=True) + eps2
                inv_dist3 = dist2 ** (-1.5)
                acc[:, i] += self.config.gravity * masses[j] * r * inv_dist3

        if self.config.damping != 0.0:
            acc -= self.config.damping * vel

        out = np.concatenate([vel.reshape(batch, 9), acc.reshape(batch, 9)], axis=1)
        return out.astype(np.float32)

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> Array:
        rng = np.random.default_rng(_split_seed(seed, split))
        base = figure_eight_state_3d().reshape(1, 18)
        states = np.repeat(base, n, axis=0).astype(np.float32)
        pos_noise = rng.normal(0.0, self.config.sample_noise_pos, size=(n, 3, 3)).astype(np.float32)
        vel_noise = rng.normal(0.0, self.config.sample_noise_vel, size=(n, 3, 3)).astype(np.float32)
        if self.config.sample_z_noise == 0.0:
            pos_noise[:, :, 2] = 0.0
            vel_noise[:, :, 2] = 0.0
        else:
            pos_noise[:, :, 2] = rng.normal(0.0, self.config.sample_z_noise, size=(n, 3)).astype(np.float32)
            vel_noise[:, :, 2] = rng.normal(0.0, self.config.sample_z_noise, size=(n, 3)).astype(np.float32)

        pos = states[:, :9].reshape(n, 3, 3) + pos_noise
        vel = states[:, 9:].reshape(n, 3, 3) + vel_noise
        pos, vel = remove_center_of_mass(pos, vel, self.config.masses)
        return np.concatenate([pos.reshape(n, 9), vel.reshape(n, 9)], axis=1).astype(np.float32)

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> Array:
        return self.sample_initial_conditions(n=n, seed=seed, split=split)

    def wrap_state(self, x: Array) -> Array:
        return np.asarray(x, dtype=np.float32)

    def state_error(self, x_true: Array, x_pred: Array) -> Array:
        xt = center_state(x_true, self.config.masses)
        xp = center_state(x_pred, self.config.masses)
        return np.sum((xt - xp) ** 2, axis=-1)

    def energy(self, x: Array) -> Array:
        xb = _as_batch(x, self.state_dim)
        pos = xb[:, :9].reshape(-1, 3, 3)
        vel = xb[:, 9:].reshape(-1, 3, 3)
        masses = np.asarray(self.config.masses, dtype=np.float32)
        kinetic = 0.5 * np.sum(masses[None, :, None] * vel * vel, axis=(1, 2))
        potential = np.zeros(xb.shape[0], dtype=np.float32)
        eps2 = float(self.config.softening) ** 2
        for i in range(3):
            for j in range(i + 1, 3):
                r = pos[:, j] - pos[:, i]
                dist = np.sqrt(np.sum(r * r, axis=1) + eps2)
                potential -= self.config.gravity * masses[i] * masses[j] / dist
        return (kinetic + potential).astype(np.float32)

    def momentum(self, x: Array) -> Array:
        xb = _as_batch(x, self.state_dim)
        vel = xb[:, 9:].reshape(-1, 3, 3)
        masses = np.asarray(self.config.masses, dtype=np.float32)
        return np.sum(masses[None, :, None] * vel, axis=1).astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "config": asdict(self.config),
            "state_layout": "[r1,r2,r3,v1,v2,v3], each vector is 3D",
            "figure_eight_reference": "Chenciner-Montgomery/Moore equal-mass figure-eight orbit, embedded in z=0 plane.",
        }


def figure_eight_state_3d() -> Array:
    """Known equal-mass figure-eight initial condition embedded in 3D.

    Values follow the commonly used Chenciner-Montgomery/Moore figure-eight
    initial condition.  Positions and velocities are placed in the ``z=0`` plane.
    """
    r1 = [0.540508553669932, 0.345263318559681, 0.0]
    r2 = [0.540508532338285, -0.345263317862853, 0.0]
    r3 = [-1.081017086008497, -0.000000000697245, 0.0]
    v1 = [-1.097122372968180, -0.233604741427372, 0.0]
    v2 = [1.097122377013713, -0.233604786311327, 0.0]
    v3 = [-0.000000004046108, 0.467209527738458, 0.0]
    return np.asarray([*r1, *r2, *r3, *v1, *v2, *v3], dtype=np.float32)


def integrate_solve_ivp(
    system: ThreeBodySystem3D,
    x0: Array,
    t_eval: Array,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Array:
    from scipy.integrate import solve_ivp

    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)

    def ode(_, y):
        return system.rhs(y.astype(np.float32)[None])[0].astype(np.float64)

    sol = solve_ivp(
        ode,
        (float(t_eval[0]), float(t_eval[-1])),
        x0,
        t_eval=np.asarray(t_eval, dtype=np.float64),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.astype(np.float32)


def integrate_rk4_fixed(system: ThreeBodySystem3D, x0: Array, steps: int, dt: float) -> Array:
    x = np.asarray(x0, dtype=np.float32).copy()
    traj = np.zeros((steps + 1, x.shape[-1]), dtype=np.float32)
    traj[0] = x
    for i in range(steps):
        k1 = system.rhs(x[None])[0]
        k2 = system.rhs((x + 0.5 * dt * k1)[None])[0]
        k3 = system.rhs((x + 0.5 * dt * k2)[None])[0]
        k4 = system.rhs((x + dt * k3)[None])[0]
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = x
    return traj


def center_state(state: Array, masses: tuple[float, float, float]) -> Array:
    arr = np.asarray(state, dtype=np.float32)
    shape = arr.shape
    flat = arr.reshape(-1, 18)
    pos = flat[:, :9].reshape(-1, 3, 3)
    vel = flat[:, 9:].reshape(-1, 3, 3)
    pos, vel = remove_center_of_mass(pos, vel, masses)
    centered = np.concatenate([pos.reshape(-1, 9), vel.reshape(-1, 9)], axis=1)
    return centered.reshape(shape).astype(np.float32)


def remove_center_of_mass(pos: Array, vel: Array, masses: tuple[float, float, float]) -> tuple[Array, Array]:
    m = np.asarray(masses, dtype=np.float32).reshape(1, 3, 1)
    total = float(np.sum(m))
    com_pos = np.sum(pos * m, axis=1, keepdims=True) / total
    com_vel = np.sum(vel * m, axis=1, keepdims=True) / total
    return (pos - com_pos).astype(np.float32), (vel - com_vel).astype(np.float32)


def _as_batch(x: Array, dim: int) -> Array:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[-1] != dim:
        raise ValueError(f"Expected last dimension {dim}, got shape {arr.shape}")
    return arr.reshape(-1, dim)


def _split_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "val": 10_000, "test": 20_000}
    return int(seed) + offsets.get(split, 30_000)
