from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray
Boundary = Literal["periodic", "reflective"]


@dataclass(frozen=True)
class SWE2DConfig:
    """Configuration for the linearized damped 2D shallow-water equations.

    The state is stored as ``state = [eta, u, v]`` with shape ``(3, ny, nx)``.

    Equations:
        eta_t + H * (u_x + v_y) = 0
        u_t   + g * eta_x       = -r*u + nu*laplace(u)
        v_t   + g * eta_y       = -r*v + nu*laplace(v)

    ``eta`` is the free-surface displacement around the resting water height.
    """

    nx: int = 64
    ny: int = 64
    lx: float = 1.0
    ly: float = 1.0
    dt: float = 2.0e-3
    gravity: float = 9.81
    depth: float = 1.0
    damping: float = 0.10
    viscosity: float = 1.0e-4
    boundary: Boundary = "periodic"

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def wave_speed(self) -> float:
        return float(np.sqrt(self.gravity * self.depth))


def make_grid(cfg: SWE2DConfig) -> tuple[Array, Array]:
    """Return meshgrid arrays X, Y with shape ``(ny, nx)``."""
    x = np.linspace(0.0, cfg.lx, cfg.nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, cfg.ly, cfg.ny, endpoint=False, dtype=np.float32)
    return np.meshgrid(x, y)


def cfl_number(cfg: SWE2DConfig) -> float:
    """Return a simple CFL estimate for the wave part of the equations."""
    return cfg.wave_speed * cfg.dt * np.sqrt((1.0 / cfg.dx**2) + (1.0 / cfg.dy**2))


def gaussian_bump_ic(
    cfg: SWE2DConfig,
    amplitude: float = 0.10,
    sigma: float = 0.06,
    center: tuple[float, float] = (0.5, 0.5),
) -> Array:
    """Create a single smooth Gaussian surface displacement with zero velocity."""
    x, y = make_grid(cfg)
    cx, cy = center
    r2 = periodic_distance_squared(x, y, cx, cy, cfg.lx, cfg.ly)
    eta = amplitude * np.exp(-0.5 * r2 / sigma**2)
    u = np.zeros_like(eta)
    v = np.zeros_like(eta)
    return np.stack([eta, u, v], axis=0).astype(np.float32)


def random_bumps_ic(
    cfg: SWE2DConfig,
    n_bumps: int = 3,
    amplitude_range: tuple[float, float] = (-0.10, 0.10),
    sigma_range: tuple[float, float] = (0.035, 0.090),
    seed: int = 0,
) -> Array:
    """Create a random smooth free-surface field from several Gaussian bumps."""
    rng = np.random.default_rng(seed)
    x, y = make_grid(cfg)
    eta = np.zeros((cfg.ny, cfg.nx), dtype=np.float32)

    for _ in range(n_bumps):
        amp = rng.uniform(*amplitude_range)
        sigma = rng.uniform(*sigma_range)
        cx = rng.uniform(0.0, cfg.lx)
        cy = rng.uniform(0.0, cfg.ly)
        r2 = periodic_distance_squared(x, y, cx, cy, cfg.lx, cfg.ly)
        eta += amp * np.exp(-0.5 * r2 / sigma**2)

    # Remove the spatial mean so the total excess mass is close to zero.
    eta -= eta.mean()
    u = np.zeros_like(eta)
    v = np.zeros_like(eta)
    return np.stack([eta, u, v], axis=0).astype(np.float32)


def periodic_distance_squared(x: Array, y: Array, cx: float, cy: float, lx: float, ly: float) -> Array:
    """Squared distance on a periodic rectangle."""
    dx = np.minimum(np.abs(x - cx), lx - np.abs(x - cx))
    dy = np.minimum(np.abs(y - cy), ly - np.abs(y - cy))
    return dx * dx + dy * dy


def rhs_linear_swe(state: Array, cfg: SWE2DConfig) -> Array:
    """Right-hand side for the linearized damped shallow-water equations."""
    eta, u, v = state

    eta_x = ddx(eta, cfg.dx, cfg.boundary)
    eta_y = ddy(eta, cfg.dy, cfg.boundary)
    u_x = ddx(u, cfg.dx, cfg.boundary)
    v_y = ddy(v, cfg.dy, cfg.boundary)

    eta_t = -cfg.depth * (u_x + v_y)
    u_t = -cfg.gravity * eta_x - cfg.damping * u + cfg.viscosity * laplacian(u, cfg.dx, cfg.dy, cfg.boundary)
    v_t = -cfg.gravity * eta_y - cfg.damping * v + cfg.viscosity * laplacian(v, cfg.dx, cfg.dy, cfg.boundary)

    return np.stack([eta_t, u_t, v_t], axis=0).astype(np.float32)


def rk4_step(state: Array, cfg: SWE2DConfig, dt: float | None = None) -> Array:
    """One RK4 step."""
    h = cfg.dt if dt is None else float(dt)
    k1 = rhs_linear_swe(state, cfg)
    k2 = rhs_linear_swe(state + 0.5 * h * k1, cfg)
    k3 = rhs_linear_swe(state + 0.5 * h * k2, cfg)
    k4 = rhs_linear_swe(state + h * k3, cfg)
    nxt = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return apply_boundary_state(nxt.astype(np.float32), cfg)


def simulate(state0: Array, cfg: SWE2DConfig, steps: int, store_every: int = 1) -> Array:
    """Simulate and return trajectory with shape ``(n_frames, 3, ny, nx)``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if store_every < 1:
        raise ValueError("store_every must be positive")

    frames = [np.asarray(state0, dtype=np.float32).copy()]
    state = frames[0]
    for step in range(1, steps + 1):
        state = rk4_step(state, cfg)
        if step % store_every == 0:
            frames.append(state.copy())
    return np.stack(frames, axis=0).astype(np.float32)


def compute_energy(state: Array, cfg: SWE2DConfig) -> float:
    """Discrete linearized shallow-water energy."""
    eta, u, v = state
    density = 0.5 * (cfg.gravity * eta**2 + cfg.depth * (u**2 + v**2))
    return float(density.sum() * cfg.dx * cfg.dy)


def compute_mass(state: Array, cfg: SWE2DConfig) -> float:
    """Total free-surface displacement integral."""
    eta = state[0]
    return float(eta.sum() * cfg.dx * cfg.dy)


def velocity_magnitude(state: Array) -> Array:
    """Return sqrt(u^2 + v^2)."""
    return np.sqrt(state[1] ** 2 + state[2] ** 2)


def compute_rmse(traj_true: Array, traj_pred: Array) -> Array:
    """RMSE per frame between two trajectories."""
    diff = np.asarray(traj_true) - np.asarray(traj_pred)
    return np.sqrt(np.mean(diff * diff, axis=(1, 2, 3)))


def ddx(a: Array, dx: float, boundary: Boundary) -> Array:
    """Centered finite difference in x."""
    if boundary == "periodic":
        return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2.0 * dx)
    return (_pad_reflect(a, axis=1, side="right") - _pad_reflect(a, axis=1, side="left")) / (2.0 * dx)


def ddy(a: Array, dy: float, boundary: Boundary) -> Array:
    """Centered finite difference in y."""
    if boundary == "periodic":
        return (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2.0 * dy)
    return (_pad_reflect(a, axis=0, side="right") - _pad_reflect(a, axis=0, side="left")) / (2.0 * dy)


def laplacian(a: Array, dx: float, dy: float, boundary: Boundary) -> Array:
    """Second-order finite-difference Laplacian."""
    if boundary == "periodic":
        ax = (np.roll(a, -1, axis=1) - 2.0 * a + np.roll(a, 1, axis=1)) / dx**2
        ay = (np.roll(a, -1, axis=0) - 2.0 * a + np.roll(a, 1, axis=0)) / dy**2
        return ax + ay

    left = _pad_reflect(a, axis=1, side="left")
    right = _pad_reflect(a, axis=1, side="right")
    down = _pad_reflect(a, axis=0, side="left")
    up = _pad_reflect(a, axis=0, side="right")
    return (right - 2.0 * a + left) / dx**2 + (up - 2.0 * a + down) / dy**2


def _pad_reflect(a: Array, axis: int, side: Literal["left", "right"]) -> Array:
    """Neighbor array with simple reflective boundary behaviour."""
    out = np.empty_like(a)
    if axis == 1:
        if side == "left":
            out[:, 1:] = a[:, :-1]
            out[:, 0] = a[:, 1]
        else:
            out[:, :-1] = a[:, 1:]
            out[:, -1] = a[:, -2]
    elif axis == 0:
        if side == "left":
            out[1:, :] = a[:-1, :]
            out[0, :] = a[1, :]
        else:
            out[:-1, :] = a[1:, :]
            out[-1, :] = a[-2, :]
    else:
        raise ValueError("axis must be 0 or 1")
    return out


def apply_boundary_state(state: Array, cfg: SWE2DConfig) -> Array:
    """Apply simple velocity reflection for reflective boundaries."""
    if cfg.boundary == "periodic":
        return state

    out = np.array(state, copy=True)
    # No-normal-flow approximation at domain boundaries.
    out[1, :, 0] = 0.0
    out[1, :, -1] = 0.0
    out[2, 0, :] = 0.0
    out[2, -1, :] = 0.0
    return out
