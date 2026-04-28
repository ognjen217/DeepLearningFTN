from __future__ import annotations

from typing import Callable

import numpy as np


def euler_step_numpy(rhs: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    """Advance one autonomous ODE step with explicit Euler integration."""

    x = np.asarray(x, dtype=np.float32)
    return (x + dt * rhs(x)).astype(np.float32)


def rk4_step_numpy(rhs: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    """Advance one autonomous ODE step with classical fourth-order Runge-Kutta."""

    x = np.asarray(x, dtype=np.float32)
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)


def simulate_autonomous_system(system, x0: np.ndarray, steps: int, dt: float, method: str = "rk4") -> np.ndarray:
    """Simulate an autonomous system from one or more initial states."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    x = np.asarray(x0, dtype=np.float32)
    traj = np.zeros((steps + 1, *x.shape), dtype=np.float32)
    traj[0] = x

    method_key = method.lower()
    if method_key == "rk4":
        step_fn = rk4_step_numpy
    elif method_key == "euler":
        step_fn = euler_step_numpy
    else:
        raise ValueError(f"Unknown integration method {method!r}; expected 'rk4' or 'euler'.")

    for i in range(steps):
        x = step_fn(system.rhs, x, dt)
        x = system.wrap_state(x).astype(np.float32)
        traj[i + 1] = x
    return traj
