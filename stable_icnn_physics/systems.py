from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import re
from typing import Any, Callable, Optional, Protocol

import numpy as np


class PhysicalSystem(Protocol):
    """Interface for autonomous continuous-time physical systems."""

    @property
    def name(self) -> str:
        ...

    @property
    def state_dim(self) -> int:
        ...

    def rhs(self, x: np.ndarray) -> np.ndarray:
        ...

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> np.ndarray:
        ...

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> np.ndarray:
        ...

    def state_error(self, x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        ...

    def wrap_state(self, x: np.ndarray) -> np.ndarray:
        return x

    def metadata(self) -> dict[str, Any]:
        ...


@dataclass
class MassSpringDamper:
    """State is `[position, velocity]`."""

    mass: float = 1.0
    damping: float = 0.3
    stiffness: float = 1.0
    position_range: tuple[float, float] = (-2.0, 2.0)
    velocity_range: tuple[float, float] = (-2.0, 2.0)

    @property
    def name(self) -> str:
        return "mass_spring_damper"

    @property
    def state_dim(self) -> int:
        return 2

    def rhs(self, x: np.ndarray) -> np.ndarray:
        x = _as_batch(x, self.state_dim)
        pos = x[:, 0]
        vel = x[:, 1]
        acc = -(self.stiffness / self.mass) * pos - (self.damping / self.mass) * vel
        return np.stack([vel, acc], axis=1).astype(np.float32)

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(_split_seed(seed, split))
        pos = rng.uniform(*self.position_range, size=n)
        vel = rng.uniform(*self.velocity_range, size=n)
        return np.stack([pos, vel], axis=1).astype(np.float32)

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> np.ndarray:
        """Sample initial conditions from the same state domain used for random datasets."""

        return self.sample_states(n=n, split=split, seed=seed)

    def state_error(self, x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        return np.sum((x_true - x_pred) ** 2, axis=-1)

    def wrap_state(self, x: np.ndarray) -> np.ndarray:
        return x

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable system metadata."""

        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "params": {
                "mass": self.mass,
                "damping": self.damping,
                "stiffness": self.stiffness,
                "position_range": list(self.position_range),
                "velocity_range": list(self.velocity_range),
            },
        }


@dataclass
class VanDerPolOscillator:
    """Van der Pol oscillator.

    State is `[position, velocity]`, with dynamics:

    `position_dot = velocity`
    `velocity_dot = mu * (1 - position**2) * velocity - position`

    For `mu > 0`, the origin is unstable and trajectories approach a stable
    limit cycle. This is useful as a nonlinear benchmark, but it is not a
    perfect match for dynamics architectures that are globally stable to the
    origin.
    """

    mu: float = 1.0
    position_range: tuple[float, float] = (-3.0, 3.0)
    velocity_range: tuple[float, float] = (-4.0, 4.0)

    @property
    def name(self) -> str:
        return "vanderpol"

    @property
    def state_dim(self) -> int:
        return 2

    def rhs(self, x: np.ndarray) -> np.ndarray:
        x = _as_batch(x, self.state_dim)
        pos = x[:, 0]
        vel = x[:, 1]
        acc = self.mu * (1.0 - pos**2) * vel - pos
        return np.stack([vel, acc], axis=1).astype(np.float32)

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(_split_seed(seed, split))
        pos = rng.uniform(*self.position_range, size=n)
        vel = rng.uniform(*self.velocity_range, size=n)
        return np.stack([pos, vel], axis=1).astype(np.float32)

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> np.ndarray:
        """Sample initial conditions from the same state domain used for random datasets."""

        return self.sample_states(n=n, split=split, seed=seed)

    def state_error(self, x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        return np.sum((x_true - x_pred) ** 2, axis=-1)

    def wrap_state(self, x: np.ndarray) -> np.ndarray:
        return x

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable system metadata."""

        note = (
            "For mu > 0, Van der Pol has an unstable origin and a stable limit cycle; "
            "it is not perfectly compatible with globally stable-to-origin Lyapunov models."
        )
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "params": {
                "mu": self.mu,
                "position_range": list(self.position_range),
                "velocity_range": list(self.velocity_range),
            },
            "notes": [note],
        }


@dataclass
class DampedPendulum:
    """Damped rigid n-link pendulum.

    State is `[theta_0, ..., theta_{n-1}, omega_0, ..., omega_{n-1}]`.
    For `n_links=1`, an analytic RHS is used. For larger `n_links`, equations are
    generated with SymPy's Kane method, following the companion code.
    """

    n_links: int = 1
    friction: float = 0.3
    gravity: float = 9.81
    lengths: Optional[np.ndarray | float] = None
    masses: np.ndarray | float = 1.0
    angle_range: tuple[float, float] = (-np.pi, np.pi)
    velocity_range: tuple[float, float] = (-np.pi, np.pi)

    def __post_init__(self) -> None:
        if self.n_links < 1:
            raise ValueError("n_links must be at least 1")

    @property
    def name(self) -> str:
        return f"damped_pendulum_n{self.n_links}"

    @property
    def state_dim(self) -> int:
        return 2 * self.n_links

    def rhs(self, x: np.ndarray) -> np.ndarray:
        x = _as_batch(x, self.state_dim)
        if self.n_links == 1:
            theta = x[:, 0]
            omega = x[:, 1]
            length = float(np.broadcast_to(1.0 if self.lengths is None else self.lengths, 1)[0])
            omega_dot = -(self.friction * omega) + (self.gravity / length) * np.sin(theta - np.pi)
            return np.stack([omega, omega_dot], axis=1).astype(np.float32)
        return self._multi_link_rhs(x).astype(np.float32)

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(_split_seed(seed, split))
        theta = rng.uniform(*self.angle_range, size=(n, self.n_links))
        omega = rng.uniform(*self.velocity_range, size=(n, self.n_links))
        return np.concatenate([theta, omega], axis=1).astype(np.float32)

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> np.ndarray:
        """Sample initial conditions for trajectory simulation."""

        return self.sample_states(n=n, split=split, seed=seed)

    def wrap_state(self, x: np.ndarray) -> np.ndarray:
        y = np.array(x, copy=True)
        theta = y[..., : self.n_links]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        y[..., : self.n_links] = theta
        return y

    def state_error(self, x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        angle_error = x_true[..., : self.n_links] - x_pred[..., : self.n_links]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        vel_error = x_true[..., self.n_links :] - x_pred[..., self.n_links :]
        return np.sum(angle_error**2, axis=-1) + np.sum(vel_error**2, axis=-1)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable system metadata."""

        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "params": {
                "n_links": self.n_links,
                "friction": self.friction,
                "gravity": self.gravity,
                "lengths": _jsonable(self.lengths),
                "masses": _jsonable(self.masses),
                "angle_range": list(self.angle_range),
                "velocity_range": list(self.velocity_range),
            },
            "angle_indices": list(range(self.n_links)),
        }

    @cached_property
    def _multi_link_rhs(self):
        return _build_multi_pendulum_rhs(
            n=self.n_links,
            gravity=self.gravity,
            lengths=self.lengths,
            masses=self.masses,
            friction=self.friction,
        )


@dataclass
class CustomStateSpaceSystem:
    """Generic autonomous state-space system `xdot = rhs_fn(x, params)`."""

    name: str
    state_dim: int
    rhs_fn: Callable[..., np.ndarray]
    state_ranges: list[tuple[float, float]]
    angle_indices: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.state_ranges) != self.state_dim:
            raise ValueError("state_ranges must contain one (low, high) pair per state dimension")
        invalid = [idx for idx in self.angle_indices if idx < 0 or idx >= self.state_dim]
        if invalid:
            raise ValueError(f"angle_indices out of range for state_dim={self.state_dim}: {invalid}")

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the configured RHS on a single state or a batch of states."""

        xb = _as_batch(x, self.state_dim)
        try:
            y = self.rhs_fn(xb, self.params)
        except TypeError:
            y = self.rhs_fn(xb)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[None, :]
        if y.shape != xb.shape:
            raise ValueError(f"rhs_fn returned shape {y.shape}; expected {xb.shape}")
        return y.astype(np.float32)

    def sample_states(self, n: int, split: str = "train", seed: int = 0) -> np.ndarray:
        """Sample states uniformly from the configured per-coordinate ranges."""

        rng = np.random.default_rng(_split_seed(seed, split))
        cols = [rng.uniform(low, high, size=n) for low, high in self.state_ranges]
        return np.stack(cols, axis=1).astype(np.float32)

    def sample_initial_conditions(self, n: int, seed: int = 0, split: str = "train") -> np.ndarray:
        """Sample initial conditions from the configured state domain."""

        return self.sample_states(n=n, split=split, seed=seed)

    def wrap_state(self, x: np.ndarray) -> np.ndarray:
        """Wrap configured angle coordinates to `[-pi, pi]`."""

        y = np.array(x, copy=True)
        if self.angle_indices:
            y[..., self.angle_indices] = (y[..., self.angle_indices] + np.pi) % (2 * np.pi) - np.pi
        return y.astype(np.float32)

    def state_error(self, x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        """Squared state error with wrapped differences for angle coordinates."""

        diff = np.asarray(x_true) - np.asarray(x_pred)
        if self.angle_indices:
            diff[..., self.angle_indices] = (diff[..., self.angle_indices] + np.pi) % (2 * np.pi) - np.pi
        return np.sum(diff**2, axis=-1)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable system metadata."""

        return {
            "name": self.name,
            "factory_name": "custom_state_space",
            "state_dim": self.state_dim,
            "params": _jsonable(self.params),
            "state_ranges": [list(r) for r in self.state_ranges],
            "angle_indices": list(self.angle_indices),
            "rhs_serialized": False,
        }


def available_systems() -> list[str]:
    """Return the built-in system names accepted by `make_system`."""

    return [
        "mass_spring_damper",
        "vanderpol",
        "damped_pendulum",
        "damped_pendulum_1",
        "damped_pendulum_2",
        "damped_pendulum_4",
        "damped_pendulum_8",
        "pendulum_1",
        "pendulum_2",
        "pendulum_4",
        "pendulum_8",
        "custom_state_space",
    ]


def make_system(system_name: str, **kwargs) -> PhysicalSystem:
    """Create a physical system by registry name.

    Names such as `damped_pendulum_4`, `damped_pendulum_n4`, and `pendulum_4`
    create `DampedPendulum(n_links=4)`.
    """

    key = system_name.lower().strip()
    aliases = {
        "oscillator": "mass_spring_damper",
        "mass_spring": "mass_spring_damper",
        "msd": "mass_spring_damper",
        "van_der_pol": "vanderpol",
        "van-der-pol": "vanderpol",
    }
    key = aliases.get(key, key)

    if key == "mass_spring_damper":
        return MassSpringDamper(**kwargs)
    if key == "vanderpol":
        return VanDerPolOscillator(**kwargs)
    if key == "damped_pendulum":
        return DampedPendulum(**kwargs)
    if key == "custom_state_space":
        return CustomStateSpaceSystem(**kwargs)

    match = re.fullmatch(r"(?:damped_)?pendulum_?n?(\d+)", key)
    if match:
        pendulum_kwargs = dict(kwargs)
        pendulum_kwargs.setdefault("n_links", int(match.group(1)))
        return DampedPendulum(**pendulum_kwargs)

    raise ValueError(f"Unknown system {system_name!r}. Available examples: {available_systems()}")


def integrate_system(
    system: PhysicalSystem,
    x0: np.ndarray,
    t_eval: np.ndarray,
    method: str = "RK45",
) -> np.ndarray:
    """Integrate a system from one initial state using SciPy."""

    from scipy.integrate import solve_ivp

    x0 = np.asarray(x0, dtype=np.float32).reshape(-1)

    def ode(_, y):
        return system.rhs(y[None, :])[0]

    sol = solve_ivp(ode, (float(t_eval[0]), float(t_eval[-1])), x0, t_eval=t_eval, method=method)
    return system.wrap_state(sol.y.T.astype(np.float32))


def _build_multi_pendulum_rhs(n: int, gravity: float, lengths, masses, friction: float):
    from sympy import Dummy, lambdify, symbols
    from sympy.physics import mechanics

    q = mechanics.dynamicsymbols(f"q:{n}")
    u = mechanics.dynamicsymbols(f"u:{n}")
    m = symbols(f"m:{n}")
    l = symbols(f"l:{n}")
    g, t = symbols("g,t")

    frame = mechanics.ReferenceFrame("A")
    point = mechanics.Point("P")
    point.set_vel(frame, 0)

    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        link_frame = frame.orientnew(f"A{i}", "Axis", [q[i], frame.z])
        link_frame.set_ang_vel(frame, u[i] * frame.z)
        link_point = point.locatenew(f"P{i}", l[i] * link_frame.x)
        link_point.v2pt_theory(point, frame, link_frame)
        particles.append(mechanics.Particle(f"Pa{i}", link_point, m[i]))
        forces.append((link_point, m[i] * g * frame.x))
        forces.append((link_frame, -friction * u[i] * frame.z))
        kinetic_odes.append(q[i].diff(t) - u[i])
        point = link_point

    kane = mechanics.KanesMethod(frame, q_ind=q, u_ind=u, kd_eqs=kinetic_odes)
    kane.kanes_equations(particles, forces)

    lengths = np.ones(n) / n if lengths is None else lengths
    lengths = np.broadcast_to(lengths, n).astype(float)
    masses = np.broadcast_to(masses, n).astype(float)

    parameters = [g] + list(l) + list(m)
    parameter_values = [gravity] + list(lengths) + list(masses)

    unknowns = [Dummy() for _ in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = kane.kindiffdict()
    mass_matrix = kane.mass_matrix_full.subs(kds).subs(unknown_dict)
    forcing = kane.forcing_full.subs(kds).subs(unknown_dict)

    mass_matrix_fn = lambdify(unknowns + parameters, mass_matrix)
    forcing_fn = lambdify(unknowns + parameters, forcing)

    def gradient(y: np.ndarray) -> np.ndarray:
        y = _as_batch(y, 2 * n)
        out = np.zeros_like(y, dtype=np.float64)
        for i in range(y.shape[0]):
            vals = np.concatenate([y[i], parameter_values])
            sol = np.linalg.solve(mass_matrix_fn(*vals), forcing_fn(*vals))
            out[i] = np.asarray(sol, dtype=np.float64).reshape(-1)
        return out.astype(np.float32)

    return gradient


def _as_batch(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] != dim:
        raise ValueError(f"Expected state dimension {dim}, got shape {x.shape}")
    return x


def _split_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "test": 10_000, "val": 20_000, "validation": 20_000}
    return int(seed) + offsets.get(split, 30_000)


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value
