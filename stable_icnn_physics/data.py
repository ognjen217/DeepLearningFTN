from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .systems import PhysicalSystem
from .integrators import euler_step_numpy, rk4_step_numpy


def generate_derivative_data(
    system: PhysicalSystem,
    n_samples: int,
    split: str = "train",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample states and compute derivative targets."""

    x = system.sample_states(n_samples, split=split, seed=seed)
    y = system.rhs(x)
    return x.astype(np.float32), y.astype(np.float32)


def save_dataset(path: str | Path, x: np.ndarray, y: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
    """Save an `(x, y)` dataset with JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        metadata=json.dumps(_jsonable(metadata or {})),
    )


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an `(x, y)` dataset, preserving the legacy return shape."""

    data = np.load(path, allow_pickle=False)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def load_dataset_metadata(path: str | Path) -> dict[str, Any]:
    """Load JSON metadata from a dataset file."""

    data = np.load(path, allow_pickle=False)
    if "metadata" not in data:
        return {}
    return _decode_metadata(data["metadata"])


def tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    import torch
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))


def dataset_path(cache_dir: str | Path, system_name: str, split: str, n_samples: int, seed: int) -> Path:
    """Legacy random derivative dataset path helper."""

    return Path(cache_dir) / f"{system_name}_{split}_n{n_samples}_seed{seed}.npz"


def simulate_trajectories(
    system: PhysicalSystem,
    n_trajectories: int,
    steps: int,
    dt: float,
    split: str = "train",
    seed: int = 0,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate trajectories and evaluate derivatives at every trajectory state."""

    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    x = system.sample_initial_conditions(n=n_trajectories, seed=seed, split=split).astype(np.float32)
    trajectories = np.zeros((n_trajectories, steps + 1, system.state_dim), dtype=np.float32)
    derivatives = np.zeros_like(trajectories)
    trajectories[:, 0, :] = system.wrap_state(x)
    derivatives[:, 0, :] = system.rhs(trajectories[:, 0, :])

    method_key = method.lower()
    if method_key == "rk4":
        step_fn = rk4_step_numpy
    elif method_key == "euler":
        step_fn = euler_step_numpy
    else:
        raise ValueError(f"Unknown integration method {method!r}; expected 'rk4' or 'euler'.")

    for i in range(steps):
        x = step_fn(system.rhs, trajectories[:, i, :], dt)
        x = system.wrap_state(x).astype(np.float32)
        trajectories[:, i + 1, :] = x
        derivatives[:, i + 1, :] = system.rhs(x)

    return trajectories.astype(np.float32), derivatives.astype(np.float32)


def trajectory_to_derivative_dataset(
    trajectories: np.ndarray,
    derivatives: np.ndarray,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert simulated trajectories into `(x_t, xdot_t)` derivative pairs."""

    trajectories = np.asarray(trajectories, dtype=np.float32)
    derivatives = np.asarray(derivatives, dtype=np.float32)
    if trajectories.shape != derivatives.shape:
        raise ValueError(f"trajectories and derivatives must have matching shape, got {trajectories.shape} and {derivatives.shape}")
    if flatten:
        return trajectories.reshape(-1, trajectories.shape[-1]), derivatives.reshape(-1, derivatives.shape[-1])
    return trajectories, derivatives


def trajectory_to_transition_dataset(trajectories: np.ndarray, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Convert simulated trajectories into `(x_t, x_{t+1})` transition pairs."""

    trajectories = np.asarray(trajectories, dtype=np.float32)
    x = trajectories[:, :-1, :]
    y = trajectories[:, 1:, :]
    if flatten:
        return x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    return x, y


def save_trajectory_dataset(
    path: str | Path,
    trajectories: np.ndarray,
    derivatives: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save trajectory arrays plus flattened derivative or transition pairs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_type = metadata.get("dataset_type", "derivative")
    if dataset_type == "transition":
        x, y = trajectory_to_transition_dataset(trajectories, flatten=True)
    elif dataset_type == "derivative":
        x, y = trajectory_to_derivative_dataset(trajectories, derivatives, flatten=True)
    else:
        raise ValueError("metadata['dataset_type'] must be 'derivative' or 'transition'")

    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        trajectories=np.asarray(trajectories, dtype=np.float32),
        derivatives=np.asarray(derivatives, dtype=np.float32),
        metadata=json.dumps(_jsonable(metadata)),
    )


def load_trajectory_dataset(path: str | Path) -> dict[str, Any]:
    """Load a trajectory dataset file including arrays and decoded metadata."""

    data = np.load(path, allow_pickle=False)
    result: dict[str, Any] = {
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "metadata": _decode_metadata(data["metadata"]) if "metadata" in data else {},
    }
    if "trajectories" in data:
        result["trajectories"] = data["trajectories"].astype(np.float32)
    if "derivatives" in data:
        result["derivatives"] = data["derivatives"].astype(np.float32)
    return result


def dataset_base_name(
    system: PhysicalSystem | str,
    split: str,
    n_samples: int | None = None,
    n_trajectories: int | None = None,
    steps: int | None = None,
    dt: float | None = None,
    seed: int = 0,
    dataset_type: str = "derivative",
) -> str:
    """Build a descriptive dataset filename stem."""

    system_name = system if isinstance(system, str) else system.name
    system_name = _safe_name(system_name)
    parts = [system_name, split]
    if n_trajectories is not None:
        parts.append(f"traj{n_trajectories}")
    elif n_samples is not None:
        parts.append(f"n{n_samples}")
    else:
        raise ValueError("Provide either n_samples or n_trajectories")
    if steps is not None:
        parts.append(f"steps{steps}")
    if dt is not None:
        parts.append(f"dt{_format_float(dt)}")
    parts.extend([f"seed{seed}", dataset_type])
    return "_".join(parts) + ".npz"


def _decode_metadata(raw) -> dict[str, Any]:
    text = raw.item() if hasattr(raw, "item") else raw
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not text:
        return {}
    try:
        return json.loads(str(text))
    except json.JSONDecodeError:
        return {"legacy_metadata": str(text)}


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)


def _format_float(value: float) -> str:
    return f"{value:g}"


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
