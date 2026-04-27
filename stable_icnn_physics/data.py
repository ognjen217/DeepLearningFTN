from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .systems import PhysicalSystem


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), y=y.astype(np.float32), metadata=str(metadata or {}))


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    import torch
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))


def dataset_path(cache_dir: str | Path, system_name: str, split: str, n_samples: int, seed: int) -> Path:
    return Path(cache_dir) / f"{system_name}_{split}_n{n_samples}_seed{seed}.npz"
