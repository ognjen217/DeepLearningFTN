from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from .swe2d import SWE2DConfig, random_bumps_ic, simulate

Split = Literal["train", "val", "test"]


def generate_trajectory_dataset(
    cfg: SWE2DConfig,
    n_trajs: int = 64,
    steps: int = 100,
    split: Split = "train",
    seed: int = 0,
    n_bumps_range: tuple[int, int] = (1, 5),
    amplitude_range: tuple[float, float] = (-0.10, 0.10),
    sigma_range: tuple[float, float] = (0.035, 0.090),
) -> np.ndarray:
    """Generate a dataset of reference shallow-water trajectories.

    Returns
    -------
    trajectories:
        Float32 array with shape ``(n_trajs, steps + 1, 3, ny, nx)``.
    """
    rng = np.random.default_rng(_split_seed(seed, split))
    trajectories = []

    for i in range(n_trajs):
        n_low, n_high = n_bumps_range
        n_bumps = int(rng.integers(n_low, n_high + 1))
        ic_seed = int(rng.integers(0, 2**31 - 1))
        state0 = random_bumps_ic(
            cfg,
            n_bumps=n_bumps,
            amplitude_range=amplitude_range,
            sigma_range=sigma_range,
            seed=ic_seed,
        )
        traj = simulate(state0, cfg, steps=steps, store_every=1)
        trajectories.append(traj)

    return np.stack(trajectories, axis=0).astype(np.float32)


def trajectory_pairs(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert trajectories into one-step pairs ``(state_t, state_{t+1})``.

    Input shape: ``(n_trajs, steps + 1, 3, ny, nx)``.
    Output shapes: ``(n_trajs * steps, 3, ny, nx)``.
    """
    arr = np.asarray(trajectories, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[2] != 3:
        raise ValueError("trajectories must have shape (n_trajs, steps + 1, 3, ny, nx)")

    x = arr[:, :-1].reshape(-1, *arr.shape[2:])
    y = arr[:, 1:].reshape(-1, *arr.shape[2:])
    return x.astype(np.float32), y.astype(np.float32)


def save_trajectory_dataset(path: str | Path, trajectories: np.ndarray, cfg: SWE2DConfig, **metadata) -> Path:
    """Save trajectories and solver metadata into a compressed NPZ file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"cfg": asdict(cfg), **metadata}
    np.savez_compressed(out, trajectories=np.asarray(trajectories, dtype=np.float32), metadata=np.array(meta, dtype=object))
    return out


def load_trajectory_dataset(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a trajectory dataset saved by :func:`save_trajectory_dataset`."""
    data = np.load(Path(path), allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    metadata = data["metadata"].item() if "metadata" in data else {}
    return trajectories, metadata


def dataset_path(
    root: str | Path,
    cfg: SWE2DConfig,
    split: Split,
    n_trajs: int,
    steps: int,
    seed: int,
) -> Path:
    """Return a stable cache path for a dataset configuration."""
    name = (
        f"swe2d_linear_{split}_n{n_trajs}_steps{steps}_"
        f"grid{cfg.ny}x{cfg.nx}_dt{cfg.dt:g}_seed{seed}.npz"
    )
    return Path(root) / name


def load_or_generate_dataset(
    root: str | Path,
    cfg: SWE2DConfig,
    n_trajs: int,
    steps: int,
    split: Split,
    seed: int = 0,
    force: bool = False,
    **generator_kwargs,
) -> tuple[np.ndarray, Path]:
    """Load a cached dataset, or generate and save it if missing."""
    path = dataset_path(root, cfg, split=split, n_trajs=n_trajs, steps=steps, seed=seed)
    if path.exists() and not force:
        trajectories, _ = load_trajectory_dataset(path)
        return trajectories, path

    trajectories = generate_trajectory_dataset(
        cfg,
        n_trajs=n_trajs,
        steps=steps,
        split=split,
        seed=seed,
        **generator_kwargs,
    )
    save_trajectory_dataset(
        path,
        trajectories,
        cfg,
        split=split,
        n_trajs=n_trajs,
        steps=steps,
        seed=seed,
    )
    return trajectories, path


def _split_seed(seed: int, split: Split) -> int:
    offsets = {"train": 0, "val": 10_000, "test": 20_000}
    return int(seed) + offsets[split]
