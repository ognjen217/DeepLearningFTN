from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from stable_icnn_physics.rollout_augmented import collect_rollout_states, train_lyapunov_v_only_on_states
from stable_icnn_physics.train import train_derivative_model

from .data import trajectory_pairs_to_derivatives
from .models import build_baseline_model, build_stable_icnn_model
from .system import ThreeBodySystem3D


def make_derivative_dataset_from_trajectories(trajectories: np.ndarray, dt: float) -> TensorDataset:
    states, derivatives = trajectory_pairs_to_derivatives(trajectories, dt)
    return TensorDataset(torch.from_numpy(states), torch.from_numpy(derivatives))


def train_phase1_models(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    stable_ckpt: str | Path,
    baseline_ckpt: str | Path,
    hidden: int = 256,
    depth: int = 3,
    lyapunov_hidden: int = 128,
    alpha: float = 1e-6,
    epochs: int = 150,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
):
    """Train phase-1 baseline and ICNN-projected dynamics on derivative MSE."""
    stable = build_stable_icnn_model(hidden=hidden, depth=depth, lyapunov_hidden=lyapunov_hidden, alpha=alpha)
    baseline = build_baseline_model(hidden=hidden, depth=depth)

    train_derivative_model(
        stable,
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        checkpoint_path=stable_ckpt,
        print_every=max(1, epochs // 10),
        use_amp=False,
    )
    train_derivative_model(
        baseline,
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        checkpoint_path=baseline_ckpt,
        print_every=max(1, epochs // 10),
        use_amp=torch.cuda.is_available(),
    )
    return stable, baseline


def train_exp7_style_v_only(
    stable_model,
    system: ThreeBodySystem3D,
    random_train_states: np.ndarray,
    random_val_states: np.ndarray,
    dt: float,
    steps: int,
    n_rollout_train: int = 256,
    n_rollout_val: int = 64,
    epochs: int = 250,
    batch_size: int = 512,
    device: str | torch.device | None = None,
):
    """Run rollout-augmented V-only training using the nominal fhat from stable_model."""
    rollout_train = collect_rollout_states(
        stable_model.fhat,
        system,
        n_trajs=n_rollout_train,
        steps=steps,
        dt=dt,
        split="train",
        seed=111,
        device=device,
    )
    rollout_val = collect_rollout_states(
        stable_model.fhat,
        system,
        n_trajs=n_rollout_val,
        steps=steps,
        dt=dt,
        split="test",
        seed=222,
        device=device,
    )
    combined_train = np.concatenate([random_train_states, rollout_train], axis=0)
    combined_val = np.concatenate([random_val_states, rollout_val], axis=0)
    history = train_lyapunov_v_only_on_states(
        stable_model,
        train_states=combined_train,
        val_states=combined_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        eta_min=1e-4,
        device=device,
        print_every=max(1, epochs // 10),
    )
    return stable_model, history, {"rollout_train": rollout_train, "rollout_val": rollout_val}
