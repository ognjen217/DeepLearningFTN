from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]


def make_pair_dataset(x: np.ndarray, y: np.ndarray, device_dtype=np.float32) -> TensorDataset:
    """Create a TensorDataset from numpy one-step state pairs."""
    return TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=device_dtype)),
        torch.from_numpy(np.asarray(y, dtype=device_dtype)),
    )


def train_next_step_model(
    dynamics: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset | None = None,
    dt: float = 2.0e-3,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    print_every: int = 5,
    use_amp: bool = False,
    grad_clip_norm: float | None = 1.0,
) -> TrainHistory:
    """Train a derivative model through the one-step update ``x + dt*f(x)``.

    This keeps the model conceptually continuous-time, but the supervised loss is
    directly on the next state from the reference solver.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dynamics.to(device)

    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.1)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)

    history = TrainHistory(train_loss=[], val_loss=[])
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        dynamics.train()
        total, count = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                with torch.enable_grad():
                    y_pred = x + float(dt) * dynamics(x)
                    loss = loss_fn(y_pred, y)

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total += loss.detach().item() * x.shape[0]
            count += x.shape[0]

        scheduler.step()
        train_loss = total / max(count, 1)
        val_loss = evaluate_next_step_mse(dynamics, val_loader, dt, device) if val_loader is not None else train_loss
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in dynamics.state_dict().items()}

        if print_every and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            print(f"epoch={epoch:04d} train={train_loss:.6g} val={val_loss:.6g} best={best_val:.6g}")

    if best_state is not None:
        dynamics.load_state_dict(best_state)

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": dynamics.state_dict(), "history": history.__dict__}, checkpoint_path)

    return history


def evaluate_next_step_mse(
    dynamics: nn.Module,
    loader: DataLoader | TensorDataset,
    dt: float,
    device: str | torch.device | None = None,
) -> float:
    """Evaluate one-step MSE through ``x + dt*f(x)``."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dynamics.to(device)
    was_training = dynamics.training
    dynamics.eval()

    if isinstance(loader, TensorDataset):
        loader = DataLoader(loader, batch_size=128, shuffle=False, pin_memory=device.type == "cuda")

    loss_fn = nn.MSELoss(reduction="sum")
    total, count = 0.0, 0
    pin = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)
        with torch.enable_grad():
            pred = x + float(dt) * dynamics(x)
        total += loss_fn(pred, y).item()
        count += y.numel()

    dynamics.train(was_training)
    return total / max(count, 1)


def split_tensor_dataset(dataset: TensorDataset, val_fraction: float = 0.1, seed: int = 0) -> tuple[TensorDataset, TensorDataset]:
    """Deterministically split a TensorDataset into train/val subsets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_fraction)))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)
