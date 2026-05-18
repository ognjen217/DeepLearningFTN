from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainHistory:
    train_loss: list[float]
    test_loss: list[float]


def train_derivative_model(
    model: nn.Module,
    train_dataset: TensorDataset,
    test_dataset: TensorDataset | None = None,
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    print_every: int = 25,
    use_amp: bool = False,
) -> TrainHistory:
    """Train a model on `(x, xdot)` derivative pairs."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)

    amp_ctx = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    history = TrainHistory(train_loss=[], test_loss=[])
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                pred = model(x)
                loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * x.shape[0]
            count += x.shape[0]

        train_loss = total / max(count, 1)
        test_loss = evaluate_derivative_mse(model, test_dataset, batch_size=batch_size, device=device) if test_dataset else train_loss
        history.train_loss.append(train_loss)
        history.test_loss.append(test_loss)

        if print_every and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            print(f"epoch={epoch:04d} train_mse={train_loss:.6g} test_mse={test_loss:.6g}")

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history.__dict__}, checkpoint_path)

    return history


def evaluate_derivative_mse(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> float:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    was_training = model.training
    model.eval()
    model.to(device)
    pin = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)
        # StableDynamics computes grad V(x), so evaluation still needs autograd.
        with torch.enable_grad():
            pred = model(x)
        total += loss_fn(pred, y).item()
        count += y.numel()
    model.train(was_training)
    return total / max(count, 1)
