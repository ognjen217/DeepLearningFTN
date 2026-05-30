from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class VOnlyTrainHistory:
    epoch: list[int]
    train_loss: list[float]
    val_loss: list[float]
    best_val_loss: list[float]


def collect_rollout_states(
    fhat: torch.nn.Module,
    system,
    n_trajs: int,
    steps: int,
    dt: float,
    split: str,
    seed: int,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Roll out the nominal model ``fhat`` and collect visited states.

    This is the distribution-shift fix used in Exp7: Lyapunov ``V`` is trained
    not only on random sampled states, but also on states produced by the model
    during autoregressive rollout.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fhat.to(device).eval()
    ics = system.sample_initial_conditions(n_trajs, split=split, seed=seed)
    all_states: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(n_trajs):
            x = ics[i].copy()
            for _ in range(steps):
                all_states.append(x.copy())
                x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                xdot = fhat(x_t).squeeze(0).detach().cpu().numpy()
                x = system.wrap_state(x + float(dt) * xdot)

    return np.asarray(all_states, dtype=np.float32)


def train_lyapunov_v_only_on_states(
    stable_model: torch.nn.Module,
    train_states: np.ndarray,
    val_states: np.ndarray | None = None,
    epochs: int = 400,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    eta_min: float = 1e-4,
    device: str | torch.device | None = None,
    print_every: int = 40,
) -> VOnlyTrainHistory:
    """Freeze ``fhat`` and train only the ICNN Lyapunov function ``V``.

    Loss:
        mean(relu(gradV(x)·fhat(x) + alpha*V(x))^2)

    The function restores the best validation ``V`` state if validation states
    are provided. It leaves the nominal dynamics unchanged.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stable_model.to(device)

    for p in stable_model.fhat.parameters():
        p.requires_grad_(False)
    stable_model.fhat.eval()

    optimizer = torch.optim.Adam(stable_model.V.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(train_states, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    val_loader = None
    if val_states is not None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(np.asarray(val_states, dtype=np.float32))),
            batch_size=max(batch_size, 1024),
            shuffle=False,
            pin_memory=pin,
        )

    history = VOnlyTrainHistory(epoch=[], train_loss=[], val_loss=[], best_val_loss=[])
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        stable_model.V.train()
        total, count = 0.0, 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device, non_blocking=pin).requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)

            fx = stable_model.fhat(x_batch)
            vx = stable_model.V(x_batch)
            gv = torch.autograd.grad(vx.sum(), x_batch, create_graph=True)[0]
            violation = (gv * fx).sum(dim=1, keepdim=True) + stable_model.alpha * vx
            loss = F.relu(violation).pow(2).mean()

            loss.backward()
            optimizer.step()

            total += loss.detach().item() * x_batch.shape[0]
            count += x_batch.shape[0]

        scheduler.step()
        train_loss = total / max(count, 1)

        should_eval = epoch == 1 or epoch == epochs or (print_every and epoch % print_every == 0)
        if should_eval:
            if val_loader is not None:
                val_loss = evaluate_v_violation_loss(stable_model, val_loader, device)
            else:
                val_loss = train_loss

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in stable_model.V.state_dict().items()}

            history.epoch.append(epoch)
            history.train_loss.append(float(train_loss))
            history.val_loss.append(float(val_loss))
            history.best_val_loss.append(float(best_val))

            if print_every:
                print(
                    f"epoch={epoch:04d} train={train_loss:.6g} "
                    f"val={val_loss:.6g} best_val={best_val:.6g}"
                )

    if best_state is not None:
        stable_model.V.load_state_dict(best_state)

    for p in stable_model.fhat.parameters():
        p.requires_grad_(True)
    stable_model.eval()
    return history


def evaluate_v_violation_loss(
    stable_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    stable_model.V.eval()
    total, count = 0.0, 0
    pin = device.type == "cuda"

    for (x_val,) in loader:
        x_val = x_val.to(device, non_blocking=pin).requires_grad_(True)
        fx = stable_model.fhat(x_val)
        vx = stable_model.V(x_val)
        gv = torch.autograd.grad(vx.sum(), x_val, create_graph=False)[0]
        violation = (gv * fx).sum(dim=1, keepdim=True) + stable_model.alpha * vx
        loss = F.relu(violation).pow(2).mean()
        total += loss.detach().item() * x_val.shape[0]
        count += x_val.shape[0]

    return total / max(count, 1)


def projection_diagnostics(stable_model: torch.nn.Module, traj: np.ndarray, device: str | torch.device | None = None) -> dict:
    """Compute projection fire rate, correction magnitude, nominal violation and V values."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stable_model.to(device).eval()

    fires = []
    correction_norms = []
    violations = []
    v_values = []

    with torch.enable_grad():
        for t in range(traj.shape[0] - 1):
            x = torch.tensor(traj[t], dtype=torch.float32, device=device).requires_grad_(True)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            fx_hat = stable_model.fhat(x)
            vx = stable_model.V(x)
            gv = torch.autograd.grad(vx.sum(), x, create_graph=False)[0]
            violation = (gv * fx_hat).sum(dim=1, keepdim=True) + stable_model.alpha * vx
            denom = gv.square().sum(dim=1, keepdim=True).clamp_min(stable_model.denom_eps)
            correction = gv * (F.relu(violation) / denom)

            fires.append((violation > 0).detach().cpu().numpy().reshape(-1))
            correction_norms.append(correction.norm(dim=1).detach().cpu().numpy().reshape(-1))
            violations.append(violation.detach().cpu().numpy().reshape(-1))
            v_values.append(vx.detach().cpu().numpy().reshape(-1))

        x_final = torch.tensor(traj[-1], dtype=torch.float32, device=device).requires_grad_(True)
        if x_final.dim() == 1:
            x_final = x_final.unsqueeze(0)
        v_values.append(stable_model.V(x_final).detach().cpu().numpy().reshape(-1))

    fire = np.asarray(fires)
    correction_norm = np.asarray(correction_norms)
    violation = np.asarray(violations)
    V = np.asarray(v_values)
    dV = V[1:] - V[:-1]

    return {
        "fire": fire,
        "correction_norm": correction_norm,
        "violation": violation,
        "V": V,
        "dV": dV,
        "projection_fire_rate": float(fire.mean()),
        "correction_norm_mean": float(correction_norm.mean()),
        "correction_norm_p95": float(np.quantile(correction_norm, 0.95)),
        "correction_norm_max": float(correction_norm.max()),
        "nominal_violation_mean": float(violation.mean()),
        "nominal_violation_max": float(violation.max()),
        "discrete_dV_frac_nonpositive": float(np.mean(dV <= 1e-5)),
        "discrete_dV_max": float(dV.max()),
    }
