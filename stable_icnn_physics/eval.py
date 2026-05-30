from __future__ import annotations

import numpy as np

from .integrators import rk4_step_numpy, simulate_autonomous_system
from .systems import PhysicalSystem


def rollout_system(system: PhysicalSystem, x0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    return simulate_autonomous_system(system, x0=x0, steps=steps, dt=dt, method="rk4")


def rollout_model(
    model,
    x0: np.ndarray,
    steps: int,
    dt: float,
    device=None,
    wrap_fn=None,
) -> np.ndarray:
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    was_training = model.training
    model.eval()
    model.to(device)

    x_np = np.asarray(x0, dtype=np.float32)
    traj = np.zeros((steps + 1, *x_np.shape), dtype=np.float32)
    traj[0] = x_np
    x = torch.from_numpy(x_np).to(device)

    for i in range(steps):
        x = _rk4_step_torch(model, x, dt).detach()
        x_np = x.cpu().numpy()
        if wrap_fn is not None:
            x_np = wrap_fn(x_np)
            x = torch.from_numpy(x_np.astype(np.float32)).to(device)
        traj[i + 1] = x_np

    model.train(was_training)
    return traj


def autoregressive_rollout_model(
    model,
    x0: np.ndarray,
    steps: int,
    dt: float,
    device=None,
    wrap_fn=None,
) -> np.ndarray:
    """Roll out a learned continuous-time model from its own previous state.

    The model predicts derivatives `xdot = f(x)`. Each next state is obtained by
    integrating that derivative field for one time step with RK4, then feeding
    the predicted state back as the next input.
    """

    return rollout_model(model=model, x0=x0, steps=steps, dt=dt, device=device, wrap_fn=wrap_fn)


def rollout_error(system: PhysicalSystem, true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    return system.state_error(true_traj, pred_traj)


def lyapunov_decrease_values(model: nn.Module, x: np.ndarray, device: str | torch.device | None = None) -> np.ndarray:
    import torch

    if not hasattr(model, "lyapunov_decrease"):
        raise TypeError("model does not expose lyapunov_decrease")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    values = model.lyapunov_decrease(xt)
    return values.detach().cpu().numpy()


def long_horizon_autoregressive_stability_test(
    system: PhysicalSystem,
    models: dict[str, object],
    x0: np.ndarray,
    steps: int,
    dt: float,
    device=None,
    store_every: int = 10,
    divergence_error: float = 1_000.0,
    divergence_norm: float = 1_000.0,
) -> dict:
    """Stream a long autoregressive stability test without storing full trajectories.

    The reference system is advanced by fixed-step RK4. Each learned model is
    advanced autoregressively from its own previous prediction using the same
    RK4 wrapper used by :func:`autoregressive_rollout_model`.

    This is an empirical finite-horizon stability test, not a formal global
    stability proof. It is designed for long horizons where storing all states
    for every model would be inconvenient.
    """
    import torch

    if steps <= 0:
        raise ValueError("steps must be positive")
    if store_every <= 0:
        raise ValueError("store_every must be positive")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x_true = np.asarray(x0, dtype=np.float32)
    if x_true.ndim == 1:
        x_true = x_true[None, :]

    model_state = {}
    training_state = {}
    for name, model in models.items():
        training_state[name] = getattr(model, "training", False)
        model.eval()
        model.to(device)
        model_state[name] = torch.from_numpy(x_true.copy()).to(device)

    sample_steps = list(range(0, steps + 1, store_every))
    if sample_steps[-1] != steps:
        sample_steps.append(steps)

    curves = {
        name: {
            "step": [],
            "time": [],
            "mean_error": [],
            "max_error": [],
            "mean_state_norm": [],
            "max_state_norm": [],
            "finite_fraction": [],
            "V_mean": [],
            "V_max": [],
        }
        for name in models
    }
    summaries = {
        name: {
            "diverged": False,
            "diverged_at_step": None,
            "max_mean_error": 0.0,
            "max_state_norm": 0.0,
            "max_error": 0.0,
        }
        for name in models
    }

    def record(step: int) -> None:
        for name, model in models.items():
            pred_np = model_state[name].detach().cpu().numpy()
            err = np.sqrt(system.state_error(x_true, pred_np))
            state_norm = np.linalg.norm(pred_np, axis=-1)
            finite = np.isfinite(pred_np).all(axis=-1)

            mean_error = float(np.mean(err))
            max_error = float(np.max(err))
            mean_norm = float(np.mean(state_norm))
            max_norm = float(np.max(state_norm))
            finite_fraction = float(np.mean(finite))

            curves[name]["step"].append(int(step))
            curves[name]["time"].append(float(step * dt))
            curves[name]["mean_error"].append(mean_error)
            curves[name]["max_error"].append(max_error)
            curves[name]["mean_state_norm"].append(mean_norm)
            curves[name]["max_state_norm"].append(max_norm)
            curves[name]["finite_fraction"].append(finite_fraction)

            if hasattr(model, "V"):
                with torch.no_grad():
                    v = model.V(model_state[name]).detach().cpu().numpy().reshape(-1)
                curves[name]["V_mean"].append(float(np.mean(v)))
                curves[name]["V_max"].append(float(np.max(v)))
            else:
                curves[name]["V_mean"].append(float("nan"))
                curves[name]["V_max"].append(float("nan"))

            summaries[name]["max_mean_error"] = max(summaries[name]["max_mean_error"], mean_error)
            summaries[name]["max_error"] = max(summaries[name]["max_error"], max_error)
            summaries[name]["max_state_norm"] = max(summaries[name]["max_state_norm"], max_norm)

            if (
                not summaries[name]["diverged"]
                and (finite_fraction < 1.0 or max_error > divergence_error or max_norm > divergence_norm)
            ):
                summaries[name]["diverged"] = True
                summaries[name]["diverged_at_step"] = int(step)

    record(0)
    sample_set = set(sample_steps)
    for step in range(1, steps + 1):
        x_true = rk4_step_numpy(system.rhs, x_true, dt)
        x_true = system.wrap_state(x_true).astype(np.float32)

        for name, model in models.items():
            with torch.enable_grad():
                x_next = _rk4_step_torch(model, model_state[name], dt).detach()
            pred_np = x_next.cpu().numpy()
            pred_np = system.wrap_state(pred_np).astype(np.float32)
            model_state[name] = torch.from_numpy(pred_np).to(device)

        if step in sample_set:
            record(step)

    for name, model in models.items():
        model.train(training_state[name])
        summaries[name]["final_mean_error"] = float(curves[name]["mean_error"][-1])
        summaries[name]["final_max_error"] = float(curves[name]["max_error"][-1])
        summaries[name]["final_mean_state_norm"] = float(curves[name]["mean_state_norm"][-1])
        summaries[name]["final_finite_fraction"] = float(curves[name]["finite_fraction"][-1])

    return {
        "steps": int(steps),
        "dt": float(dt),
        "store_every": int(store_every),
        "divergence_error": float(divergence_error),
        "divergence_norm": float(divergence_norm),
        "summaries": summaries,
        "curves": curves,
    }


def _rk4_step_torch(model, x, dt: float):
    k1 = model(x)
    k2 = model(x + 0.5 * dt * k1)
    k3 = model(x + 0.5 * dt * k2)
    k4 = model(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
