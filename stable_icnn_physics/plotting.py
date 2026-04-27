from __future__ import annotations

import numpy as np
import torch


def plot_loss(history, ax=None, title: str = "Derivative MSE"):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.plot(history.train_loss, label="train")
    ax.plot(history.test_loss, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    return ax


def plot_rollout_error(errors: dict[str, np.ndarray], ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    for label, error in errors.items():
        ax.plot(np.mean(error, axis=tuple(range(1, error.ndim))) if error.ndim > 1 else error, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("state squared error")
    ax.set_yscale("log")
    ax.legend()
    return ax


def plot_vector_field(system, model=None, xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), density=25, ax=None, title="Vector field"):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    xs = np.linspace(*xlim, density)
    ys = np.linspace(*ylim, density)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    if model is None:
        vec = system.rhs(points)
    else:
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            vec = model(torch.from_numpy(points)).detach().cpu().numpy()
        model.train(was_training)
    ax.streamplot(grid_x, grid_y, vec[:, 0].reshape(grid_x.shape), vec[:, 1].reshape(grid_y.shape), density=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    return ax


def plot_lyapunov_contours(model, xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), density=80, ax=None, title="Learned Lyapunov V"):
    import matplotlib.pyplot as plt

    if not hasattr(model, "V"):
        raise TypeError("model does not expose V")
    ax = ax or plt.gca()
    xs = np.linspace(*xlim, density)
    ys = np.linspace(*ylim, density)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    with torch.enable_grad():
        values = model.V(torch.from_numpy(points)).detach().cpu().numpy().reshape(grid_x.shape)
    ax.contour(grid_x, grid_y, values, levels=20)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    return ax

