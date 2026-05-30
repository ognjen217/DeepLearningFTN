from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def positions_from_traj(traj: np.ndarray) -> np.ndarray:
    """Extract positions with shape ``(frames, batch, 3, 2)`` or ``(frames, 3, 2)``."""
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 12:
        return arr[:, :6].reshape(arr.shape[0], 3, 2)
    if arr.ndim == 3 and arr.shape[-1] == 12:
        return arr[:, :, :6].reshape(arr.shape[0], arr.shape[1], 3, 2)
    raise ValueError("traj must have shape (frames, 12) or (frames, batch, 12)")


def plot_three_body_paths(
    traj_true: np.ndarray,
    traj_pred: np.ndarray | None = None,
    traj_id: int = 0,
    title: str = "Three-body paths",
):
    """Plot 2D body paths for a true trajectory and optional predicted trajectory."""
    true_pos = positions_from_traj(traj_true)
    if true_pos.ndim == 4:
        true_pos = true_pos[:, traj_id]

    pred_pos = None
    if traj_pred is not None:
        pred_pos = positions_from_traj(traj_pred)
        if pred_pos.ndim == 4:
            pred_pos = pred_pos[:, traj_id]

    fig, ax = plt.subplots(figsize=(6, 6))
    for body in range(3):
        ax.plot(true_pos[:, body, 0], true_pos[:, body, 1], label=f"true body {body+1}")
        ax.scatter(true_pos[0, body, 0], true_pos[0, body, 1], marker="o", s=30)
        ax.scatter(true_pos[-1, body, 0], true_pos[-1, body, 1], marker="x", s=40)
        if pred_pos is not None:
            ax.plot(pred_pos[:, body, 0], pred_pos[:, body, 1], linestyle="--", label=f"pred body {body+1}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def animate_three_body_comparison(
    traj_true: np.ndarray,
    traj_pred: np.ndarray | None = None,
    traj_id: int = 0,
    interval_ms: int = 40,
    title: str = "Three-body rollout",
    trail: int = 80,
):
    """Animate true and optional predicted three-body trajectories."""
    true_pos = positions_from_traj(traj_true)
    if true_pos.ndim == 4:
        true_pos = true_pos[:, traj_id]

    pred_pos = None
    if traj_pred is not None:
        pred_pos = positions_from_traj(traj_pred)
        if pred_pos.ndim == 4:
            pred_pos = pred_pos[:, traj_id]

    all_pos = true_pos if pred_pos is None else np.concatenate([true_pos, pred_pos], axis=0)
    xmin, ymin = all_pos.reshape(-1, 2).min(axis=0)
    xmax, ymax = all_pos.reshape(-1, 2).max(axis=0)
    pad = 0.15 * max(xmax - xmin, ymax - ymin, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    true_points = [ax.plot([], [], marker="o", linestyle="", markersize=7, label=f"true {i+1}")[0] for i in range(3)]
    true_trails = [ax.plot([], [], linewidth=1.5)[0] for _ in range(3)]

    pred_points = []
    pred_trails = []
    if pred_pos is not None:
        pred_points = [ax.plot([], [], marker="x", linestyle="", markersize=7, label=f"pred {i+1}")[0] for i in range(3)]
        pred_trails = [ax.plot([], [], linestyle="--", linewidth=1.2)[0] for _ in range(3)]

    ax.legend(loc="best", fontsize=8)
    frames = true_pos.shape[0]

    def update(i: int):
        start = max(0, i - trail)
        artists = []
        for b in range(3):
            true_points[b].set_data([true_pos[i, b, 0]], [true_pos[i, b, 1]])
            true_trails[b].set_data(true_pos[start : i + 1, b, 0], true_pos[start : i + 1, b, 1])
            artists.extend([true_points[b], true_trails[b]])
            if pred_pos is not None:
                pred_points[b].set_data([pred_pos[i, b, 0]], [pred_pos[i, b, 1]])
                pred_trails[b].set_data(pred_pos[start : i + 1, b, 0], pred_pos[start : i + 1, b, 1])
                artists.extend([pred_points[b], pred_trails[b]])
        fig.suptitle(f"{title} | frame={i}/{frames-1}")
        return artists

    return FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)


def save_gif(animation: FuncAnimation, path: str | Path, fps: int = 25) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=fps))
    return out
