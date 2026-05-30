from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def positions(traj: np.ndarray) -> np.ndarray:
    """Return positions with shape ``(..., 3, 3)`` from states with last dim 18."""
    arr = np.asarray(traj, dtype=np.float32)
    if arr.shape[-1] != 18:
        raise ValueError("trajectory/state last dimension must be 18")
    return arr[..., :9].reshape(*arr.shape[:-1], 3, 3)


def plot_3d_trajectories(
    true_traj: np.ndarray,
    pred_traj: np.ndarray | None = None,
    title: str = "Three-body 3D trajectories",
):
    true_pos = positions(true_traj)
    pred_pos = None if pred_traj is None else positions(pred_traj)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for body in range(3):
        ax.plot(true_pos[:, body, 0], true_pos[:, body, 1], true_pos[:, body, 2], label=f"true {body+1}")
        ax.scatter(true_pos[0, body, 0], true_pos[0, body, 1], true_pos[0, body, 2], marker="o", s=30)
        ax.scatter(true_pos[-1, body, 0], true_pos[-1, body, 1], true_pos[-1, body, 2], marker="x", s=40)
        if pred_pos is not None:
            ax.plot(
                pred_pos[:, body, 0],
                pred_pos[:, body, 1],
                pred_pos[:, body, 2],
                linestyle="--",
                label=f"pred {body+1}",
            )

    _set_equal_3d(ax, true_pos if pred_pos is None else np.concatenate([true_pos, pred_pos], axis=0))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=8)
    return fig, ax


def animate_3d_comparison(
    true_traj: np.ndarray,
    pred_traj: np.ndarray | None = None,
    interval_ms: int = 30,
    title: str = "Three-body rollout",
    trail: int = 120,
):
    true_pos = positions(true_traj)
    pred_pos = None if pred_traj is None else positions(pred_traj)
    all_pos = true_pos if pred_pos is None else np.concatenate([true_pos, pred_pos], axis=0)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    _set_equal_3d(ax, all_pos)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    true_points = [ax.plot([], [], [], marker="o", linestyle="", markersize=7, label=f"true {i+1}")[0] for i in range(3)]
    true_trails = [ax.plot([], [], [], linewidth=1.5)[0] for _ in range(3)]

    pred_points = []
    pred_trails = []
    if pred_pos is not None:
        pred_points = [ax.plot([], [], [], marker="x", linestyle="", markersize=7, label=f"pred {i+1}")[0] for i in range(3)]
        pred_trails = [ax.plot([], [], [], linestyle="--", linewidth=1.2)[0] for _ in range(3)]

    ax.legend(fontsize=8)
    frames = true_pos.shape[0]

    def update(i: int):
        start = max(0, i - trail)
        artists = []
        for b in range(3):
            true_points[b].set_data([true_pos[i, b, 0]], [true_pos[i, b, 1]])
            true_points[b].set_3d_properties([true_pos[i, b, 2]])
            true_trails[b].set_data(true_pos[start : i + 1, b, 0], true_pos[start : i + 1, b, 1])
            true_trails[b].set_3d_properties(true_pos[start : i + 1, b, 2])
            artists.extend([true_points[b], true_trails[b]])
            if pred_pos is not None:
                pred_points[b].set_data([pred_pos[i, b, 0]], [pred_pos[i, b, 1]])
                pred_points[b].set_3d_properties([pred_pos[i, b, 2]])
                pred_trails[b].set_data(pred_pos[start : i + 1, b, 0], pred_pos[start : i + 1, b, 1])
                pred_trails[b].set_3d_properties(pred_pos[start : i + 1, b, 2])
                artists.extend([pred_points[b], pred_trails[b]])
        fig.suptitle(f"{title} | frame={i}/{frames-1}")
        return artists

    return FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)


def save_gif(animation: FuncAnimation, path: str | Path, fps: int = 25) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=fps))
    return out


def _set_equal_3d(ax, xyz: np.ndarray) -> None:
    pts = np.asarray(xyz).reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * max(float(np.max(maxs - mins)), 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
