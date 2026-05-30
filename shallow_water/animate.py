from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from .swe2d import SWE2DConfig, compute_energy, compute_mass, compute_rmse, velocity_magnitude


def animate_eta_comparison(
    traj_true: np.ndarray,
    traj_pred: np.ndarray | None,
    cfg: SWE2DConfig,
    interval_ms: int = 40,
    title: str = "2D shallow-water rollout",
):
    """Create a matplotlib animation comparing true and predicted eta fields.

    If ``traj_pred`` is None, the animation shows the true eta field, velocity
    magnitude, and energy/mass curves. If a prediction is provided, panels show:

        true eta | predicted eta | absolute eta error
        rollout RMSE curve | energy curve | mass curve
    """
    true = np.asarray(traj_true)
    pred = None if traj_pred is None else np.asarray(traj_pred)

    if true.ndim != 4 or true.shape[1] != 3:
        raise ValueError("traj_true must have shape (frames, 3, ny, nx)")
    if pred is not None and pred.shape != true.shape:
        raise ValueError("traj_pred must have the same shape as traj_true")

    frames = true.shape[0]
    t = np.arange(frames) * cfg.dt

    eta_true = true[:, 0]
    vmax = float(max(np.max(np.abs(eta_true)), np.max(np.abs(pred[:, 0])) if pred is not None else 0.0))
    vmax = max(vmax, 1.0e-8)

    true_energy = np.array([compute_energy(s, cfg) for s in true])
    true_mass = np.array([compute_mass(s, cfg) for s in true])

    if pred is not None:
        pred_energy = np.array([compute_energy(s, cfg) for s in pred])
        pred_mass = np.array([compute_mass(s, cfg) for s in pred])
        rmse = compute_rmse(true, pred)
        err_max = float(np.max(np.abs(true[:, 0] - pred[:, 0])))
        err_max = max(err_max, 1.0e-8)

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        ax_true, ax_pred, ax_err = axes[0]
        ax_rmse, ax_energy, ax_mass = axes[1]

        im_true = ax_true.imshow(true[0, 0], origin="lower", vmin=-vmax, vmax=vmax)
        im_pred = ax_pred.imshow(pred[0, 0], origin="lower", vmin=-vmax, vmax=vmax)
        im_err = ax_err.imshow(np.abs(true[0, 0] - pred[0, 0]), origin="lower", vmin=0.0, vmax=err_max)

        ax_true.set_title("true eta")
        ax_pred.set_title("predicted eta")
        ax_err.set_title("absolute eta error")
        fig.colorbar(im_true, ax=ax_true, fraction=0.046)
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046)
        fig.colorbar(im_err, ax=ax_err, fraction=0.046)

        ax_rmse.set_title("rollout RMSE")
        ax_rmse.set_xlabel("time [s]")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_xlim(t[0], t[-1])
        ax_rmse.set_ylim(0.0, max(float(rmse.max()) * 1.05, 1.0e-8))
        rmse_line, = ax_rmse.plot([], [])

        ax_energy.set_title("energy")
        ax_energy.set_xlabel("time [s]")
        ax_energy.set_xlim(t[0], t[-1])
        emin = min(true_energy.min(), pred_energy.min())
        emax = max(true_energy.max(), pred_energy.max())
        ax_energy.set_ylim(emin * 0.95, emax * 1.05 + 1.0e-12)
        e_true_line, = ax_energy.plot([], [], label="true")
        e_pred_line, = ax_energy.plot([], [], label="pred")
        ax_energy.legend(loc="best")

        ax_mass.set_title("mass")
        ax_mass.set_xlabel("time [s]")
        ax_mass.set_xlim(t[0], t[-1])
        mmin = min(true_mass.min(), pred_mass.min())
        mmax = max(true_mass.max(), pred_mass.max())
        pad = max(abs(mmax - mmin) * 0.10, 1.0e-8)
        ax_mass.set_ylim(mmin - pad, mmax + pad)
        m_true_line, = ax_mass.plot([], [], label="true")
        m_pred_line, = ax_mass.plot([], [], label="pred")
        ax_mass.legend(loc="best")

        for ax in axes.ravel():
            ax.grid(False)

        def update(i: int):
            im_true.set_data(true[i, 0])
            im_pred.set_data(pred[i, 0])
            im_err.set_data(np.abs(true[i, 0] - pred[i, 0]))
            rmse_line.set_data(t[: i + 1], rmse[: i + 1])
            e_true_line.set_data(t[: i + 1], true_energy[: i + 1])
            e_pred_line.set_data(t[: i + 1], pred_energy[: i + 1])
            m_true_line.set_data(t[: i + 1], true_mass[: i + 1])
            m_pred_line.set_data(t[: i + 1], pred_mass[: i + 1])
            fig.suptitle(f"{title} | frame={i}/{frames-1} | t={t[i]:.3f}s | RMSE={rmse[i]:.4g}")
            return (
                im_true,
                im_pred,
                im_err,
                rmse_line,
                e_true_line,
                e_pred_line,
                m_true_line,
                m_pred_line,
            )

        return FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax_eta, ax_vel, ax_energy, ax_mass = axes.ravel()
    velmag = np.array([velocity_magnitude(s) for s in true])
    vel_vmax = max(float(velmag.max()), 1.0e-8)

    im_eta = ax_eta.imshow(true[0, 0], origin="lower", vmin=-vmax, vmax=vmax)
    im_vel = ax_vel.imshow(velmag[0], origin="lower", vmin=0.0, vmax=vel_vmax)
    fig.colorbar(im_eta, ax=ax_eta, fraction=0.046)
    fig.colorbar(im_vel, ax=ax_vel, fraction=0.046)
    ax_eta.set_title("eta")
    ax_vel.set_title("velocity magnitude")

    ax_energy.set_title("energy")
    ax_energy.set_xlabel("time [s]")
    ax_energy.set_xlim(t[0], t[-1])
    ax_energy.set_ylim(true_energy.min() * 0.95, true_energy.max() * 1.05 + 1.0e-12)
    e_line, = ax_energy.plot([], [])

    ax_mass.set_title("mass")
    ax_mass.set_xlabel("time [s]")
    ax_mass.set_xlim(t[0], t[-1])
    pad = max(abs(true_mass.max() - true_mass.min()) * 0.10, 1.0e-8)
    ax_mass.set_ylim(true_mass.min() - pad, true_mass.max() + pad)
    m_line, = ax_mass.plot([], [])

    def update(i: int):
        im_eta.set_data(true[i, 0])
        im_vel.set_data(velmag[i])
        e_line.set_data(t[: i + 1], true_energy[: i + 1])
        m_line.set_data(t[: i + 1], true_mass[: i + 1])
        fig.suptitle(f"{title} | frame={i}/{frames-1} | t={t[i]:.3f}s")
        return im_eta, im_vel, e_line, m_line

    return FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)


def save_gif(animation: FuncAnimation, path: str | Path, fps: int = 25) -> Path:
    """Save a matplotlib animation as a GIF using Pillow."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=fps))
    return out
