"""Three-body ICNN training pipeline.

Run from repository root:

    python notebooks/10_three_body_icnn_training.py

The script generates perturbed figure-eight trajectories, trains a baseline MLP
and ICNN-projected dynamics model, then performs Exp7-style rollout-augmented
V-only training and saves rollout plots/summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stable_icnn_physics.eval import autoregressive_rollout_model
from stable_icnn_physics.train import evaluate_derivative_mse
from stable_icnn_physics.rollout_augmented import projection_diagnostics
from three_body import ThreeBodyConfig, ThreeBodySystem3D
from three_body.data import dataset_path, generate_trajectory_dataset, load_trajectory_dataset, save_trajectory_dataset, trajectory_pairs_to_derivatives
from three_body.eval import rmse_per_step, summarize_rollout
from three_body.models import build_baseline_model, build_stable_icnn_model
from three_body.train import make_derivative_dataset_from_trajectories, train_exp7_style_v_only, train_phase1_models
from three_body.viz import animate_3d_comparison, plot_3d_trajectories, save_gif

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

DATA_DIR = REPO_ROOT / "data" / "three_body"
CKPT_DIR = REPO_ROOT / "checkpoints" / "three_body"
RESULTS_DIR = REPO_ROOT / "results" / "three_body"
PLOTS_DIR = RESULTS_DIR / "plots"
for d in [DATA_DIR, CKPT_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 0
TAG = "three_body_figure8_icnn"
DT = 0.005
STEPS = 700
TRAIN_TRAJS = 96
VAL_TRAJS = 16
TEST_TRAJS = 8
HIDDEN = 256
DEPTH = 3
LYAPUNOV_HIDDEN = 128
ALPHA = 1e-6
PHASE1_EPOCHS = 150
V_ONLY_EPOCHS = 250
BATCH_SIZE = 512

cfg = ThreeBodyConfig(softening=1e-3, sample_noise_pos=0.02, sample_noise_vel=0.02, sample_z_noise=0.0)
system = ThreeBodySystem3D(cfg)


def load_or_generate(path: Path, n: int, split: str):
    if path.exists():
        traj, t, _ = load_trajectory_dataset(path)
        print("loaded", path)
        return traj, t
    traj, t = generate_trajectory_dataset(system, n, STEPS, DT, split=split, seed=SEED, solver="rk4")
    meta = {"system": system.metadata(), "split": split, "dt": DT, "steps": STEPS, "solver": "rk4"}
    save_trajectory_dataset(path, traj, t, meta)
    print("saved", path)
    return traj, t


def main():
    train_path = dataset_path(DATA_DIR, "train_perturbed", TRAIN_TRAJS, STEPS, DT, SEED, "rk4")
    val_path = dataset_path(DATA_DIR, "val_perturbed", VAL_TRAJS, STEPS, DT, SEED, "rk4")
    test_path = dataset_path(DATA_DIR, "test_perturbed", TEST_TRAJS, STEPS, DT, SEED, "rk4")
    train_traj, t = load_or_generate(train_path, TRAIN_TRAJS, "train")
    val_traj, _ = load_or_generate(val_path, VAL_TRAJS, "val")
    test_traj, _ = load_or_generate(test_path, TEST_TRAJS, "test")

    train_ds = make_derivative_dataset_from_trajectories(train_traj, DT)
    val_ds = make_derivative_dataset_from_trajectories(val_traj, DT)
    test_ds = make_derivative_dataset_from_trajectories(test_traj, DT)
    x_train_states, _ = trajectory_pairs_to_derivatives(train_traj, DT)
    x_val_states, _ = trajectory_pairs_to_derivatives(val_traj, DT)

    stable_ckpt = CKPT_DIR / f"{TAG}_phase1_stable.pt"
    baseline_ckpt = CKPT_DIR / f"{TAG}_baseline.pt"
    exp7_ckpt = CKPT_DIR / f"{TAG}_exp7style_stable.pt"

    if stable_ckpt.exists() and baseline_ckpt.exists():
        stable_phase1 = build_stable_icnn_model(hidden=HIDDEN, depth=DEPTH, lyapunov_hidden=LYAPUNOV_HIDDEN, alpha=ALPHA)
        baseline = build_baseline_model(hidden=HIDDEN, depth=DEPTH)
        stable_phase1.load_state_dict(torch.load(stable_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
        stable_phase1.to(DEVICE).eval(); baseline.to(DEVICE).eval()
    else:
        stable_phase1, baseline = train_phase1_models(train_ds, val_ds, stable_ckpt, baseline_ckpt, HIDDEN, DEPTH, LYAPUNOV_HIDDEN, ALPHA, PHASE1_EPOCHS, BATCH_SIZE, device=DEVICE)

    if exp7_ckpt.exists():
        stable_exp7 = build_stable_icnn_model(hidden=HIDDEN, depth=DEPTH, lyapunov_hidden=LYAPUNOV_HIDDEN, alpha=ALPHA)
        stable_exp7.load_state_dict(torch.load(exp7_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
        stable_exp7.to(DEVICE).eval()
    else:
        stable_exp7 = build_stable_icnn_model(hidden=HIDDEN, depth=DEPTH, lyapunov_hidden=LYAPUNOV_HIDDEN, alpha=ALPHA)
        stable_exp7.load_state_dict(torch.load(stable_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
        stable_exp7.to(DEVICE).eval()
        stable_exp7, history_v, _ = train_exp7_style_v_only(stable_exp7, system, x_train_states, x_val_states, DT, STEPS, 128, 32, V_ONLY_EPOCHS, BATCH_SIZE, DEVICE)
        torch.save({"model_state": stable_exp7.state_dict(), "history": history_v.__dict__}, exp7_ckpt)

    true_traj = test_traj[0]
    x0 = true_traj[0:1]
    baseline_traj = autoregressive_rollout_model(baseline, x0, STEPS, DT, device=DEVICE)[:, 0]
    fhat_traj = autoregressive_rollout_model(stable_exp7.fhat, x0, STEPS, DT, device=DEVICE)[:, 0]
    phase1_traj = autoregressive_rollout_model(stable_phase1, x0, STEPS, DT, device=DEVICE)[:, 0]
    exp7_traj = autoregressive_rollout_model(stable_exp7, x0, STEPS, DT, device=DEVICE)[:, 0]

    summary = {
        "experiment": TAG,
        "system": system.metadata(),
        "dt": DT,
        "steps": STEPS,
        "derivative_mse": {
            "baseline": evaluate_derivative_mse(baseline, test_ds, device=DEVICE),
            "fhat": evaluate_derivative_mse(stable_exp7.fhat, test_ds, device=DEVICE),
            "phase1_stable": evaluate_derivative_mse(stable_phase1, test_ds, device=DEVICE),
            "exp7_stable": evaluate_derivative_mse(stable_exp7, test_ds, device=DEVICE),
        },
        "rollout": {
            "baseline": summarize_rollout(system, true_traj, baseline_traj).as_dict(),
            "fhat": summarize_rollout(system, true_traj, fhat_traj).as_dict(),
            "phase1_stable": summarize_rollout(system, true_traj, phase1_traj).as_dict(),
            "exp7_stable": summarize_rollout(system, true_traj, exp7_traj).as_dict(),
        },
        "projection_exp7": {k: v for k, v in projection_diagnostics(stable_exp7, exp7_traj[:, None], device=DEVICE).items() if k not in {"fire", "correction_norm", "violation", "V", "dV"}},
    }
    (RESULTS_DIR / f"{TAG}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    err_baseline = rmse_per_step(system, true_traj, baseline_traj)
    err_fhat = rmse_per_step(system, true_traj, fhat_traj)
    err_phase1 = rmse_per_step(system, true_traj, phase1_traj)
    err_exp7 = rmse_per_step(system, true_traj, exp7_traj)
    plt.figure(figsize=(9, 5))
    plt.plot(t, err_baseline, label="baseline")
    plt.plot(t, err_fhat, label="fhat only")
    plt.plot(t, err_phase1, label="phase1 stable")
    plt.plot(t, err_exp7, label="Exp7-style stable")
    plt.xlabel("time"); plt.ylabel("RMSE"); plt.title("Three-body autoregressive rollout error")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{TAG}_rollout_rmse.png", dpi=160)

    fig, _ = plot_3d_trajectories(true_traj, exp7_traj, title="Solver vs Exp7-style ICNN")
    fig.savefig(PLOTS_DIR / f"{TAG}_solver_vs_exp7_paths.png", dpi=160)
    anim = animate_3d_comparison(true_traj, exp7_traj, interval_ms=20, title="Solver vs Exp7-style ICNN", trail=150)
    save_gif(anim, RESULTS_DIR / f"{TAG}_solver_vs_exp7.gif", fps=30)


if __name__ == "__main__":
    main()
