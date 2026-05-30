"""Train baseline and ICNN dynamics on perturbed three-body figure-eight data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stable_icnn_physics.eval import autoregressive_rollout_model
from stable_icnn_physics.rollout_augmented import projection_diagnostics
from stable_icnn_physics.train import evaluate_derivative_mse
from three_body import ThreeBodyConfig, ThreeBodySystem3D
from three_body.data import (
    dataset_path,
    generate_trajectory_dataset,
    load_trajectory_dataset,
    save_trajectory_dataset,
    trajectory_pairs_to_derivatives,
)
from three_body.eval import rmse_per_step, summarize_rollout
from three_body.models import build_baseline_model, build_stable_icnn_model
from three_body.train import make_derivative_dataset_from_trajectories, train_exp7_style_v_only, train_phase1_models
from three_body.viz import animate_3d_comparison, plot_3d_trajectories, save_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small smoke-test configuration.")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--train-trajs", type=int, default=96)
    parser.add_argument("--val-trajs", type=int, default=16)
    parser.add_argument("--test-trajs", type=int, default=8)
    parser.add_argument("--phase1-epochs", type=int, default=150)
    parser.add_argument("--v-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lyapunov-hidden", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--softening", type=float, default=1e-3)
    parser.add_argument("--noise-pos", type=float, default=0.02)
    parser.add_argument("--noise-vel", type=float, default=0.02)
    parser.add_argument("--noise-z", type=float, default=0.0)
    parser.add_argument("--rollout-train", type=int, default=128)
    parser.add_argument("--rollout-val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default="three_body_figure8_icnn")
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.steps = 200
        args.train_trajs = 16
        args.val_trajs = 4
        args.test_trajs = 2
        args.phase1_epochs = 20
        args.v_epochs = 20
        args.batch_size = 256
        args.rollout_train = 16
        args.rollout_val = 4
        args.tag = args.tag + "_quick"
    return args


def load_or_generate(system: ThreeBodySystem3D, path: Path, n_trajs: int, split: str, args: argparse.Namespace):
    if path.exists():
        traj, t, _ = load_trajectory_dataset(path)
        print("loaded", path)
        return traj, t
    traj, t = generate_trajectory_dataset(system, n_trajs, args.steps, args.dt, split=split, seed=args.seed, solver="rk4")
    meta = {"system": system.metadata(), "split": split, "dt": args.dt, "steps": args.steps, "solver": "rk4"}
    save_trajectory_dataset(path, traj, t, meta)
    print("saved", path)
    return traj, t


def main() -> None:
    args = apply_quick(parse_args())
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")
    print("device:", device)
    print("args:", vars(args))

    data_dir = REPO_ROOT / "data" / "three_body"
    ckpt_dir = REPO_ROOT / "checkpoints" / "three_body"
    results_dir = REPO_ROOT / "results" / "three_body"
    plots_dir = results_dir / "plots"
    for d in [data_dir, ckpt_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = ThreeBodyConfig(
        softening=args.softening,
        sample_noise_pos=args.noise_pos,
        sample_noise_vel=args.noise_vel,
        sample_z_noise=args.noise_z,
    )
    system = ThreeBodySystem3D(cfg)

    train_path = dataset_path(data_dir, "train_perturbed", args.train_trajs, args.steps, args.dt, args.seed, "rk4")
    val_path = dataset_path(data_dir, "val_perturbed", args.val_trajs, args.steps, args.dt, args.seed, "rk4")
    test_path = dataset_path(data_dir, "test_perturbed", args.test_trajs, args.steps, args.dt, args.seed, "rk4")

    train_traj, t = load_or_generate(system, train_path, args.train_trajs, "train", args)
    val_traj, _ = load_or_generate(system, val_path, args.val_trajs, "val", args)
    test_traj, _ = load_or_generate(system, test_path, args.test_trajs, "test", args)

    train_ds = make_derivative_dataset_from_trajectories(train_traj, args.dt)
    val_ds = make_derivative_dataset_from_trajectories(val_traj, args.dt)
    test_ds = make_derivative_dataset_from_trajectories(test_traj, args.dt)
    x_train_states, _ = trajectory_pairs_to_derivatives(train_traj, args.dt)
    x_val_states, _ = trajectory_pairs_to_derivatives(val_traj, args.dt)

    stable_ckpt = ckpt_dir / f"{args.tag}_phase1_stable.pt"
    baseline_ckpt = ckpt_dir / f"{args.tag}_baseline.pt"
    exp7_ckpt = ckpt_dir / f"{args.tag}_exp7style_stable.pt"

    if args.force_retrain or not (stable_ckpt.exists() and baseline_ckpt.exists()):
        stable_phase1, baseline = train_phase1_models(
            train_ds,
            val_ds,
            stable_ckpt,
            baseline_ckpt,
            hidden=args.hidden,
            depth=args.depth,
            lyapunov_hidden=args.lyapunov_hidden,
            alpha=args.alpha,
            epochs=args.phase1_epochs,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        stable_phase1 = build_stable_icnn_model(hidden=args.hidden, depth=args.depth, lyapunov_hidden=args.lyapunov_hidden, alpha=args.alpha)
        baseline = build_baseline_model(hidden=args.hidden, depth=args.depth)
        stable_phase1.load_state_dict(torch.load(stable_ckpt, map_location=device, weights_only=True)["model_state"])
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True)["model_state"])
        stable_phase1.to(device).eval()
        baseline.to(device).eval()
        print("loaded phase-1 checkpoints")

    if args.force_retrain or not exp7_ckpt.exists():
        stable_exp7 = build_stable_icnn_model(hidden=args.hidden, depth=args.depth, lyapunov_hidden=args.lyapunov_hidden, alpha=args.alpha)
        stable_exp7.load_state_dict(torch.load(stable_ckpt, map_location=device, weights_only=True)["model_state"])
        stable_exp7.to(device).eval()
        stable_exp7, history_v, _ = train_exp7_style_v_only(
            stable_exp7,
            system,
            x_train_states,
            x_val_states,
            args.dt,
            args.steps,
            n_rollout_train=args.rollout_train,
            n_rollout_val=args.rollout_val,
            epochs=args.v_epochs,
            batch_size=args.batch_size,
            device=device,
        )
        torch.save({"model_state": stable_exp7.state_dict(), "history": history_v.__dict__}, exp7_ckpt)
    else:
        stable_exp7 = build_stable_icnn_model(hidden=args.hidden, depth=args.depth, lyapunov_hidden=args.lyapunov_hidden, alpha=args.alpha)
        stable_exp7.load_state_dict(torch.load(exp7_ckpt, map_location=device, weights_only=True)["model_state"])
        stable_exp7.to(device).eval()
        print("loaded Exp7-style checkpoint")

    true_traj = test_traj[0]
    x0 = true_traj[0:1]
    baseline_traj = autoregressive_rollout_model(baseline, x0, args.steps, args.dt, device=device)[:, 0]
    fhat_traj = autoregressive_rollout_model(stable_exp7.fhat, x0, args.steps, args.dt, device=device)[:, 0]
    phase1_traj = autoregressive_rollout_model(stable_phase1, x0, args.steps, args.dt, device=device)[:, 0]
    exp7_traj = autoregressive_rollout_model(stable_exp7, x0, args.steps, args.dt, device=device)[:, 0]

    summary = {
        "experiment": args.tag,
        "args": vars(args),
        "system": system.metadata(),
        "derivative_mse": {
            "baseline": float(evaluate_derivative_mse(baseline, test_ds, device=device)),
            "fhat": float(evaluate_derivative_mse(stable_exp7.fhat, test_ds, device=device)),
            "phase1_stable": float(evaluate_derivative_mse(stable_phase1, test_ds, device=device)),
            "exp7_stable": float(evaluate_derivative_mse(stable_exp7, test_ds, device=device)),
        },
        "rollout": {
            "baseline": summarize_rollout(system, true_traj, baseline_traj).as_dict(),
            "fhat": summarize_rollout(system, true_traj, fhat_traj).as_dict(),
            "phase1_stable": summarize_rollout(system, true_traj, phase1_traj).as_dict(),
            "exp7_stable": summarize_rollout(system, true_traj, exp7_traj).as_dict(),
        },
        "projection_exp7": {
            k: v for k, v in projection_diagnostics(stable_exp7, exp7_traj[:, None], device=device).items()
            if k not in {"fire", "correction_norm", "violation", "V", "dV"}
        },
    }

    summary_path = results_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved:", summary_path)

    errors = {
        "baseline": rmse_per_step(system, true_traj, baseline_traj),
        "fhat only": rmse_per_step(system, true_traj, fhat_traj),
        "phase1 stable": rmse_per_step(system, true_traj, phase1_traj),
        "Exp7-style stable": rmse_per_step(system, true_traj, exp7_traj),
    }
    plt.figure(figsize=(9, 5))
    for name, err in errors.items():
        plt.plot(t, err, label=name)
    plt.xlabel("time")
    plt.ylabel("RMSE")
    plt.title("Three-body autoregressive rollout error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = plots_dir / f"{args.tag}_rollout_rmse.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    fig, _ = plot_3d_trajectories(true_traj, exp7_traj, title="Solver vs Exp7-style ICNN")
    path = plots_dir / f"{args.tag}_solver_vs_exp7_paths.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved:", path)

    if not args.no_gif:
        anim = animate_3d_comparison(true_traj, exp7_traj, interval_ms=20, title="Solver vs Exp7-style ICNN", trail=150)
        gif_path = results_dir / f"{args.tag}_solver_vs_exp7.gif"
        save_gif(anim, gif_path, fps=30)
        print("saved:", gif_path)


if __name__ == "__main__":
    main()
