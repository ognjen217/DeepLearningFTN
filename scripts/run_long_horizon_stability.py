"""Run a long-horizon autoregressive stability test for the core ICNN experiment.

This script assumes that `notebooks/01_core_icnn_lyapunov_experiments.ipynb`
has already generated the three core checkpoints:

    checkpoints/core_p4_full_exp5_exp7_phase1_exp5_stable.pt
    checkpoints/core_p4_full_exp5_exp7_phase1_exp5_baseline.pt
    checkpoints/core_p4_full_exp5_exp7_exp7style_stable.pt

Example:

    python scripts/run_long_horizon_stability.py --steps 5000 --store-every 10
    python scripts/run_long_horizon_stability.py --steps 25000 --store-every 50

The test is empirical: it rolls learned models autoregressively for a long
finite horizon and records error/norm/V curves. It is not a formal proof of
global stability.
"""
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

from stable_icnn_physics import BaselineDynamicsMLP, build_stable_model, make_system
from stable_icnn_physics.eval import long_horizon_autoregressive_stability_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000, help="Number of autoregressive steps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Integration time step.")
    parser.add_argument("--store-every", type=int, default=10, help="Store metrics every N steps.")
    parser.add_argument("--eval-trajs", type=int, default=16, help="Number of test initial conditions.")
    parser.add_argument("--seed", type=int, default=123, help="Seed offset for test initial conditions.")
    parser.add_argument("--divergence-error", type=float, default=1_000.0)
    parser.add_argument("--divergence-norm", type=float, default=1_000.0)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto.")
    parser.add_argument("--tag", default="core_p4_full_exp5_exp7")
    return parser.parse_args()


def make_stable(state_dim: int):
    return build_stable_model(
        dim=state_dim,
        hidden=200,
        depth=3,
        lyapunov_hidden=100,
        lyapunov_eps=0.01,
        alpha=1e-5,
        rehu_width=0.01,
    )


def load_checkpoint_model(model, path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            "Run notebooks/01_core_icnn_lyapunov_experiments.ipynb first."
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model


def save_plots(result: dict, out_dir: Path, tag: str, steps: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for name, curve in result["curves"].items():
        plt.plot(curve["time"], curve["mean_error"], label=name)
    plt.xlabel("time [s]")
    plt.ylabel("mean autoregressive error")
    plt.title(f"Long-horizon autoregressive stability: {steps} steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"{tag}_long_horizon_mean_error.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    plt.figure(figsize=(10, 5))
    for name, curve in result["curves"].items():
        plt.semilogy(curve["time"], np.asarray(curve["mean_error"]) + 1e-12, label=name)
    plt.xlabel("time [s]")
    plt.ylabel("mean autoregressive error, log scale")
    plt.title(f"Long-horizon autoregressive stability: {steps} steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"{tag}_long_horizon_mean_error_log.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    plt.figure(figsize=(10, 5))
    for name, curve in result["curves"].items():
        plt.plot(curve["time"], curve["mean_state_norm"], label=name)
    plt.xlabel("time [s]")
    plt.ylabel("mean predicted state norm")
    plt.title("Long-horizon state norm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"{tag}_long_horizon_state_norm.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    stable_names = [name for name, curve in result["curves"].items() if not np.all(np.isnan(curve["V_mean"]))]
    if stable_names:
        plt.figure(figsize=(10, 5))
        for name in stable_names:
            curve = result["curves"][name]
            plt.plot(curve["time"], curve["V_mean"], label=f"{name} V_mean")
        plt.xlabel("time [s]")
        plt.ylabel("V(x)")
        plt.title("ICNN Lyapunov value over long rollout")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"{tag}_long_horizon_lyapunov_value.png"
        plt.savefig(path, dpi=160)
        plt.close()
        print("saved:", path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    system = make_system("damped_pendulum_4", friction=0.3, gravity=9.81)
    state_dim = system.state_dim
    x0 = system.sample_initial_conditions(args.eval_trajs, split="test", seed=args.seed)

    ckpt_dir = REPO_ROOT / "checkpoints"
    phase1_stable_ckpt = ckpt_dir / f"{args.tag}_phase1_exp5_stable.pt"
    baseline_ckpt = ckpt_dir / f"{args.tag}_phase1_exp5_baseline.pt"
    exp7_ckpt = ckpt_dir / f"{args.tag}_exp7style_stable.pt"

    phase1_stable = load_checkpoint_model(make_stable(state_dim), phase1_stable_ckpt, device)
    exp7_stable = load_checkpoint_model(make_stable(state_dim), exp7_ckpt, device)
    baseline = load_checkpoint_model(BaselineDynamicsMLP(dim=state_dim, hidden=200, depth=3), baseline_ckpt, device)

    models = {
        "baseline": baseline,
        "fhat_only": exp7_stable.fhat,
        "phase1_stable": phase1_stable,
        "exp7_stable": exp7_stable,
    }

    print(f"running long-horizon test: steps={args.steps}, dt={args.dt}, store_every={args.store_every}")
    result = long_horizon_autoregressive_stability_test(
        system=system,
        models=models,
        x0=x0,
        steps=args.steps,
        dt=args.dt,
        device=device,
        store_every=args.store_every,
        divergence_error=args.divergence_error,
        divergence_norm=args.divergence_norm,
    )

    result["experiment"] = args.tag
    result["system"] = system.metadata()
    result["eval_trajs"] = args.eval_trajs
    result["seed"] = args.seed

    out_dir = REPO_ROOT / "results" / "core_icnn_long_horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.tag}_long_horizon_steps{args.steps}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))
    print("saved:", out_json)

    save_plots(result, out_dir, f"{args.tag}_steps{args.steps}", args.steps)


if __name__ == "__main__":
    main()
