"""Generate and validate a reference figure-eight three-body trajectory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_body import ThreeBodyConfig, ThreeBodySystem3D, figure_eight_state_3d
from three_body.data import generate_reference_figure_eight, save_trajectory_dataset
from three_body.eval import rmse_per_step
from three_body.system import integrate_rk4_fixed
from three_body.viz import animate_3d_comparison, plot_3d_trajectories, save_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--softening", type=float, default=0.0)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = REPO_ROOT / "data" / "three_body"
    results_dir = REPO_ROOT / "results" / "three_body" / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    system = ThreeBodySystem3D(ThreeBodyConfig(softening=args.softening))
    x0 = figure_eight_state_3d()

    print("generating solve_ivp/DOP853 reference...")
    traj_solve_ivp, t = generate_reference_figure_eight(system, x0, steps=args.steps, dt=args.dt, solver="solve_ivp")
    true_traj = traj_solve_ivp[0]
    print("generating fixed RK4 trajectory...")
    rk4_traj = integrate_rk4_fixed(system, x0, steps=args.steps, dt=args.dt)

    err_rk4 = rmse_per_step(system, true_traj, rk4_traj)
    energy_true = system.energy(true_traj)
    energy_rk4 = system.energy(rk4_traj)

    summary = {
        "system": system.metadata(),
        "steps": args.steps,
        "dt": args.dt,
        "rk4_final_rmse_vs_solve_ivp": float(err_rk4[-1]),
        "rk4_mean_rmse_vs_solve_ivp": float(err_rk4.mean()),
        "solve_ivp_energy_start": float(energy_true[0]),
        "solve_ivp_energy_end": float(energy_true[-1]),
        "rk4_energy_start": float(energy_rk4[0]),
        "rk4_energy_end": float(energy_rk4[-1]),
    }

    out_npz = data_dir / "figure8_reference_solve_ivp.npz"
    save_trajectory_dataset(out_npz, traj_solve_ivp, t, metadata={**summary, "solver": "solve_ivp_DOP853"})
    out_json = results_dir / "figure8_reference_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved:", out_npz)
    print("saved:", out_json)

    plt.figure(figsize=(8, 4))
    plt.plot(t, err_rk4)
    plt.xlabel("time")
    plt.ylabel("RMSE vs solve_ivp")
    plt.title("Fixed RK4 vs DOP853 reference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = results_dir / "figure8_rk4_vs_solve_ivp_rmse.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    fig, _ = plot_3d_trajectories(true_traj, rk4_traj, title="Figure-eight: solve_ivp vs fixed RK4")
    path = results_dir / "figure8_solve_ivp_vs_rk4_paths.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved:", path)

    if args.make_gif:
        anim = animate_3d_comparison(true_traj, rk4_traj, interval_ms=20, title="Figure-eight: solve_ivp vs RK4", trail=180)
        gif_path = results_dir / "figure8_solve_ivp_vs_rk4.gif"
        save_gif(anim, gif_path, fps=args.fps)
        print("saved:", gif_path)


if __name__ == "__main__":
    main()
