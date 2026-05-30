"""Benchmark standard three-body solvers against trained neural rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from three_body import ThreeBodyConfig, ThreeBodySystem3D, figure_eight_state_3d
from three_body.eval import benchmark_n_step
from three_body.models import build_baseline_model, build_stable_icnn_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lyapunov-hidden", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--softening", type=float, default=1e-3)
    parser.add_argument("--tag", default="three_body_figure8_icnn")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_model(model, path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run scripts/run_three_body_training.py first.")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state"])
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = REPO_ROOT / "checkpoints" / "three_body"
    results_dir = REPO_ROOT / "results" / "three_body"
    results_dir.mkdir(parents=True, exist_ok=True)

    system = ThreeBodySystem3D(ThreeBodyConfig(softening=args.softening))
    x0 = figure_eight_state_3d()

    baseline = load_model(
        build_baseline_model(hidden=args.hidden, depth=args.depth),
        ckpt_dir / f"{args.tag}_baseline.pt",
        device,
    )
    stable = load_model(
        build_stable_icnn_model(
            hidden=args.hidden,
            depth=args.depth,
            lyapunov_hidden=args.lyapunov_hidden,
            alpha=args.alpha,
        ),
        ckpt_dir / f"{args.tag}_exp7style_stable.pt",
        device,
    )

    models = {
        "baseline_nn": baseline,
        "fhat_only": stable.fhat,
        "icnn_projected": stable,
    }
    results = benchmark_n_step(system, x0, steps=args.steps, dt=args.dt, models=models, device=device)
    results["metadata"] = {
        "tag": args.tag,
        "steps": args.steps,
        "dt": args.dt,
        "device": str(device),
        "system": system.metadata(),
    }

    df = pd.DataFrame({k: v for k, v in results.items() if isinstance(v, dict) and k != "metadata"}).T
    print(df)

    out_json = results_dir / f"{args.tag}_realtime_benchmark_steps{args.steps}.json"
    out_csv = results_dir / f"{args.tag}_realtime_benchmark_steps{args.steps}.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    df.to_csv(out_csv)
    print("saved:", out_json)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
