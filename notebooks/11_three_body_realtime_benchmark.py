"""Real-time N-step benchmark for three-body solver vs neural rollouts."""
from __future__ import annotations

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = REPO_ROOT / "checkpoints" / "three_body"
RESULTS_DIR = REPO_ROOT / "results" / "three_body"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAG = "three_body_figure8_icnn"
DT = 0.005
STEPS = 1000
HIDDEN = 256
DEPTH = 3
LYAPUNOV_HIDDEN = 128
ALPHA = 1e-6

system = ThreeBodySystem3D(ThreeBodyConfig(softening=1e-3))
x0 = figure_eight_state_3d()

baseline = build_baseline_model(hidden=HIDDEN, depth=DEPTH)
stable = build_stable_icnn_model(hidden=HIDDEN, depth=DEPTH, lyapunov_hidden=LYAPUNOV_HIDDEN, alpha=ALPHA)

baseline.load_state_dict(torch.load(CKPT_DIR / f"{TAG}_baseline.pt", map_location=DEVICE, weights_only=True)["model_state"])
stable.load_state_dict(torch.load(CKPT_DIR / f"{TAG}_exp7style_stable.pt", map_location=DEVICE, weights_only=True)["model_state"])
baseline.to(DEVICE).eval()
stable.to(DEVICE).eval()

models = {
    "baseline_nn": baseline,
    "fhat_only": stable.fhat,
    "icnn_projected": stable,
}

results = benchmark_n_step(system, x0, steps=STEPS, dt=DT, models=models, device=DEVICE)
df = pd.DataFrame(results).T
print(df)

out_json = RESULTS_DIR / f"{TAG}_realtime_benchmark.json"
out_csv = RESULTS_DIR / f"{TAG}_realtime_benchmark.csv"
out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
df.to_csv(out_csv)
print("saved:", out_json)
print("saved:", out_csv)
