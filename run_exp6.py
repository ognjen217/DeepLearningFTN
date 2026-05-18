"""
Experiment 6: two-phase training.

Diagnosis from diagnose_fhat.py:
- fhat rollout = 0.577  (excellent, better than baseline 1.684)
- stable rollout = 377.9 (projection fires 91.62% of steps, max magnitude 8.063)
- Root cause: V has been learned implicitly via MSE on f=fhat-correction.
  This does NOT force grad_V to point in the right direction — it only
  makes corrections small at training points.

Fix: two-phase training.
  Phase 1: Joint training (already done in exp5) → gives good fhat.
  Phase 2: Freeze fhat, train V with explicit Lyapunov violation loss:
           min_V  E_x [ relu(grad_V(x)·fhat(x) + alpha*V(x))^2 ]
  This directly forces grad_V to be a valid Lyapunov descent direction for fhat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from stable_icnn_physics import build_stable_model, BaselineDynamicsMLP, make_system
from stable_icnn_physics.data import dataset_base_name, load_dataset, generate_derivative_data, save_dataset, tensor_dataset
from stable_icnn_physics.eval import (
    autoregressive_rollout_model,
    lyapunov_decrease_values,
    rollout_error,
    rollout_system,
)
from stable_icnn_physics.train import evaluate_derivative_mse, train_lyapunov_only

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR   = REPO_ROOT / "data" / "cache"
CKPT_DIR    = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
TOLERANCE   = 1e-5
SEED        = 0

print(f"Device: {DEVICE}  |  torch {torch.__version__}")

P4_SYSTEM = "damped_pendulum_4"
P4_KWARGS = {"friction": 0.3, "gravity": 9.81}
P4_DT     = 0.02
P4_STEPS  = 300

p4r = make_system(P4_SYSTEM, **P4_KWARGS)
state_dim = p4r.state_dim

# ── Load existing random p4 data ──────────────────────────────────────────────
for split, n in [("train", 50_000), ("test", 10_000)]:
    path = CACHE_DIR / dataset_base_name(p4r, split=split, n_samples=n, seed=SEED, dataset_type="derivative")
    if not path.exists():
        print(f"  generating {split} data ...")
        x, y = generate_derivative_data(p4r, n_samples=n, split=split, seed=SEED)
        save_dataset(path, x, y)
    else:
        print(f"  reusing  {split}: {path.name}")

p4r_train_path = CACHE_DIR / dataset_base_name(p4r, split="train", n_samples=50_000, seed=SEED, dataset_type="derivative")
p4r_test_path  = CACHE_DIR / dataset_base_name(p4r, split="test",  n_samples=10_000, seed=SEED, dataset_type="derivative")
x_p4r_train, _ = load_dataset(p4r_train_path)
x_p4r_test, y_p4r_test = load_dataset(p4r_test_path)
p4r_train_ds = tensor_dataset(*load_dataset(p4r_train_path))
p4r_test_ds  = tensor_dataset(*load_dataset(p4r_test_path))
print(f"  train: {x_p4r_train.shape}  test: {x_p4r_test.shape}")

x0_p4   = p4r.sample_initial_conditions(16, split="test", seed=SEED + 123)
true_p4 = rollout_system(p4r, x0_p4, steps=P4_STEPS, dt=P4_DT)

# ── Load phase-1 checkpoint (exp5: large model, alpha=1e-5) ──────────────────
print("\n[Phase 1] Loading exp5 checkpoint (fhat already trained) ...")
phase1_ckpt = CKPT_DIR / "p4_random_large_e500_alpha1e5_stable.pt"
if not phase1_ckpt.exists():
    raise FileNotFoundError(f"Run run_exp5.py first: {phase1_ckpt}")

stable = build_stable_model(
    dim=state_dim, hidden=200, depth=3,
    lyapunov_hidden=100, lyapunov_eps=0.01,
    alpha=1e-5, rehu_width=0.01,
)
stable.load_state_dict(torch.load(phase1_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
stable.to(DEVICE)

mse_before = evaluate_derivative_mse(stable, p4r_test_ds, device=DEVICE)
print(f"  Loaded. Derivative MSE (full stable model): {mse_before:.4g}")

# ── Phase 2: train V only ─────────────────────────────────────────────────────
print("\n[Phase 2] Freeze fhat, train V with Lyapunov violation loss ...")
phase2_ckpt = CKPT_DIR / "p4_twophase_stable.pt"
train_lyapunov_only(
    stable, p4r_train_ds,
    epochs=300,
    batch_size=256,
    learning_rate=1e-3,
    device=DEVICE,
    checkpoint_path=phase2_ckpt,
    print_every=30,
)

# Reload from checkpoint (clean state)
stable2 = build_stable_model(
    dim=state_dim, hidden=200, depth=3,
    lyapunov_hidden=100, lyapunov_eps=0.01,
    alpha=1e-5, rehu_width=0.01,
)
stable2.load_state_dict(torch.load(phase2_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
stable2.to(DEVICE).eval()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[Eval] Rolling out ...")
wrap = p4r.wrap_state

traj_stable2  = autoregressive_rollout_model(stable2, x0_p4, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)
traj_fhat     = autoregressive_rollout_model(stable2.fhat, x0_p4, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)

err_stable2   = rollout_error(p4r, true_p4, traj_stable2).mean(axis=1)
err_fhat      = rollout_error(p4r, true_p4, traj_fhat).mean(axis=1)

mse_after = evaluate_derivative_mse(stable2, p4r_test_ds, device=DEVICE)

decrease   = lyapunov_decrease_values(stable2, x_p4r_test[:2048], device=DEVICE).ravel()
max_viol   = float(decrease.max())
frac_sat   = float(np.mean(decrease <= TOLERANCE))

# Projection fire rate on rollout states
def projection_fire_rate(model, traj):
    fires, total = 0, 0
    with torch.enable_grad():
        for t in range(traj.shape[0] - 1):
            x = torch.tensor(traj[t], dtype=torch.float32, device=DEVICE).requires_grad_(True)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            fx = model.fhat(x)
            vx = model.V(x)
            grad_v = torch.autograd.grad(vx.sum(), x, create_graph=False)[0]
            violation = (grad_v * fx).sum(dim=1) + model.alpha * vx.squeeze(1)
            fires += (violation > 0).sum().item()
            total += x.shape[0]
    return fires / max(total, 1)

fire_rate = projection_fire_rate(stable2, traj_stable2)

print("\n" + "="*60)
print("EXPERIMENT 6: TWO-PHASE TRAINING")
print("="*60)
print(f"\nDerivative MSE after phase 2: {mse_after:.4g}  (before: {mse_before:.4g})")
print(f"\nRollout error (final step):")
print(f"  stable (two-phase):  {err_stable2[-1]:.4g}")
print(f"  fhat only:           {err_fhat[-1]:.4g}")
print(f"  [ref] exp5 stable:   377.9   baseline: 1.684")
print(f"\nRollout error (mean):")
print(f"  stable (two-phase):  {err_stable2.mean():.4g}")
print(f"  fhat only:           {err_fhat.mean():.4g}")
print(f"\nLyapunov:")
print(f"  max_violation={max_viol:.4g}   fraction_satisfied={frac_sat:.4f}")
print(f"\nProjection fire rate during rollout: {fire_rate:.4f} ({fire_rate*100:.2f}%)")
print(f"  [ref] exp5: 91.62%")

summary = {
    "experiment": "p4_twophase",
    "system": P4_SYSTEM,
    "alpha": 1e-5,
    "phase1_tag": "p4_random_large_e500_alpha1e5",
    "phase2_epochs": 300,
    "derivative_mse_stable": float(mse_after),
    "final_rollout_error_stable": float(err_stable2[-1]),
    "final_rollout_error_fhat":   float(err_fhat[-1]),
    "mean_rollout_error_stable":  float(err_stable2.mean()),
    "mean_rollout_error_fhat":    float(err_fhat.mean()),
    "lyapunov_max_violation":     max_viol,
    "lyapunov_fraction_satisfied": frac_sat,
    "projection_fire_rate":        fire_rate,
}
(RESULTS_DIR / "p4_twophase_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n  saved → {RESULTS_DIR / 'p4_twophase_summary.json'}")
