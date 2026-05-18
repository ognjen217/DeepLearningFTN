"""
Diagnostic: roll out stable model's fhat WITHOUT the Lyapunov projection.

If fhat rollout ≈ baseline rollout (~1.7):
  → projection is 100% responsible; fhat itself is fine.
If fhat rollout >> baseline rollout:
  → Lyapunov constraint during training is distorting fhat.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from stable_icnn_physics import build_stable_model, BaselineDynamicsMLP, make_system
from stable_icnn_physics.data import dataset_base_name, load_dataset, tensor_dataset
from stable_icnn_physics.eval import (
    autoregressive_rollout_model,
    lyapunov_decrease_values,
    rollout_error,
    rollout_system,
)
from stable_icnn_physics.train import evaluate_derivative_mse

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CKPT_DIR  = REPO_ROOT / "checkpoints"
SEED = 0

print(f"Device: {DEVICE}")

P4_SYSTEM = "damped_pendulum_4"
P4_KWARGS = {"friction": 0.3, "gravity": 9.81}
P4_DT     = 0.02
P4_STEPS  = 300

system = make_system(P4_SYSTEM, **P4_KWARGS)
state_dim = system.state_dim

# ── Load exp5 checkpoints ─────────────────────────────────────────────────────
TAG = "p4_random_large_e500_alpha1e5"
stable_ckpt   = CKPT_DIR / f"{TAG}_stable.pt"
baseline_ckpt = CKPT_DIR / f"{TAG}_baseline.pt"

stable = build_stable_model(
    dim=state_dim, hidden=200, depth=3,
    lyapunov_hidden=100, lyapunov_eps=0.01,
    alpha=1e-5, rehu_width=0.01,
)
baseline = BaselineDynamicsMLP(dim=state_dim, hidden=200, depth=3)

stable.load_state_dict(torch.load(stable_ckpt,   map_location=DEVICE, weights_only=True)["model_state"])
baseline.load_state_dict(torch.load(baseline_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
stable.to(DEVICE).eval()
baseline.to(DEVICE).eval()

# ── Load test data ────────────────────────────────────────────────────────────
p4r = make_system(P4_SYSTEM, **P4_KWARGS)
p4r_test_path = CACHE_DIR / dataset_base_name(p4r, split="test", n_samples=10_000, seed=SEED, dataset_type="derivative")
x_p4r_test, y_p4r_test = load_dataset(p4r_test_path)
test_ds = tensor_dataset(x_p4r_test, y_p4r_test)

# ── Setup rollout ─────────────────────────────────────────────────────────────
x0 = p4r.sample_initial_conditions(16, split="test", seed=SEED + 123)
true_traj = rollout_system(p4r, x0, steps=P4_STEPS, dt=P4_DT)
wrap = system.wrap_state

# ── Three rollouts ────────────────────────────────────────────────────────────
print("\nRolling out baseline ...")
base_traj = autoregressive_rollout_model(baseline, x0, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)
err_base  = rollout_error(system, true_traj, base_traj).mean(axis=1)

print("Rolling out stable (WITH projection) ...")
stable_traj = autoregressive_rollout_model(stable, x0, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)
err_stable  = rollout_error(system, true_traj, stable_traj).mean(axis=1)

print("Rolling out stable.fhat (WITHOUT projection) ...")
fhat_traj = autoregressive_rollout_model(stable.fhat, x0, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)
err_fhat  = rollout_error(system, true_traj, fhat_traj).mean(axis=1)

# ── Derivative MSE for fhat vs baseline ──────────────────────────────────────
mse_base  = evaluate_derivative_mse(baseline,   test_ds, device=DEVICE)
mse_stable = evaluate_derivative_mse(stable,    test_ds, device=DEVICE)
mse_fhat  = evaluate_derivative_mse(stable.fhat, test_ds, device=DEVICE)

# ── Projection statistics ─────────────────────────────────────────────────────
# Count how often projection fires during the stable rollout
def count_projections(model, traj, device):
    """Count steps where relu(violation) > 0."""
    fires = 0
    total = 0
    max_corr = 0.0
    with torch.enable_grad():
        for t in range(traj.shape[0] - 1):
            x = torch.tensor(traj[t], dtype=torch.float32, device=device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.detach().requires_grad_(True)
            fx = model.fhat(x)
            vx = model.V(x)
            grad_v = torch.autograd.grad(vx.sum(), x, create_graph=False)[0]
            violation = (grad_v * fx).sum(dim=1) + model.alpha * vx.squeeze(1)
            fired = (violation > 0).sum().item()
            corr_mag = (violation.clamp_min(0) / grad_v.square().sum(dim=1).clamp_min(1e-8)).max().item()
            fires += fired
            total += x.shape[0]
            max_corr = max(max_corr, corr_mag)
    return fires / max(total, 1), max_corr

print("\nCounting projection fires during stable rollout ...")
fire_rate, max_corr_mag = count_projections(stable, stable_traj, DEVICE)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC RESULTS (exp5 checkpoints, alpha=1e-5)")
print("="*60)
print(f"\nDerivative MSE:")
print(f"  baseline:         {mse_base:.4g}")
print(f"  stable (full):    {mse_stable:.4g}")
print(f"  stable.fhat only: {mse_fhat:.4g}")
print(f"\nFinal rollout error (step {P4_STEPS}):")
print(f"  baseline:         {err_base[-1]:.4g}")
print(f"  stable (full):    {err_stable[-1]:.4g}")
print(f"  stable.fhat only: {err_fhat[-1]:.4g}")
print(f"\nMean rollout error:")
print(f"  baseline:         {err_base.mean():.4g}")
print(f"  stable (full):    {err_stable.mean():.4g}")
print(f"  stable.fhat only: {err_fhat.mean():.4g}")
print(f"\nProjection statistics (during stable rollout):")
print(f"  fire rate: {fire_rate:.4f} ({fire_rate*100:.2f}% of steps)")
print(f"  max correction magnitude: {max_corr_mag:.4g}")
print("\nInterpretation:")
if err_fhat[-1] < err_stable[-1] * 0.1:
    print("  >> fhat rollout << stable rollout: PROJECTION is the culprit.")
    print("     fhat itself is fine; the Lyapunov corrections cause divergence.")
elif abs(err_fhat[-1] - err_stable[-1]) / err_stable[-1] < 0.2:
    print("  >> fhat rollout ≈ stable rollout: fhat is also degraded.")
    print("     Lyapunov constraint during TRAINING is distorting fhat.")
else:
    print("  >> Intermediate — both effects present.")
