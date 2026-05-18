"""
Dva eksperimenta:
  1. mass_spring_damper  — 2D, random sampling, α=1e-3  (verifikacija)
  2. damped_pendulum_4   — 8D, existing traj data, α=1e-5 (popravka)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from stable_icnn_physics import BaselineDynamicsMLP, build_stable_model, make_system
from stable_icnn_physics.data import (
    dataset_base_name,
    generate_derivative_data,
    load_dataset,
    save_dataset,
    tensor_dataset,
)
from stable_icnn_physics.eval import (
    autoregressive_rollout_model,
    lyapunov_decrease_values,
    rollout_error,
    rollout_system,
)
from stable_icnn_physics.train import evaluate_derivative_mse, train_derivative_model

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR  = REPO_ROOT / "data" / "cache"
CKPT_DIR   = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
TOLERANCE   = 1e-5

print(f"Device: {DEVICE}  |  torch {torch.__version__}")


# ── helpers ──────────────────────────────────────────────────────────────────

def build_models(state_dim, hidden, depth, lyapunov_hidden, lyapunov_eps, alpha, rehu_width):
    # StableDynamics uses create_graph=True internally → double backward →
    # torch.compile (AOT autograd) does not support this. Baseline is fine.
    stable = build_stable_model(
        dim=state_dim, hidden=hidden, depth=depth,
        lyapunov_hidden=lyapunov_hidden, lyapunov_eps=lyapunov_eps,
        alpha=alpha, rehu_width=rehu_width,
    )
    baseline = torch.compile(BaselineDynamicsMLP(dim=state_dim, hidden=hidden, depth=depth))
    return stable, baseline


def load_raw_models(state_dim, hidden, depth, lyapunov_hidden, lyapunov_eps, alpha, rehu_width,
                    stable_ckpt, baseline_ckpt):
    """Load uncompiled models for rollout/Lyapunov evaluation."""
    stable = build_stable_model(
        dim=state_dim, hidden=hidden, depth=depth,
        lyapunov_hidden=lyapunov_hidden, lyapunov_eps=lyapunov_eps,
        alpha=alpha, rehu_width=rehu_width,
    )
    baseline = BaselineDynamicsMLP(dim=state_dim, hidden=hidden, depth=depth)
    stable.load_state_dict(torch.load(stable_ckpt,   map_location=DEVICE, weights_only=True)["model_state"])
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
    stable.to(DEVICE).eval()
    baseline.to(DEVICE).eval()
    return stable, baseline


def run_experiment(
    tag: str,
    system_name: str,
    system_kwargs: dict,
    train_ds, test_ds,
    x_test: np.ndarray,
    x0_rollout: np.ndarray,
    true_traj: np.ndarray,
    dt: float,
    rollout_steps: int,
    # hyperparams
    epochs=200, batch_size=256, lr=1e-3,
    hidden=100, depth=2,
    lyapunov_hidden=60, lyapunov_eps=0.01,
    alpha=1e-3, rehu_width=0.01,
):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  system={system_name}  α={alpha}  epochs={epochs}")
    print(f"{'='*60}")

    system = make_system(system_name, **system_kwargs)
    state_dim = system.state_dim
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    stable_ckpt   = CKPT_DIR / f"{tag}_stable.pt"
    baseline_ckpt = CKPT_DIR / f"{tag}_baseline.pt"

    stable, baseline = build_models(state_dim, hidden, depth, lyapunov_hidden, lyapunov_eps, alpha, rehu_width)
    n_stable   = sum(p.numel() for p in stable.parameters())
    n_baseline = sum(p.numel() for p in baseline.parameters())
    print(f"  params: stable={n_stable}  baseline={n_baseline}")

    print("  [train] stable  (use_amp=False) ...")
    train_derivative_model(
        stable, train_ds, test_ds,
        epochs=epochs, batch_size=batch_size, learning_rate=lr,
        device=DEVICE, checkpoint_path=stable_ckpt,
        print_every=max(1, epochs // 8), use_amp=False,
    )

    print("  [train] baseline (use_amp=True) ...")
    train_derivative_model(
        baseline, train_ds, test_ds,
        epochs=epochs, batch_size=batch_size, learning_rate=lr,
        device=DEVICE, checkpoint_path=baseline_ckpt,
        print_every=max(1, epochs // 8), use_amp=True,
    )

    # ── evaluation ───────────────────────────────────────────────────────────
    stable_raw, baseline_raw = load_raw_models(
        state_dim, hidden, depth, lyapunov_hidden, lyapunov_eps, alpha, rehu_width,
        stable_ckpt, baseline_ckpt,
    )

    dmse_stable   = evaluate_derivative_mse(stable_raw,   test_ds, device=DEVICE)
    dmse_baseline = evaluate_derivative_mse(baseline_raw, test_ds, device=DEVICE)
    print(f"  derivative MSE: stable={dmse_stable:.4g}  baseline={dmse_baseline:.4g}")

    wrap = system.wrap_state
    stable_traj   = autoregressive_rollout_model(stable_raw,   x0_rollout, steps=rollout_steps, dt=dt, device=DEVICE, wrap_fn=wrap)
    baseline_traj = autoregressive_rollout_model(baseline_raw, x0_rollout, steps=rollout_steps, dt=dt, device=DEVICE, wrap_fn=wrap)

    err_stable   = rollout_error(system, true_traj, stable_traj).mean(axis=1)   # (steps+1,)
    err_baseline = rollout_error(system, true_traj, baseline_traj).mean(axis=1)
    print(f"  final rollout error: stable={err_stable[-1]:.4g}  baseline={err_baseline[-1]:.4g}")
    print(f"  mean  rollout error: stable={err_stable.mean():.4g}  baseline={err_baseline.mean():.4g}")

    decrease = lyapunov_decrease_values(stable_raw, x_test[:2048], device=DEVICE).ravel()
    max_viol  = float(decrease.max())
    frac_sat  = float(np.mean(decrease <= TOLERANCE))
    print(f"  Lyapunov max_violation={max_viol:.4g}  fraction_satisfied={frac_sat:.4f}")

    summary = {
        "experiment": tag,
        "system": system_name,
        "alpha": alpha,
        "derivative_mse_stable":        float(dmse_stable),
        "derivative_mse_baseline":      float(dmse_baseline),
        "final_rollout_error_stable":   float(err_stable[-1]),
        "final_rollout_error_baseline": float(err_baseline[-1]),
        "mean_rollout_error_stable":    float(err_stable.mean()),
        "mean_rollout_error_baseline":  float(err_baseline.mean()),
        "lyapunov_max_violation":       max_viol,
        "lyapunov_fraction_satisfied":  frac_sat,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved → {out}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: mass_spring_damper (2D, random sampling, α=1e-3)
# ═══════════════════════════════════════════════════════════════════════════

MSD_SYSTEM  = "mass_spring_damper"
MSD_KWARGS  = {}
MSD_DT      = 0.05
MSD_STEPS   = 300
SEED        = 0

print("\n[1/2] Preparing mass_spring_damper data ...")
msd = make_system(MSD_SYSTEM)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

for split, n in [("train", 50_000), ("test", 10_000)]:
    path = CACHE_DIR / dataset_base_name(msd, split=split, n_samples=n, seed=SEED, dataset_type="derivative")
    if not path.exists():
        x, y = generate_derivative_data(msd, n_samples=n, split=split, seed=SEED)
        save_dataset(path, x, y)
        print(f"  generated {split}: {path.name}")
    else:
        print(f"  reusing  {split}: {path.name}")

msd_train_path = CACHE_DIR / dataset_base_name(msd, split="train", n_samples=50_000, seed=SEED, dataset_type="derivative")
msd_test_path  = CACHE_DIR / dataset_base_name(msd, split="test",  n_samples=10_000, seed=SEED, dataset_type="derivative")
x_msd_train, y_msd_train = load_dataset(msd_train_path)
x_msd_test,  y_msd_test  = load_dataset(msd_test_path)
msd_train_ds = tensor_dataset(x_msd_train, y_msd_train)
msd_test_ds  = tensor_dataset(x_msd_test,  y_msd_test)

x0_msd   = msd.sample_initial_conditions(16, split="test", seed=SEED + 123)
true_msd  = rollout_system(msd, x0_msd, steps=MSD_STEPS, dt=MSD_DT)

summary1 = run_experiment(
    tag="msd_random_alpha1e3",
    system_name=MSD_SYSTEM, system_kwargs=MSD_KWARGS,
    train_ds=msd_train_ds, test_ds=msd_test_ds,
    x_test=x_msd_test,
    x0_rollout=x0_msd, true_traj=true_msd,
    dt=MSD_DT, rollout_steps=MSD_STEPS,
    alpha=1e-3,
)

# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: damped_pendulum_4, existing data, α=1e-5
# ═══════════════════════════════════════════════════════════════════════════

P4_SYSTEM = "damped_pendulum_4"
P4_KWARGS = {"friction": 0.3, "gravity": 9.81}
P4_DT     = 0.02
P4_STEPS  = 300

print("\n[2/2] Loading existing damped_pendulum_4 data ...")
p4 = make_system(P4_SYSTEM, **P4_KWARGS)

p4_train_path = CACHE_DIR / dataset_base_name(p4, split="train", n_trajectories=200, steps=1000, dt=P4_DT, seed=SEED, dataset_type="derivative")
p4_test_path  = CACHE_DIR / dataset_base_name(p4, split="test",  n_trajectories=50,  steps=1000, dt=P4_DT, seed=SEED, dataset_type="derivative")

if not p4_train_path.exists() or not p4_test_path.exists():
    raise FileNotFoundError("Existing pendulum_4 data not found. Run 01_generate_data.ipynb first.")

x_p4_train, y_p4_train = load_dataset(p4_train_path)
x_p4_test,  y_p4_test  = load_dataset(p4_test_path)
p4_train_ds = tensor_dataset(x_p4_train, y_p4_train)
p4_test_ds  = tensor_dataset(x_p4_test,  y_p4_test)
print(f"  train: {x_p4_train.shape}  test: {x_p4_test.shape}")

x0_p4   = p4.sample_initial_conditions(16, split="test", seed=SEED + 123)
true_p4  = rollout_system(p4, x0_p4, steps=P4_STEPS, dt=P4_DT)

summary2 = run_experiment(
    tag="p4_traj_alpha1e5",
    system_name=P4_SYSTEM, system_kwargs=P4_KWARGS,
    train_ds=p4_train_ds, test_ds=p4_test_ds,
    x_test=x_p4_test,
    x0_rollout=x0_p4, true_traj=true_p4,
    dt=P4_DT, rollout_steps=P4_STEPS,
    alpha=1e-5,
)

# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: damped_pendulum_4, RANDOM sampling, α=1e-3
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3/3] Generating random derivative data for damped_pendulum_4 ...")
p4r = make_system(P4_SYSTEM, **P4_KWARGS)

for split, n in [("train", 50_000), ("test", 10_000)]:
    path = CACHE_DIR / dataset_base_name(p4r, split=split, n_samples=n, seed=SEED, dataset_type="derivative")
    if not path.exists():
        x, y = generate_derivative_data(p4r, n_samples=n, split=split, seed=SEED)
        save_dataset(path, x, y)
        print(f"  generated {split}: {path.name}")
    else:
        print(f"  reusing  {split}: {path.name}")

p4r_train_path = CACHE_DIR / dataset_base_name(p4r, split="train", n_samples=50_000, seed=SEED, dataset_type="derivative")
p4r_test_path  = CACHE_DIR / dataset_base_name(p4r, split="test",  n_samples=10_000, seed=SEED, dataset_type="derivative")
x_p4r_train, y_p4r_train = load_dataset(p4r_train_path)
x_p4r_test,  y_p4r_test  = load_dataset(p4r_test_path)
p4r_train_ds = tensor_dataset(x_p4r_train, y_p4r_train)
p4r_test_ds  = tensor_dataset(x_p4r_test,  y_p4r_test)
print(f"  train: {x_p4r_train.shape}  test: {x_p4r_test.shape}")

# Rollout from same initial conditions as trajectory experiment for fair comparison
summary3 = run_experiment(
    tag="p4_random_alpha1e3",
    system_name=P4_SYSTEM, system_kwargs=P4_KWARGS,
    train_ds=p4r_train_ds, test_ds=p4r_test_ds,
    x_test=x_p4r_test,
    x0_rollout=x0_p4, true_traj=true_p4,
    dt=P4_DT, rollout_steps=P4_STEPS,
    alpha=1e-3,
)

# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: damped_pendulum_4, RANDOM sampling, larger model, more epochs
# Hypothesis: stable≈baseline MSE with hidden=100/epochs=200 means model is
# undertrained/underpowered for 8D chaotic system (original paper used 1000 epochs).
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4/4] p4 random — larger model (hidden=200, depth=3), 500 epochs ...")
summary4 = run_experiment(
    tag="p4_random_large_e500",
    system_name=P4_SYSTEM, system_kwargs=P4_KWARGS,
    train_ds=p4r_train_ds, test_ds=p4r_test_ds,
    x_test=x_p4r_test,
    x0_rollout=x0_p4, true_traj=true_p4,
    dt=P4_DT, rollout_steps=P4_STEPS,
    alpha=1e-3,
    epochs=500,
    hidden=200,
    depth=3,
    lyapunov_hidden=100,
)

# ═══════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for s in [summary1, summary2, summary3, summary4]:
    print(f"\n{s['experiment']}  (α={s['alpha']})")
    print(f"  deriv MSE:    stable={s['derivative_mse_stable']:.4g}  baseline={s['derivative_mse_baseline']:.4g}")
    print(f"  rollout err:  stable={s['final_rollout_error_stable']:.4g}  baseline={s['final_rollout_error_baseline']:.4g}")
    print(f"  Lyapunov:     max_viol={s['lyapunov_max_violation']:.4g}  frac_sat={s['lyapunov_fraction_satisfied']:.4f}")
