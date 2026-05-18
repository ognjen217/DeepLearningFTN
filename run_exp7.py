"""
Experiment 7: rollout-augmented Lyapunov training.

Root cause confirmed in exp6:
  - V achieves 100% Lyapunov satisfaction on 10k random test points
  - But projection fires at 86% of rollout states
  - Distribution shift: rollout trajectory is a 1D manifold in 8D space,
    barely overlapping with 50k uniformly-random training points.
  - V's gradient is correct everywhere we trained it, wrong everywhere the
    rollout actually goes.

Fix: generate rollout states using the good fhat (rollout error=0.577),
     train V with violation loss on BOTH random states AND rollout states.
     V must learn correct gradients specifically in the rollout distribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stable_icnn_physics import build_stable_model, make_system
from stable_icnn_physics.data import dataset_base_name, load_dataset, generate_derivative_data, save_dataset, tensor_dataset
from stable_icnn_physics.eval import (
    autoregressive_rollout_model,
    lyapunov_decrease_values,
    rollout_error,
    rollout_system,
)
from stable_icnn_physics.train import evaluate_derivative_mse

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

# ── Load existing random data ─────────────────────────────────────────────────
for split, n in [("train", 50_000), ("test", 10_000)]:
    path = CACHE_DIR / dataset_base_name(p4r, split=split, n_samples=n, seed=SEED, dataset_type="derivative")
    if not path.exists():
        x, y = generate_derivative_data(p4r, n_samples=n, split=split, seed=SEED)
        save_dataset(path, x, y)
    else:
        print(f"  reusing  {split}: {path.name}")

p4r_train_path = CACHE_DIR / dataset_base_name(p4r, split="train", n_samples=50_000, seed=SEED, dataset_type="derivative")
p4r_test_path  = CACHE_DIR / dataset_base_name(p4r, split="test",  n_samples=10_000, seed=SEED, dataset_type="derivative")
x_p4r_train, _ = load_dataset(p4r_train_path)
x_p4r_test, _  = load_dataset(p4r_test_path)
p4r_test_ds    = tensor_dataset(*load_dataset(p4r_test_path))
print(f"  random data: train={x_p4r_train.shape}  test={x_p4r_test.shape}")

x0_eval  = p4r.sample_initial_conditions(16, split="test", seed=SEED + 123)
true_p4  = rollout_system(p4r, x0_eval, steps=P4_STEPS, dt=P4_DT)

# ── Load phase-1 fhat (exp5 checkpoint) ───────────────────────────────────────
print("\n[Setup] Loading exp5 fhat checkpoint ...")
phase1_ckpt = CKPT_DIR / "p4_random_large_e500_alpha1e5_stable.pt"
if not phase1_ckpt.exists():
    raise FileNotFoundError(f"Run run_exp5.py first: {phase1_ckpt}")

def make_stable():
    return build_stable_model(
        dim=state_dim, hidden=200, depth=3,
        lyapunov_hidden=100, lyapunov_eps=0.01,
        alpha=1e-5, rehu_width=0.01,
    )

stable = make_stable()
stable.load_state_dict(torch.load(phase1_ckpt, map_location=DEVICE, weights_only=True)["model_state"])
stable.to(DEVICE).eval()

# ── Generate rollout states from fhat ────────────────────────────────────────
print("\n[Step 1] Generating rollout states using fhat (no projection) ...")

N_TRAJ_TRAIN = 500    # trajectories to collect rollout states from
N_TRAJ_VAL   = 100

def collect_rollout_states(fhat, system, n_trajs, steps, dt, split, seed):
    """Roll out fhat (Euler) from n_trajs initial conditions, collect all states."""
    ics = system.sample_initial_conditions(n_trajs, split=split, seed=seed)
    all_states = []
    with torch.no_grad():
        for i in range(n_trajs):
            x = ics[i].copy()
            for _ in range(steps):
                all_states.append(x.copy())
                x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                xdot = fhat(x_t).squeeze(0).cpu().numpy()
                x = system.wrap_state(x + dt * xdot)
    return np.array(all_states, dtype=np.float32)

rollout_train_states = collect_rollout_states(stable.fhat, p4r, N_TRAJ_TRAIN, P4_STEPS, P4_DT, "train", SEED + 1)
rollout_val_states   = collect_rollout_states(stable.fhat, p4r, N_TRAJ_VAL,   P4_STEPS, P4_DT, "test",  SEED + 2)
print(f"  rollout states: train={rollout_train_states.shape}  val={rollout_val_states.shape}")

# Combine random + rollout states for V training
combined_train = np.concatenate([x_p4r_train, rollout_train_states], axis=0)
combined_val   = np.concatenate([x_p4r_test,  rollout_val_states],   axis=0)
print(f"  combined: train={combined_train.shape}  val={combined_val.shape}")

combined_train_t = torch.from_numpy(combined_train)
combined_val_t   = torch.from_numpy(combined_val)

# ── Phase 2: train V with rollout-augmented data ──────────────────────────────
print("\n[Step 2] Training V on combined (random + rollout) states ...")

for p in stable.fhat.parameters():
    p.requires_grad_(False)
stable.fhat.eval()

optimizer = torch.optim.Adam(stable.V.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-4)

pin = DEVICE == "cuda"
EPOCHS_V = 400
BATCH_V  = 512

loader_train = DataLoader(TensorDataset(combined_train_t), batch_size=BATCH_V, shuffle=True, pin_memory=pin)
loader_val   = DataLoader(TensorDataset(combined_val_t),   batch_size=1024,   shuffle=False, pin_memory=pin)

best_val_loss = float("inf")
best_state    = None

for epoch in range(1, EPOCHS_V + 1):
    stable.V.train()
    total, count = 0.0, 0
    for (x_batch,) in loader_train:
        x_batch = x_batch.to(DEVICE, non_blocking=pin).requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            fx  = stable.fhat(x_batch)
            vx  = stable.V(x_batch)
            gv  = torch.autograd.grad(vx.sum(), x_batch, create_graph=True)[0]
            vio = (gv * fx).sum(dim=1, keepdim=True) + stable.alpha * vx
            loss = F.relu(vio).pow(2).mean()
        loss.backward()
        optimizer.step()
        total += loss.detach().item() * x_batch.shape[0]
        count += x_batch.shape[0]
    scheduler.step()

    if epoch % 40 == 0 or epoch == 1 or epoch == EPOCHS_V:
        stable.V.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for (x_val,) in loader_val:
                x_val = x_val.to(DEVICE, non_blocking=pin).requires_grad_(True)
                with torch.enable_grad():
                    fx  = stable.fhat(x_val)
                    vx  = stable.V(x_val)
                    gv  = torch.autograd.grad(vx.sum(), x_val, create_graph=False)[0]
                    vio = (gv * fx).sum(dim=1, keepdim=True) + stable.alpha * vx
                    vloss = F.relu(vio).pow(2).mean()
                val_total += vloss.item() * x_val.shape[0]
                val_count += x_val.shape[0]
        val_loss = val_total / max(val_count, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in stable.V.state_dict().items()}
        print(f"epoch={epoch:04d}  train={total/max(count,1):.5g}  val={val_loss:.5g}  best_val={best_val_loss:.5g}")

# Restore best V
stable.V.load_state_dict(best_state)
stable.eval()

# Re-enable fhat gradients
for p in stable.fhat.parameters():
    p.requires_grad_(True)

# Save
phase2_ckpt = CKPT_DIR / "p4_rollout_aug_stable.pt"
torch.save({"model_state": stable.state_dict()}, phase2_ckpt)
print(f"  saved → {phase2_ckpt}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[Eval] Rolling out ...")
stable.eval()
wrap = p4r.wrap_state

traj_stable = autoregressive_rollout_model(stable, x0_eval, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)
traj_fhat   = autoregressive_rollout_model(stable.fhat, x0_eval, steps=P4_STEPS, dt=P4_DT, device=DEVICE, wrap_fn=wrap)

err_stable = rollout_error(p4r, true_p4, traj_stable).mean(axis=1)
err_fhat   = rollout_error(p4r, true_p4, traj_fhat).mean(axis=1)

decrease = lyapunov_decrease_values(stable, x_p4r_test[:2048], device=DEVICE).ravel()
max_viol = float(decrease.max())
frac_sat = float(np.mean(decrease <= TOLERANCE))

# Fire rate on rollout trajectory
def fire_rate_on_traj(model, traj, device):
    fires, total = 0, 0
    with torch.enable_grad():
        for t in range(traj.shape[0] - 1):
            x = torch.tensor(traj[t], dtype=torch.float32, device=device).requires_grad_(True)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            fx = model.fhat(x)
            vx = model.V(x)
            gv = torch.autograd.grad(vx.sum(), x, create_graph=False)[0]
            vio = (gv * fx).sum(dim=1) + model.alpha * vx.squeeze(1)
            fires += (vio > 0).sum().item()
            total += x.shape[0]
    return fires / max(total, 1)

fire_rate = fire_rate_on_traj(stable, traj_stable, DEVICE)

print("\n" + "="*60)
print("EXPERIMENT 7: ROLLOUT-AUGMENTED LYAPUNOV TRAINING")
print("="*60)
print(f"\nRollout error (final step {P4_STEPS}):")
print(f"  stable (exp7):   {err_stable[-1]:.4g}")
print(f"  fhat only:       {err_fhat[-1]:.4g}")
print(f"  [ref] exp6:      472.4    exp5: 377.9    baseline: 1.684")
print(f"\nRollout error (mean):")
print(f"  stable (exp7):   {err_stable.mean():.4g}")
print(f"  fhat only:       {err_fhat.mean():.4g}")
print(f"\nLyapunov:")
print(f"  max_violation={max_viol:.4g}   fraction_satisfied={frac_sat:.4f}")
print(f"\nProjection fire rate: {fire_rate:.4f} ({fire_rate*100:.2f}%)")
print(f"  [ref] exp6: 86.29%   exp5: 91.62%")

summary = {
    "experiment": "p4_rollout_aug",
    "alpha": 1e-5,
    "n_rollout_train_trajs": N_TRAJ_TRAIN,
    "n_rollout_steps": P4_STEPS,
    "final_rollout_error_stable": float(err_stable[-1]),
    "final_rollout_error_fhat":   float(err_fhat[-1]),
    "mean_rollout_error_stable":  float(err_stable.mean()),
    "lyapunov_max_violation":     max_viol,
    "lyapunov_fraction_satisfied": frac_sat,
    "projection_fire_rate":        fire_rate,
}
(RESULTS_DIR / "p4_rollout_aug_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n  saved → {RESULTS_DIR / 'p4_rollout_aug_summary.json'}")
