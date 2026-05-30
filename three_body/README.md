# Three-body ICNN experiment branch

This branch is an experimental extension of the core ICNN Lyapunov project.  It
uses a 3D Newtonian three-body system as a visually interesting stress test for
baseline neural dynamics and ICNN-projected stable dynamics.

## Problem formulation

State dimension is 18:

```text
[r1, r2, r3, v1, v2, v3]
```

where each `ri` and `vi` is a 3D vector. The reference dynamics are

```text
r_i_dot = v_i
v_i_dot = G * sum_{j != i} m_j (r_j - r_i) / (||r_j-r_i||^2 + eps^2)^(3/2)
```

The default reference initial condition is the equal-mass figure-eight orbit
embedded in the `z=0` plane. For training, the branch uses small perturbations
around that trajectory and optional softening to avoid singular accelerations.

## Important interpretation note

The pure Newtonian three-body problem is conservative and can be chaotic. The
ICNN projection used in the core project enforces a Lyapunov decrease condition,
so it is not naturally energy-conserving. Therefore this branch should be read
as a nonlinear stress test and visualization sandbox, not as a replacement for
the main project experiments.

## Recommended workflow

From the repository root:

```bash
git switch feature/three-body-icnn
```

### 1. Validate the reference solver

```bash
python scripts/run_three_body_reference.py --steps 1400 --dt 0.005 --make-gif
```

Outputs:

```text
data/three_body/figure8_reference_solve_ivp.npz
results/three_body/reference/
```

### 2. Train baseline and ICNN models

Start with a quick smoke test:

```bash
python scripts/run_three_body_training.py --quick --no-gif
```

Then run a fuller experiment:

```bash
python scripts/run_three_body_training.py \
  --train-trajs 96 \
  --val-trajs 16 \
  --test-trajs 8 \
  --steps 700 \
  --dt 0.005 \
  --phase1-epochs 150 \
  --v-epochs 250 \
  --batch-size 512
```

Outputs:

```text
data/three_body/
checkpoints/three_body/
results/three_body/
```

### 3. Benchmark solver vs neural rollout speed

After training:

```bash
python scripts/run_three_body_benchmark.py --steps 1000 --dt 0.005
```

This compares:

```text
solve_ivp / DOP853
fixed-step RK4
baseline neural rollout
fhat-only neural rollout
ICNN-projected rollout
```

## Main metrics

Training/evaluation saves:

- derivative MSE,
- autoregressive rollout RMSE,
- energy drift,
- momentum drift,
- ICNN projection fire rate,
- correction norm,
- 3D trajectory plots,
- optional GIF animations,
- solver-vs-network timing.

## Expected behavior

The expected first result is not necessarily that the ICNN model beats the
baseline in pure trajectory tracking. Since the reference system is
conservative, a Lyapunov-decreasing projection can over-damp the learned vector
field. The key things to inspect are:

1. whether `fhat` learns a reasonable local vector field,
2. whether projection fire rate is too high,
3. whether the correction norm is destroying the orbit,
4. whether lowering `alpha` or using V-only rollout augmentation improves the
   projected rollout.
