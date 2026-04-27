# Stable ICNN Dynamics for Physical Systems

This project implements the physical-system part of *Learning Stable Deep Dynamics Models*
using a small, reusable PyTorch package plus notebooks.

The main model learns a nominal dynamics network `fhat(x)` and projects it so that the
learned vector field is non-expansive under a positive definite ICNN Lyapunov function:

```text
f(x) = fhat(x) - gradV(x) * relu(gradV(x) dot fhat(x) + alpha * V(x)) / ||gradV(x)||^2
```

Video texture generation is intentionally not included.

## Layout

- `stable_icnn_physics/`: reusable package code.
- `notebooks/01_generate_data.ipynb`: generate cached derivative datasets.
- `notebooks/02_train_models.ipynb`: train stable and baseline models.
- `notebooks/03_evaluate_results.ipynb`: evaluate derivative fit, rollout error, and plots.
- `tests/`: pytest checks for systems and stability constraints.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest
```

Open the notebooks from the repository root:

```bash
jupyter lab
```

The first examples are intentionally small enough for CPU coursework runs. Use the notebook
configuration cells to change the physical system, system parameters, model sizes, and training
settings.

