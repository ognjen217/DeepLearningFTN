"""2D shallow-water simulation and learning utilities.

The package starts with a linearized, damped 2D shallow-water-wave solver
that is intentionally simple enough for notebook experiments and animations.
It also contains lightweight CNN dynamics models for first learning experiments.
"""

from .data import (
    dataset_path,
    generate_trajectory_dataset,
    load_or_generate_dataset,
    load_trajectory_dataset,
    save_trajectory_dataset,
    trajectory_pairs,
)
from .eval import energy_projection_stats, rollout_torch_dynamics, summarize_rollout
from .models import (
    ConvResNetDynamics,
    EnergyProjectedDynamics,
    PhysicalEnergy,
    ResidualBlock,
    SWE2DModelConfig,
    make_cnn_dynamics,
    make_energy_projected_cnn,
)
from .swe2d import (
    SWE2DConfig,
    cfl_number,
    compute_energy,
    compute_mass,
    compute_rmse,
    gaussian_bump_ic,
    make_grid,
    random_bumps_ic,
    rhs_linear_swe,
    rk4_step,
    simulate,
    velocity_magnitude,
)

__all__ = [
    "ConvResNetDynamics",
    "EnergyProjectedDynamics",
    "PhysicalEnergy",
    "ResidualBlock",
    "SWE2DConfig",
    "SWE2DModelConfig",
    "cfl_number",
    "compute_energy",
    "compute_mass",
    "compute_rmse",
    "dataset_path",
    "energy_projection_stats",
    "gaussian_bump_ic",
    "generate_trajectory_dataset",
    "load_or_generate_dataset",
    "load_trajectory_dataset",
    "make_cnn_dynamics",
    "make_energy_projected_cnn",
    "make_grid",
    "random_bumps_ic",
    "rhs_linear_swe",
    "rk4_step",
    "rollout_torch_dynamics",
    "save_trajectory_dataset",
    "simulate",
    "summarize_rollout",
    "trajectory_pairs",
    "velocity_magnitude",
]
