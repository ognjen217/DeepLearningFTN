"""Three-body ICNN experiment package.

This package is intentionally separate from the core coursework package.  It
contains a 3D Newtonian three-body reference system, dataset utilities,
visualization helpers, and wrappers for baseline/ICNN dynamics experiments.
"""

from .system import (
    ThreeBodyConfig,
    ThreeBodySystem3D,
    center_state,
    figure_eight_state_3d,
    integrate_solve_ivp,
    integrate_rk4_fixed,
)
from .data import (
    generate_trajectory_dataset,
    load_trajectory_dataset,
    save_trajectory_dataset,
    trajectory_pairs_to_derivatives,
)
from .models import build_baseline_model, build_stable_icnn_model

__all__ = [
    "ThreeBodyConfig",
    "ThreeBodySystem3D",
    "build_baseline_model",
    "build_stable_icnn_model",
    "center_state",
    "figure_eight_state_3d",
    "generate_trajectory_dataset",
    "integrate_rk4_fixed",
    "integrate_solve_ivp",
    "load_trajectory_dataset",
    "save_trajectory_dataset",
    "trajectory_pairs_to_derivatives",
]
