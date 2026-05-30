"""2D shallow-water simulation utilities.

This package starts with a linearized, damped 2D shallow-water-wave solver
that is intentionally simple enough for notebook experiments and animations.
"""

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
    "SWE2DConfig",
    "cfl_number",
    "compute_energy",
    "compute_mass",
    "compute_rmse",
    "gaussian_bump_ic",
    "make_grid",
    "random_bumps_ic",
    "rhs_linear_swe",
    "rk4_step",
    "simulate",
    "velocity_magnitude",
]
