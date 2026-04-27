"""Stable ICNN dynamics models for physical systems."""

from .systems import DampedPendulum, MassSpringDamper, PhysicalSystem

__all__ = [
    "BaselineDynamicsMLP",
    "DampedPendulum",
    "ICNN",
    "MassSpringDamper",
    "NominalMLP",
    "PhysicalSystem",
    "PositiveDefiniteLyapunov",
    "ReHU",
    "StableDynamics",
]


def __getattr__(name):
    if name in {
        "BaselineDynamicsMLP",
        "ICNN",
        "NominalMLP",
        "PositiveDefiniteLyapunov",
        "ReHU",
        "StableDynamics",
    }:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
