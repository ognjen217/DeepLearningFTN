"""Stable ICNN dynamics models for physical systems."""

from .systems import (
    CustomStateSpaceSystem,
    DampedPendulum,
    MassSpringDamper,
    PhysicalSystem,
    VanDerPolOscillator,
    available_systems,
    make_system,
)

__all__ = [
    "BaselineDynamicsMLP",
    "CustomStateSpaceSystem",
    "DampedPendulum",
    "ICNN",
    "MassSpringDamper",
    "NominalMLP",
    "PhysicalSystem",
    "PositiveDefiniteLyapunov",
    "ReHU",
    "StableDynamics",
    "VanDerPolOscillator",
    "available_systems",
    "build_stable_model",
    "make_system",
]


def __getattr__(name):
    if name in {
        "BaselineDynamicsMLP",
        "ICNN",
        "NominalMLP",
        "PositiveDefiniteLyapunov",
        "ReHU",
        "StableDynamics",
        "build_stable_model",
    }:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
