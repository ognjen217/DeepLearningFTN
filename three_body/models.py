from __future__ import annotations

from stable_icnn_physics import BaselineDynamicsMLP, build_stable_model


def build_baseline_model(state_dim: int = 18, hidden: int = 256, depth: int = 3) -> BaselineDynamicsMLP:
    """Build the unconstrained baseline MLP derivative model."""
    return BaselineDynamicsMLP(dim=state_dim, hidden=hidden, depth=depth)


def build_stable_icnn_model(
    state_dim: int = 18,
    hidden: int = 256,
    depth: int = 3,
    lyapunov_hidden: int = 128,
    lyapunov_eps: float = 0.01,
    alpha: float = 1e-6,
    rehu_width: float = 0.01,
):
    """Build the ICNN-projected stable dynamics model used in the core project."""
    return build_stable_model(
        dim=state_dim,
        hidden=hidden,
        depth=depth,
        lyapunov_hidden=lyapunov_hidden,
        lyapunov_eps=lyapunov_eps,
        alpha=alpha,
        rehu_width=rehu_width,
    )
