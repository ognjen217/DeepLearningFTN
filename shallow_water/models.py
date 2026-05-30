from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SWE2DModelConfig:
    """Configuration shared by the CNN dynamics models."""

    dt: float = 2.0e-3
    gravity: float = 9.81
    depth: float = 1.0
    alpha: float = 1.0e-5
    denom_eps: float = 1.0e-8


class ResidualBlock(nn.Module):
    """Small same-resolution residual convolution block."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ConvResNetDynamics(nn.Module):
    """CNN derivative predictor for SWE2D states.

    Input and output shape: ``(batch, 3, ny, nx)``.
    The network predicts a continuous-time derivative ``xdot = fhat(x)``.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, depth: int = 4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(depth)])
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

        # Start close to zero derivative. This usually makes early rollout safer.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = self.blocks(z)
        return self.out_proj(z)


class NextStepFromDerivative(nn.Module):
    """Wrap a derivative model as a one-step predictor ``x_next = x + dt*f(x)``."""

    def __init__(self, dynamics: nn.Module, dt: float):
        super().__init__()
        self.dynamics = dynamics
        self.dt = float(dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dt * self.dynamics(x)


class PhysicalEnergy(nn.Module):
    """Physical quadratic energy for linearized shallow-water waves.

    E = 0.5 * mean(g * eta^2 + H * (u^2 + v^2))

    The spatial mean is used instead of dx*dy integral so the scale stays stable
    when changing grid resolution. This is sufficient for gradient projection.
    """

    def __init__(self, gravity: float = 9.81, depth: float = 1.0):
        super().__init__()
        self.gravity = float(gravity)
        self.depth = float(depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta = x[:, 0:1]
        u = x[:, 1:2]
        v = x[:, 2:3]
        density = 0.5 * (self.gravity * eta.square() + self.depth * (u.square() + v.square()))
        return density.mean(dim=(1, 2, 3), keepdim=True).flatten(1)


class EnergyProjectedDynamics(nn.Module):
    """Project a nominal CNN derivative field to satisfy an energy decrease condition.

    f(x) = fhat(x) - gradE(x) * relu(gradE(x)·fhat(x) + alpha*E(x)) / ||gradE(x)||^2

    This is the same idea as the ICNN Lyapunov projection, but here the Lyapunov
    function is the known physical shallow-water energy instead of a learned ICNN.
    """

    def __init__(self, fhat: nn.Module, energy: PhysicalEnergy, alpha: float = 1.0e-5, denom_eps: float = 1.0e-8):
        super().__init__()
        self.fhat = fhat
        self.energy = energy
        self.alpha = float(alpha)
        self.denom_eps = float(denom_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        fx = self.fhat(x)
        e = self.energy(x)
        grad_e = torch.autograd.grad(e.sum(), x, create_graph=True, only_inputs=True)[0]
        violation = (grad_e * fx).flatten(1).sum(dim=1, keepdim=True) + self.alpha * e
        denom = grad_e.flatten(1).square().sum(dim=1, keepdim=True).clamp_min(self.denom_eps)
        correction = grad_e * (F.relu(violation).view(-1, 1, 1, 1) / denom.view(-1, 1, 1, 1))
        return fx - correction

    def energy_decrease(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        fx = self.forward(x)
        e = self.energy(x)
        grad_e = torch.autograd.grad(e.sum(), x, create_graph=True, only_inputs=True)[0]
        return (grad_e * fx).flatten(1).sum(dim=1, keepdim=True) + self.alpha * e

    def projection_diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.detach().requires_grad_(True)
        fx_hat = self.fhat(x)
        e = self.energy(x)
        grad_e = torch.autograd.grad(e.sum(), x, create_graph=False, only_inputs=True)[0]
        violation = (grad_e * fx_hat).flatten(1).sum(dim=1, keepdim=True) + self.alpha * e
        denom = grad_e.flatten(1).square().sum(dim=1, keepdim=True).clamp_min(self.denom_eps)
        correction = grad_e * (F.relu(violation).view(-1, 1, 1, 1) / denom.view(-1, 1, 1, 1))
        return {
            "violation": violation.detach(),
            "fire": (violation > 0).detach(),
            "correction_norm": correction.flatten(1).norm(dim=1).detach(),
            "energy": e.detach(),
        }


def make_cnn_dynamics(hidden_channels: int = 64, depth: int = 4) -> ConvResNetDynamics:
    return ConvResNetDynamics(in_channels=3, hidden_channels=hidden_channels, depth=depth)


def make_energy_projected_cnn(
    hidden_channels: int = 64,
    depth: int = 4,
    gravity: float = 9.81,
    depth_water: float = 1.0,
    alpha: float = 1.0e-5,
) -> EnergyProjectedDynamics:
    fhat = make_cnn_dynamics(hidden_channels=hidden_channels, depth=depth)
    energy = PhysicalEnergy(gravity=gravity, depth=depth_water)
    return EnergyProjectedDynamics(fhat=fhat, energy=energy, alpha=alpha)
