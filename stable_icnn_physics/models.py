from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ReHU(nn.Module):
    """Rectified Huber unit: a smooth ReLU variant from the paper."""

    def __init__(self, width: float = 0.01):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = float(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quadratic = x.square() / (2.0 * self.width)
        linear = x - self.width / 2.0
        return torch.where(x <= 0, torch.zeros_like(x), torch.where(x < self.width, quadratic, linear))


class NominalMLP(nn.Module):
    """Plain MLP used as nominal dynamics `fhat`."""

    def __init__(self, dim: int, hidden: int = 100, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineDynamicsMLP(NominalMLP):
    """Unconstrained dynamics model for empirical comparison."""


class ICNN(nn.Module):
    """Fully input-convex neural network with positive hidden-to-hidden weights."""

    def __init__(self, layer_sizes: list[int], activation: nn.Module | None = None):
        super().__init__()
        if len(layer_sizes) < 3:
            raise ValueError("ICNN needs input, at least one hidden layer, and output size")
        if layer_sizes[-1] != 1:
            raise ValueError("This ICNN implementation is scalar-valued; final size must be 1")

        self.input_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(out_dim, layer_sizes[0])) for out_dim in layer_sizes[1:]]
        )
        self.hidden_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(layer_sizes[i + 1], layer_sizes[i])) for i in range(1, len(layer_sizes) - 1)]
        )
        self.biases = nn.ParameterList([nn.Parameter(torch.empty(out_dim)) for out_dim in layer_sizes[1:]])
        self.activation = activation if activation is not None else ReHU(width=0.01)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.input_weights:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        for weight in self.hidden_weights:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        for i, bias in enumerate(self.biases):
            fan_in = self.input_weights[i].shape[1]
            bound = fan_in**-0.5
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.input_weights[0], self.biases[0])
        z = self.activation(z)

        for input_weight, hidden_weight, bias in zip(
            self.input_weights[1:-1], self.hidden_weights[:-1], self.biases[1:-1]
        ):
            positive_hidden_weight = F.softplus(hidden_weight)
            z = F.linear(x, input_weight, bias) + F.linear(z, positive_hidden_weight) / hidden_weight.shape[0]
            z = self.activation(z)

        positive_hidden_weight = F.softplus(self.hidden_weights[-1])
        return F.linear(x, self.input_weights[-1], self.biases[-1]) + (
            F.linear(z, positive_hidden_weight) / self.hidden_weights[-1].shape[0]
        )


class PositiveDefiniteLyapunov(nn.Module):
    """Positive definite Lyapunov function built from an ICNN."""

    def __init__(self, icnn: ICNN, eps: float = 0.01, output_activation: nn.Module | None = None):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.icnn = icnn
        self.eps = float(eps)
        self.output_activation = output_activation if output_activation is not None else ReHU(width=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
        shifted = self.icnn(x) - self.icnn(zero)
        return self.output_activation(shifted) + self.eps * x.square().sum(dim=1, keepdim=True)


class StableDynamics(nn.Module):
    """Dynamics projected to satisfy the Lyapunov decrease condition."""

    def __init__(self, fhat: nn.Module, V: nn.Module, alpha: float = 1e-3, denom_eps: float = 1e-8):
        super().__init__()
        self.fhat = fhat
        self.V = V
        self.alpha = float(alpha)
        self.denom_eps = float(denom_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        fx = self.fhat(x)
        vx = self.V(x)
        grad_v = torch.autograd.grad(vx.sum(), x, create_graph=True, only_inputs=True)[0]
        violation = (grad_v * fx).sum(dim=1, keepdim=True) + self.alpha * vx
        denom = grad_v.square().sum(dim=1, keepdim=True).clamp_min(self.denom_eps)
        return fx - grad_v * (F.relu(violation) / denom)

    def lyapunov_decrease(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        fx = self.forward(x)
        vx = self.V(x)
        grad_v = torch.autograd.grad(vx.sum(), x, create_graph=True, only_inputs=True)[0]
        return (grad_v * fx).sum(dim=1, keepdim=True) + self.alpha * vx


def build_stable_model(
    dim: int,
    hidden: int = 100,
    depth: int = 2,
    lyapunov_hidden: int = 60,
    lyapunov_eps: float = 0.01,
    alpha: float = 1e-3,
    rehu_width: float = 0.01,
) -> StableDynamics:
    fhat = NominalMLP(dim=dim, hidden=hidden, depth=depth)
    icnn = ICNN([dim, lyapunov_hidden, lyapunov_hidden, 1], activation=ReHU(width=rehu_width))
    V = PositiveDefiniteLyapunov(icnn, eps=lyapunov_eps, output_activation=ReHU(width=rehu_width))
    return StableDynamics(fhat=fhat, V=V, alpha=alpha)

