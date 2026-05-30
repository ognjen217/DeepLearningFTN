from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ReHU(nn.Module):
    """Rectified Huber unit used by the ICNN Lyapunov network."""

    def __init__(self, width: float = 0.01):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = float(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quadratic = x.square() / (2.0 * self.width)
        linear = x - self.width / 2.0
        return torch.where(x <= 0, torch.zeros_like(x), torch.where(x < self.width, quadratic, linear))


class ConvEncoder(nn.Module):
    """Small convolutional encoder for SWE2D states.

    Input shape: ``(batch, 3, ny, nx)``.
    Output shape: ``(batch, z_dim)``.

    ``ny`` and ``nx`` must be divisible by 8.
    """

    def __init__(self, z_dim: int = 64, grid_shape: tuple[int, int] = (32, 32), hidden: int = 32):
        super().__init__()
        ny, nx = grid_shape
        if ny % 8 != 0 or nx % 8 != 0:
            raise ValueError("grid_shape must be divisible by 8")
        self.z_dim = int(z_dim)
        self.grid_shape = (int(ny), int(nx))
        self.hidden = int(hidden)
        self.feature_shape = (hidden * 4, ny // 8, nx // 8)
        flat_dim = self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]

        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.fc = nn.Linear(flat_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        return self.fc(z.flatten(1))


class ConvDecoder(nn.Module):
    """Convolutional decoder for latent SWE2D states."""

    def __init__(self, z_dim: int = 64, grid_shape: tuple[int, int] = (32, 32), hidden: int = 32):
        super().__init__()
        ny, nx = grid_shape
        if ny % 8 != 0 or nx % 8 != 0:
            raise ValueError("grid_shape must be divisible by 8")
        self.z_dim = int(z_dim)
        self.grid_shape = (int(ny), int(nx))
        self.hidden = int(hidden)
        self.feature_shape = (hidden * 4, ny // 8, nx // 8)
        flat_dim = self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]

        self.fc = nn.Sequential(nn.Linear(z_dim, flat_dim), nn.GELU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.shape[0], *self.feature_shape)
        return self.deconv(h)


class ConvAutoencoder(nn.Module):
    """Deterministic convolutional autoencoder for SWE2D states."""

    def __init__(self, z_dim: int = 64, grid_shape: tuple[int, int] = (32, 32), hidden: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(z_dim=z_dim, grid_shape=grid_shape, hidden=hidden)
        self.decoder = ConvDecoder(z_dim=z_dim, grid_shape=grid_shape, hidden=hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class LatentNominalMLP(nn.Module):
    """Nominal latent dynamics ``zhat_dot = fhat(z)``."""

    def __init__(self, z_dim: int = 64, hidden: int = 128, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ICNN(nn.Module):
    """Scalar input-convex neural network in latent space.

    Convexity is enforced by passing hidden-to-hidden weights through softplus.
    """

    def __init__(self, layer_sizes: list[int], activation: nn.Module | None = None):
        super().__init__()
        if len(layer_sizes) < 3:
            raise ValueError("ICNN needs input, at least one hidden layer, and output size")
        if layer_sizes[-1] != 1:
            raise ValueError("ICNN must be scalar-valued")

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
        for w in self.input_weights:
            nn.init.kaiming_uniform_(w, a=5**0.5)
        for w in self.hidden_weights:
            nn.init.kaiming_uniform_(w, a=5**0.5)
        for i, b in enumerate(self.biases):
            fan_in = self.input_weights[i].shape[1]
            nn.init.uniform_(b, -fan_in**-0.5, fan_in**-0.5)

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


class PositiveDefiniteLatentLyapunov(nn.Module):
    """Positive definite ICNN Lyapunov function in latent space."""

    def __init__(self, icnn: ICNN, eps: float = 0.01, output_activation: nn.Module | None = None):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.icnn = icnn
        self.eps = float(eps)
        self.output_activation = output_activation if output_activation is not None else ReHU(width=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, z.shape[-1], device=z.device, dtype=z.dtype)
        shifted = self.icnn(z) - self.icnn(zero)
        return self.output_activation(shifted) + self.eps * z.square().sum(dim=1, keepdim=True)


class LatentStableDynamics(nn.Module):
    """Lyapunov-projected latent dynamics using an ICNN V(z)."""

    def __init__(self, fhat: nn.Module, V: nn.Module, alpha: float = 1.0e-5, denom_eps: float = 1.0e-8):
        super().__init__()
        self.fhat = fhat
        self.V = V
        self.alpha = float(alpha)
        self.denom_eps = float(denom_eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        fz = self.fhat(z)
        vz = self.V(z)
        grad_v = torch.autograd.grad(vz.sum(), z, create_graph=True, only_inputs=True)[0]
        violation = (grad_v * fz).sum(dim=1, keepdim=True) + self.alpha * vz
        denom = grad_v.square().sum(dim=1, keepdim=True).clamp_min(self.denom_eps)
        return fz - grad_v * (F.relu(violation) / denom)

    def lyapunov_decrease(self, z: torch.Tensor) -> torch.Tensor:
        z = z.detach().requires_grad_(True)
        fz = self.forward(z)
        vz = self.V(z)
        grad_v = torch.autograd.grad(vz.sum(), z, create_graph=True, only_inputs=True)[0]
        return (grad_v * fz).sum(dim=1, keepdim=True) + self.alpha * vz

    def projection_diagnostics(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = z.detach().requires_grad_(True)
        fz_hat = self.fhat(z)
        vz = self.V(z)
        grad_v = torch.autograd.grad(vz.sum(), z, create_graph=False, only_inputs=True)[0]
        violation = (grad_v * fz_hat).sum(dim=1, keepdim=True) + self.alpha * vz
        denom = grad_v.square().sum(dim=1, keepdim=True).clamp_min(self.denom_eps)
        correction = grad_v * (F.relu(violation) / denom)
        return {
            "violation": violation.detach(),
            "fire": (violation > 0).detach(),
            "correction_norm": correction.norm(dim=1).detach(),
            "V": vz.detach(),
        }


class LatentICNNSWEModel(nn.Module):
    """Autoencoder + ICNN Lyapunov-stable latent dynamics for SWE2D.

    The ICNN is not applied directly to the full grid. Instead:

        x_t -> encoder -> z_t
        z_{t+1} = z_t + dt * f_stable(z_t)
        z_{t+1} -> decoder -> x_{t+1}

    This follows the same motivation as stable latent dynamics for video-like
    systems: high-dimensional fields are first compressed, and stability is
    imposed in a lower-dimensional latent state.
    """

    def __init__(
        self,
        z_dim: int = 64,
        grid_shape: tuple[int, int] = (32, 32),
        ae_hidden: int = 32,
        dyn_hidden: int = 128,
        dyn_depth: int = 3,
        lyapunov_hidden: int = 128,
        lyapunov_eps: float = 0.01,
        alpha: float = 1.0e-5,
        rehu_width: float = 0.01,
    ):
        super().__init__()
        self.autoencoder = ConvAutoencoder(z_dim=z_dim, grid_shape=grid_shape, hidden=ae_hidden)
        fhat = LatentNominalMLP(z_dim=z_dim, hidden=dyn_hidden, depth=dyn_depth)
        icnn = ICNN([z_dim, lyapunov_hidden, lyapunov_hidden, 1], activation=ReHU(width=rehu_width))
        V = PositiveDefiniteLatentLyapunov(icnn, eps=lyapunov_eps, output_activation=ReHU(width=rehu_width))
        self.latent_dynamics = LatentStableDynamics(fhat=fhat, V=V, alpha=alpha)

    @property
    def encoder(self) -> ConvEncoder:
        return self.autoencoder.encoder

    @property
    def decoder(self) -> ConvDecoder:
        return self.autoencoder.decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)

    def latent_step(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        return z + float(dt) * self.latent_dynamics(z)

    def forward_next(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        z = self.encode(x)
        z_next = self.latent_step(z, dt)
        return self.decode(z_next)

    def forward(self, x: torch.Tensor, dt: float = 2.0e-3) -> torch.Tensor:
        return self.forward_next(x, dt=dt)


def make_latent_icnn_swe_model(
    z_dim: int = 64,
    grid_shape: tuple[int, int] = (32, 32),
    ae_hidden: int = 32,
    dyn_hidden: int = 128,
    dyn_depth: int = 3,
    lyapunov_hidden: int = 128,
    alpha: float = 1.0e-5,
) -> LatentICNNSWEModel:
    return LatentICNNSWEModel(
        z_dim=z_dim,
        grid_shape=grid_shape,
        ae_hidden=ae_hidden,
        dyn_hidden=dyn_hidden,
        dyn_depth=dyn_depth,
        lyapunov_hidden=lyapunov_hidden,
        alpha=alpha,
    )
