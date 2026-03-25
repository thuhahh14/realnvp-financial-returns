import math
import torch
import torch.nn as nn


class STNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        self.s_net = STNet(dim, hidden_dim, dim)
        self.t_net = STNet(dim, hidden_dim, dim)

    def forward(self, x):
        x_masked = x * self.mask
        s = self.s_net(x_masked) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)

        s = torch.tanh(s)

        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)

        s = torch.tanh(s)

        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = (-s).sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, dim: int, n_coupling: int = 6, hidden_dim: int = 64):
        super().__init__()
        masks = []

        for i in range(n_coupling):
            base_mask = torch.tensor(([1, 0] * ((dim + 1) // 2))[:dim], dtype=torch.float32)
            if i % 2 == 1:
                base_mask = 1 - base_mask
            masks.append(base_mask)

        self.layers = nn.ModuleList([
            AffineCoupling(dim, mask, hidden_dim) for mask in masks
        ])
        self.dim = dim

    def forward_transform(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_det += ld
        return z, log_det

    def inverse_transform(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det += ld
        return x, log_det

    def log_prob(self, x):
        z, log_det = self.forward_transform(x)
        log_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return log_base + log_det

    def sample(self, n: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.inverse_transform(z)
        return x
