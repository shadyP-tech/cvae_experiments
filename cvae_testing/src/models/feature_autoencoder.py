from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAutoencoder(nn.Module):
    """Small deterministic MLP autoencoder for feature-embedding reconstruction."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def reconstruction_mse_per_sample(model: FeatureAutoencoder, x: torch.Tensor) -> torch.Tensor:
    recon = model(x)
    return (recon - x).pow(2).mean(dim=1)
