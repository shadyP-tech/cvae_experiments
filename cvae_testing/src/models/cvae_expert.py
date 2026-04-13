from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAEExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        metadata_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.metadata_dim = int(metadata_dim)
        if self.metadata_dim < 0:
            raise ValueError("metadata_dim must be >= 0")

        self.conditioning_enabled = self.metadata_dim > 0
        enc_input_dim = self.input_dim + (self.metadata_dim if self.conditioning_enabled else 0)
        dec_input_dim = self.latent_dim + (self.metadata_dim if self.conditioning_enabled else 0)

        self.enc = nn.Linear(enc_input_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(dec_input_dim, self.hidden_dim)
        self.dec2 = nn.Linear(self.hidden_dim, self.input_dim)

    def _concat_metadata(self, x: torch.Tensor, m: torch.Tensor | None, stage: str) -> torch.Tensor:
        if not self.conditioning_enabled:
            return x
        if m is None:
            raise ValueError(f"Metadata tensor is required for conditioning at stage '{stage}'.")
        if x.ndim != 2 or m.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors for conditioning at stage '{stage}', got x.ndim={x.ndim}, m.ndim={m.ndim}."
            )
        if x.shape[0] != m.shape[0]:
            raise ValueError(
                f"Batch-size mismatch at stage '{stage}': x has {x.shape[0]} rows, metadata has {m.shape[0]} rows."
            )
        if m.shape[1] != self.metadata_dim:
            raise ValueError(
                f"Metadata width mismatch at stage '{stage}': expected {self.metadata_dim}, got {m.shape[1]}."
            )
        return torch.cat([x, m], dim=1)

    def encode(self, x: torch.Tensor, m: torch.Tensor | None = None):
        x_enc = self._concat_metadata(x, m, stage="encoder")
        h = F.relu(self.enc(x_enc))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, m: torch.Tensor | None = None):
        z_dec = self._concat_metadata(z, m, stage="decoder")
        h = F.relu(self.dec1(z_dec))
        return self.dec2(h)

    def forward(self, x: torch.Tensor, m: torch.Tensor | None = None):
        mu, logvar = self.encode(x, m=m)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, m=m)
        return recon, mu, logvar


def elbo_components(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
):
    recon = F.mse_loss(recon_x, x, reduction="none").sum(dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon, kl


def negative_elbo(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    recon, kl = elbo_components(recon_x, x, mu, logvar)
    return (recon + kl).mean()
