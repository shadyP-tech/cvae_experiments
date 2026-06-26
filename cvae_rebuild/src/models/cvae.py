from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CVAELoss:
    nelbo: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


class ClassConditionedCVAE(nn.Module):
    """Class-conditioned CVAE with one-hot labels in encoder and decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        n_classes: int = 2,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_hidden_layers != 2:
            raise ValueError("Locked v1 CVAE uses exactly two hidden layers.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)
        enc_in = self.input_dim + self.n_classes
        dec_in = self.latent_dim + self.n_classes
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def class_one_hot(self, y: torch.Tensor) -> torch.Tensor:
        y = y.long().view(-1)
        if y.numel() and (int(y.min()) < 0 or int(y.max()) >= self.n_classes):
            raise ValueError("Class label out of range.")
        return F.one_hot(y, num_classes=self.n_classes).to(dtype=torch.float32, device=y.device)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _ensure_2d(x, self.input_dim)
        y_one_hot = self.class_one_hot(y).to(device=x.device)
        h = self.encoder(torch.cat([x, y_one_hot], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + (torch.randn_like(std) * std)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], -1)
        y_one_hot = self.class_one_hot(y).to(device=z.device)
        return self.decoder(torch.cat([z, y_one_hot], dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def nelbo_for_class(self, x: torch.Tensor, y: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        mu, logvar = self.encode(x, y)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return recon_loss + kl

    def marginal_nelbo(
        self,
        x: torch.Tensor,
        *,
        class_prior: Sequence[float] | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        x = _ensure_2d(x, self.input_dim)
        if class_prior is None:
            prior = torch.full((self.n_classes,), 1.0 / self.n_classes, dtype=x.dtype, device=x.device)
        else:
            prior = torch.tensor([float(v) for v in class_prior], dtype=x.dtype, device=x.device)
            prior = prior / prior.sum()
        values = []
        for cls in range(self.n_classes):
            y = torch.full((x.shape[0],), cls, dtype=torch.long, device=x.device)
            values.append(-self.nelbo_for_class(x, y, deterministic=deterministic) + torch.log(prior[cls]))
        return -torch.logsumexp(torch.stack(values, dim=1), dim=1)


def loss_for_batch(model: ClassConditionedCVAE, x: torch.Tensor, y: torch.Tensor) -> CVAELoss:
    recon, mu, logvar = model(x, y)
    recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return CVAELoss(nelbo=(recon_loss + kl).mean(), recon=recon_loss.mean(), kl=kl.mean())


def _ensure_2d(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != int(input_dim):
        raise ValueError(f"Expected x shape [n,{input_dim}], got {tuple(x.shape)}.")
    return x.to(dtype=torch.float32)
