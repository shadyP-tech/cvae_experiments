"""CVAE with a class-conditional compressed aggregate-posterior mixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..latent_mixture_prior import ClassConditionalLowRankMixturePrior
from .cvae import ClassConditionedCVAE


@dataclass(frozen=True)
class MixturePriorLoss:
    """Dimension-normalized objective components."""

    total: torch.Tensor
    distortion: torch.Tensor
    rate: torch.Tensor
    beta: float | None
    objective: str


class AggregateMatchedMixturePriorCVAE(ClassConditionedCVAE):
    """Isolated CVAE model used by the v3 aggregate-prior study."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        n_classes: int = 2,
        num_hidden_layers: int = 2,
        *,
        n_components: int = 2,
        mixture_rank: int = 2,
        weight_floor: float = 0.05,
        variance_floor: float = 1e-4,
    ) -> None:
        if int(n_classes) != 2:
            raise ValueError("The aggregate mixture prior requires two classes.")
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_classes=n_classes,
            num_hidden_layers=num_hidden_layers,
        )
        self.latent_prior = ClassConditionalLowRankMixturePrior(
            latent_dim,
            n_components=n_components,
            rank=mixture_rank,
            weight_floor=weight_floor,
            variance_floor=variance_floor,
        )

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("latent_prior."):
                yield parameter

    def mixture_prior_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.latent_prior.parameters()

    def forward_with_epsilon(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mu, posterior_logvar = self.encode(x, y)
        noise = torch.as_tensor(
            epsilon,
            dtype=posterior_mu.dtype,
            device=posterior_mu.device,
        )
        if noise.shape != posterior_mu.shape:
            raise ValueError("epsilon must match posterior shape.")
        latent = posterior_mu + torch.exp(0.5 * posterior_logvar) * noise
        return self.decode(latent, y), posterior_mu, posterior_logvar

    def objective_components(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction, posterior_mu, posterior_logvar = self.forward_with_epsilon(
            x,
            y,
            epsilon=epsilon,
        )
        distortion = F.mse_loss(
            reconstruction,
            x,
            reduction="none",
        ).mean(dim=-1)
        rate = self.latent_prior.kl_upper_bound(
            posterior_mu,
            posterior_logvar,
            y,
        )
        return distortion, rate

    def fixed_beta_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epsilon: torch.Tensor,
        beta: float,
    ) -> MixturePriorLoss:
        if float(beta) < 0.0:
            raise ValueError("beta must be nonnegative.")
        distortion, rate = self.objective_components(x, y, epsilon=epsilon)
        mean_distortion = distortion.mean()
        mean_rate = rate.mean()
        return MixturePriorLoss(
            total=mean_distortion + float(beta) * mean_rate,
            distortion=mean_distortion,
            rate=mean_rate,
            beta=float(beta),
            objective="fixed_beta",
        )

    def sample_prior(
        self,
        labels: torch.Tensor,
        *,
        epsilon: torch.Tensor,
        component_uniform: torch.Tensor,
    ) -> torch.Tensor:
        return self.latent_prior.sample(
            labels,
            epsilon=epsilon,
            component_uniform=component_uniform,
        )

    def nelbo_for_class(  # type: ignore[override]
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        raise RuntimeError(
            "The mixture variational KL bound is not an exact NELBO. "
            "Use objective_components() for training or an explicit IWAE "
            "estimator for density evaluation."
        )
