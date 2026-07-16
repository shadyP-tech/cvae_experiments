"""CVAE variant with a jointly learned class-conditional diagonal prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..latent_priors import (
    ConditionalPriorDiagnostics,
    LearnedClassConditionalDiagonalPrior,
    sample_diagonal_gaussian,
)
from .cvae import ClassConditionedCVAE


@dataclass(frozen=True)
class LearnedConditionalPriorLoss:
    """Dimension-normalized stochastic beta-objective components."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    beta: float


class LearnedConditionalPriorCVAE(ClassConditionedCVAE):
    """Isolated v2 CVAE with ``p(z|y)`` learned through the KL term.

    The inherited encoder and decoder are unchanged.  The prior submodule is
    registered only on this subclass, so locked v1 model state and checkpoint
    schemas remain untouched.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        n_classes: int = 2,
        num_hidden_layers: int = 2,
    ) -> None:
        if int(n_classes) != 2:
            raise ValueError("Learned conditional-prior v2 requires two classes.")
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_classes=n_classes,
            num_hidden_layers=num_hidden_layers,
        )
        # Zero initialization consumes no RNG and preserves paired base-model
        # initialization when the shared state is copied from arm A.
        self.latent_prior = LearnedClassConditionalDiagonalPrior(self.latent_dim)

    @property
    def prior_mu(self) -> nn.Parameter:
        return self.latent_prior.prior_mu

    @property
    def prior_rho(self) -> nn.Parameter:
        return self.latent_prior.prior_rho

    @property
    def prior_logvar(self) -> torch.Tensor:
        return self.latent_prior.effective_logvar()

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        """Yield encoder/decoder parameters for the existing AdamW group."""

        for name, parameter in self.named_parameters():
            if not name.startswith("latent_prior."):
                yield parameter

    def learned_prior_parameters(self) -> Iterator[nn.Parameter]:
        """Yield prior-only parameters for the zero-weight-decay group."""

        yield from self.latent_prior.parameters()

    def kl_to_prior(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.latent_prior.kl_from_posterior(
            posterior_mu,
            posterior_logvar,
            labels,
        )

    def posterior_sample(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        *,
        epsilon: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return sample_diagonal_gaussian(
            posterior_mu,
            posterior_logvar,
            epsilon=epsilon,
            generator=generator,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epsilon: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mu, posterior_logvar = self.encode(x, y)
        z = self.posterior_sample(
            posterior_mu,
            posterior_logvar,
            epsilon=epsilon,
            generator=generator,
        )
        return self.decode(z, y), posterior_mu, posterior_logvar

    def loss_for_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        beta: float,
        epsilon: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LearnedConditionalPriorLoss:
        """Return the v2 isotropic beta objective for one minibatch."""

        if float(beta) < 0.0:
            raise ValueError("beta must be nonnegative.")
        reconstruction, posterior_mu, posterior_logvar = self(
            x,
            y,
            epsilon=epsilon,
            generator=generator,
        )
        reconstruction_per_row = F.mse_loss(
            reconstruction,
            x,
            reduction="none",
        ).mean(dim=1)
        kl_per_row = self.kl_to_prior(posterior_mu, posterior_logvar, y)
        reconstruction_mean = reconstruction_per_row.mean()
        kl_mean = kl_per_row.mean()
        return LearnedConditionalPriorLoss(
            total=reconstruction_mean + float(beta) * kl_mean,
            reconstruction=reconstruction_mean,
            kl=kl_mean,
            beta=float(beta),
        )

    def nelbo_for_class(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        deterministic: bool = True,
        epsilon: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the class-conditional NELBO score used by the base API.

        Training uses the dimension-normalized beta objective above.  NELBO
        scoring retains summed reconstruction and KL terms so its subsequent
        class-prior marginalization remains on the correct additive scale.
        """

        posterior_mu, posterior_logvar = self.encode(x, y)
        if deterministic:
            if epsilon is not None or generator is not None:
                raise ValueError(
                    "Deterministic NELBO scoring does not consume sampling noise."
                )
            z = posterior_mu
        else:
            z = self.posterior_sample(
                posterior_mu,
                posterior_logvar,
                epsilon=epsilon,
                generator=generator,
            )
        reconstruction = self.decode(z, y)
        reconstruction_loss = F.mse_loss(
            reconstruction,
            x,
            reduction="none",
        ).sum(dim=1)
        kl = self.kl_to_prior(
            posterior_mu,
            posterior_logvar,
            y,
        ) * float(self.latent_dim)
        return reconstruction_loss + kl

    def marginal_nelbo(
        self,
        x: torch.Tensor,
        *,
        class_prior: Sequence[float] | None = None,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Marginalize the learned-prior-aware class scores over ``p(y)``."""

        x = _ensure_input(x, self.input_dim)
        prior = _class_prior_tensor(
            class_prior,
            n_classes=self.n_classes,
            dtype=x.dtype,
            device=x.device,
        )
        values: list[torch.Tensor] = []
        for class_label in range(self.n_classes):
            labels = torch.full(
                (x.shape[0],),
                class_label,
                dtype=torch.long,
                device=x.device,
            )
            values.append(
                -self.nelbo_for_class(
                    x,
                    labels,
                    deterministic=deterministic,
                    generator=generator if not deterministic else None,
                )
                + torch.log(prior[class_label])
            )
        return -torch.logsumexp(torch.stack(values, dim=1), dim=1)

    def sample_prior(
        self,
        labels: torch.Tensor,
        *,
        epsilon: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.latent_prior.sample(
            labels,
            epsilon=epsilon,
            generator=generator,
        )

    def prior_state_diagnostics(
        self,
        posterior_mu: torch.Tensor,
        labels: torch.Tensor,
    ) -> ConditionalPriorDiagnostics:
        return self.latent_prior.state_diagnostics(posterior_mu, labels)


def _ensure_input(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != int(input_dim):
        raise ValueError(f"Expected x shape [n,{input_dim}], got {tuple(x.shape)}.")
    return x.to(dtype=torch.float32)


def _class_prior_tensor(
    values: Sequence[float] | None,
    *,
    n_classes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        return torch.full(
            (n_classes,),
            1.0 / float(n_classes),
            dtype=dtype,
            device=device,
        )
    if len(values) != n_classes:
        raise ValueError("class_prior length must equal n_classes.")
    prior = torch.tensor([float(value) for value in values], dtype=dtype, device=device)
    if not bool(torch.isfinite(prior).all()) or bool((prior <= 0.0).any()):
        raise ValueError("class_prior entries must be finite and strictly positive.")
    return prior / prior.sum()


__all__ = ["LearnedConditionalPriorCVAE", "LearnedConditionalPriorLoss"]
