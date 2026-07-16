"""Learned class-conditional diagonal latent-prior primitives.

This module is intentionally independent of the locked v1 CVAE model and
training code.  It implements the bounded learned-prior contract used by the
separate Stage-20 v2 source-inner study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


PRIOR_LOGVAR_LIMIT = 6.0
PRIOR_SATURATION_THRESHOLD = 5.9
ACTIVE_UNIT_THRESHOLD = 0.01
CLASS_SEPARATION_THRESHOLD = 1e-4
N_CLASSES = 2


@dataclass(frozen=True)
class ConditionalPriorDiagnostics:
    """Final-state mechanism diagnostics for a learned conditional prior."""

    standardized_active_unit_scores: tuple[float, ...]
    active_unit_mask: tuple[bool, ...]
    active_unit_count: int
    normalized_symmetric_kl: float
    near_class_independent: bool
    saturated: bool
    saturation_count: int
    max_abs_logvar: float
    finite: bool
    active_unit_threshold: float = ACTIVE_UNIT_THRESHOLD
    class_separation_threshold: float = CLASS_SEPARATION_THRESHOLD
    saturation_threshold: float = PRIOR_SATURATION_THRESHOLD

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_learned_conditional_prior_diagnostics_v1",
            "standardized_active_unit_scores": list(
                self.standardized_active_unit_scores
            ),
            "active_unit_mask": list(self.active_unit_mask),
            "active_unit_count": self.active_unit_count,
            "normalized_symmetric_kl": self.normalized_symmetric_kl,
            "near_class_independent": self.near_class_independent,
            "saturated": self.saturated,
            "saturation_count": self.saturation_count,
            "max_abs_logvar": self.max_abs_logvar,
            "finite": self.finite,
            "active_unit_threshold": self.active_unit_threshold,
            "class_separation_threshold": self.class_separation_threshold,
            "saturation_threshold": self.saturation_threshold,
        }


def normalized_diagonal_gaussian_kl(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """Return per-row ``KL(q || p)`` normalized by latent dimension."""

    _validate_aligned_gaussians(
        posterior_mu,
        posterior_logvar,
        prior_mu,
        prior_logvar,
    )
    latent_dim = int(posterior_mu.shape[-1])
    variance_ratio = torch.exp(posterior_logvar - prior_logvar)
    squared_mean_term = (posterior_mu - prior_mu).pow(2) * torch.exp(
        -prior_logvar
    )
    per_dimension = (
        prior_logvar
        - posterior_logvar
        + variance_ratio
        + squared_mean_term
        - 1.0
    )
    return 0.5 * per_dimension.sum(dim=-1) / float(latent_dim)


def sample_diagonal_gaussian(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    epsilon: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a diagonal Gaussian using explicit or generator-backed noise."""

    if mean.shape != logvar.shape or mean.ndim < 1 or mean.shape[-1] <= 0:
        raise ValueError("mean and logvar must be aligned nonempty tensors.")
    if epsilon is not None and generator is not None:
        raise ValueError("Provide epsilon or generator, not both.")
    if epsilon is None:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
    else:
        noise = torch.as_tensor(epsilon, dtype=mean.dtype, device=mean.device)
        if noise.shape != mean.shape:
            raise ValueError(
                f"epsilon shape {tuple(noise.shape)} does not match "
                f"Gaussian shape {tuple(mean.shape)}."
            )
    return mean + torch.exp(0.5 * logvar) * noise


def standardized_active_unit_scores(
    posterior_mu: torch.Tensor,
    labels: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """Return class-balanced, prior-scale-normalized posterior-mean variance.

    For latent dimension ``j`` this computes

    ``0.5 * sum_y Var_{x:y}[(mu_q(x,y)-mu_p(y))/sigma_p(y)]``

    with population variance (``ddof=0``).  Both binary classes must be
    represented in the source-fit rows.
    """

    _validate_prior_state(prior_mu, prior_logvar)
    if posterior_mu.ndim != 2 or posterior_mu.shape[1] != prior_mu.shape[1]:
        raise ValueError("posterior_mu must have shape [rows, latent_dim].")
    y = _validated_labels(
        labels,
        n_rows=int(posterior_mu.shape[0]),
        device=posterior_mu.device,
    )
    class_scores: list[torch.Tensor] = []
    for class_label in range(N_CLASSES):
        class_mask = y == class_label
        if not bool(class_mask.any()):
            raise ValueError("Active-unit diagnostics require both classes.")
        standardized = (
            posterior_mu[class_mask]
            - prior_mu[class_label].to(
                dtype=posterior_mu.dtype,
                device=posterior_mu.device,
            )
        ) * torch.exp(
            -0.5
            * prior_logvar[class_label].to(
                dtype=posterior_mu.dtype,
                device=posterior_mu.device,
            )
        )
        class_scores.append(standardized.var(dim=0, unbiased=False))
    return 0.5 * (class_scores[0] + class_scores[1])


def normalized_symmetric_diagonal_kl(
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """Return latent-normalized symmetric KL between the two class priors."""

    _validate_prior_state(prior_mu, prior_logvar)
    forward = normalized_diagonal_gaussian_kl(
        prior_mu[0],
        prior_logvar[0],
        prior_mu[1],
        prior_logvar[1],
    )
    reverse = normalized_diagonal_gaussian_kl(
        prior_mu[1],
        prior_logvar[1],
        prior_mu[0],
        prior_logvar[0],
    )
    return 0.5 * (forward + reverse)


def conditional_prior_diagnostics(
    posterior_mu: torch.Tensor,
    labels: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    *,
    prior_rho: torch.Tensor | None = None,
    active_unit_threshold: float = ACTIVE_UNIT_THRESHOLD,
    class_separation_threshold: float = CLASS_SEPARATION_THRESHOLD,
    saturation_threshold: float = PRIOR_SATURATION_THRESHOLD,
) -> ConditionalPriorDiagnostics:
    """Compute the hash-bound final-state learned-prior diagnostics."""

    if active_unit_threshold < 0.0 or class_separation_threshold < 0.0:
        raise ValueError("Diagnostic thresholds must be nonnegative.")
    if not 0.0 < saturation_threshold < PRIOR_LOGVAR_LIMIT:
        raise ValueError("Saturation threshold must lie inside logvar bounds.")
    scores = standardized_active_unit_scores(
        posterior_mu,
        labels,
        prior_mu,
        prior_logvar,
    )
    symmetric_kl = normalized_symmetric_diagonal_kl(prior_mu, prior_logvar)
    saturation_mask = prior_logvar.abs() >= float(saturation_threshold)
    finite_tensors = [posterior_mu, prior_mu, prior_logvar]
    if prior_rho is not None:
        if prior_rho.shape != prior_mu.shape:
            raise ValueError("prior_rho must match prior_mu shape.")
        finite_tensors.append(prior_rho)
    finite = all(bool(torch.isfinite(value).all()) for value in finite_tensors)
    active_mask = scores > float(active_unit_threshold)
    detached_scores = scores.detach().cpu()
    detached_active = active_mask.detach().cpu()
    symmetric_value = float(symmetric_kl.detach().cpu())
    return ConditionalPriorDiagnostics(
        standardized_active_unit_scores=tuple(
            float(value) for value in detached_scores.tolist()
        ),
        active_unit_mask=tuple(bool(value) for value in detached_active.tolist()),
        active_unit_count=int(detached_active.sum().item()),
        normalized_symmetric_kl=symmetric_value,
        near_class_independent=(
            finite and symmetric_value <= float(class_separation_threshold)
        ),
        saturated=bool(saturation_mask.any()),
        saturation_count=int(saturation_mask.sum().detach().cpu().item()),
        max_abs_logvar=float(prior_logvar.abs().max().detach().cpu()),
        finite=finite,
        active_unit_threshold=float(active_unit_threshold),
        class_separation_threshold=float(class_separation_threshold),
        saturation_threshold=float(saturation_threshold),
    )


class LearnedClassConditionalDiagonalPrior(nn.Module):
    """Binary learned diagonal Gaussian prior with smooth logvar bounds."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive.")
        self.latent_dim = int(latent_dim)
        self.n_classes = N_CLASSES
        self.prior_mu = nn.Parameter(
            torch.zeros(self.n_classes, self.latent_dim, dtype=torch.float32)
        )
        self.prior_rho = nn.Parameter(
            torch.zeros(self.n_classes, self.latent_dim, dtype=torch.float32)
        )

    def effective_logvar(self) -> torch.Tensor:
        return PRIOR_LOGVAR_LIMIT * torch.tanh(
            self.prior_rho / PRIOR_LOGVAR_LIMIT
        )

    def parameters_for_labels(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = _validated_labels(labels, device=self.prior_mu.device)
        return self.prior_mu[y], self.effective_logvar()[y]

    def kl_from_posterior(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if posterior_mu.ndim != 2:
            raise ValueError("Posterior parameters must have shape [batch, latent_dim].")
        if posterior_mu.shape[1] != self.latent_dim:
            raise ValueError("Posterior latent dimension differs from the prior.")
        prior_mu, prior_logvar = self.parameters_for_labels(labels)
        return normalized_diagonal_gaussian_kl(
            posterior_mu,
            posterior_logvar,
            prior_mu.to(dtype=posterior_mu.dtype, device=posterior_mu.device),
            prior_logvar.to(
                dtype=posterior_logvar.dtype,
                device=posterior_logvar.device,
            ),
        )

    def sample(
        self,
        labels: torch.Tensor,
        *,
        epsilon: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not bool(torch.isfinite(self.prior_mu).all()) or not bool(
            torch.isfinite(self.prior_rho).all()
        ):
            raise FloatingPointError("Learned prior parameters are nonfinite.")
        prior_mu, prior_logvar = self.parameters_for_labels(labels)
        return sample_diagonal_gaussian(
            prior_mu,
            prior_logvar,
            epsilon=epsilon,
            generator=generator,
        )

    def state_diagnostics(
        self,
        posterior_mu: torch.Tensor,
        labels: torch.Tensor,
    ) -> ConditionalPriorDiagnostics:
        return conditional_prior_diagnostics(
            posterior_mu,
            labels,
            self.prior_mu,
            self.effective_logvar(),
            prior_rho=self.prior_rho,
        )

    def state_payload(self) -> Mapping[str, object]:
        return {
            "schema_version": "midogpp_learned_conditional_diagonal_prior_v1",
            "n_classes": self.n_classes,
            "latent_dim": self.latent_dim,
            "logvar_parameterization": "6*tanh(rho/6)",
            "logvar_limit": PRIOR_LOGVAR_LIMIT,
            "prior_mu": self.prior_mu.detach().cpu().tolist(),
            "prior_rho": self.prior_rho.detach().cpu().tolist(),
            "effective_logvar": self.effective_logvar().detach().cpu().tolist(),
        }


def _validated_labels(
    labels: torch.Tensor,
    *,
    n_rows: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    if torch.is_floating_point(labels) or labels.dtype == torch.bool:
        raise ValueError("Class labels must use an integer dtype.")
    y = labels.to(device=device, dtype=torch.long).reshape(-1)
    if n_rows is not None and y.numel() != int(n_rows):
        raise ValueError("Class labels are not aligned with posterior rows.")
    if y.numel() and (int(y.min()) < 0 or int(y.max()) >= N_CLASSES):
        raise ValueError("Class labels must be binary values 0/1.")
    return y


def _validate_prior_state(
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> None:
    expected_prefix = (N_CLASSES,)
    if (
        prior_mu.ndim != 2
        or prior_mu.shape != prior_logvar.shape
        or prior_mu.shape[0] != expected_prefix[0]
        or prior_mu.shape[1] <= 0
    ):
        raise ValueError("Prior parameters must have shape [2, latent_dim].")


def _validate_aligned_gaussians(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> None:
    shape = posterior_mu.shape
    if (
        posterior_mu.ndim < 1
        or shape[-1] <= 0
        or posterior_logvar.shape != shape
        or prior_mu.shape != shape
        or prior_logvar.shape != shape
    ):
        raise ValueError("All Gaussian parameters must have aligned latent shapes.")


__all__ = [
    "ACTIVE_UNIT_THRESHOLD",
    "CLASS_SEPARATION_THRESHOLD",
    "PRIOR_LOGVAR_LIMIT",
    "PRIOR_SATURATION_THRESHOLD",
    "ConditionalPriorDiagnostics",
    "LearnedClassConditionalDiagonalPrior",
    "conditional_prior_diagnostics",
    "normalized_diagonal_gaussian_kl",
    "normalized_symmetric_diagonal_kl",
    "sample_diagonal_gaussian",
    "standardized_active_unit_scores",
]
