"""Trace-normalized stochastic CVAE beta objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ISOTROPIC_OBJECTIVE = "stochastic_isotropic_v1"
TASK_FISHER_OBJECTIVE = "stochastic_task_fisher_v1"


@dataclass(frozen=True)
class BetaObjectiveLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    beta: float


def beta_objective(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
    metric: torch.Tensor | None = None,
) -> BetaObjectiveLoss:
    """Compute the normalized beta objective without an ELBO/NELBO claim."""

    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must have identical shapes.")
    if target.ndim != 2:
        raise ValueError("CVAE beta objective expects [batch,input_dim] tensors.")
    residual = reconstruction - target
    if metric is None:
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction="none").mean(dim=1)
    else:
        if metric.ndim != 2 or metric.shape != (target.shape[1], target.shape[1]):
            raise ValueError("Task metric must be a square input-dimensional matrix.")
        metric = metric.to(dtype=residual.dtype, device=residual.device)
        reconstruction_loss = torch.einsum("bi,ij,bj->b", residual, metric, residual) / float(target.shape[1])
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / float(mu.shape[1])
    reconstruction_mean = reconstruction_loss.mean()
    kl_mean = kl.mean()
    return BetaObjectiveLoss(
        total=reconstruction_mean + (float(beta) * kl_mean),
        reconstruction=reconstruction_mean,
        kl=kl_mean,
        beta=float(beta),
    )


def validate_trace_normalized_metric(metric: object, *, input_dim: int, atol: float = 1e-5) -> None:
    """Fail closed when a reconstruction metric is not finite, PSD, or trace-normalized."""

    import numpy as np

    matrix = np.asarray(metric, dtype=np.float64)
    if matrix.shape != (int(input_dim), int(input_dim)):
        raise ValueError(f"Expected metric shape {(input_dim, input_dim)}, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Task metric contains nonfinite values.")
    if not np.allclose(matrix, matrix.T, atol=atol, rtol=0.0):
        raise ValueError("Task metric must be symmetric.")
    if float(np.linalg.eigvalsh(matrix).min()) < -atol:
        raise ValueError("Task metric must be positive semidefinite.")
    if not np.isclose(float(np.trace(matrix)), float(input_dim), atol=atol, rtol=0.0):
        raise ValueError("Task metric trace must equal input_dim.")
