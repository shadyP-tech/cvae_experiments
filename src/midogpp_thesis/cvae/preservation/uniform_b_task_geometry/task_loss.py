"""Pure Torch prior-sample task-distribution objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from ....common.hashing import stable_hash
from ...keyed_training import AuxiliaryContext, AuxiliaryResult
from ...protocol import ProtocolError
from .task_geometry import FoldTaskGeometry, TaskGeometryState


@dataclass(frozen=True)
class TaskTermScales:
    mmd: float
    margin: float
    gradient: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.mmd, self.margin, self.gradient)
        ):
            raise ProtocolError("Task-term scales must be finite and positive.")

    @property
    def state_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": "midogpp_task_term_scales_v1",
                "mmd": self.mmd,
                "margin": self.margin,
                "gradient": self.gradient,
            }
        )


@dataclass(frozen=True)
class TaskLossWeights:
    mmd: float
    margin: float
    gradient: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.mmd, self.margin, self.gradient)
        ):
            raise ProtocolError("Task-loss weights must be finite and nonnegative.")


class FrozenTaskObjective:
    """Hashed source-only objective compatible with the keyed trainer."""

    def __init__(
        self,
        geometry: TaskGeometryState,
        *,
        scales: TaskTermScales,
        weights: TaskLossWeights,
        cdf_temperature: float,
        device: str,
    ) -> None:
        self.geometry = geometry
        self.scales = scales
        self.weights = weights
        self.cdf_temperature = float(cdf_temperature)
        self.device = str(device)
        if self.cdf_temperature <= 0.0:
            raise ProtocolError("CDF temperature must be positive.")
        self._folds = tuple(
            _TorchFold(fold, device=self.device) for fold in geometry.folds
        )
        self._identity_hash = stable_hash(
            {
                "schema_version": "midogpp_frozen_task_objective_v1",
                "geometry_hash": geometry.state_hash,
                "scale_hash": scales.state_hash,
                "weights": {
                    "mmd": weights.mmd,
                    "margin": weights.margin,
                    "gradient": weights.gradient,
                },
                "cdf_temperature": self.cdf_temperature,
            }
        )

    @property
    def identity_hash(self) -> str:
        return self._identity_hash

    def __call__(self, context: AuxiliaryContext) -> AuxiliaryResult:
        generated = context.model.decode(
            context.prior_z,
            context.requested_labels,
        )
        terms = task_terms(
            generated,
            context.requested_labels,
            folds=self._folds,
            cdf_temperature=self.cdf_temperature,
        )
        loss = (
            self.weights.mmd * self.scales.mmd * terms["mmd"]
            + self.weights.margin * self.scales.margin * terms["margin"]
            + self.weights.gradient
            * self.scales.gradient
            * terms["gradient"]
        )
        diagnostics = {
            "task_mmd": float(terms["mmd"].detach().cpu()),
            "task_margin": float(terms["margin"].detach().cpu()),
            "task_gradient": float(terms["gradient"].detach().cpu()),
        }
        return AuxiliaryResult(loss=loss, diagnostics=diagnostics)


def calibrate_task_scales(
    generated: torch.Tensor,
    labels: torch.Tensor,
    geometry: TaskGeometryState,
    *,
    cdf_temperature: float,
    device: str,
    floor: float = 1e-6,
    maximum: float = 1e6,
) -> TaskTermScales:
    folds = tuple(_TorchFold(fold, device=device) for fold in geometry.folds)
    values = task_terms(
        generated,
        labels,
        folds=folds,
        cdf_temperature=cdf_temperature,
    )
    resolved = {
        key: min(maximum, 1.0 / max(floor, float(value.detach().cpu())))
        for key, value in values.items()
    }
    return TaskTermScales(
        mmd=resolved["mmd"],
        margin=resolved["margin"],
        gradient=resolved["gradient"],
    )


def task_terms(
    generated: torch.Tensor,
    labels: torch.Tensor,
    *,
    folds: tuple["_TorchFold", ...],
    cdf_temperature: float,
) -> Mapping[str, torch.Tensor]:
    if (
        generated.ndim != 2
        or generated.shape[1] != 128
        or labels.ndim != 1
        or len(generated) != len(labels)
        or int((labels == 0).sum()) == 0
        or int((labels == 1).sum()) == 0
    ):
        raise ProtocolError("Task-loss batch must be balanced binary [n,128].")
    totals = {
        "mmd": torch.zeros((), dtype=generated.dtype, device=generated.device),
        "margin": torch.zeros((), dtype=generated.dtype, device=generated.device),
        "gradient": torch.zeros((), dtype=generated.dtype, device=generated.device),
    }
    for fold in folds:
        phi = fold.transform(generated)
        logits = phi @ fold.coef + fold.intercept
        augmented = torch.cat(
            [
                phi,
                torch.ones(
                    (len(phi), 1),
                    dtype=phi.dtype,
                    device=phi.device,
                ),
            ],
            dim=1,
        )
        for cls in (0, 1):
            generated_cls = generated[labels == cls]
            reference_cls = fold.reference[fold.reference_labels == cls]
            totals["mmd"] = totals["mmd"] + _multiscale_mmd(
                generated_cls,
                reference_cls,
                fold.bandwidths,
            )
            class_logits = logits[labels == cls]
            observed_cdf = torch.sigmoid(
                (
                    fold.cdf_grids[cls][:, None]
                    - class_logits[None, :]
                )
                / float(cdf_temperature)
            ).mean(dim=1)
            totals["margin"] = totals["margin"] + torch.mean(
                (observed_cdf - fold.cdf_targets[cls]).square()
            )
        probabilities = torch.sigmoid(logits)
        gradient = torch.zeros(
            augmented.shape[1],
            dtype=augmented.dtype,
            device=augmented.device,
        )
        for cls in (0, 1):
            mask = labels == cls
            gradient = gradient + 0.5 * (
                augmented[mask]
                * (probabilities[mask] - labels[mask].to(probabilities.dtype))[
                    :, None
                ]
            ).mean(dim=0)
        discrepancy = fold.hessian_inverse_sqrt @ (
            gradient - fold.reference_gradient
        )
        totals["gradient"] = totals["gradient"] + torch.mean(
            discrepancy.square()
        )
    normalization = float(len(folds) * 2)
    return {
        "mmd": totals["mmd"] / normalization,
        "margin": totals["margin"] / normalization,
        "gradient": totals["gradient"] / float(len(folds)),
    }


def _multiscale_mmd(
    generated: torch.Tensor,
    reference: torch.Tensor,
    bandwidths: torch.Tensor,
) -> torch.Tensor:
    xx = torch.cdist(generated, generated).square()
    yy = torch.cdist(reference, reference).square()
    xy = torch.cdist(generated, reference).square()
    total = torch.zeros((), dtype=generated.dtype, device=generated.device)
    for bandwidth in bandwidths:
        denominator = 2.0 * bandwidth.square().clamp_min(1e-12)
        total = total + (
            torch.exp(-xx / denominator).mean()
            + torch.exp(-yy / denominator).mean()
            - 2.0 * torch.exp(-xy / denominator).mean()
        )
    return total / float(len(bandwidths))


class _TorchFold:
    def __init__(self, fold: FoldTaskGeometry, *, device: str) -> None:
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(  # noqa: E731
            value,
            dtype=dtype,
            device=device,
        ).detach()
        self.scaler_mean = tensor(fold.scaler_mean)
        self.scaler_scale = tensor(fold.scaler_scale)
        self.components = tensor(fold.nystrom_components)
        self.normalization = tensor(fold.nystrom_normalization)
        self.gamma = float(fold.nystrom_gamma)
        self.coef = tensor(fold.teacher_coef)
        self.intercept = tensor(fold.teacher_intercept)
        self.hessian_inverse_sqrt = tensor(fold.hessian_inverse_sqrt)
        self.reference = tensor(fold.reference_projected)
        self.reference_labels = tensor(fold.reference_labels, torch.long)
        self.reference_gradient = tensor(fold.reference_gradient)
        self.cdf_grids = tensor(fold.cdf_grids)
        self.cdf_targets = tensor(fold.cdf_targets)
        self.bandwidths = tensor(fold.mmd_bandwidths)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        scaled = (values - self.scaler_mean) / self.scaler_scale
        squared = torch.cdist(scaled, self.components).square()
        kernel = torch.exp(-self.gamma * squared)
        return kernel @ self.normalization.T


__all__ = (
    "FrozenTaskObjective",
    "TaskLossWeights",
    "TaskTermScales",
    "calibrate_task_scales",
    "task_terms",
)
