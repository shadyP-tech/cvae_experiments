"""Control-anchored dense residual soft weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError


WEIGHT_SEMANTICS = (
    "uniform_anchored_residual_softmax_negative_calibrated_energy_"
    "automatic_max_weight_and_effective_source_constraints"
)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_SOURCE_WEIGHT = 0.25
DEFAULT_MIN_EFFECTIVE_SOURCES = 6.0


@dataclass(frozen=True)
class ResidualSoftWeights:
    candidate_sources: tuple[str, ...]
    calibrated_energy_by_source: Mapping[str, float]
    uniform_weights: Mapping[str, float]
    direction_weights: Mapping[str, float]
    weights: Mapping[str, float]
    requested_rho: float
    applied_rho: float
    temperature: float
    max_source_weight: float
    minimum_effective_sources: float
    effective_source_count: float
    active_constraints: tuple[str, ...]
    weight_semantics: str = WEIGHT_SEMANTICS


def residual_soft_weights(
    calibrated_energy_by_source: Mapping[str, float],
    *,
    rho: float,
    temperature: float = DEFAULT_TEMPERATURE,
    max_source_weight: float = DEFAULT_MAX_SOURCE_WEIGHT,
    minimum_effective_sources: float = DEFAULT_MIN_EFFECTIVE_SOURCES,
    tie_tolerance: float = 0.0,
) -> ResidualSoftWeights:
    """Tilt uniform weights toward low energy and shrink to density bounds.

    The requested residual strength is reduced analytically when necessary so
    that no source exceeds ``max_source_weight`` and
    ``1 / sum_e(w_e**2)`` remains at least ``minimum_effective_sources``.
    ``rho == 0`` and exact score ties return the same uniform values directly,
    rather than relying on a numerically approximate softmax.
    """

    normalized_scores: dict[str, float] = {}
    for raw_source, raw_value in calibrated_energy_by_source.items():
        source = str(raw_source)
        if not source or source in normalized_scores:
            raise ProtocolError("Residual router source keys must be unique and nonempty.")
        normalized_scores[source] = float(raw_value)
    sources = tuple(sorted(normalized_scores))
    values = np.asarray([normalized_scores[source] for source in sources], dtype=np.float64)
    requested_rho = float(rho)
    tau = float(temperature)
    max_weight = float(max_source_weight)
    min_effective = float(minimum_effective_sources)
    tolerance = float(tie_tolerance)
    n_sources = len(sources)
    if (
        not n_sources
        or not np.isfinite(values).all()
        or not np.isfinite(requested_rho)
        or requested_rho < 0.0
        or requested_rho > 1.0
        or not np.isfinite(tau)
        or tau <= 0.0
        or not np.isfinite(max_weight)
        or max_weight <= 0.0
        or max_weight > 1.0
        or not np.isfinite(min_effective)
        or min_effective < 1.0
        or min_effective > float(n_sources)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ProtocolError("Residual soft-weight contract is invalid.")
    uniform_value = 1.0 / float(n_sources)
    if max_weight + 1e-15 < uniform_value:
        raise ProtocolError("Maximum source weight excludes the uniform anchor.")
    uniform = np.full(n_sources, uniform_value, dtype=np.float64)

    if requested_rho == 0.0:
        return _uniform_result(
            sources=sources,
            values=values,
            uniform=uniform,
            requested_rho=requested_rho,
            tau=tau,
            max_weight=max_weight,
            min_effective=min_effective,
            reason="rho_zero_uniform",
        )
    if float(values.max() - values.min()) <= tolerance:
        return _uniform_result(
            sources=sources,
            values=values,
            uniform=uniform,
            requested_rho=requested_rho,
            tau=tau,
            max_weight=max_weight,
            min_effective=min_effective,
            reason="exact_score_tie_uniform",
        )

    logits = -values / tau
    logits -= float(logits.max())
    direction = np.exp(logits)
    direction /= float(direction.sum())
    delta = direction - uniform
    rho_limit = 1.0
    constraints: list[str] = []

    positive_delta = delta > 0.0
    if np.any(positive_delta):
        cap_limits = (max_weight - uniform[positive_delta]) / delta[positive_delta]
        max_weight_limit = max(0.0, float(cap_limits.min()))
        if max_weight_limit < rho_limit:
            rho_limit = max_weight_limit
        if requested_rho > max_weight_limit:
            constraints.append("max_source_weight")

    squared_direction = float(np.dot(delta, delta))
    if squared_direction > 0.0:
        allowed_concentration = 1.0 / min_effective - 1.0 / float(n_sources)
        effective_limit = float(
            np.sqrt(max(0.0, allowed_concentration) / squared_direction)
        )
        if effective_limit < rho_limit:
            rho_limit = effective_limit
        if requested_rho > effective_limit:
            constraints.append("minimum_effective_sources")

    applied_rho = min(requested_rho, rho_limit)
    applied_rho = min(1.0, max(0.0, float(applied_rho)))
    weights = uniform + applied_rho * delta
    weights /= float(weights.sum())
    effective = float(1.0 / np.dot(weights, weights))
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or float(weights.max()) > max_weight + 1e-12
        or effective < min_effective - 1e-10
    ):
        raise ProtocolError("Automatic residual density constraints failed closed.")
    return ResidualSoftWeights(
        candidate_sources=sources,
        calibrated_energy_by_source=_mapping(sources, values),
        uniform_weights=_mapping(sources, uniform),
        direction_weights=_mapping(sources, direction),
        weights=_mapping(sources, weights),
        requested_rho=requested_rho,
        applied_rho=applied_rho,
        temperature=tau,
        max_source_weight=max_weight,
        minimum_effective_sources=min_effective,
        effective_source_count=effective,
        active_constraints=tuple(constraints),
    )


def _uniform_result(
    *,
    sources: tuple[str, ...],
    values: np.ndarray,
    uniform: np.ndarray,
    requested_rho: float,
    tau: float,
    max_weight: float,
    min_effective: float,
    reason: str,
) -> ResidualSoftWeights:
    return ResidualSoftWeights(
        candidate_sources=sources,
        calibrated_energy_by_source=_mapping(sources, values),
        uniform_weights=_mapping(sources, uniform),
        direction_weights=_mapping(sources, uniform),
        weights=_mapping(sources, uniform),
        requested_rho=requested_rho,
        applied_rho=0.0,
        temperature=tau,
        max_source_weight=max_weight,
        minimum_effective_sources=min_effective,
        effective_source_count=float(len(sources)),
        active_constraints=(reason,),
    )


def _mapping(sources: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {
        source: float(value)
        for source, value in zip(sources, values, strict=True)
    }


__all__ = (
    "DEFAULT_MAX_SOURCE_WEIGHT",
    "DEFAULT_MIN_EFFECTIVE_SOURCES",
    "DEFAULT_TEMPERATURE",
    "WEIGHT_SEMANTICS",
    "ResidualSoftWeights",
    "residual_soft_weights",
)
