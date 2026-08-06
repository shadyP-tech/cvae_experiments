"""Conservative utility-gradient routing under dense-mixture constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from ...protocol import ProtocolError
from .ridge import validate_psd_covariance


DEFAULT_KAPPA = 1.0
DEFAULT_L2_PENALTY = 0.01
DEFAULT_MAX_SOURCE_WEIGHT = 0.25
DEFAULT_MIN_EFFECTIVE_SOURCES = 6.0
DEFAULT_OBJECTIVE_TOLERANCE = 1e-10
DEFAULT_SOLVER_TOLERANCE = 1e-12
DEFAULT_MAX_ITERATIONS = 1000


@dataclass(frozen=True)
class RobustRoutingSolution:
    candidate_sources: tuple[str, ...]
    predicted_marginal_utility: Mapping[str, float]
    uniform_weights: Mapping[str, float]
    delta: Mapping[str, float]
    weights: Mapping[str, float]
    objective_value: float
    expected_gain: float
    uncertainty_penalty: float
    l2_penalty_value: float
    effective_source_count: float
    maximum_source_weight: float
    used_uniform_fallback: bool
    fallback_reason: str | None
    solver_success: bool
    solver_message: str
    solver_iterations: int


def robust_local_utility_weights(
    marginal_utility_by_source: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]] | Sequence[Sequence[float]] | np.ndarray,
    *,
    covariance_source_order: Sequence[str] | None = None,
    kappa: float = DEFAULT_KAPPA,
    l2_penalty: float = DEFAULT_L2_PENALTY,
    max_source_weight: float = DEFAULT_MAX_SOURCE_WEIGHT,
    min_effective_sources: float = DEFAULT_MIN_EFFECTIVE_SOURCES,
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE,
    solver_tolerance: float = DEFAULT_SOLVER_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> RobustRoutingSolution:
    """Optimize a robust local utility improvement over equal union.

    The optimized variable is ``d = w-u``.  The objective is

    ``d' m - kappa * sqrt(d' Sigma d) - l2_penalty * ||d||^2``

    under ``sum(d)=0``, ``w>=0``, ``max(w)<=0.25``, and
    ``sum(w^2)<=1/6`` by default.  Invalid statistical inputs fail closed with
    :class:`ProtocolError`.  Numerical solver failure, post-solve infeasibility,
    or a nonpositive robust gain returns the equal-union vector bit-exactly.
    """

    sources, marginal = _marginal_vector(marginal_utility_by_source)
    n_sources = len(sources)
    sigma = _ordered_covariance(
        covariance,
        sources=sources,
        covariance_source_order=covariance_source_order,
    )
    risk_multiplier = float(kappa)
    shrinkage = float(l2_penalty)
    cap = float(max_source_weight)
    minimum_effective = float(min_effective_sources)
    gain_tolerance = float(objective_tolerance)
    ftol = float(solver_tolerance)
    iterations = int(max_iterations)
    if (
        not np.isfinite(risk_multiplier)
        or risk_multiplier < 0.0
        or not np.isfinite(shrinkage)
        or shrinkage < 0.0
        or not np.isfinite(cap)
        or cap <= 0.0
        or cap > 1.0
        or not np.isfinite(minimum_effective)
        or minimum_effective < 1.0
        or not np.isfinite(gain_tolerance)
        or gain_tolerance < 0.0
        or not np.isfinite(ftol)
        or ftol <= 0.0
        or iterations <= 0
    ):
        raise ProtocolError("Robust local utility optimizer configuration is invalid.")
    uniform = np.full(n_sources, 1.0 / n_sources, dtype=np.float64)
    concentration_limit = 1.0 / minimum_effective
    if float(uniform.max()) > cap + 1e-15 or float(np.dot(uniform, uniform)) > (
        concentration_limit + 1e-15
    ):
        raise ProtocolError("Density constraints exclude the equal-union anchor.")

    def objective(weights: np.ndarray) -> float:
        delta = weights - uniform
        quadratic = max(0.0, float(delta @ sigma @ delta))
        return -(
            float(np.dot(delta, marginal))
            - risk_multiplier * float(np.sqrt(quadratic))
            - shrinkage * float(np.dot(delta, delta))
        )

    def objective_jacobian(weights: np.ndarray) -> np.ndarray:
        delta = weights - uniform
        quadratic = max(0.0, float(delta @ sigma @ delta))
        risk_gradient = (
            sigma @ delta / np.sqrt(quadratic)
            if quadratic > np.finfo(np.float64).eps
            else np.zeros_like(delta)
        )
        # Jacobian of the minimized negative robust objective.
        return -marginal + risk_multiplier * risk_gradient + 2.0 * shrinkage * delta

    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(concentration_limit - np.dot(weights, weights)),
            "jac": lambda weights: -2.0 * weights,
        },
    )
    try:
        result = minimize(
            objective,
            uniform.copy(),
            method="SLSQP",
            jac=objective_jacobian,
            bounds=tuple((0.0, cap) for _ in sources),
            constraints=constraints,
            options={"ftol": ftol, "maxiter": iterations, "disp": False},
        )
    except Exception as exc:  # scipy failures are converted to a safe control.
        return _uniform_fallback(
            sources=sources,
            marginal=marginal,
            reason="solver_failure_uniform",
            solver_message=f"{type(exc).__name__}: {exc}",
        )
    if not bool(result.success) or not np.isfinite(result.x).all():
        return _uniform_fallback(
            sources=sources,
            marginal=marginal,
            reason="solver_failure_uniform",
            solver_message=str(result.message),
            solver_iterations=int(getattr(result, "nit", 0)),
        )
    weights = np.asarray(result.x, dtype=np.float64)
    feasibility_tolerance = max(1e-8, 100.0 * ftol)
    if (
        abs(float(weights.sum()) - 1.0) > feasibility_tolerance
        or float(weights.min()) < -feasibility_tolerance
        or float(weights.max()) > cap + feasibility_tolerance
        or float(np.dot(weights, weights)) > concentration_limit + feasibility_tolerance
    ):
        return _uniform_fallback(
            sources=sources,
            marginal=marginal,
            reason="postsolve_infeasible_uniform",
            solver_message=str(result.message),
            solver_iterations=int(getattr(result, "nit", 0)),
        )
    # Remove harmless equality drift, then re-check without projecting an
    # actually infeasible solution into the admissible set.
    weights = weights / float(weights.sum())
    if (
        float(weights.min()) < -feasibility_tolerance
        or float(weights.max()) > cap + feasibility_tolerance
        or float(np.dot(weights, weights)) > concentration_limit + feasibility_tolerance
    ):
        return _uniform_fallback(
            sources=sources,
            marginal=marginal,
            reason="postsolve_infeasible_uniform",
            solver_message=str(result.message),
            solver_iterations=int(getattr(result, "nit", 0)),
        )
    delta = weights - uniform
    expected_gain, uncertainty, penalty_value, robust_gain = _objective_terms(
        delta,
        marginal,
        sigma,
        kappa=risk_multiplier,
        l2_penalty=shrinkage,
    )
    if not np.isfinite(robust_gain) or robust_gain <= gain_tolerance:
        return _uniform_fallback(
            sources=sources,
            marginal=marginal,
            reason="nonpositive_robust_gain_uniform",
            solver_message=str(result.message),
            solver_iterations=int(getattr(result, "nit", 0)),
        )
    effective = float(1.0 / np.dot(weights, weights))
    return RobustRoutingSolution(
        candidate_sources=sources,
        predicted_marginal_utility=_mapping(sources, marginal),
        uniform_weights=_mapping(sources, uniform),
        delta=_mapping(sources, delta),
        weights=_mapping(sources, weights),
        objective_value=robust_gain,
        expected_gain=expected_gain,
        uncertainty_penalty=uncertainty,
        l2_penalty_value=penalty_value,
        effective_source_count=effective,
        maximum_source_weight=float(weights.max()),
        used_uniform_fallback=False,
        fallback_reason=None,
        solver_success=True,
        solver_message=str(result.message),
        solver_iterations=int(getattr(result, "nit", 0)),
    )


def _objective_terms(
    delta: np.ndarray,
    marginal: np.ndarray,
    covariance: np.ndarray,
    *,
    kappa: float,
    l2_penalty: float,
) -> tuple[float, float, float, float]:
    expected_gain = float(np.dot(delta, marginal))
    uncertainty = float(kappa * np.sqrt(max(0.0, float(delta @ covariance @ delta))))
    penalty_value = float(l2_penalty * np.dot(delta, delta))
    return expected_gain, uncertainty, penalty_value, expected_gain - uncertainty - penalty_value


def _uniform_fallback(
    *,
    sources: tuple[str, ...],
    marginal: np.ndarray,
    reason: str,
    solver_message: str,
    solver_iterations: int = 0,
) -> RobustRoutingSolution:
    uniform = np.full(len(sources), 1.0 / len(sources), dtype=np.float64)
    zero = np.zeros(len(sources), dtype=np.float64)
    return RobustRoutingSolution(
        candidate_sources=sources,
        predicted_marginal_utility=_mapping(sources, marginal),
        uniform_weights=_mapping(sources, uniform),
        delta=_mapping(sources, zero),
        weights=_mapping(sources, uniform),
        objective_value=0.0,
        expected_gain=0.0,
        uncertainty_penalty=0.0,
        l2_penalty_value=0.0,
        effective_source_count=float(len(sources)),
        maximum_source_weight=float(uniform[0]),
        used_uniform_fallback=True,
        fallback_reason=reason,
        solver_success=reason == "nonpositive_robust_gain_uniform",
        solver_message=solver_message,
        solver_iterations=solver_iterations,
    )


def _marginal_vector(
    values: Mapping[str, float],
) -> tuple[tuple[str, ...], np.ndarray]:
    normalized = {str(source): float(value) for source, value in values.items()}
    sources = tuple(sorted(normalized))
    marginal = np.asarray([normalized[source] for source in sources], dtype=np.float64)
    if (
        not sources
        or len(normalized) != len(values)
        or any(not source or source.strip() != source for source in sources)
        or not np.isfinite(marginal).all()
    ):
        raise ProtocolError("Predicted marginal utilities must be finite and canonically keyed.")
    return sources, marginal


def _ordered_covariance(
    covariance: Mapping[str, Mapping[str, float]] | Sequence[Sequence[float]] | np.ndarray,
    *,
    sources: tuple[str, ...],
    covariance_source_order: Sequence[str] | None,
) -> np.ndarray:
    if isinstance(covariance, Mapping):
        if covariance_source_order is not None or set(covariance) != set(sources):
            raise ProtocolError("Mapped covariance must exactly cover the marginal sources.")
        try:
            matrix = np.asarray(
                [
                    [float(covariance[row][column]) for column in sources]
                    for row in sources
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Mapped covariance rows must cover every source.") from exc
        if any(set(covariance[row]) != set(sources) for row in sources):
            raise ProtocolError("Mapped covariance rows must cover every source exactly.")
    else:
        matrix = np.asarray(covariance, dtype=np.float64)
        if covariance_source_order is not None:
            declared = tuple(str(source) for source in covariance_source_order)
            if len(declared) != len(set(declared)) or set(declared) != set(sources):
                raise ProtocolError("Covariance source order must match marginal sources.")
            indices = tuple(declared.index(source) for source in sources)
            if matrix.shape == (len(sources), len(sources)):
                matrix = matrix[np.ix_(indices, indices)]
    return validate_psd_covariance(
        matrix, dimension=len(sources), name="marginal utility covariance"
    )


def _mapping(sources: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {
        source: float(value) for source, value in zip(sources, values, strict=True)
    }


__all__ = (
    "DEFAULT_KAPPA",
    "DEFAULT_L2_PENALTY",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_SOURCE_WEIGHT",
    "DEFAULT_MIN_EFFECTIVE_SOURCES",
    "DEFAULT_OBJECTIVE_TOLERANCE",
    "DEFAULT_SOLVER_TOLERANCE",
    "RobustRoutingSolution",
    "robust_local_utility_weights",
)
