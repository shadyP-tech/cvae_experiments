"""Closed-form empirical-Bayes shrinkage of target-local action residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .hashing import canonical_hash
from .influence.contracts import ActionMetricVector, MetricStandardError
from .protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class EmpiricalBayesEstimate:
    action_id: str
    donor_metrics: ActionMetricVector
    local_residual: ActionMetricVector
    donor_standard_error: MetricStandardError
    local_standard_error: MetricStandardError
    between_center_variance: tuple[float, float, float]
    shrinkage_weight: tuple[float, float, float]
    posterior_metrics: ActionMetricVector
    posterior_standard_error: MetricStandardError
    estimate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action_id = str(self.action_id)
        between = tuple(float(value) for value in self.between_center_variance)
        weights = tuple(float(value) for value in self.shrinkage_weight)
        if (
            not action_id
            or len(between) != 3
            or not all(math.isfinite(value) and value >= 0.0 for value in between)
            or len(weights) != 3
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in weights)
        ):
            raise ProtocolError("SCALE-BP empirical-Bayes estimate drifted.")
        expected_weights = []
        expected_values = []
        expected_se = []
        for donor, residual, donor_se, local_se, tau2 in zip(
            self.donor_metrics.as_tuple(),
            self.local_residual.as_tuple(),
            self.donor_standard_error.as_tuple(),
            self.local_standard_error.as_tuple(),
            between,
            strict=True,
        ):
            local_variance = local_se * local_se
            denominator = tau2 + local_variance
            weight = 0.0 if denominator <= 0.0 else tau2 / denominator
            posterior_local_variance = (
                0.0 if denominator <= 0.0 else tau2 * local_variance / denominator
            )
            expected_weights.append(weight)
            expected_values.append(donor + weight * residual)
            expected_se.append(math.sqrt(donor_se * donor_se + posterior_local_variance))
        if (
            any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, expected in zip(weights, expected_weights, strict=True)
            )
            or any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, expected in zip(
                    self.posterior_metrics.as_tuple(), expected_values, strict=True
                )
            )
            or any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, expected in zip(
                    self.posterior_standard_error.as_tuple(), expected_se, strict=True
                )
            )
        ):
            raise ProtocolError("SCALE-BP empirical-Bayes algebra drifted.")
        payload = {
            "schema_version": "scale_bp_empirical_bayes_estimate_v1",
            "action_id": action_id,
            "donor_metrics": self.donor_metrics.to_payload(),
            "local_residual": self.local_residual.to_payload(),
            "donor_standard_error": self.donor_standard_error.to_payload(),
            "local_standard_error": self.local_standard_error.to_payload(),
            "between_center_variance": between,
            "shrinkage_weight": weights,
            "posterior_metrics": self.posterior_metrics.to_payload(),
            "posterior_standard_error": self.posterior_standard_error.to_payload(),
        }
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "between_center_variance", between)
        object.__setattr__(self, "shrinkage_weight", weights)
        object.__setattr__(self, "estimate_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_empirical_bayes_estimate_v1",
            "action_id": self.action_id,
            "donor_metrics": self.donor_metrics.to_payload(),
            "local_residual": self.local_residual.to_payload(),
            "donor_standard_error": self.donor_standard_error.to_payload(),
            "local_standard_error": self.local_standard_error.to_payload(),
            "between_center_variance": self.between_center_variance,
            "shrinkage_weight": self.shrinkage_weight,
            "posterior_metrics": self.posterior_metrics.to_payload(),
            "posterior_standard_error": self.posterior_standard_error.to_payload(),
            "estimate_hash": self.estimate_hash,
        }


def shrink_action_value(
    *,
    action_id: str,
    donor_metrics: ActionMetricVector,
    local_residual: ActionMetricVector,
    donor_standard_error: MetricStandardError,
    local_standard_error: MetricStandardError,
    between_center_variance: object,
) -> EmpiricalBayesEstimate:
    """Combine a frozen donor prior with an H\\c local residual correction."""

    try:
        between = tuple(float(value) for value in between_center_variance)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError("SCALE-BP between-center variance drifted.") from exc
    if len(between) != 3 or not all(
        math.isfinite(value) and value >= 0.0 for value in between
    ):
        raise ProtocolError("SCALE-BP between-center variance drifted.")
    weights = []
    posterior = []
    posterior_se = []
    for donor, residual, donor_se, local_se, tau2 in zip(
        donor_metrics.as_tuple(),
        local_residual.as_tuple(),
        donor_standard_error.as_tuple(),
        local_standard_error.as_tuple(),
        between,
        strict=True,
    ):
        local_variance = local_se * local_se
        denominator = tau2 + local_variance
        weight = 0.0 if denominator <= 0.0 else tau2 / denominator
        posterior_local_variance = (
            0.0 if denominator <= 0.0 else tau2 * local_variance / denominator
        )
        weights.append(weight)
        posterior.append(donor + weight * residual)
        posterior_se.append(
            math.sqrt(donor_se * donor_se + posterior_local_variance)
        )
    return EmpiricalBayesEstimate(
        action_id=str(action_id),
        donor_metrics=donor_metrics,
        local_residual=local_residual,
        donor_standard_error=donor_standard_error,
        local_standard_error=local_standard_error,
        between_center_variance=between,
        shrinkage_weight=tuple(weights),
        posterior_metrics=ActionMetricVector.from_iterable(posterior),
        posterior_standard_error=MetricStandardError.from_iterable(posterior_se),
    )


__all__ = ("EmpiricalBayesEstimate", "shrink_action_value")
