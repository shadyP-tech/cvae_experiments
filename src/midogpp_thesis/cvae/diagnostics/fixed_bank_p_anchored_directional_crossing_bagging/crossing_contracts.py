"""Typed contracts for label-free crossings, donor fits, and P-anchored output."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    CROSSING_FEATURE_NAMES,
    DIRECTION_IDS,
    ENDPOINT_METHOD_IDS,
)
from .hashing import canonical_hash, require_sha256


def _finite(values: tuple[float, ...], *, size: int, name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != size or any(not math.isfinite(value) for value in converted):
        raise ProtocolError(f"PDCB {name} drifted.")
    return converted


@dataclass(frozen=True, order=True)
class CrossingDescriptor:
    target_center: str
    case_id: str
    sample_id: str
    alternative: str
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    endpoint_prediction_hash: str
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(CROSSING_FEATURE_NAMES),
            name="crossing descriptor",
        )
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.sample_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.feature_names != CROSSING_FEATURE_NAMES
        ):
            raise ProtocolError("PDCB crossing identity drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.sample_id, self.alternative

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pdcb_crossing_descriptor_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "alternative": self.alternative,
            "direction": self.direction,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class DonorCrossingRow:
    outer_target_center: str
    donor_center: str
    case_id: str
    sample_id: str
    alternative: str
    direction: str
    feature_values: tuple[float, ...]
    helpful: int
    bacc_contribution_delta: float
    log_loss_delta: float
    descriptor_hash: str

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(CROSSING_FEATURE_NAMES),
            name="donor crossing row",
        )
        if (
            self.outer_target_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_target_center == self.donor_center
            or not self.case_id
            or not self.sample_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.helpful not in (0, 1)
            or not math.isfinite(float(self.bacc_contribution_delta))
            or not math.isfinite(float(self.log_loss_delta))
        ):
            raise ProtocolError("PDCB donor crossing response drifted.")
        require_sha256(self.descriptor_hash, "crossing_descriptor_hash")
        object.__setattr__(self, "feature_values", values)

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.outer_target_center,
            self.donor_center,
            self.case_id,
            self.sample_id,
            self.alternative,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pdcb_donor_crossing_row_v1",
            **self.__dict__,
            "feature_values": list(self.feature_values),
            "raw_label_persisted": False,
        }


@dataclass(frozen=True)
class CrossingHelpfulnessModel:
    outer_target_center: str
    training_centers: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge_alpha: float
    training_row_count_by_center: Mapping[str, int]
    positive_row_count: int
    negative_row_count: int
    iterations: int
    converged: bool
    model_hash: str
    fit_status: str = "FIT"

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in self.training_centers)
        counts = {
            str(center): int(count)
            for center, count in self.training_row_count_by_center.items()
        }
        if (
            self.outer_target_center not in CENTERS
            or self.outer_target_center in centers
            or not centers
            or len(centers) != len(set(centers))
            or any(center not in CENTERS for center in centers)
            or self.feature_names != CROSSING_FEATURE_NAMES
            or len(self.feature_mean) != len(CROSSING_FEATURE_NAMES)
            or len(self.feature_scale) != len(CROSSING_FEATURE_NAMES)
            or len(self.coefficients) != 1 + len(CROSSING_FEATURE_NAMES)
            or tuple(counts) != centers
            or any(count < 0 for count in counts.values())
            or self.positive_row_count < 0
            or self.negative_row_count < 0
            or self.ridge_alpha <= 0.0
            or self.iterations < 0
            or self.fit_status
            not in {
                "FIT",
                "IRLS_NONCONVERGENCE_P_FALLBACK",
                "NO_ACTIONABLE_DONOR_CROSSINGS_P_FALLBACK",
                "SINGLE_CLASS_DONOR_EVIDENCE_P_FALLBACK",
            }
            or (self.fit_status == "FIT" and (not self.converged or self.iterations <= 0))
            or (self.fit_status != "FIT" and self.converged)
            or (
                self.fit_status
                in {
                    "NO_ACTIONABLE_DONOR_CROSSINGS_P_FALLBACK",
                    "SINGLE_CLASS_DONOR_EVIDENCE_P_FALLBACK",
                }
                and self.iterations != 0
            )
            or (
                self.fit_status == "IRLS_NONCONVERGENCE_P_FALLBACK"
                and self.iterations <= 0
            )
        ):
            raise ProtocolError("PDCB helpfulness model drifted.")
        if any(
            not math.isfinite(value)
            for value in (*self.feature_mean, *self.feature_scale, *self.coefficients)
        ) or any(value <= 0.0 for value in self.feature_scale):
            raise ProtocolError("PDCB helpfulness model contains invalid numerics.")
        require_sha256(self.model_hash, "crossing_model_hash")
        object.__setattr__(self, "training_centers", centers)
        object.__setattr__(self, "training_row_count_by_center", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pdcb_helpfulness_model_v1",
            "outer_target_center": self.outer_target_center,
            "training_centers": list(self.training_centers),
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "ridge_alpha": self.ridge_alpha,
            "training_row_count_by_center": dict(self.training_row_count_by_center),
            "positive_row_count": self.positive_row_count,
            "negative_row_count": self.negative_row_count,
            "iterations": self.iterations,
            "converged": self.converged,
            "fit_status": self.fit_status,
            "center_dummy_effects_used": False,
            "structural_no_crossing_rows_used": False,
            "equal_total_weight_per_donor_center": True,
            "equal_total_weight_per_case_within_donor_center": True,
            "P_fallback_forced": self.fit_status != "FIT",
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True, order=True)
class CrossingPrediction:
    descriptor_hash: str
    full_probability: float
    deletion_probabilities: tuple[tuple[str, float], ...]
    robust_probability: float
    positive_deletion_fraction: float
    raw_weight: float
    model_hashes: tuple[str, ...]
    prediction_hash: str = field(init=False, compare=True)

    @property
    def full_delete_sign_agreement(self) -> float:
        full_positive = self.full_probability > 0.5
        return sum((value > 0.5) == full_positive for _center, value in self.deletion_probabilities) / len(self.deletion_probabilities)

    @property
    def deletion_mad(self) -> float:
        values = sorted(value for _center, value in self.deletion_probabilities)
        median = 0.5 * (values[3] + values[4])
        deviations = sorted(abs(value - median) for value in values)
        return 0.5 * (deviations[3] + deviations[4])

    @property
    def deletion_iqr(self) -> float:
        values = sorted(value for _center, value in self.deletion_probabilities)
        lower = 0.5 * (values[1] + values[2])
        upper = 0.5 * (values[5] + values[6])
        return upper - lower

    def __post_init__(self) -> None:
        deletions = tuple((str(center), float(value)) for center, value in self.deletion_probabilities)
        numeric = (
            float(self.full_probability),
            *(value for _center, value in deletions),
            float(self.robust_probability),
            float(self.positive_deletion_fraction),
            float(self.raw_weight),
        )
        if (
            len(deletions) != 8
            or len({center for center, _value in deletions}) != 8
            or any(center not in CENTERS for center, _value in deletions)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric)
        ):
            raise ProtocolError("PDCB crossing prediction drifted.")
        require_sha256(self.descriptor_hash, "crossing_descriptor_hash")
        for digest in self.model_hashes:
            require_sha256(digest, "crossing_model_hash")
        object.__setattr__(self, "deletion_probabilities", deletions)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pdcb_crossing_prediction_v1",
            "descriptor_hash": self.descriptor_hash,
            "full_probability": self.full_probability,
            "deletion_probabilities": [
                {"deleted_center": center, "probability": value}
                for center, value in self.deletion_probabilities
            ],
            "robust_probability": self.robust_probability,
            "positive_deletion_fraction": self.positive_deletion_fraction,
            "full_delete_sign_agreement": self.full_delete_sign_agreement,
            "deletion_mad": self.deletion_mad,
            "deletion_iqr": self.deletion_iqr,
            "raw_weight": self.raw_weight,
            "model_hashes": list(self.model_hashes),
            "deletion_predictions_are_correlated_robustness_checks": True,
            "confidence_bound_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True, order=True)
class ComposedCasePrediction:
    target_center: str
    case_id: str
    policy_id: str
    sample_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    portfolio_weights: tuple[float, ...]
    alternative_weights: tuple[tuple[str, tuple[float, ...]], ...]
    alternative_mean_weights: tuple[tuple[str, float], ...]
    crossing_prediction_hashes: tuple[str, ...]
    endpoint_prediction_hash: str
    composition_residual_max_abs: float
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        probabilities = _finite(self.probabilities, size=len(samples), name="composition")
        p_weights = _finite(
            self.portfolio_weights,
            size=len(samples),
            name="portfolio weights",
        )
        alternative_weights = tuple(
            (
                str(method),
                _finite(tuple(values), size=len(samples), name="alternative weights"),
            )
            for method, values in self.alternative_weights
        )
        weights = tuple((str(method), float(value)) for method, value in self.alternative_mean_weights)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.policy_id
            or not samples
            or len(samples) != len(set(samples))
            or any(not 0.0 <= value <= 1.0 for value in (*probabilities, *p_weights))
            or tuple(method for method, _value in weights) != ALTERNATIVE_METHOD_IDS
            or tuple(method for method, _values in alternative_weights)
            != ALTERNATIVE_METHOD_IDS
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for _method, value in weights)
            or any(
                not 0.0 <= value <= 1.0
                for _method, values in alternative_weights
                for value in values
            )
            or any(
                abs(
                    p_weights[index]
                    + sum(values[index] for _method, values in alternative_weights)
                    - 1.0
                )
                > 1.0e-12
                for index in range(len(samples))
            )
            or any(
                method != observed
                or abs(mean - sum(values) / len(values)) > 1.0e-12
                for (method, mean), (observed, values) in zip(
                    weights, alternative_weights, strict=True
                )
            )
            or not math.isfinite(float(self.composition_residual_max_abs))
            or not 0.0 <= float(self.composition_residual_max_abs) <= 1.0e-12
        ):
            raise ProtocolError("PDCB composed prediction drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        for digest in self.crossing_prediction_hashes:
            require_sha256(digest, "crossing_prediction_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "portfolio_weights", p_weights)
        object.__setattr__(self, "alternative_weights", alternative_weights)
        object.__setattr__(self, "alternative_mean_weights", weights)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pdcb_composed_case_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "sample_ids": list(self.sample_ids),
            "probabilities": list(self.probabilities),
            "portfolio_weights": list(self.portfolio_weights),
            "alternative_weights": {
                method: list(values) for method, values in self.alternative_weights
            },
            "alternative_mean_weights": dict(self.alternative_mean_weights),
            "crossing_prediction_hashes": list(self.crossing_prediction_hashes),
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "composition_residual_max_abs": self.composition_residual_max_abs,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


__all__ = (
    "ComposedCasePrediction",
    "CrossingDescriptor",
    "CrossingHelpfulnessModel",
    "CrossingPrediction",
    "DonorCrossingRow",
)
