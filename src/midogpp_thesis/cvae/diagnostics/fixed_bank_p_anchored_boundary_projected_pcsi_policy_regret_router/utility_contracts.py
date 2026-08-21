"""Typed contracts for complete signed-utility routing surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    PORTFOLIO_METHOD_ID,
    SIGN_PRESERVING_SHRINKAGE,
    UTILITY_CELL_IDS,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash, require_sha256


def _finite(values: tuple[float, ...], *, size: int, name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != size or any(not math.isfinite(value) for value in converted):
        raise ProtocolError(f"PCSI-PARC {name} drifted.")
    return converted


@dataclass(frozen=True, order=True)
class UtilityDescriptor:
    """One label-free case x alternative x direction candidate."""

    target_center: str
    case_id: str
    alternative: str
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    crossing_sample_ids: tuple[str, ...]
    endpoint_prediction_hash: str
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(UTILITY_FEATURE_NAMES),
            name="utility descriptor",
        )
        sample_ids = tuple(str(value) for value in self.crossing_sample_ids)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or self.feature_names != UTILITY_FEATURE_NAMES
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise ProtocolError("PCSI-PARC utility descriptor identity drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "crossing_sample_ids", sample_ids)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.alternative, self.direction

    @property
    def crossing_count(self) -> int:
        return len(self.crossing_sample_ids)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_utility_descriptor_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "alternative": self.alternative,
            "direction": self.direction,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "crossing_sample_ids": list(self.crossing_sample_ids),
            "crossing_count": self.crossing_count,
            "structural_zero": self.crossing_count == 0,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class DonorUtilityRow:
    """Observed signed contribution of one complete donor candidate."""

    outer_target_center: str
    donor_center: str
    case_id: str
    alternative: str
    direction: str
    feature_values: tuple[float, ...]
    crossing_count: int
    bacc_contribution_delta: float
    brier_contribution_delta: float
    log_loss_contribution_delta: float
    descriptor_hash: str

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(UTILITY_FEATURE_NAMES),
            name="donor utility row",
        )
        responses = (
            float(self.bacc_contribution_delta),
            float(self.brier_contribution_delta),
            float(self.log_loss_contribution_delta),
        )
        if (
            self.outer_target_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_target_center == self.donor_center
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or type(self.crossing_count) is not int
            or self.crossing_count < 0
            or any(not math.isfinite(value) for value in responses)
            or (
                self.crossing_count == 0
                and any(abs(value) > 1.0e-15 for value in responses)
            )
        ):
            raise ProtocolError("PCSI-PARC donor utility response drifted.")
        require_sha256(self.descriptor_hash, "utility_descriptor_hash")
        object.__setattr__(self, "feature_values", values)

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.outer_target_center,
            self.donor_center,
            self.case_id,
            self.alternative,
            self.direction,
        )

    def response(self, response_id: str) -> float:
        if response_id not in UTILITY_RESPONSE_IDS:
            raise ProtocolError("PCSI-PARC requested an unknown utility response.")
        return float(getattr(self, response_id))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_donor_utility_row_v1",
            **self.__dict__,
            "feature_values": list(self.feature_values),
            "structural_zero": self.crossing_count == 0,
            "raw_label_persisted": False,
        }


@dataclass(frozen=True)
class SignedUtilityModel:
    outer_target_center: str
    training_centers: tuple[str, ...]
    feature_names: tuple[str, ...]
    cell_ids: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    cell_intercepts: tuple[tuple[str, tuple[float, ...]], ...]
    slope_coefficients: tuple[tuple[str, tuple[float, ...]], ...]
    ridge_alpha: float
    training_row_count_by_center: Mapping[str, int]
    model_hash: str
    fit_status: str = "FIT"

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in self.training_centers)
        counts = {
            str(center): int(count)
            for center, count in self.training_row_count_by_center.items()
        }
        intercepts = dict(self.cell_intercepts)
        slopes = dict(self.slope_coefficients)
        numeric = (
            *self.feature_mean,
            *self.feature_scale,
            *(value for values in intercepts.values() for value in values),
            *(value for values in slopes.values() for value in values),
        )
        if (
            self.outer_target_center not in CENTERS
            or self.outer_target_center in centers
            or not centers
            or len(centers) != len(set(centers))
            or any(center not in CENTERS for center in centers)
            or self.feature_names != UTILITY_FEATURE_NAMES
            or self.cell_ids != UTILITY_CELL_IDS
            or len(self.feature_mean) != len(UTILITY_FEATURE_NAMES)
            or len(self.feature_scale) != len(UTILITY_FEATURE_NAMES)
            or tuple(intercepts) != UTILITY_RESPONSE_IDS
            or tuple(slopes) != UTILITY_RESPONSE_IDS
            or any(len(values) != len(UTILITY_CELL_IDS) for values in intercepts.values())
            or any(len(values) != len(UTILITY_FEATURE_NAMES) for values in slopes.values())
            or tuple(counts) != centers
            or any(count <= 0 for count in counts.values())
            or self.ridge_alpha <= 0.0
            or self.fit_status not in {"FIT", "NO_DONOR_ROWS_P_FALLBACK"}
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in self.feature_scale)
        ):
            raise ProtocolError("PCSI-PARC signed utility model drifted.")
        require_sha256(self.model_hash, "signed_utility_model_hash")
        object.__setattr__(self, "training_centers", centers)
        object.__setattr__(self, "training_row_count_by_center", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_signed_utility_model_v1",
            "outer_target_center": self.outer_target_center,
            "response_ids": list(UTILITY_RESPONSE_IDS),
            "training_centers": list(self.training_centers),
            "feature_names": list(self.feature_names),
            "cell_ids": list(self.cell_ids),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "cell_intercepts": {
                response: list(values) for response, values in self.cell_intercepts
            },
            "slope_coefficients": {
                response: list(values) for response, values in self.slope_coefficients
            },
            "ridge_alpha": self.ridge_alpha,
            "training_row_count_by_center": dict(self.training_row_count_by_center),
            "fit_status": self.fit_status,
            "center_dummy_effects_used": False,
            "cell_intercepts_penalized": False,
            "structural_zero_rows_used": True,
            "equal_total_weight_per_donor_center": True,
            "equal_total_weight_per_case_within_donor_center": True,
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True, order=True)
class UtilityPrediction:
    descriptor_hash: str
    full_values: tuple[tuple[str, float], ...]
    deletion_values: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    residual_bias: tuple[tuple[str, float], ...]
    residual_scale: tuple[tuple[str, float], ...]
    robust_values: tuple[tuple[str, float], ...]
    stability_fractions: tuple[tuple[str, float], ...]
    model_hashes: tuple[str, ...]
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        full = dict(self.full_values)
        deletions = {response: tuple(values) for response, values in self.deletion_values}
        bias = dict(self.residual_bias)
        scale = dict(self.residual_scale)
        robust = dict(self.robust_values)
        fractions = dict(self.stability_fractions)
        if (
            tuple(full) != UTILITY_RESPONSE_IDS
            or tuple(deletions) != UTILITY_RESPONSE_IDS
            or tuple(bias) != UTILITY_RESPONSE_IDS
            or tuple(scale) != UTILITY_RESPONSE_IDS
            or tuple(robust) != UTILITY_RESPONSE_IDS
            or tuple(fractions) != UTILITY_RESPONSE_IDS
            or any(len(values) != 8 for values in deletions.values())
            or any(len({center for center, _value in values}) != 8 for values in deletions.values())
            or any(
                center not in CENTERS
                for values in deletions.values()
                for center, _value in values
            )
            or any(
                not math.isfinite(float(value))
                for value in (
                    *full.values(),
                    *bias.values(),
                    *scale.values(),
                    *robust.values(),
                    *(value for values in deletions.values() for _center, value in values),
                )
            )
            or any(value < 0.0 for value in scale.values())
            or any(not 0.0 <= value <= 1.0 for value in fractions.values())
        ):
            raise ProtocolError("PCSI-PARC utility prediction drifted.")
        require_sha256(self.descriptor_hash, "utility_descriptor_hash")
        for digest in self.model_hashes:
            require_sha256(digest, "signed_utility_model_hash")
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def full(self, response_id: str) -> float:
        return dict(self.full_values)[response_id]

    def robust(self, response_id: str) -> float:
        return dict(self.robust_values)[response_id]

    def fraction(self, response_id: str) -> float:
        return dict(self.stability_fractions)[response_id]

    def scale(self, response_id: str) -> float:
        return dict(self.residual_scale)[response_id]

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_utility_prediction_v1",
            "descriptor_hash": self.descriptor_hash,
            "full_values": dict(self.full_values),
            "deletion_values": {
                response: [
                    {"deleted_center": center, "value": value}
                    for center, value in values
                ]
                for response, values in self.deletion_values
            },
            "residual_bias": dict(self.residual_bias),
            "residual_scale": dict(self.residual_scale),
            "robust_values": dict(self.robust_values),
            "stability_fractions": dict(self.stability_fractions),
            "model_hashes": list(self.model_hashes),
            "delete_donor_predictions_are_stability_checks": True,
            "residual_calibration_is_inner_held_donor_only": True,
            "confidence_bound_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True, order=True)
class DirectionalDecision:
    target_center: str
    case_id: str
    policy_id: str
    direction: str
    selected_alternative: str
    selected_score: float
    candidate_prediction_hashes: tuple[str, ...]
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.policy_id not in COMPOSED_POLICY_IDS
            or self.direction not in DIRECTION_IDS
            or self.selected_alternative
            not in (*ALTERNATIVE_METHOD_IDS, PORTFOLIO_METHOD_ID)
            or not math.isfinite(float(self.selected_score))
        ):
            raise ProtocolError("PCSI-PARC directional decision drifted.")
        for digest in self.candidate_prediction_hashes:
            require_sha256(digest, "utility_prediction_hash")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_directional_decision_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "direction": self.direction,
            "selected_alternative": self.selected_alternative,
            "selected_score": self.selected_score,
            "candidate_prediction_hashes": list(self.candidate_prediction_hashes),
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class ComposedCasePrediction:
    target_center: str
    case_id: str
    policy_id: str
    sample_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    decisions: tuple[DirectionalDecision, ...]
    switched_sample_counts: tuple[tuple[str, int], ...]
    endpoint_prediction_hash: str
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        probabilities = _finite(self.probabilities, size=len(samples), name="composition")
        counts = dict(self.switched_sample_counts)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.policy_id not in COMPOSED_POLICY_IDS
            or not samples
            or len(samples) != len(set(samples))
            or any(not 0.0 <= value <= 1.0 for value in probabilities)
            or tuple(row.direction for row in self.decisions) != DIRECTION_IDS
            or any(
                row.target_center != self.target_center
                or row.case_id != self.case_id
                or row.policy_id != self.policy_id
                for row in self.decisions
            )
            or tuple(counts) != DIRECTION_IDS
            or any(type(value) is not int or value < 0 for value in counts.values())
        ):
            raise ProtocolError("PCSI-PARC composed prediction drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_composed_case_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "sample_ids": list(self.sample_ids),
            "probabilities": list(self.probabilities),
            "decisions": [row.to_payload() for row in self.decisions],
            "switched_sample_counts": dict(self.switched_sample_counts),
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "sign_preserving_shrinkage": SIGN_PRESERVING_SHRINKAGE,
            "one_alternative_per_direction": True,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


__all__ = (
    "ComposedCasePrediction",
    "DirectionalDecision",
    "DonorUtilityRow",
    "SignedUtilityModel",
    "UtilityDescriptor",
    "UtilityPrediction",
)
