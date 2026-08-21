"""Typed contracts for projected/unprojected PCSI-RACR utility surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    ACTION_GEOMETRY_IDS,
    ALTERNATIVE_METHOD_IDS,
    CENTERS,
    DIRECTION_IDS,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash, require_sha256


def _finite(values: object, *, size: int, name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != size or any(not math.isfinite(value) for value in converted):
        raise ProtocolError(f"PCSI-RACR {name} drifted.")
    return converted


@dataclass(frozen=True, order=True)
class ProjectedUtilityDescriptor:
    target_center: str
    case_id: str
    geometry_id: str
    direction: str
    representative: str
    equivalence_members: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    crossing_sample_ids: tuple[str, ...]
    action_hash: str
    endpoint_prediction_hash: str
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = _finite(
            self.feature_values,
            size=len(UTILITY_FEATURE_NAMES),
            name="projected utility descriptor",
        )
        members = tuple(str(value) for value in self.equivalence_members)
        crossings = tuple(str(value) for value in self.crossing_sample_ids)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.geometry_id not in ACTION_GEOMETRY_IDS
            or self.direction not in DIRECTION_IDS
            or self.representative not in ALTERNATIVE_METHOD_IDS
            or not members
            or members[0] != self.representative
            or len(members) != len(set(members))
            or any(value not in ALTERNATIVE_METHOD_IDS for value in members)
            or self.feature_names != UTILITY_FEATURE_NAMES
            or len(crossings) != len(set(crossings))
        ):
            raise ProtocolError("PCSI-RACR projected descriptor identity drifted.")
        require_sha256(self.action_hash, "projected_action_hash")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "equivalence_members", members)
        object.__setattr__(self, "crossing_sample_ids", crossings)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    @property
    def crossing_count(self) -> int:
        return len(self.crossing_sample_ids)

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.target_center,
            self.case_id,
            self.geometry_id,
            self.direction,
            self.action_hash,
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_projected_descriptor_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "geometry_id": self.geometry_id,
            "direction": self.direction,
            "representative": self.representative,
            "equivalence_members": list(self.equivalence_members),
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "crossing_sample_ids": list(self.crossing_sample_ids),
            "crossing_count": self.crossing_count,
            "structural_zero": self.crossing_count == 0,
            "action_hash": self.action_hash,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class ProjectedDonorUtilityRow:
    outer_target_center: str
    donor_center: str
    case_id: str
    geometry_id: str
    direction: str
    representative: str
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
            name="projected donor utility row",
        )
        responses = tuple(float(getattr(self, value)) for value in UTILITY_RESPONSE_IDS)
        if (
            self.outer_target_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_target_center == self.donor_center
            or not self.case_id
            or self.geometry_id not in ACTION_GEOMETRY_IDS
            or self.direction not in DIRECTION_IDS
            or self.representative not in ALTERNATIVE_METHOD_IDS
            or type(self.crossing_count) is not int
            or self.crossing_count < 0
            or any(not math.isfinite(value) for value in responses)
            or (
                self.crossing_count == 0
                and any(abs(value) > 1.0e-15 for value in responses)
            )
        ):
            raise ProtocolError("PCSI-RACR projected donor response drifted.")
        require_sha256(self.descriptor_hash, "projected_descriptor_hash")
        object.__setattr__(self, "feature_values", values)

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.outer_target_center,
            self.donor_center,
            self.case_id,
            self.geometry_id,
            self.direction,
            self.descriptor_hash,
        )

    def response(self, response_id: str) -> float:
        if response_id not in UTILITY_RESPONSE_IDS:
            raise ProtocolError("PCSI-RACR requested an unknown response.")
        return float(getattr(self, response_id))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_projected_donor_row_v1",
            **self.__dict__,
            "feature_values": list(self.feature_values),
            "raw_label_persisted": False,
        }


@dataclass(frozen=True)
class ProjectedUtilityModel:
    outer_target_center: str
    geometry_id: str
    training_centers: tuple[str, ...]
    feature_names: tuple[str, ...]
    direction_ids: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    direction_intercepts: tuple[tuple[str, tuple[float, ...]], ...]
    slope_coefficients: tuple[tuple[str, tuple[float, ...]], ...]
    ridge_alpha: float
    training_row_count_by_center: Mapping[str, int]
    model_hash: str
    fit_status: str = "FIT"

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in self.training_centers)
        counts = {str(key): int(value) for key, value in self.training_row_count_by_center.items()}
        intercepts = dict(self.direction_intercepts)
        slopes = dict(self.slope_coefficients)
        numeric = (
            *self.feature_mean,
            *self.feature_scale,
            *(value for values in intercepts.values() for value in values),
            *(value for values in slopes.values() for value in values),
        )
        if (
            self.outer_target_center not in CENTERS
            or self.geometry_id not in ACTION_GEOMETRY_IDS
            or self.outer_target_center in centers
            or not centers
            or len(centers) != len(set(centers))
            or any(center not in CENTERS for center in centers)
            or self.feature_names != UTILITY_FEATURE_NAMES
            or self.direction_ids != DIRECTION_IDS
            or len(self.feature_mean) != len(UTILITY_FEATURE_NAMES)
            or len(self.feature_scale) != len(UTILITY_FEATURE_NAMES)
            or tuple(intercepts) != UTILITY_RESPONSE_IDS
            or tuple(slopes) != UTILITY_RESPONSE_IDS
            or any(len(values) != len(DIRECTION_IDS) for values in intercepts.values())
            or any(len(values) != len(UTILITY_FEATURE_NAMES) for values in slopes.values())
            or tuple(counts) != centers
            or any(value <= 0 for value in counts.values())
            or self.ridge_alpha <= 0.0
            or self.fit_status != "FIT"
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in self.feature_scale)
        ):
            raise ProtocolError("PCSI-RACR projected model contract drifted.")
        require_sha256(self.model_hash, "projected_model_hash")
        object.__setattr__(self, "training_centers", centers)
        object.__setattr__(self, "training_row_count_by_center", MappingProxyType(counts))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_projected_model_v1",
            "outer_target_center": self.outer_target_center,
            "geometry_id": self.geometry_id,
            "training_centers": list(self.training_centers),
            "feature_names": list(self.feature_names),
            "direction_ids": list(self.direction_ids),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "direction_intercepts": {key: list(value) for key, value in self.direction_intercepts},
            "slope_coefficients": {key: list(value) for key, value in self.slope_coefficients},
            "ridge_alpha": self.ridge_alpha,
            "training_row_count_by_center": dict(self.training_row_count_by_center),
            "fit_status": self.fit_status,
            "intercept_count": 2,
            "center_dummy_effects_used": False,
            "direction_intercepts_penalized": False,
            "equal_total_weight_per_donor_center": True,
            "equal_total_weight_per_case_within_donor_center": True,
            "equal_total_weight_per_equivalence_class_within_case": True,
            "model_hash": self.model_hash,
        }

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Rebuild the immutable count mapping after a spawn-process transfer."""

        return (
            type(self),
            (
                self.outer_target_center,
                self.geometry_id,
                self.training_centers,
                self.feature_names,
                self.direction_ids,
                self.feature_mean,
                self.feature_scale,
                self.direction_intercepts,
                self.slope_coefficients,
                self.ridge_alpha,
                dict(self.training_row_count_by_center),
                self.model_hash,
                self.fit_status,
            ),
        )


@dataclass(frozen=True, order=True)
class ProjectedUtilityPrediction:
    descriptor_hash: str
    geometry_id: str
    full_values: tuple[tuple[str, float], ...]
    deletion_values: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    residual_bias: tuple[tuple[str, float], ...]
    residual_scale: tuple[tuple[str, float], ...]
    robust_values: tuple[tuple[str, float], ...]
    model_hashes: tuple[str, ...]
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        full = dict(self.full_values)
        deletions = dict(self.deletion_values)
        bias = dict(self.residual_bias)
        scale = dict(self.residual_scale)
        robust = dict(self.robust_values)
        if (
            self.geometry_id not in ACTION_GEOMETRY_IDS
            or tuple(full) != UTILITY_RESPONSE_IDS
            or tuple(deletions) != UTILITY_RESPONSE_IDS
            or tuple(bias) != UTILITY_RESPONSE_IDS
            or tuple(scale) != UTILITY_RESPONSE_IDS
            or tuple(robust) != UTILITY_RESPONSE_IDS
            or not deletions
            or any(len(values) == 0 for values in deletions.values())
            or any(value < 0.0 for value in scale.values())
            or any(
                not math.isfinite(float(value))
                for value in (
                    *full.values(),
                    *bias.values(),
                    *scale.values(),
                    *robust.values(),
                    *(item for values in deletions.values() for _center, item in values),
                )
            )
        ):
            raise ProtocolError("PCSI-RACR projected prediction drifted.")
        require_sha256(self.descriptor_hash, "projected_descriptor_hash")
        for digest in self.model_hashes:
            require_sha256(digest, "projected_model_hash")
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def robust(self, response_id: str) -> float:
        return dict(self.robust_values)[response_id]

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_projected_prediction_v1",
            "descriptor_hash": self.descriptor_hash,
            "geometry_id": self.geometry_id,
            "full_values": dict(self.full_values),
            "deletion_values": {
                response: [{"deleted_center": center, "value": value} for center, value in values]
                for response, values in self.deletion_values
            },
            "residual_bias": dict(self.residual_bias),
            "residual_scale": dict(self.residual_scale),
            "robust_values": dict(self.robust_values),
            "model_hashes": list(self.model_hashes),
            "confidence_bound_claimed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


__all__ = (
    "ProjectedDonorUtilityRow",
    "ProjectedUtilityDescriptor",
    "ProjectedUtilityModel",
    "ProjectedUtilityPrediction",
)
