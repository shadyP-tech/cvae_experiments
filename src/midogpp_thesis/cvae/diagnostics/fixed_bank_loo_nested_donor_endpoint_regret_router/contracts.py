"""Typed, hash-bound contracts for the endpoint-regret diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .constants import (
    CENTERS,
    ENDPOINT_METHOD_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_ROW_COUNT,
    PORTFOLIO_METHOD_ID,
    REGRET_FEATURE_NAMES,
    SEED_PAIR_COUNT,
    physical_action_ids,
)
from .hashing import canonical_hash, require_digest, require_sha256


def _finite_probability_array(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or bool(np.any((array < 0.0) | (array > 1.0)))
    ):
        raise ProtocolError("Probability array shape or range drifted.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CenterProbabilitySurface:
    """Label-free exact-nine probability arrays for one target center."""

    center: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    seed_probabilities: Mapping[str, np.ndarray]
    probability_store_hash: str
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center)
        samples = tuple(str(value) for value in self.sample_ids)
        cases = tuple(str(value) for value in self.case_ids)
        actions = physical_action_ids(center)
        if (
            center not in CENTERS
            or not samples
            or len(samples) != len(cases)
            or len(samples) != len(set(samples))
            or any(not value for value in cases)
            or tuple(self.seed_probabilities) != actions
        ):
            raise ProtocolError("Center probability identity or action order drifted.")
        require_digest(self.probability_store_hash, "probability_store_hash")
        arrays = {
            action: _finite_probability_array(
                self.seed_probabilities[action], shape=(SEED_PAIR_COUNT, len(samples))
            )
            for action in actions
        }
        payload = {
            "schema_version": "fixed_bank_nested_regret_center_surface_v1",
            "center": center,
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "probability_store_hash": self.probability_store_hash,
            "action_array_sha256": {
                action: sha256_array(arrays[action]) for action in actions
            },
            "storage_dtype": "float32",
            "reduction_dtype": "float64",
            "labels_used": False,
        }
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "seed_probabilities", MappingProxyType(arrays))
        object.__setattr__(self, "surface_hash", canonical_hash(payload))

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.case_ids))

    def positions(self, case_id: object) -> np.ndarray:
        case = str(case_id)
        result = np.flatnonzero(np.asarray(self.case_ids, dtype=object) == case)
        if not len(result):
            raise ProtocolError("Requested case is absent from the center surface.")
        return result

    def exact_nine_mean(self, action_id: object) -> np.ndarray:
        try:
            values = self.seed_probabilities[str(action_id)]
        except KeyError as exc:
            raise ProtocolError("Requested physical action is absent.") from exc
        return np.mean(values.astype(np.float64, copy=False), axis=0, dtype=np.float64)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            CenterProbabilitySurface,
            (
                self.center,
                self.sample_ids,
                self.case_ids,
                dict(self.seed_probabilities),
                self.probability_store_hash,
            ),
        )


@dataclass(frozen=True)
class PhysicalProbabilitySurface:
    centers: Mapping[str, CenterProbabilitySurface]
    probability_store_hash: str
    strict_canonical_topology: bool = True
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        require_digest(self.probability_store_hash, "probability_store_hash")
        rows = {str(key): value for key, value in self.centers.items()}
        if tuple(rows) != CENTERS or any(
            not isinstance(value, CenterProbabilitySurface)
            or value.center != center
            or value.probability_store_hash != self.probability_store_hash
            for center, value in rows.items()
        ):
            raise ProtocolError("Physical surface must contain all nine centers in order.")
        if self.strict_canonical_topology:
            counts = {center: len(rows[center].cases) for center in CENTERS}
            if (
                counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER)
                or sum(len(rows[center].sample_ids) for center in CENTERS)
                != EXPECTED_TEST_ROW_COUNT
            ):
                raise ProtocolError("Physical surface canonical test topology drifted.")
        payload = {
            "schema_version": "fixed_bank_nested_regret_physical_surface_v1",
            "probability_store_hash": self.probability_store_hash,
            "center_surface_hashes": {
                center: rows[center].surface_hash for center in CENTERS
            },
            "strict_canonical_topology": self.strict_canonical_topology,
            "labels_used": False,
        }
        object.__setattr__(self, "centers", MappingProxyType(rows))
        object.__setattr__(self, "surface_hash", canonical_hash(payload))


@dataclass(frozen=True, order=True)
class BinaryLabel:
    center: str
    case_id: str
    sample_id: str
    value: int
    scope: str

    def __post_init__(self) -> None:
        if (
            self.center not in CENTERS
            or not self.case_id
            or not self.sample_id
            or self.value not in (0, 1)
            or not self.scope
        ):
            raise ProtocolError("Scoped binary label drifted.")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.center, self.case_id, self.sample_id


@dataclass(frozen=True)
class EndpointCasePrediction:
    center: str
    case_id: str
    sample_ids: tuple[str, ...]
    probabilities: Mapping[str, tuple[float, ...]]
    state_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        rows = {
            str(method): tuple(float(value) for value in probabilities)
            for method, probabilities in self.probabilities.items()
        }
        if (
            self.center not in CENTERS
            or not self.case_id
            or not samples
            or len(samples) != len(set(samples))
            or tuple(rows) != ENDPOINT_METHOD_IDS
            or any(
                len(values) != len(samples)
                or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
                for values in rows.values()
            )
        ):
            raise ProtocolError("Endpoint prediction topology drifted.")
        require_sha256(self.state_hash, "endpoint_state_hash")
        payload = {
            "schema_version": "fixed_bank_nested_regret_endpoint_prediction_v1",
            "center": self.center,
            "case_id": self.case_id,
            "sample_ids": list(samples),
            "probabilities": {method: list(rows[method]) for method in ENDPOINT_METHOD_IDS},
            "state_hash": self.state_hash,
        }
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", MappingProxyType(rows))
        object.__setattr__(self, "prediction_hash", canonical_hash(payload))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            EndpointCasePrediction,
            (
                self.center,
                self.case_id,
                self.sample_ids,
                dict(self.probabilities),
                self.state_hash,
            ),
        )


@dataclass(frozen=True, order=True)
class CandidateDescriptor:
    target_center: str
    case_id: str
    alternative: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    nested_prediction_hashes: tuple[str, ...]
    descriptor_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.alternative not in ENDPOINT_METHOD_IDS
            or self.feature_names != REGRET_FEATURE_NAMES
            or len(values) != len(REGRET_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in values)
            or len(self.nested_prediction_hashes) != len(set(self.nested_prediction_hashes))
        ):
            raise ProtocolError("Endpoint-regret descriptor drifted.")
        for digest in self.nested_prediction_hashes:
            require_sha256(digest, "nested_prediction_hash")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._unhashed()))

    @property
    def is_candidate(self) -> bool:
        return (
            self.alternative != PORTFOLIO_METHOD_ID
            and self.values[3] > 0.0
            and self.values[-1] == 1.0
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_nested_regret_descriptor_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "alternative": self.alternative,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "nested_prediction_hashes": list(self.nested_prediction_hashes),
            "labels_persisted": False,
            "support_regret_field_is_sum_not_mean": True,
            "voter_dispersion_is_not_a_confidence_bound": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "descriptor_hash": self.descriptor_hash}


@dataclass(frozen=True, order=True)
class DonorRegretRow:
    donor_center: str
    case_id: str
    alternative: str
    feature_values: tuple[float, ...]
    bacc_regret: float
    log_loss_delta: float
    center_case_count: int
    descriptor_hash: str

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.feature_values)
        if (
            self.donor_center not in CENTERS
            or not self.case_id
            or self.alternative not in ENDPOINT_METHOD_IDS
            or len(values) != len(REGRET_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in values)
            or not math.isfinite(float(self.bacc_regret))
            or not math.isfinite(float(self.log_loss_delta))
            or isinstance(self.center_case_count, bool)
            or self.center_case_count <= 0
        ):
            raise ProtocolError("Donor-regret row drifted.")
        require_sha256(self.descriptor_hash, "descriptor_hash")
        object.__setattr__(self, "feature_values", values)

    @property
    def key(self) -> tuple[str, str]:
        return self.donor_center, self.case_id


@dataclass(frozen=True)
class CenterBalancedRidgeModel:
    response_name: str
    training_centers: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge_alpha: float
    center_effect_alpha: float
    training_row_count_by_center: Mapping[str, int]
    model_hash: str

    def __post_init__(self) -> None:
        centers = tuple(str(value) for value in self.training_centers)
        counts = {str(key): int(value) for key, value in self.training_row_count_by_center.items()}
        if (
            self.response_name not in {"bacc_regret", "log_loss_delta"}
            or not centers
            or len(centers) != len(set(centers))
            or any(center not in CENTERS for center in centers)
            or self.feature_names != REGRET_FEATURE_NAMES
            or len(self.feature_mean) != len(REGRET_FEATURE_NAMES)
            or len(self.feature_scale) != len(REGRET_FEATURE_NAMES)
            or len(self.coefficients) != 1 + len(REGRET_FEATURE_NAMES) + len(centers)
            or any(scale <= 0.0 or not math.isfinite(scale) for scale in self.feature_scale)
            or tuple(counts) != centers
            or any(value <= 0 for value in counts.values())
            or self.ridge_alpha <= 0.0
            or self.center_effect_alpha <= 0.0
        ):
            raise ProtocolError("Center-balanced Ridge model drifted.")
        require_sha256(self.model_hash, "model_hash")
        object.__setattr__(self, "training_centers", centers)
        object.__setattr__(self, "training_row_count_by_center", MappingProxyType(counts))


@dataclass(frozen=True, order=True)
class RouteDecision:
    target_center: str
    case_id: str
    policy_id: str
    alternative: str
    selected_method: str
    predicted_bacc_regret: float
    predicted_log_loss_delta: float
    delete_bacc_positive_count: int
    delete_log_loss_safe_count: int
    support_regret_sum_pp: float
    support_voter_dispersion_for_sum_pp: float
    reason: str
    descriptor_hash: str
    model_hashes: tuple[str, ...]
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.policy_id
            or self.alternative not in ENDPOINT_METHOD_IDS
            or self.selected_method not in ENDPOINT_METHOD_IDS
            or any(
                not math.isfinite(value)
                for value in (
                    self.predicted_bacc_regret,
                    self.predicted_log_loss_delta,
                    self.support_regret_sum_pp,
                    self.support_voter_dispersion_for_sum_pp,
                )
            )
            or self.delete_bacc_positive_count not in range(9)
            or self.delete_log_loss_safe_count not in range(9)
            or not self.reason
        ):
            raise ProtocolError("Nested donor route decision drifted.")
        require_sha256(self.descriptor_hash, "descriptor_hash")
        for digest in self.model_hashes:
            require_sha256(digest, "model_hash")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_nested_regret_route_decision_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "policy_id": self.policy_id,
            "alternative": self.alternative,
            "selected_method": self.selected_method,
            "predicted_bacc_regret": self.predicted_bacc_regret,
            "predicted_log_loss_delta": self.predicted_log_loss_delta,
            "delete_bacc_positive_count": self.delete_bacc_positive_count,
            "delete_log_loss_safe_count": self.delete_log_loss_safe_count,
            "support_regret_sum_pp": self.support_regret_sum_pp,
            "support_voter_dispersion_for_sum_pp": self.support_voter_dispersion_for_sum_pp,
            "reason": self.reason,
            "descriptor_hash": self.descriptor_hash,
            "model_hashes": list(self.model_hashes),
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


def labels_by_sample(
    labels: Sequence[BinaryLabel], *, expected_scope: str | None = None
) -> Mapping[tuple[str, str, str], BinaryLabel]:
    rows = tuple(labels)
    result = {row.key: row for row in rows}
    if not rows or len(result) != len(rows):
        raise ProtocolError("Scoped labels are empty or duplicated.")
    scopes = {row.scope for row in rows}
    if len(scopes) != 1 or (
        expected_scope is not None and scopes != {str(expected_scope)}
    ):
        raise ProtocolError("Scoped labels mix or mismatch capabilities.")
    return MappingProxyType(result)


__all__ = (
    "BinaryLabel",
    "CandidateDescriptor",
    "CenterBalancedRidgeModel",
    "CenterProbabilitySurface",
    "DonorRegretRow",
    "EndpointCasePrediction",
    "PhysicalProbabilitySurface",
    "RouteDecision",
    "labels_by_sample",
)
