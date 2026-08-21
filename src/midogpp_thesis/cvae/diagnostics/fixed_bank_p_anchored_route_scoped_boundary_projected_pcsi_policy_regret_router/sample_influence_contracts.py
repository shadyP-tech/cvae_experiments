"""Immutable contracts for target-local fingerprints and BACC influence scores."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    DIRECTION_IDS,
    FINGERPRINT_FEATURE_COUNT,
    FINGERPRINT_STATISTIC_IDS,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    TARGET_POSTERIOR_C,
    TARGET_POSTERIOR_MAX_ITER,
    TARGET_POSTERIOR_RANDOM_STATE,
    TARGET_POSTERIOR_SOLVER,
    physical_action_ids,
)
from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True)
class PhysicalFingerprintSurface:
    """One label-free, sample-aligned physical-action fingerprint surface."""

    center: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: np.ndarray
    source_surface_hash: str
    control_id: str
    fingerprint_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        cases = tuple(str(value) for value in self.case_ids)
        names = tuple(str(value) for value in self.feature_names)
        values = np.ascontiguousarray(self.feature_values, dtype=np.float64)
        expected_names = tuple(
            f"{action}::{statistic}"
            for action in physical_action_ids(self.center)
            for statistic in FINGERPRINT_STATISTIC_IDS
        ) if self.center in CENTERS else ()
        if (
            self.center not in CENTERS
            or not samples
            or len(samples) != len(cases)
            or len(samples) != len(set(samples))
            or len(names) != FINGERPRINT_FEATURE_COUNT
            or names != expected_names
            or len(names) != len(set(names))
            or values.shape != (len(samples), FINGERPRINT_FEATURE_COUNT)
            or not np.isfinite(values).all()
            or bool(np.any((values[:, 0::3] < 0.0) | (values[:, 0::3] > 1.0)))
            or bool(np.any((values[:, 1::3] < 0.0) | (values[:, 1::3] > 0.5)))
            or bool(np.any((values[:, 2::3] < 0.0) | (values[:, 2::3] > 1.0)))
            or self.control_id not in {
                PRIMARY_FINGERPRINT_CONTROL_ID,
                BLOCKED_FINGERPRINT_CONTROL_ID,
            }
        ):
            raise ProtocolError("PCSI-RACR physical fingerprint topology drifted.")
        require_sha256(self.source_surface_hash, "physical_source_surface_hash")
        values.setflags(write=False)
        payload = {
            "schema_version": "fixed_bank_pcsi_racr_physical_fingerprint_v1",
            "center": self.center,
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "feature_names": list(names),
            "feature_array_sha256": sha256_array(values),
            "source_surface_hash": self.source_surface_hash,
            "control_id": self.control_id,
            "labels_used": False,
        }
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "fingerprint_hash", canonical_hash(payload))

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.case_ids))

    def positions(self, case_id: object) -> np.ndarray:
        positions = np.flatnonzero(
            np.asarray(self.case_ids, dtype=object) == str(case_id)
        )
        if not len(positions):
            raise ProtocolError("PCSI-RACR fingerprint requested an absent case.")
        return positions

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_physical_fingerprint_summary_v1",
            "center": self.center,
            "sample_count": len(self.sample_ids),
            "case_count": len(self.cases),
            "feature_names": list(self.feature_names),
            "feature_array_sha256": sha256_array(self.feature_values),
            "source_surface_hash": self.source_surface_hash,
            "control_id": self.control_id,
            "fingerprint_hash": self.fingerprint_hash,
            "raw_feature_rows_persisted": False,
            "labels_used": False,
        }


@dataclass(frozen=True)
class TargetLocalPosteriorModel:
    """A route-local H-c posterior model; no model is shared across cases."""

    target_center: str
    held_case_id: str
    support_case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    support_row_count: int
    support_n_positive: int
    support_n_negative: int
    fingerprint_hash: str
    support_identity_hash: str
    iterations: int
    converged: bool
    fit_status: str = "FIT"
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(str(value) for value in self.support_case_ids)
        names = tuple(str(value) for value in self.feature_names)
        mean = tuple(float(value) for value in self.feature_mean)
        scale = tuple(float(value) for value in self.feature_scale)
        coefficients = tuple(float(value) for value in self.coefficients)
        numeric = (*mean, *scale, *coefficients, float(self.intercept))
        if (
            self.target_center not in CENTERS
            or not self.held_case_id
            or self.held_case_id in cases
            or not cases
            or len(cases) != len(set(cases))
            or len(names) != FINGERPRINT_FEATURE_COUNT
            or len(mean) != len(names)
            or len(scale) != len(names)
            or len(coefficients) != len(names)
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in scale)
            or type(self.support_row_count) is not int
            or type(self.support_n_positive) is not int
            or type(self.support_n_negative) is not int
            or self.support_row_count
            != self.support_n_positive + self.support_n_negative
            or min(self.support_n_positive, self.support_n_negative) <= 0
            or type(self.iterations) is not int
            or self.iterations <= 0
            or type(self.converged) is not bool
            or self.converged is not True
            or self.fit_status != "FIT"
        ):
            raise ProtocolError("PCSI-RACR target-local posterior model drifted.")
        require_sha256(self.fingerprint_hash, "fingerprint_hash")
        require_sha256(self.support_identity_hash, "support_identity_hash")
        object.__setattr__(self, "support_case_ids", cases)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_target_local_posterior_model_v1",
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "support_case_ids": list(self.support_case_ids),
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "support_row_count": self.support_row_count,
            "support_n_positive": self.support_n_positive,
            "support_n_negative": self.support_n_negative,
            "fingerprint_hash": self.fingerprint_hash,
            "support_identity_hash": self.support_identity_hash,
            "C": TARGET_POSTERIOR_C,
            "class_weight": "balanced",
            "solver": TARGET_POSTERIOR_SOLVER,
            "max_iter": TARGET_POSTERIOR_MAX_ITER,
            "random_state": TARGET_POSTERIOR_RANDOM_STATE,
            "iterations": self.iterations,
            "converged": self.converged,
            "fit_status": self.fit_status,
            "route_local_not_shared": True,
            "held_case_labels_used": False,
            "raw_support_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


@dataclass(frozen=True)
class TargetLocalPosteriorPrediction:
    target_center: str
    case_id: str
    sample_ids: tuple[str, ...]
    balanced_probabilities: tuple[float, ...]
    natural_probabilities: tuple[float, ...]
    model_hash: str
    fingerprint_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        balanced = tuple(float(value) for value in self.balanced_probabilities)
        natural = tuple(float(value) for value in self.natural_probabilities)
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not samples
            or len(samples) != len(set(samples))
            or len(balanced) != len(samples)
            or len(natural) != len(samples)
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (*balanced, *natural)
            )
        ):
            raise ProtocolError("PCSI-RACR target-local posterior prediction drifted.")
        require_sha256(self.model_hash, "target_posterior_model_hash")
        require_sha256(self.fingerprint_hash, "fingerprint_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "balanced_probabilities", balanced)
        object.__setattr__(self, "natural_probabilities", natural)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_target_local_posterior_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_ids": list(self.sample_ids),
            "balanced_probabilities": list(self.balanced_probabilities),
            "natural_probabilities": list(self.natural_probabilities),
            "model_hash": self.model_hash,
            "fingerprint_hash": self.fingerprint_hash,
            "held_case_labels_used": False,
            "final_classifier_prediction": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True, order=True)
class InfluencePrediction:
    descriptor_hash: str
    target_center: str
    case_id: str
    alternative: str
    direction: str
    crossing_count: int
    target_score: float
    posterior_prediction_hash: str
    influence_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or self.alternative not in ALTERNATIVE_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or type(self.crossing_count) is not int
            or self.crossing_count < 0
            or not math.isfinite(float(self.target_score))
            or (self.crossing_count == 0 and abs(float(self.target_score)) > 1.0e-15)
        ):
            raise ProtocolError("PCSI-RACR sample-influence prediction drifted.")
        require_sha256(self.descriptor_hash, "utility_descriptor_hash")
        require_sha256(
            self.posterior_prediction_hash, "target_posterior_prediction_hash"
        )
        object.__setattr__(self, "target_score", float(self.target_score))
        object.__setattr__(self, "influence_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.alternative, self.direction

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_sample_influence_prediction_v1",
            "descriptor_hash": self.descriptor_hash,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "alternative": self.alternative,
            "direction": self.direction,
            "crossing_count": self.crossing_count,
            "target_score": self.target_score,
            "posterior_prediction_hash": self.posterior_prediction_hash,
            "score_estimand": "crossfit_target_local_balanced_accuracy_influence",
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "influence_hash": self.influence_hash}


def index_predictions(
    rows: tuple[TargetLocalPosteriorPrediction, ...],
) -> Mapping[tuple[str, str], TargetLocalPosteriorPrediction]:
    indexed = {(row.target_center, row.case_id): row for row in rows}
    if len(indexed) != len(rows):
        raise ProtocolError("PCSI-RACR posterior predictions duplicate a route.")
    return MappingProxyType(indexed)


__all__ = (
    "InfluencePrediction",
    "PhysicalFingerprintSurface",
    "TargetLocalPosteriorModel",
    "TargetLocalPosteriorPrediction",
    "index_predictions",
)
