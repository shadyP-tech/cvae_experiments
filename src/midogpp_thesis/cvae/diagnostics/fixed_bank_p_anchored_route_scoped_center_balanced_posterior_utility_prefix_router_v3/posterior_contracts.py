"""Plain, pickle-safe contracts for route-local posterior fits.

CBPUPR fits exactly one posterior on ``H-c`` for each fingerprint control.  A
pseudo ``(H,J,d)`` route references the already sealed ``J-d`` fit and adds
outer-H lineage; it does not create another model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .canonical_probabilities import canonical_float32_probabilities
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CANONICAL_PHYSICAL_ROW_ORDER,
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
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
from .row_order import (
    require_canonical_center_row_order,
    require_canonical_sample_ids,
)


CONTROL_IDS = (
    PRIMARY_FINGERPRINT_CONTROL_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
)


@dataclass(frozen=True)
class PhysicalFingerprintSurface:
    center: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: np.ndarray
    source_surface_hash: str
    control_id: str
    fingerprint_hash: str = field(init=False)

    def __post_init__(self) -> None:
        topology_error = "CBPUPR physical fingerprint topology drifted."
        samples, cases = require_canonical_center_row_order(
            self.sample_ids,
            self.case_ids,
            error_message=topology_error,
        )
        names = tuple(str(value) for value in self.feature_names)
        values = np.ascontiguousarray(self.feature_values, dtype=np.float64)
        expected_names = tuple(
            f"{action}::{statistic}"
            for action in physical_action_ids(self.center)
            for statistic in FINGERPRINT_STATISTIC_IDS
        ) if self.center in CENTERS else ()
        if (
            self.center not in CENTERS
            or names != expected_names
            or len(names) != FINGERPRINT_FEATURE_COUNT
            or values.shape != (len(samples), FINGERPRINT_FEATURE_COUNT)
            or not np.isfinite(values).all()
            or self.control_id not in CONTROL_IDS
        ):
            raise ProtocolError(topology_error)
        require_sha256(self.source_surface_hash, "physical_source_surface_hash")
        values.setflags(write=False)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(
            self,
            "fingerprint_hash",
            canonical_hash(
                {
                    "schema_version": "fixed_bank_cbpupr_fingerprint_v1",
                    "center": self.center,
                    "sample_ids": list(samples),
                    "case_ids": list(cases),
                    "row_order": CANONICAL_PHYSICAL_ROW_ORDER,
                    "feature_names": list(names),
                    "feature_array_sha256": sha256_array(values),
                    "source_surface_hash": self.source_surface_hash,
                    "control_id": self.control_id,
                    "labels_used": False,
                }
            ),
        )

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.case_ids))

    def positions(self, case_id: object) -> np.ndarray:
        result = np.flatnonzero(np.asarray(self.case_ids, dtype=object) == str(case_id))
        if not len(result):
            raise ProtocolError("CBPUPR fingerprint requested an absent case.")
        return result

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_fingerprint_summary_v1",
            "center": self.center,
            "sample_count": len(self.sample_ids),
            "case_count": len(self.cases),
            "row_order": CANONICAL_PHYSICAL_ROW_ORDER,
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
    target_center: str
    held_case_id: str
    control_id: str
    training_case_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_row_count: int
    training_n_positive: int
    training_n_negative: int
    fingerprint_hash: str
    training_identity_hash: str
    iterations: int
    converged: bool
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(str(value) for value in self.training_case_ids))
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
            or self.control_id not in CONTROL_IDS
            or len(names) != FINGERPRINT_FEATURE_COUNT
            or len(mean) != len(names)
            or len(scale) != len(names)
            or len(coefficients) != len(names)
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in scale)
            or self.training_row_count
            != self.training_n_positive + self.training_n_negative
            or min(self.training_n_positive, self.training_n_negative) <= 0
            or type(self.iterations) is not int
            or self.iterations <= 0
            or self.converged is not True
        ):
            raise ProtocolError("CBPUPR target-local posterior model drifted.")
        require_sha256(self.fingerprint_hash, "fingerprint_hash")
        require_sha256(self.training_identity_hash, "training_identity_hash")
        object.__setattr__(self, "training_case_ids", cases)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_target_posterior_model_v1",
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "control_id": self.control_id,
            "training_case_ids": list(self.training_case_ids),
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "training_row_count": self.training_row_count,
            "training_n_positive": self.training_n_positive,
            "training_n_negative": self.training_n_negative,
            "fingerprint_hash": self.fingerprint_hash,
            "training_identity_hash": self.training_identity_hash,
            "C": TARGET_POSTERIOR_C,
            "class_weight": "balanced",
            "solver": TARGET_POSTERIOR_SOLVER,
            "max_iter": TARGET_POSTERIOR_MAX_ITER,
            "random_state": TARGET_POSTERIOR_RANDOM_STATE,
            "iterations": self.iterations,
            "converged": self.converged,
            "inner_crossfit_or_OOF_reliability_used": False,
            "fit_once_per_target_case_control": True,
            "sealed_prediction_may_be_structurally_referenced_by_H_specific_pseudo_routes": True,
            "held_case_labels_used": False,
            "raw_support_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


@dataclass(frozen=True)
class CasePosteriorPrediction:
    target_center: str
    held_case_id: str
    control_id: str
    sample_ids: tuple[str, ...]
    natural_probabilities: tuple[float, ...]
    model_hash: str
    fingerprint_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        prediction_error = "CBPUPR posterior prediction drifted."
        samples = require_canonical_sample_ids(
            self.sample_ids,
            error_message=prediction_error,
        )
        values_array = canonical_float32_probabilities(
            self.natural_probabilities,
            expected_length=len(samples),
        )
        values = tuple(float(value) for value in values_array)
        if (
            self.target_center not in CENTERS
            or not self.held_case_id
            or self.control_id not in CONTROL_IDS
        ):
            raise ProtocolError(prediction_error)
        require_sha256(self.model_hash, "posterior_model_hash")
        require_sha256(self.fingerprint_hash, "fingerprint_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "natural_probabilities", values)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_case_posterior_v1",
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "control_id": self.control_id,
            "sample_ids": list(self.sample_ids),
            "natural_probabilities": list(self.natural_probabilities),
            "model_hash": self.model_hash,
            "fingerprint_hash": self.fingerprint_hash,
            "held_case_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True)
class PseudoPosteriorReference:
    outer_target_center: str
    pseudo_target_center: str
    held_case_id: str
    control_id: str
    posterior_prediction_hash: str
    reference_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.outer_target_center not in CENTERS
            or self.pseudo_target_center not in CENTERS
            or self.outer_target_center == self.pseudo_target_center
            or not self.held_case_id
            or self.control_id not in CONTROL_IDS
        ):
            raise ProtocolError("CBPUPR pseudo posterior lineage drifted.")
        require_sha256(self.posterior_prediction_hash, "posterior_prediction_hash")
        object.__setattr__(self, "reference_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_pseudo_posterior_reference_v1",
            "outer_target_center": self.outer_target_center,
            "pseudo_target_center": self.pseudo_target_center,
            "held_case_id": self.held_case_id,
            "control_id": self.control_id,
            "posterior_prediction_hash": self.posterior_prediction_hash,
            "posterior_fit_scope": "J_minus_d",
            "outer_H_support_rows_or_labels_enter_J_minus_d_posterior_fit_or_normalization": False,
            "outer_H_frozen_label_free_expert_fingerprint_covariates_present": True,
            "posterior_is_outer_H_covariate_invariant": False,
            "outer_H_specific_posterior_refit_performed": False,
            "pseudo_case_d_rows_or_labels_enter_own_posterior_fit": False,
            "posterior_refit": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "reference_hash": self.reference_hash}


def build_pseudo_posterior_references(
    predictions: Sequence[CasePosteriorPrediction],
) -> tuple[PseudoPosteriorReference, ...]:
    rows = tuple(predictions)
    indexed = {(row.target_center, row.held_case_id, row.control_id): row for row in rows}
    cases_by_center = {
        center: tuple(
            sorted(
                key[1]
                for key in indexed
                if key[0] == center and key[2] == PRIMARY_FINGERPRINT_CONTROL_ID
            )
        )
        for center in CENTERS
    }
    expected_keys = {
        (center, case, control)
        for center in CENTERS
        for case in cases_by_center[center]
        for control in CONTROL_IDS
    }
    if (
        len(rows) != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or len(indexed) != len(rows)
        or set(indexed) != expected_keys
        or any(
            len(cases_by_center[center]) != EXPECTED_CASE_COUNTS_BY_CENTER[center]
            for center in CENTERS
        )
    ):
        raise ProtocolError("CBPUPR posterior prediction rectangle is incomplete.")
    try:
        result = tuple(
            PseudoPosteriorReference(
                outer,
                pseudo,
                case,
                control,
                indexed[(pseudo, case, control)].prediction_hash,
            )
            for outer in CENTERS
            for pseudo in CENTERS
            if pseudo != outer
            for case in cases_by_center[pseudo]
            for control in CONTROL_IDS
        )
    except KeyError as exc:  # defensive fail-closed boundary
        raise ProtocolError("CBPUPR pseudo posterior reference is absent.") from exc
    route_count = len(
        {(row.outer_target_center, row.pseudo_target_center, row.held_case_id) for row in result}
    )
    if (
        route_count != EXPECTED_PSEUDO_ROUTE_COUNT
        or len(result) != len(CONTROL_IDS) * EXPECTED_PSEUDO_ROUTE_COUNT
        or len({row.reference_hash for row in result}) != len(result)
    ):
        raise ProtocolError("CBPUPR pseudo posterior reference topology drifted.")
    return result


def index_predictions(
    rows: Sequence[CasePosteriorPrediction],
) -> dict[tuple[str, str, str], CasePosteriorPrediction]:
    indexed = {(row.target_center, row.held_case_id, row.control_id): row for row in rows}
    if len(indexed) != len(tuple(rows)):
        raise ProtocolError("CBPUPR posterior predictions duplicate a route/control.")
    return indexed


__all__ = (
    "CONTROL_IDS",
    "CasePosteriorPrediction",
    "PhysicalFingerprintSurface",
    "PseudoPosteriorReference",
    "TargetLocalPosteriorModel",
    "build_pseudo_posterior_references",
    "index_predictions",
)
