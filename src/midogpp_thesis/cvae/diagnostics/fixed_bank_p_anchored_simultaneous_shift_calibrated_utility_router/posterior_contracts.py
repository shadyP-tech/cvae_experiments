"""Immutable contracts for grouped support cross-fitting and held-case posteriors."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    FINGERPRINT_FEATURE_COUNT,
    FINGERPRINT_STATISTIC_IDS,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    RELIABILITY_AUC_MIN,
    RELIABILITY_BRIER_SKILL_MIN,
    SUPPORT_CROSSFIT_FOLD_COUNT,
    TARGET_POSTERIOR_C,
    TARGET_POSTERIOR_MAX_ITER,
    TARGET_POSTERIOR_RANDOM_STATE,
    TARGET_POSTERIOR_SOLVER,
    physical_action_ids,
)
from .hashing import canonical_hash, require_sha256


CONTROL_IDS = (
    PRIMARY_FINGERPRINT_CONTROL_ID,
    BLOCKED_FINGERPRINT_CONTROL_ID,
)


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
        expected_names = (
            tuple(
                f"{action}::{statistic}"
                for action in physical_action_ids(self.center)
                for statistic in FINGERPRINT_STATISTIC_IDS
            )
            if self.center in CENTERS
            else ()
        )
        if (
            self.center not in CENTERS
            or not samples
            or len(samples) != len(cases)
            or len(samples) != len(set(samples))
            or names != expected_names
            or len(names) != FINGERPRINT_FEATURE_COUNT
            or len(names) != len(set(names))
            or values.shape != (len(samples), FINGERPRINT_FEATURE_COUNT)
            or not np.isfinite(values).all()
            or bool(np.any((values[:, 0::3] < 0.0) | (values[:, 0::3] > 1.0)))
            or bool(np.any((values[:, 1::3] < 0.0) | (values[:, 1::3] > 0.5)))
            or bool(np.any((values[:, 2::3] < 0.0) | (values[:, 2::3] > 1.0)))
            or self.control_id not in CONTROL_IDS
        ):
            raise ProtocolError("PSSCUR physical fingerprint topology drifted.")
        require_sha256(self.source_surface_hash, "physical_source_surface_hash")
        values.setflags(write=False)
        payload = {
            "schema_version": "fixed_bank_psscur_physical_fingerprint_v1",
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
            raise ProtocolError("PSSCUR fingerprint requested an absent case.")
        return positions

    def summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_physical_fingerprint_summary_v1",
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


@dataclass(frozen=True, order=True)
class SupportFoldPlan:
    """One deterministic whole-case fold inside the legal H-c support."""

    target_center: str
    held_case_id: str
    fold_id: int
    training_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    fingerprint_hash: str
    plan_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        training = tuple(sorted(str(value) for value in self.training_case_ids))
        validation = tuple(sorted(str(value) for value in self.validation_case_ids))
        if (
            self.target_center not in CENTERS
            or not self.held_case_id
            or type(self.fold_id) is not int
            or not 0 <= self.fold_id < SUPPORT_CROSSFIT_FOLD_COUNT
            or not training
            or not validation
            or self.held_case_id in training
            or self.held_case_id in validation
            or set(training) & set(validation)
            or len(training) != len(set(training))
            or len(validation) != len(set(validation))
        ):
            raise ProtocolError("PSSCUR support-fold plan drifted.")
        require_sha256(self.fingerprint_hash, "fingerprint_hash")
        object.__setattr__(self, "training_case_ids", training)
        object.__setattr__(self, "validation_case_ids", validation)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_support_fold_plan_v1",
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "fold_id": self.fold_id,
            "training_case_ids": list(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
            "fingerprint_hash": self.fingerprint_hash,
            "whole_case_grouped": True,
            "held_case_excluded": True,
            "labels_used_to_assign_folds": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_hash": self.plan_hash}


@dataclass(frozen=True)
class TargetLocalPosteriorModel:
    """One fold model trained on H-c minus its validation cases."""

    target_center: str
    held_case_id: str
    fold_id: int
    training_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
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
    fold_plan_hash: str
    iterations: int
    converged: bool
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(str(value) for value in self.training_case_ids)
        validation = tuple(str(value) for value in self.validation_case_ids)
        names = tuple(str(value) for value in self.feature_names)
        mean = tuple(float(value) for value in self.feature_mean)
        scale = tuple(float(value) for value in self.feature_scale)
        coefficients = tuple(float(value) for value in self.coefficients)
        numeric = (*mean, *scale, *coefficients, float(self.intercept))
        if (
            self.target_center not in CENTERS
            or not self.held_case_id
            or type(self.fold_id) is not int
            or not 0 <= self.fold_id < SUPPORT_CROSSFIT_FOLD_COUNT
            or self.held_case_id in training
            or self.held_case_id in validation
            or not training
            or not validation
            or set(training) & set(validation)
            or len(training) != len(set(training))
            or len(validation) != len(set(validation))
            or len(names) != FINGERPRINT_FEATURE_COUNT
            or len(mean) != len(names)
            or len(scale) != len(names)
            or len(coefficients) != len(names)
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in scale)
            or type(self.training_row_count) is not int
            or type(self.training_n_positive) is not int
            or type(self.training_n_negative) is not int
            or self.training_row_count
            != self.training_n_positive + self.training_n_negative
            or min(self.training_n_positive, self.training_n_negative) <= 0
            or type(self.iterations) is not int
            or self.iterations <= 0
            or self.converged is not True
        ):
            raise ProtocolError("PSSCUR target-local posterior model drifted.")
        for digest, name in (
            (self.fingerprint_hash, "fingerprint_hash"),
            (self.training_identity_hash, "training_identity_hash"),
            (self.fold_plan_hash, "fold_plan_hash"),
        ):
            require_sha256(digest, name)
        object.__setattr__(self, "training_case_ids", training)
        object.__setattr__(self, "validation_case_ids", validation)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "model_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_target_local_posterior_model_v1",
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "fold_id": self.fold_id,
            "training_case_ids": list(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
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
            "fold_plan_hash": self.fold_plan_hash,
            "C": TARGET_POSTERIOR_C,
            "class_weight": "balanced",
            "solver": TARGET_POSTERIOR_SOLVER,
            "max_iter": TARGET_POSTERIOR_MAX_ITER,
            "random_state": TARGET_POSTERIOR_RANDOM_STATE,
            "iterations": self.iterations,
            "converged": self.converged,
            "route_local_not_shared": True,
            "held_case_labels_used": False,
            "validation_case_labels_used_in_fit": False,
            "raw_support_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "model_hash": self.model_hash}


@dataclass(frozen=True)
class CasePosteriorPrediction:
    target_center: str
    route_held_case_id: str
    predicted_case_id: str
    fold_id: int
    prediction_role: str
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
            or not self.route_held_case_id
            or not self.predicted_case_id
            or type(self.fold_id) is not int
            or not 0 <= self.fold_id < SUPPORT_CROSSFIT_FOLD_COUNT
            or self.prediction_role not in {"HELD_ROUTE", "SUPPORT_OOF"}
            or (
                self.prediction_role == "HELD_ROUTE"
                and self.predicted_case_id != self.route_held_case_id
            )
            or (
                self.prediction_role == "SUPPORT_OOF"
                and self.predicted_case_id == self.route_held_case_id
            )
            or not samples
            or len(samples) != len(set(samples))
            or len(balanced) != len(samples)
            or len(natural) != len(samples)
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (*balanced, *natural)
            )
        ):
            raise ProtocolError("PSSCUR posterior prediction drifted.")
        require_sha256(self.model_hash, "target_posterior_model_hash")
        require_sha256(self.fingerprint_hash, "fingerprint_hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "balanced_probabilities", balanced)
        object.__setattr__(self, "natural_probabilities", natural)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_case_posterior_prediction_v1",
            "target_center": self.target_center,
            "route_held_case_id": self.route_held_case_id,
            "predicted_case_id": self.predicted_case_id,
            "fold_id": self.fold_id,
            "prediction_role": self.prediction_role,
            "sample_ids": list(self.sample_ids),
            "balanced_probabilities": list(self.balanced_probabilities),
            "natural_probabilities": list(self.natural_probabilities),
            "model_hash": self.model_hash,
            "fingerprint_hash": self.fingerprint_hash,
            "predicted_case_labels_used": False,
            "final_classifier_prediction": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True)
class RoutePosteriorEnsemble:
    """Five held-case predictions plus route-local OOF reliability."""

    target_center: str
    held_case_id: str
    control_id: str
    fold_plan_hashes: tuple[str, ...]
    model_hashes: tuple[str, ...]
    held_prediction_hashes: tuple[str, ...]
    held_sample_ids: tuple[str, ...]
    held_natural_probabilities_by_fold: tuple[tuple[float, ...], ...]
    support_row_count: int
    support_n_positive: int
    support_n_negative: int
    oof_sample_count: int
    oof_auc: float
    oof_brier: float
    oof_prevalence_brier: float
    oof_brier_skill: float
    oof_identity_hash: str
    oof_prediction_hash: str
    ensemble_hash: str = field(init=False)

    def __post_init__(self) -> None:
        fold_count = SUPPORT_CROSSFIT_FOLD_COUNT
        samples = tuple(str(value) for value in self.held_sample_ids)
        probabilities = tuple(
            tuple(float(value) for value in row)
            for row in self.held_natural_probabilities_by_fold
        )
        metrics = (
            float(self.oof_auc),
            float(self.oof_brier),
            float(self.oof_prevalence_brier),
            float(self.oof_brier_skill),
        )
        if (
            self.target_center not in CENTERS
            or not self.held_case_id
            or self.control_id not in CONTROL_IDS
            or len(self.fold_plan_hashes) != fold_count
            or len(self.model_hashes) != fold_count
            or len(self.held_prediction_hashes) != fold_count
            or len(probabilities) != fold_count
            or not samples
            or len(samples) != len(set(samples))
            or any(len(row) != len(samples) for row in probabilities)
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for row in probabilities
                for value in row
            )
            or type(self.support_row_count) is not int
            or type(self.support_n_positive) is not int
            or type(self.support_n_negative) is not int
            or self.support_row_count
            != self.support_n_positive + self.support_n_negative
            or min(self.support_n_positive, self.support_n_negative) <= 0
            or self.oof_sample_count != self.support_row_count
            or any(not math.isfinite(value) for value in metrics)
            or not 0.0 <= self.oof_auc <= 1.0
            or self.oof_brier < 0.0
            or self.oof_prevalence_brier < 0.0
        ):
            raise ProtocolError("PSSCUR route posterior ensemble drifted.")
        for digest in (
            *self.fold_plan_hashes,
            *self.model_hashes,
            *self.held_prediction_hashes,
            self.oof_identity_hash,
            self.oof_prediction_hash,
        ):
            require_sha256(digest, "route_posterior_digest")
        object.__setattr__(self, "held_sample_ids", samples)
        object.__setattr__(
            self, "held_natural_probabilities_by_fold", probabilities
        )
        object.__setattr__(self, "oof_auc", metrics[0])
        object.__setattr__(self, "oof_brier", metrics[1])
        object.__setattr__(self, "oof_prevalence_brier", metrics[2])
        object.__setattr__(self, "oof_brier_skill", metrics[3])
        object.__setattr__(self, "ensemble_hash", canonical_hash(self._unhashed()))

    @property
    def reliability_pass(self) -> bool:
        return (
            self.oof_auc > RELIABILITY_AUC_MIN
            and self.oof_brier_skill > RELIABILITY_BRIER_SKILL_MIN
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_route_posterior_ensemble_v1",
            "target_center": self.target_center,
            "held_case_id": self.held_case_id,
            "control_id": self.control_id,
            "fold_plan_hashes": list(self.fold_plan_hashes),
            "model_hashes": list(self.model_hashes),
            "held_prediction_hashes": list(self.held_prediction_hashes),
            "held_sample_ids": list(self.held_sample_ids),
            "held_natural_probabilities_by_fold": [
                list(row) for row in self.held_natural_probabilities_by_fold
            ],
            "support_row_count": self.support_row_count,
            "support_n_positive": self.support_n_positive,
            "support_n_negative": self.support_n_negative,
            "oof_sample_count": self.oof_sample_count,
            "oof_auc": self.oof_auc,
            "oof_brier": self.oof_brier,
            "oof_prevalence_brier": self.oof_prevalence_brier,
            "oof_brier_skill": self.oof_brier_skill,
            "oof_identity_hash": self.oof_identity_hash,
            "oof_prediction_hash": self.oof_prediction_hash,
            "reliability_pass": self.reliability_pass,
            "whole_case_grouped_crossfit": True,
            "held_case_labels_used": False,
            "raw_support_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "ensemble_hash": self.ensemble_hash}


def index_ensembles(
    rows: tuple[RoutePosteriorEnsemble, ...],
) -> Mapping[tuple[str, str], RoutePosteriorEnsemble]:
    indexed = {(row.target_center, row.held_case_id): row for row in rows}
    if len(indexed) != len(rows):
        raise ProtocolError("PSSCUR posterior ensembles duplicate a route.")
    return MappingProxyType(indexed)


__all__ = (
    "CasePosteriorPrediction",
    "PhysicalFingerprintSurface",
    "RoutePosteriorEnsemble",
    "SupportFoldPlan",
    "TargetLocalPosteriorModel",
    "index_ensembles",
)
