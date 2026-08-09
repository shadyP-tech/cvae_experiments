"""Family, fitted-fold, and crossfit contracts for the fixed-bank audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    BLOCKED_PERMUTATION_SHIFT,
    CENTERS,
    CONTROL_FAMILY_IDS,
    EXACT_BACC_DELTA,
    EXACT_CROSSFIT_SCHEMA,
    EXACT_FAMILY_IDS,
    EXACT_FAMILY_PREDICTORS,
    EXACT_FOLD_SCHEMA,
    EXACT_PREDICTION_SCHEMA,
    FAMILY_DESIGN_SCHEMA,
    FAMILY_SPEC_SCHEMA,
    GLOBAL_SOURCE_EXACT_CONTROL,
    OUTER_INFERENCE_UNIT_COUNT,
    PERMUTATION_CONTROL_FAMILY_IDS,
    PERMUTATION_PARENT,
    PRIMARY_R_FAMILY_ID,
    RIDGE_ALPHA,
    SECONDARY_CHALLENGER_FAMILY_IDS,
    SMOOTH_BACC_DELTA,
    SMOOTH_CROSSFIT_SCHEMA,
    SMOOTH_DESCRIPTIVE_FAMILY_IDS,
    SMOOTH_FAMILY_PREDICTORS,
    SMOOTH_FOLD_SCHEMA,
    SMOOTH_PREDICTION_SCHEMA,
    candidate_sources,
    expected_training_row_count,
)
from .serialization import (
    canonical_array_hash,
    canonical_hash,
    finite,
    require_sha256,
)


def _scientific_role(family_id: str) -> str:
    if family_id == PRIMARY_R_FAMILY_ID:
        return "primary_r_arm"
    if family_id in SECONDARY_CHALLENGER_FAMILY_IDS:
        return "secondary_challenger_descriptive"
    if family_id in CONTROL_FAMILY_IDS:
        return "control"
    if family_id in SMOOTH_DESCRIPTIVE_FAMILY_IDS:
        return "smooth_descriptive_only"
    raise ProtocolError("Unknown fixed-bank family ID.")


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    predictor_names: tuple[str, ...]
    response_name: str
    source_effects_included: bool
    scientific_role: str
    blocked_permutation_parent: str | None = None
    blocked_permutation_shift: int | None = None

    def __post_init__(self) -> None:
        exact = self.family_id in EXACT_FAMILY_IDS
        smooth = self.family_id in SMOOTH_DESCRIPTIVE_FAMILY_IDS
        if exact == smooth:
            raise ProtocolError("Family must belong to exactly one response surface.")
        expected_predictors = (
            EXACT_FAMILY_PREDICTORS[self.family_id]
            if exact
            else SMOOTH_FAMILY_PREDICTORS[self.family_id]
        )
        expected_response = EXACT_BACC_DELTA if exact else SMOOTH_BACC_DELTA
        expected_source = self.family_id != EXACT_FAMILY_IDS[0]
        parent = PERMUTATION_PARENT.get(self.family_id)
        expected_shift = BLOCKED_PERMUTATION_SHIFT if parent is not None else None
        if (
            self.predictor_names != expected_predictors
            or self.response_name != expected_response
            or self.source_effects_included is not expected_source
            or self.scientific_role != _scientific_role(self.family_id)
            or self.blocked_permutation_parent != parent
            or self.blocked_permutation_shift != expected_shift
            or len(self.predictor_names) > 3
        ):
            raise ProtocolError("Fixed-bank family specification drifted.")

    @property
    def publication_gate_eligible(self) -> bool:
        return self.family_id == PRIMARY_R_FAMILY_ID

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": FAMILY_SPEC_SCHEMA,
            "family_id": self.family_id,
            "predictor_names": list(self.predictor_names),
            "local_predictor_count": len(self.predictor_names),
            "response_name": self.response_name,
            "source_effects_included": self.source_effects_included,
            "scientific_role": self.scientific_role,
            "blocked_permutation_parent": self.blocked_permutation_parent,
            "blocked_permutation_shift": self.blocked_permutation_shift,
            "publication_gate_eligible": self.publication_gate_eligible,
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
            "ridge_alpha": RIDGE_ALPHA,
            "hyperparameter_selection": "none_fixed_predeclared",
        }


@dataclass(frozen=True, eq=False)
class FamilyDesign:
    spec: FamilySpec
    row_keys: tuple[tuple[str, str, str], ...]
    values: np.ndarray
    source_feature_row_hashes: tuple[str, ...]
    donor_feature_row_hashes: tuple[str, ...]
    feature_surface_hash: str
    design_hash: str = field(init=False)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values, dtype=np.float64).copy()
        if matrix.shape != (len(self.row_keys), len(self.spec.predictor_names)):
            raise ProtocolError("Fixed-bank family design geometry drifted.")
        if not np.isfinite(matrix).all() or len(set(self.row_keys)) != len(
            self.row_keys
        ):
            raise ProtocolError("Fixed-bank family design is non-finite or duplicated.")
        if (
            len(self.source_feature_row_hashes) != len(self.row_keys)
            or len(self.donor_feature_row_hashes) != len(self.row_keys)
        ):
            raise ProtocolError("Fixed-bank family design provenance drifted.")
        source_hashes = tuple(
            require_sha256(value, "source_feature_row_hash")
            for value in self.source_feature_row_hashes
        )
        donor_hashes = tuple(
            require_sha256(value, "donor_feature_row_hash")
            for value in self.donor_feature_row_hashes
        )
        surface_hash = require_sha256(
            self.feature_surface_hash, "feature_surface_hash"
        )
        if self.spec.blocked_permutation_parent is None and donor_hashes != source_hashes:
            raise ProtocolError("Unpermuted family design changed row provenance.")
        if self.spec.blocked_permutation_parent is not None and donor_hashes == source_hashes:
            raise ProtocolError("Blocked permutation failed to change donors.")
        matrix.setflags(write=False)
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "source_feature_row_hashes", source_hashes)
        object.__setattr__(self, "donor_feature_row_hashes", donor_hashes)
        object.__setattr__(self, "feature_surface_hash", surface_hash)
        object.__setattr__(self, "design_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": FAMILY_DESIGN_SCHEMA,
            "family_spec": self.spec.to_payload(),
            "row_keys": [list(key) for key in self.row_keys],
            "source_feature_row_hashes": list(self.source_feature_row_hashes),
            "donor_feature_row_hashes": list(self.donor_feature_row_hashes),
            "feature_surface_hash": self.feature_surface_hash,
            "values_sha256": canonical_array_hash(self.values),
            "candidate_source_categorical_block_added_during_fit": (
                self.spec.source_effects_included
            ),
            "within_query_blocked_permutation": (
                self.spec.blocked_permutation_parent is not None
            ),
            "utility_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "family_design_hash": self.design_hash}


@dataclass(frozen=True)
class CrossfitFoldAudit:
    family_id: str
    response_name: str
    outer_target_id: str
    query_id: str
    training_row_keys: tuple[tuple[str, str, str], ...]
    legal_candidate_source_history_counts: tuple[tuple[str, int], ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    coefficient_covariance_sha256: str
    residual_variance: float
    family_design_hash: str
    model_hash: str = field(init=False)
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.response_name == EXACT_BACC_DELTA:
            allowed = EXACT_FAMILY_IDS
        elif self.response_name == SMOOTH_BACC_DELTA:
            allowed = SMOOTH_DESCRIPTIVE_FAMILY_IDS
        else:
            raise ProtocolError("Unknown fixed-bank fold response.")
        if self.family_id not in allowed:
            raise ProtocolError("Fixed-bank fold family/response mismatch.")
        candidates = candidate_sources(self.outer_target_id, self.query_id)
        held = {self.outer_target_id, self.query_id}
        expected_count = expected_training_row_count()
        if (
            len(self.training_row_keys) != expected_count
            or len(set(self.training_row_keys)) != expected_count
            or any(not held.isdisjoint(key) for key in self.training_row_keys)
        ):
            raise ProtocolError("Fixed-bank fold violated strict all-role H/q exclusion.")
        histories = dict(self.legal_candidate_source_history_counts)
        per_source = (len(CENTERS) - 3) * (len(CENTERS) - 4)
        if (
            tuple(histories) != candidates
            or any(histories[source] != per_source for source in candidates)
        ):
            raise ProtocolError("Known-candidate e history coverage drifted.")
        mean = tuple(finite(value, "feature_mean") for value in self.feature_mean)
        scale = tuple(finite(value, "feature_scale") for value in self.feature_scale)
        coefficients = tuple(
            finite(value, "coefficients") for value in self.coefficients
        )
        if (
            len(self.feature_names) != len(mean)
            or len(mean) != len(scale)
            or len(mean) != len(coefficients)
            or len(set(self.feature_names)) != len(self.feature_names)
            or any(value <= 0.0 for value in scale)
        ):
            raise ProtocolError("Fixed-bank fitted parameter geometry drifted.")
        covariance_hash = require_sha256(
            self.coefficient_covariance_sha256,
            "coefficient_covariance_sha256",
        )
        design_hash = require_sha256(self.family_design_hash, "family_design_hash")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", finite(self.intercept, "intercept"))
        object.__setattr__(
            self,
            "residual_variance",
            finite(self.residual_variance, "residual_variance"),
        )
        object.__setattr__(self, "coefficient_covariance_sha256", covariance_hash)
        object.__setattr__(self, "family_design_hash", design_hash)
        model_payload = {
            "response_name": self.response_name,
            "family_id": self.family_id,
            "training_row_keys": [list(key) for key in self.training_row_keys],
            "feature_names": list(self.feature_names),
            "feature_mean": list(mean),
            "feature_scale": list(scale),
            "intercept": self.intercept,
            "coefficients": list(coefficients),
            "coefficient_covariance_sha256": covariance_hash,
            "residual_variance": self.residual_variance,
            "ridge_alpha": RIDGE_ALPHA,
        }
        object.__setattr__(self, "model_hash", canonical_hash(model_payload))
        object.__setattr__(self, "fold_hash", canonical_hash(self._unhashed()))

    @property
    def held_pair(self) -> tuple[str, str]:
        return self.outer_target_id, self.query_id

    def _unhashed(self) -> dict[str, object]:
        schema = (
            EXACT_FOLD_SCHEMA
            if self.response_name == EXACT_BACC_DELTA
            else SMOOTH_FOLD_SCHEMA
        )
        return {
            "schema_version": schema,
            "family_id": self.family_id,
            "response_name": self.response_name,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "excluded_domain_ids": [self.outer_target_id, self.query_id],
            "training_row_keys": [list(key) for key in self.training_row_keys],
            "training_row_count": len(self.training_row_keys),
            "legal_candidate_source_history_counts": [
                list(value) for value in self.legal_candidate_source_history_counts
            ],
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "coefficient_covariance_sha256": self.coefficient_covariance_sha256,
            "residual_variance": self.residual_variance,
            "family_design_hash": self.family_design_hash,
            "model_hash": self.model_hash,
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_cluster_unit": "outer_target_query",
            "strict_H_q_all_role_exclusion": True,
            "candidate_e_history_retained_for_known_bank": True,
            "same_model_scores_all_legal_e": True,
            "scaling_fit_on_training_fold_only": True,
            "hyperparameter_selection": "none_fixed_predeclared",
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
            "exact_terminal": self.response_name == EXACT_BACC_DELTA,
            "smooth_descriptive_only": self.response_name == SMOOTH_BACC_DELTA,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "fold_hash": self.fold_hash}


@dataclass(frozen=True)
class CrossfitPredictionRow:
    family_id: str
    response_name: str
    outer_target_id: str
    query_id: str
    candidate_source: str
    predicted_delta: float
    prediction_standard_error: float
    observed_delta: float
    fold_hash: str
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        candidates = candidate_sources(self.outer_target_id, self.query_id)
        if self.candidate_source not in candidates:
            raise ProtocolError("Crossfit prediction candidate is illegal.")
        if self.response_name == EXACT_BACC_DELTA:
            allowed = EXACT_FAMILY_IDS
        elif self.response_name == SMOOTH_BACC_DELTA:
            allowed = SMOOTH_DESCRIPTIVE_FAMILY_IDS
        else:
            raise ProtocolError("Crossfit prediction response drifted.")
        if self.family_id not in allowed:
            raise ProtocolError("Crossfit prediction family/response mismatch.")
        predicted = finite(self.predicted_delta, "predicted_delta")
        standard_error = finite(
            self.prediction_standard_error, "prediction_standard_error"
        )
        observed = finite(self.observed_delta, "observed_delta")
        if standard_error < 0.0 or observed < -1.0 or observed > 1.0:
            raise ProtocolError("Crossfit prediction values drifted.")
        object.__setattr__(self, "predicted_delta", predicted)
        object.__setattr__(self, "prediction_standard_error", standard_error)
        object.__setattr__(self, "observed_delta", observed)
        object.__setattr__(self, "fold_hash", require_sha256(self.fold_hash, "fold_hash"))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def _unhashed(self) -> dict[str, object]:
        schema = (
            EXACT_PREDICTION_SCHEMA
            if self.response_name == EXACT_BACC_DELTA
            else SMOOTH_PREDICTION_SCHEMA
        )
        return {
            "schema_version": schema,
            "family_id": self.family_id,
            "response_name": self.response_name,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "predicted_delta": self.predicted_delta,
            "prediction_standard_error": self.prediction_standard_error,
            "observed_delta": self.observed_delta,
            "fold_hash": self.fold_hash,
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
            "response_is_terminal_exact": self.response_name == EXACT_BACC_DELTA,
            "smooth_descriptive_only": self.response_name == SMOOTH_BACC_DELTA,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class ExactCrossfitResult:
    predictions: tuple[CrossfitPredictionRow, ...]
    fold_audits: tuple[CrossfitFoldAudit, ...]
    family_ids: tuple[str, ...]
    feature_surface_hash: str
    exact_response_surface_hash: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_crossfit_result(
            self.predictions,
            self.fold_audits,
            self.family_ids,
            allowed=EXACT_FAMILY_IDS,
            response_name=EXACT_BACC_DELTA,
        )
        object.__setattr__(self, "feature_surface_hash", require_sha256(self.feature_surface_hash, "feature_surface_hash"))
        object.__setattr__(self, "exact_response_surface_hash", require_sha256(self.exact_response_surface_hash, "exact_response_surface_hash"))
        object.__setattr__(self, "result_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": EXACT_CROSSFIT_SCHEMA,
            "family_ids": list(self.family_ids),
            "feature_surface_hash": self.feature_surface_hash,
            "exact_response_surface_hash": self.exact_response_surface_hash,
            "prediction_row_hashes": [row.row_hash for row in self.predictions],
            "fold_hashes": [row.fold_hash for row in self.fold_audits],
            "strict_H_q_all_role_exclusion": True,
            "training_row_count_per_fold": expected_training_row_count(),
            "candidate_e_history_retained_for_known_bank": True,
            "same_model_scores_all_legal_e": True,
            "exact_response_is_only_model_input": True,
            "smooth_response_used": False,
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "exact_crossfit_hash": self.result_hash}

    @property
    def table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.predictions)

    @property
    def fold_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.fold_audits)


@dataclass(frozen=True)
class SmoothCrossfitResult:
    predictions: tuple[CrossfitPredictionRow, ...]
    fold_audits: tuple[CrossfitFoldAudit, ...]
    family_ids: tuple[str, ...]
    feature_surface_hash: str
    smooth_response_surface_hash: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_crossfit_result(
            self.predictions,
            self.fold_audits,
            self.family_ids,
            allowed=SMOOTH_DESCRIPTIVE_FAMILY_IDS,
            response_name=SMOOTH_BACC_DELTA,
        )
        object.__setattr__(self, "feature_surface_hash", require_sha256(self.feature_surface_hash, "feature_surface_hash"))
        object.__setattr__(self, "smooth_response_surface_hash", require_sha256(self.smooth_response_surface_hash, "smooth_response_surface_hash"))
        object.__setattr__(self, "result_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": SMOOTH_CROSSFIT_SCHEMA,
            "family_ids": list(self.family_ids),
            "feature_surface_hash": self.feature_surface_hash,
            "smooth_response_surface_hash": self.smooth_response_surface_hash,
            "prediction_row_hashes": [row.row_hash for row in self.predictions],
            "fold_hashes": [row.fold_hash for row in self.fold_audits],
            "strict_H_q_all_role_exclusion": True,
            "training_row_count_per_fold": expected_training_row_count(),
            "smooth_response_is_wholly_separate_descriptive_result": True,
            "exact_model_or_decision_influence": False,
            "terminal_decision_authorized": False,
            "known_fixed_bank_reuse": True,
            "unseen_expert_transfer": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "smooth_crossfit_hash": self.result_hash}

    @property
    def table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.predictions)

    @property
    def fold_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.fold_audits)


def _validate_crossfit_result(
    predictions: Sequence[CrossfitPredictionRow],
    folds: Sequence[CrossfitFoldAudit],
    family_ids: tuple[str, ...],
    *,
    allowed: tuple[str, ...],
    response_name: str,
) -> None:
    if (
        not family_ids
        or len(set(family_ids)) != len(family_ids)
        or any(family not in allowed for family in family_ids)
        or len(predictions) != len(family_ids) * 504
        or len(folds) != len(family_ids) * 72
        or any(row.response_name != response_name for row in predictions)
        or any(row.response_name != response_name for row in folds)
    ):
        raise ProtocolError("Fixed-bank crossfit result coverage drifted.")
    fold_by_family_pair = {
        (row.family_id, row.outer_target_id, row.query_id): row for row in folds
    }
    if len(fold_by_family_pair) != len(folds):
        raise ProtocolError("Fixed-bank crossfit folds are duplicated.")
    for prediction in predictions:
        key = (
            prediction.family_id,
            prediction.outer_target_id,
            prediction.query_id,
        )
        fold = fold_by_family_pair.get(key)
        if fold is None or prediction.fold_hash != fold.fold_hash:
            raise ProtocolError("Prediction did not use the shared H/q fold model.")


def family_spec(family_id: str) -> FamilySpec:
    if family_id in EXACT_FAMILY_IDS:
        response = EXACT_BACC_DELTA
        predictors = EXACT_FAMILY_PREDICTORS[family_id]
    elif family_id in SMOOTH_DESCRIPTIVE_FAMILY_IDS:
        response = SMOOTH_BACC_DELTA
        predictors = SMOOTH_FAMILY_PREDICTORS[family_id]
    else:
        raise ProtocolError("Unknown fixed-bank family ID.")
    parent = PERMUTATION_PARENT.get(family_id)
    return FamilySpec(
        family_id=family_id,
        predictor_names=predictors,
        response_name=response,
        source_effects_included=family_id != EXACT_FAMILY_IDS[0],
        scientific_role=_scientific_role(family_id),
        blocked_permutation_parent=parent,
        blocked_permutation_shift=(
            BLOCKED_PERMUTATION_SHIFT if parent is not None else None
        ),
    )


__all__ = (
    "CrossfitFoldAudit",
    "CrossfitPredictionRow",
    "ExactCrossfitResult",
    "FamilyDesign",
    "FamilySpec",
    "SmoothCrossfitResult",
    "family_spec",
)
