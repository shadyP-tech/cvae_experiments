"""Family-design, fitted-fold, and crossfit-result DTOs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .constants import (
    CENTERS,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
    CYCLIC_PERMUTATION_SHIFT,
    EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT,
    FAMILY_IDS,
    PRIMARY_RESPONSE_NAME,
    RESPONSE_NAMES,
    RIDGE_ALPHA,
    SCREENING_FAMILY_IDS,
    expected_strict_training_row_count,
)
from .contract_validation import finite, sha256


@dataclass(frozen=True)
class ProxyFamilySpec:
    family_id: str
    predictor_names: tuple[str, ...]
    family_role: str
    cyclic_shift: int | None = None

    def __post_init__(self) -> None:
        if self.family_id not in FAMILY_IDS or len(self.predictor_names) > 3:
            raise ProtocolError("Family identity or fixed capacity drifted.")
        expected_role = (
            "screening_candidate"
            if self.family_id in SCREENING_FAMILY_IDS
            else "control"
        )
        if self.family_role != expected_role:
            raise ProtocolError("Family role drifted.")
        expected_shift = (
            CYCLIC_PERMUTATION_SHIFT
            if self.family_id == CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL
            else None
        )
        if self.cyclic_shift != expected_shift:
            raise ProtocolError("Family cyclic permutation contract drifted.")

    @property
    def predictor_count(self) -> int:
        return len(self.predictor_names)

    def to_payload(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "predictor_names": list(self.predictor_names),
            "predictor_count": self.predictor_count,
            "family_role": self.family_role,
            "cyclic_shift": self.cyclic_shift,
            "maximum_predictors": 3,
            "hyperparameter_selection": "none_fixed_predeclared",
        }


@dataclass(frozen=True)
class ProxyFamilyDesign:
    spec: ProxyFamilySpec
    row_keys: tuple[tuple[str, str, str], ...]
    values: np.ndarray
    source_row_hashes: tuple[str, ...]
    design_hash: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values, dtype=np.float64).copy()
        if matrix.shape != (len(self.row_keys), self.spec.predictor_count):
            raise ProtocolError("Family design geometry drifted.")
        if not np.isfinite(matrix).all():
            raise ProtocolError("Family design contains non-finite values.")
        if len(self.source_row_hashes) != len(self.row_keys):
            raise ProtocolError("Family design provenance drifted.")
        matrix.setflags(write=False)
        object.__setattr__(self, "values", matrix)


@dataclass(frozen=True)
class CrossfitFoldAudit:
    family_id: str
    response_name: str
    predicted_row_key: tuple[str, str, str]
    excluded_domain_ids: tuple[str, ...]
    training_row_keys: tuple[tuple[str, str, str], ...]
    training_row_count: int
    ridge_alpha: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    family_design_hash: str
    fold_hash: str

    def __post_init__(self) -> None:
        if self.family_id not in FAMILY_IDS or self.response_name not in RESPONSE_NAMES:
            raise ProtocolError("Crossfit fold family/response identity drifted.")
        if (
            len(self.predicted_row_key) != 3
            or len(set(self.predicted_row_key)) != 3
            or any(value not in CENTERS for value in self.predicted_row_key)
            or set(self.excluded_domain_ids) != set(self.predicted_row_key)
        ):
            raise ProtocolError("Crossfit fold predicted H/q/e identity drifted.")
        if (
            type(self.training_row_count) is not int
            or self.training_row_count
            != expected_strict_training_row_count(CENTERS)
            or len(self.training_row_keys) != self.training_row_count
            or len(set(self.training_row_keys)) != self.training_row_count
            or any(
                not set(self.predicted_row_key).isdisjoint(key)
                for key in self.training_row_keys
            )
        ):
            raise ProtocolError("Crossfit fold strict training geometry drifted.")
        if not np.isclose(float(self.ridge_alpha), RIDGE_ALPHA, atol=0.0):
            raise ProtocolError("Crossfit fold ridge alpha drifted.")
        mean = tuple(finite(value, "feature_mean") for value in self.feature_mean)
        scale = tuple(finite(value, "feature_scale") for value in self.feature_scale)
        coefficients = tuple(
            finite(value, "coefficients") for value in self.coefficients
        )
        intercept = finite(self.intercept, "intercept")
        if (
            len(mean) != len(scale)
            or len(mean) != len(coefficients)
            or len(mean) > 3
            or any(value <= 0.0 for value in scale)
        ):
            raise ProtocolError("Crossfit fold fitted-parameter geometry drifted.")
        design_hash = sha256(self.family_design_hash, "family_design_hash")
        fold_hash = sha256(self.fold_hash, "fold_hash")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "family_design_hash", design_hash)
        object.__setattr__(self, "fold_hash", fold_hash)
        if canonical_sha256(self._unhashed_payload()) != fold_hash:
            raise ProtocolError("Crossfit fold fitted provenance hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_case_aware_crossfit_fold_v1",
            "family_id": self.family_id,
            "response_name": self.response_name,
            "predicted_row_key": list(self.predicted_row_key),
            "excluded_domain_ids": list(self.excluded_domain_ids),
            "training_row_keys": [list(key) for key in self.training_row_keys],
            "training_row_count": self.training_row_count,
            "ridge_alpha": self.ridge_alpha,
            "ridge_cluster_unit": "outer_target_query",
            "strict_H_q_e_exclusion_from_all_training_roles": True,
            "scaling_fit_on_training_fold_only": True,
            "hyperparameter_selection": "none_fixed_predeclared",
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "family_design_hash": self.family_design_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "fold_hash": self.fold_hash}


@dataclass(frozen=True)
class CrossfitPredictionRow:
    family_id: str
    response_name: str
    outer_target_id: str
    query_id: str
    candidate_source: str
    predicted_delta: float
    observed_delta: float
    predictor_count: int
    training_row_count: int
    fold_hash: str
    row_hash: str

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_case_aware_crossfit_prediction_v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "response_is_primary": self.response_name == PRIMARY_RESPONSE_NAME,
            "smooth_response_is_diagnostic_only": (
                self.response_name != PRIMARY_RESPONSE_NAME
            ),
            "technical_seed_rows_are_independent_observations": False,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class CaseAwareCrossfitResult:
    predictions: tuple[CrossfitPredictionRow, ...]
    fold_audits: tuple[CrossfitFoldAudit, ...]
    family_ids: tuple[str, ...]
    response_names: tuple[str, ...]
    feature_surface_hash: str
    response_surface_hash: str
    result_hash: str

    @property
    def table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.predictions)

    @property
    def fold_audit_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.fold_audits)

    @property
    def fold_lock_hash(self) -> str:
        return canonical_sha256(self._fold_lock_unhashed_payload())

    def _fold_lock_unhashed_payload(self) -> dict[str, object]:
        training_counts = sorted({row.training_row_count for row in self.fold_audits})
        if training_counts != [EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT]:
            raise ProtocolError("Crossfit fold-lock training geometry drifted.")
        return {
            "schema_version": "midogpp_stage90_case_aware_crossfit_fold_lock_v1",
            "family_ids": list(self.family_ids),
            "response_names": list(self.response_names),
            "feature_surface_hash": self.feature_surface_hash,
            "response_surface_hash": self.response_surface_hash,
            "crossfit_result_hash": self.result_hash,
            "fold_count": len(self.fold_audits),
            "ordered_fold_hashes": [row.fold_hash for row in self.fold_audits],
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_cluster_unit": "outer_target_query",
            "strict_H_q_e_exclusion_from_all_training_roles": True,
            "training_row_count_per_fold": training_counts[0],
            "hyperparameter_selection": "none_fixed_predeclared",
            "exact_response_is_primary": True,
            "smooth_response_is_diagnostic_only": True,
            "technical_seed_rows_are_independent_observations": False,
        }

    def fold_lock_payload(self) -> dict[str, object]:
        unhashed = self._fold_lock_unhashed_payload()
        return {**unhashed, "crossfit_fold_lock_hash": canonical_sha256(unhashed)}


__all__ = (
    "CaseAwareCrossfitResult",
    "CrossfitFoldAudit",
    "CrossfitPredictionRow",
    "ProxyFamilyDesign",
    "ProxyFamilySpec",
)
