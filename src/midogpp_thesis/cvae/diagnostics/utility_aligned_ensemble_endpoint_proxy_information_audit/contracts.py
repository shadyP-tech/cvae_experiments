"""Pure contracts for the consumed-data proxy-information audit.

The feature and response boundaries are deliberately separate.  A
``ProxyFeatureRow`` contains label-free, support-only primitives for exactly
one candidate ``(H, q, e)``.  A ``ProxyUtilityRow`` contains the terminal
development response opened only after those feature rows have been sealed.
The 4,536 technical seed cells are provenance, never observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_"
    "ensemble_endpoint_proxy_information_audit.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_"
    "proxy_information_audit_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_utility_aligned_"
    "ensemble_endpoint_proxy_information_audit_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
ROUTING_STATUS = "PROXY_INFORMATION_AUDIT_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_utility_aligned_ensemble_endpoint_proxy_information_"
    "audit_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_utility_aligned_ensemble_endpoint_proxy_information_"
    "audit_validation_manifest_v1"
)
METADATA_PROFILE_ARTIFACT_ID = "midogpp_routing_metadata_profiles_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    METADATA_PROFILE_ARTIFACT_ID,
)

# Center 4 is outside the frozen bank.  Ordering is hash-significant.
CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTER = "4"
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
SEED_PAIRS = tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))
SEED_PAIR_COUNT = len(SEED_PAIRS)

VALIDATION_SPLIT = "val"
FIXED_SUPPORT_CASE_COUNT_PER_CENTER = 2
EXPECTED_TOTAL_CASE_COUNT = 44
SUPPORT_SPLIT_SEED = 20_260_806
SUPPORT_PARTITION_NAMESPACE = (
    "midogpp_utility_aligned_ensemble_endpoint_proxy_information_audit_"
    "support_v1"
)

INNER_CANDIDATE_COUNT = 7
EXPECTED_PROXY_FEATURE_ROW_COUNT = 504
EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT = 504
EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT = 4_536
EXPECTED_QUERY_METRIC_ROW_COUNT_PER_FAMILY = 72
EXPECTED_OUTER_METRIC_ROW_COUNT_PER_FAMILY = 9
EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT = 120

PROXY_FEATURE_SCHEMA = "midogpp_stage90_proxy_information_feature_row_v1"
PROXY_UTILITY_SCHEMA = "midogpp_stage90_proxy_information_utility_row_v1"
PROBABILITY_ROLE_USED = "support_only"
RIDGE_ALPHA = 1.0
CYCLIC_PERMUTATION_SEED = 90_902_026
CYCLIC_PERMUTATION_SHIFT = 1
OUTER_INFERENCE_UNIT_COUNT = len(CENTERS)
STUDENT_T_975_DF8 = 2.306004135204166

EQUAL_UNION_NULL = "equal_union_null"
METADATA_ONLY_CONTROL = "metadata_only_control"
ABSOLUTE_SHIFT_CONTROL = "absolute_shift_control"
RICH_DISTRIBUTIONAL_COMPACT = "rich_distributional_compact"
DIRECTIONAL_ACTION_COMPACT = "directional_action_compact"
HYBRID_COMPACT = "hybrid_compact"
CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL = (
    "cyclic_directional_permutation_control"
)
FAMILY_IDS = (
    EQUAL_UNION_NULL,
    METADATA_ONLY_CONTROL,
    ABSOLUTE_SHIFT_CONTROL,
    RICH_DISTRIBUTIONAL_COMPACT,
    DIRECTIONAL_ACTION_COMPACT,
    HYBRID_COMPACT,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
)
SCREENING_FAMILY_IDS = (
    RICH_DISTRIBUTIONAL_COMPACT,
    DIRECTIONAL_ACTION_COMPACT,
    HYBRID_COMPACT,
)
CONTROL_FAMILY_IDS = (
    METADATA_ONLY_CONTROL,
    ABSOLUTE_SHIFT_CONTROL,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENTROPY_LIMIT = float(np.log(2.0))
_TOLERANCE = 1.0e-12


def candidate_sources(outer_target_id: object, query_id: object) -> tuple[str, ...]:
    """Return the canonical legal seven-source list for ``(H, q)``."""

    outer = _center(outer_target_id, "outer_target_id")
    query = _center(query_id, "query_id")
    if outer == query:
        raise ProtocolError("Proxy-audit H and q must be distinct.")
    return tuple(center for center in CENTERS if center not in {outer, query})


@dataclass(frozen=True)
class ProxyFeatureRow:
    """One label-free candidate row after exact-nine support aggregation."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    candidate_source_count: int
    support_partition_hash: str
    support_case_count: int
    support_row_count: int
    seed_pair_count: int
    seed_feature_row_hashes: tuple[str, ...]
    base_support_vector_hashes: tuple[str, ...]
    tail_support_vector_hashes: tuple[str, ...]
    metadata_similarity: float
    absolute_ensemble_shift: float
    reconstruction_mean_within_query_z: float
    kl_mean_within_query_z: float
    log_distribution_mmd_within_query_z: float
    signed_margin_projection: float
    threshold_flip_rate: float
    mean_entropy_change: float
    development_prediction_seal_hash: str
    probability_role_used: str = PROBABILITY_ROLE_USED
    labels_used: bool = False
    evaluation_probabilities_used_as_features: bool = False
    technical_seed_rows_are_independent_observations: bool = False
    proxy_feature_row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _center(self.outer_target_id, "outer_target_id")
        query = _center(self.query_id, "query_id")
        source = _center(self.candidate_source, "candidate_source")
        if outer == query or source in {outer, query}:
            raise ProtocolError("Proxy feature row requires distinct H/q/e domains.")
        if type(self.candidate_source_count) is not int or self.candidate_source_count != 7:
            raise ProtocolError("Proxy feature row requires seven legal candidates.")
        if type(self.support_case_count) is not int or self.support_case_count != 2:
            raise ProtocolError("Proxy feature row requires fixed support-2.")
        if type(self.support_row_count) is not int or self.support_row_count <= 0:
            raise ProtocolError("Proxy feature support row count must be positive.")
        if type(self.seed_pair_count) is not int or self.seed_pair_count != 9:
            raise ProtocolError("Proxy feature row requires exact-nine seed provenance.")
        support_hash = _hash(self.support_partition_hash, "support_partition_hash")
        seal_hash = _hash(
            self.development_prediction_seal_hash,
            "development_prediction_seal_hash",
        )
        seed_hashes = _exact_nine_hashes(
            self.seed_feature_row_hashes, "seed_feature_row_hashes"
        )
        base_hashes = _exact_nine_hashes(
            self.base_support_vector_hashes, "base_support_vector_hashes"
        )
        tail_hashes = _exact_nine_hashes(
            self.tail_support_vector_hashes, "tail_support_vector_hashes"
        )
        values = {
            "metadata_similarity": _bounded(
                self.metadata_similarity, "metadata_similarity", 0.0, 1.0
            ),
            "absolute_ensemble_shift": _bounded(
                self.absolute_ensemble_shift,
                "absolute_ensemble_shift",
                0.0,
                1.0,
            ),
            "reconstruction_mean_within_query_z": _finite(
                self.reconstruction_mean_within_query_z,
                "reconstruction_mean_within_query_z",
            ),
            "kl_mean_within_query_z": _finite(
                self.kl_mean_within_query_z, "kl_mean_within_query_z"
            ),
            "log_distribution_mmd_within_query_z": _finite(
                self.log_distribution_mmd_within_query_z,
                "log_distribution_mmd_within_query_z",
            ),
            "signed_margin_projection": _bounded(
                self.signed_margin_projection,
                "signed_margin_projection",
                -1.0,
                1.0,
            ),
            "threshold_flip_rate": _bounded(
                self.threshold_flip_rate, "threshold_flip_rate", 0.0, 1.0
            ),
            "mean_entropy_change": _bounded(
                self.mean_entropy_change,
                "mean_entropy_change",
                -_ENTROPY_LIMIT,
                _ENTROPY_LIMIT,
            ),
        }
        if (
            self.probability_role_used != PROBABILITY_ROLE_USED
            or self.labels_used is not False
            or self.evaluation_probabilities_used_as_features is not False
            or self.technical_seed_rows_are_independent_observations is not False
        ):
            raise ProtocolError("Proxy feature row crossed the label/probability boundary.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "support_partition_hash", support_hash)
        object.__setattr__(self, "development_prediction_seal_hash", seal_hash)
        object.__setattr__(self, "seed_feature_row_hashes", seed_hashes)
        object.__setattr__(self, "base_support_vector_hashes", base_hashes)
        object.__setattr__(self, "tail_support_vector_hashes", tail_hashes)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "proxy_feature_row_hash", canonical_sha256(self._unhashed_payload())
        )

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": PROXY_FEATURE_SCHEMA,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "support_case_count": self.support_case_count,
            "support_row_count": self.support_row_count,
            "seed_pair_count": self.seed_pair_count,
            "seed_feature_row_hashes": list(self.seed_feature_row_hashes),
            "base_support_vector_hashes": list(self.base_support_vector_hashes),
            "tail_support_vector_hashes": list(self.tail_support_vector_hashes),
            "metadata_similarity": self.metadata_similarity,
            "absolute_ensemble_shift": self.absolute_ensemble_shift,
            "reconstruction_mean_within_query_z": (
                self.reconstruction_mean_within_query_z
            ),
            "kl_mean_within_query_z": self.kl_mean_within_query_z,
            "log_distribution_mmd_within_query_z": (
                self.log_distribution_mmd_within_query_z
            ),
            "signed_margin_projection": self.signed_margin_projection,
            "threshold_flip_rate": self.threshold_flip_rate,
            "mean_entropy_change": self.mean_entropy_change,
            "development_prediction_seal_hash": (
                self.development_prediction_seal_hash
            ),
            "probability_role_used": self.probability_role_used,
            "labels_used": self.labels_used,
            "evaluation_probabilities_used_as_features": (
                self.evaluation_probabilities_used_as_features
            ),
            "technical_seed_rows_are_independent_observations": (
                self.technical_seed_rows_are_independent_observations
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed_payload(),
            "proxy_feature_row_hash": self.proxy_feature_row_hash,
        }


@dataclass(frozen=True)
class ProxyFeatureSurface:
    rows: tuple[ProxyFeatureRow, ...]
    row_keys: tuple[tuple[str, str, str], ...]
    surface_hash: str


@dataclass(frozen=True)
class ProxyUtilityRow:
    """One scored exact-nine endpoint response, never a seed response."""

    outer_target_id: str
    query_id: str
    candidate_source: str
    candidate_source_count: int
    support_partition_hash: str
    utility_delta: float
    response_hash: str
    support_eval_disjoint: bool = True
    predictions_sealed_before_labels: bool = True
    source_expert_frozen: bool = True
    target_labels_used_for_routing: bool = False

    def __post_init__(self) -> None:
        outer = _center(self.outer_target_id, "outer_target_id")
        query = _center(self.query_id, "query_id")
        source = _center(self.candidate_source, "candidate_source")
        if outer == query or source in {outer, query}:
            raise ProtocolError("Proxy utility row requires distinct H/q/e domains.")
        if type(self.candidate_source_count) is not int or self.candidate_source_count != 7:
            raise ProtocolError("Proxy utility response cardinality drifted.")
        support_hash = _hash(self.support_partition_hash, "support_partition_hash")
        response_hash = _hash(self.response_hash, "response_hash")
        utility = _bounded(self.utility_delta, "utility_delta", -1.0, 1.0)
        if (
            self.support_eval_disjoint is not True
            or self.predictions_sealed_before_labels is not True
            or self.source_expert_frozen is not True
            or self.target_labels_used_for_routing is not False
        ):
            raise ProtocolError("Proxy utility response violates the scoring boundary.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "support_partition_hash", support_hash)
        object.__setattr__(self, "response_hash", response_hash)
        object.__setattr__(self, "utility_delta", utility)

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": PROXY_UTILITY_SCHEMA,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "candidate_source_count": self.candidate_source_count,
            "support_partition_hash": self.support_partition_hash,
            "utility_delta": self.utility_delta,
            "response_hash": self.response_hash,
            "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
            "technical_seed_rows_are_independent_observations": False,
            "support_eval_disjoint": self.support_eval_disjoint,
            "predictions_sealed_before_labels": self.predictions_sealed_before_labels,
            "source_expert_frozen": self.source_expert_frozen,
            "target_labels_used_for_routing": self.target_labels_used_for_routing,
        }


@dataclass(frozen=True)
class ProxyUtilitySurface:
    rows: tuple[ProxyUtilityRow, ...]
    row_keys: tuple[tuple[str, str, str], ...]
    surface_hash: str


@dataclass(frozen=True)
class ProxyFamilySpec:
    family_id: str
    predictor_names: tuple[str, ...]
    family_role: str
    cyclic_shift: int | None = None

    def __post_init__(self) -> None:
        if self.family_id not in FAMILY_IDS or len(self.predictor_names) > 3:
            raise ProtocolError("Proxy family identity/capacity drifted.")
        expected_role = (
            "screening_candidate"
            if self.family_id in SCREENING_FAMILY_IDS
            else "negative_or_baseline_control"
        )
        if self.family_role != expected_role:
            raise ProtocolError("Proxy family role drifted.")
        expected_shift = (
            CYCLIC_PERMUTATION_SHIFT
            if self.family_id == CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL
            else None
        )
        if self.cyclic_shift != expected_shift:
            raise ProtocolError("Proxy family cyclic-shift contract drifted.")

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
            raise ProtocolError("Proxy family design shape drifted.")
        if not np.isfinite(matrix).all():
            raise ProtocolError("Proxy family design contains non-finite values.")
        if len(self.source_row_hashes) != len(self.row_keys):
            raise ProtocolError("Proxy family design provenance drifted.")
        matrix.setflags(write=False)
        object.__setattr__(self, "values", matrix)


@dataclass(frozen=True)
class CrossfitFoldAudit:
    family_id: str
    predicted_row_key: tuple[str, str, str]
    excluded_domain_ids: tuple[str, ...]
    training_row_keys: tuple[tuple[str, str, str], ...]
    training_outer_target_ids: tuple[str, ...]
    training_query_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]
    training_row_count: int
    ridge_alpha: float
    learned_scaling_fit_on_training_fold_only: bool
    precomputed_candidate_list_transforms_are_label_free: bool
    fold_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_proxy_information_fold_audit_v1",
            "family_id": self.family_id,
            "predicted_row_key": list(self.predicted_row_key),
            "excluded_domain_ids": list(self.excluded_domain_ids),
            "training_row_keys": [list(key) for key in self.training_row_keys],
            "training_outer_target_ids": list(self.training_outer_target_ids),
            "training_query_ids": list(self.training_query_ids),
            "training_source_ids": list(self.training_source_ids),
            "training_row_count": self.training_row_count,
            "ridge_alpha": self.ridge_alpha,
            "ridge_cluster_unit": "outer_target_query",
            "hyperparameter_selection": "none_fixed_predeclared",
            "learned_scaling_fit_on_training_fold_only": (
                self.learned_scaling_fit_on_training_fold_only
            ),
            "precomputed_candidate_list_transforms_are_label_free": (
                self.precomputed_candidate_list_transforms_are_label_free
            ),
            "strict_H_q_e_exclusion_from_all_training_roles": True,
            "fold_hash": self.fold_hash,
        }


@dataclass(frozen=True)
class CrossfitPredictionRow:
    family_id: str
    outer_target_id: str
    query_id: str
    candidate_source: str
    predicted_utility_delta: float
    observed_utility_delta: float
    predictor_count: int
    training_row_count: int
    fold_hash: str
    row_hash: str

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target_id, self.query_id, self.candidate_source

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_proxy_information_crossfit_prediction_v1",
            "family_id": self.family_id,
            "outer_target_id": self.outer_target_id,
            "query_id": self.query_id,
            "candidate_source": self.candidate_source,
            "predicted_utility_delta": self.predicted_utility_delta,
            "observed_utility_delta": self.observed_utility_delta,
            "predictor_count": self.predictor_count,
            "training_row_count": self.training_row_count,
            "fold_hash": self.fold_hash,
            "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
            "technical_seed_rows_are_independent_observations": False,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class CrossfitFoldLock:
    family_ids: tuple[str, ...]
    feature_surface_hash: str
    utility_surface_hash: str
    fold_count: int
    ridge_alpha: float
    ordered_fold_hashes: tuple[str, ...]
    lock_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_proxy_information_crossfit_fold_lock_v1",
            "family_ids": list(self.family_ids),
            "feature_surface_hash": self.feature_surface_hash,
            "utility_surface_hash": self.utility_surface_hash,
            "fold_count": self.fold_count,
            "ridge_alpha": self.ridge_alpha,
            "ridge_cluster_unit": "outer_target_query",
            "hyperparameter_selection": "none_fixed_predeclared",
            "ordered_fold_hashes": list(self.ordered_fold_hashes),
            "strict_H_q_e_exclusion_from_all_training_roles": True,
            "scaling_fit_on_training_fold_only": True,
            "response_row_count": EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
            "descriptive_seed_row_count": EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT,
            "descriptive_seed_rows_may_feed_model": False,
            "crossfit_fold_lock_hash": self.lock_hash,
        }


@dataclass(frozen=True)
class ProxyCrossfitResult:
    predictions: tuple[CrossfitPredictionRow, ...]
    fold_audits: tuple[CrossfitFoldAudit, ...]
    fold_lock: CrossfitFoldLock
    feature_surface_hash: str
    utility_surface_hash: str
    result_hash: str

    @property
    def crossfit_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.predictions)


@dataclass(frozen=True)
class QueryMetricRow:
    family_id: str
    outer_target_id: str
    query_id: str
    candidate_count: int
    exact_top1: float
    tie_aware_top1: float
    spearman: float
    spearman_defined: bool
    normalized_oracle_regret: float
    pairwise_accuracy: float
    calibration_intercept: float
    calibration_slope: float
    calibration_slope_defined: bool
    rmse: float
    selected_source: str
    oracle_sources: tuple[str, ...]
    predicted_top_sources: tuple[str, ...]
    row_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_proxy_information_query_metrics_v1",
            **{name: getattr(self, name) for name in (
                "family_id", "outer_target_id", "query_id", "candidate_count",
                "exact_top1", "tie_aware_top1", "spearman", "spearman_defined",
                "normalized_oracle_regret", "pairwise_accuracy",
                "calibration_intercept", "calibration_slope",
                "calibration_slope_defined", "rmse", "selected_source",
            )},
            "oracle_sources": list(self.oracle_sources),
            "predicted_top_sources": list(self.predicted_top_sources),
            "inference_unit": "descriptive_query_nested_within_outer_H",
            "technical_seed_rows_are_independent_observations": False,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class OuterMetricRow:
    family_id: str
    outer_target_id: str
    query_count: int
    mean_exact_top1: float
    mean_tie_aware_top1: float
    mean_spearman: float
    defined_spearman_query_count: int
    mean_normalized_oracle_regret: float
    mean_pairwise_accuracy: float
    calibration_intercept: float
    calibration_slope: float
    calibration_slope_defined: bool
    rmse: float
    row_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_proxy_information_outer_metrics_v1",
            **{name: getattr(self, name) for name in (
                "family_id", "outer_target_id", "query_count",
                "mean_exact_top1", "mean_tie_aware_top1", "mean_spearman",
                "defined_spearman_query_count", "mean_normalized_oracle_regret",
                "mean_pairwise_accuracy", "calibration_intercept",
                "calibration_slope", "calibration_slope_defined", "rmse",
            )},
            "inference_unit": "outer_target_center",
            "query_rows_are_nested_descriptive_units": True,
            "technical_seed_rows_are_independent_observations": False,
            "row_hash": self.row_hash,
        }


@dataclass(frozen=True)
class FamilySummaryRow:
    family_id: str
    family_role: str
    predictor_count: int
    outer_count: int
    mean_exact_top1: float
    mean_tie_aware_top1: float
    mean_spearman: float
    spearman_ci95_lower: float
    spearman_ci95_upper: float
    mean_normalized_oracle_regret: float
    normalized_oracle_regret_ci95_lower: float
    normalized_oracle_regret_ci95_upper: float
    mean_pairwise_accuracy: float
    pairwise_accuracy_ci95_lower: float
    pairwise_accuracy_ci95_upper: float
    mean_calibration_intercept: float
    mean_calibration_slope: float
    mean_rmse: float
    spearman_gate_passed: bool
    pairwise_gate_passed: bool
    regret_gate_passed: bool
    beats_all_regret_controls: bool
    screening_eligible: bool
    screening_passed: bool
    row_hash: str

    def to_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": "midogpp_stage90_proxy_information_family_summary_v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "row_hash"},
            "inference_unit": "outer_target_center",
            "query_rows_are_nested_descriptive_units": True,
            "candidate_and_seed_rows_are_not_inference_units": True,
            "confidence_interval": "two_sided_student_t_95_percent_n9",
            "screening_gate_may_authorize_policy": False,
            "row_hash": self.row_hash,
        }
        return payload


@dataclass(frozen=True)
class ProxyInformationAuditResult:
    crossfit: ProxyCrossfitResult
    query_metrics: tuple[QueryMetricRow, ...]
    outer_metrics: tuple[OuterMetricRow, ...]
    family_summaries: tuple[FamilySummaryRow, ...]
    proxy_information_gate_passed: bool
    informative_family_ids: tuple[str, ...]
    result_hash: str

    @property
    def fold_lock(self) -> CrossfitFoldLock:
        return self.crossfit.fold_lock

    @property
    def crossfit_table_rows(self) -> tuple[dict[str, object], ...]:
        return self.crossfit.crossfit_table_rows

    @property
    def query_metric_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.query_metrics)

    @property
    def outer_metric_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.outer_metrics)

    @property
    def family_summary_table_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(row.to_payload() for row in self.family_summaries)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_proxy_information_audit_result_v1",
            "experiment_id": EXPERIMENT_ID,
            "feature_surface_hash": self.crossfit.feature_surface_hash,
            "utility_surface_hash": self.crossfit.utility_surface_hash,
            "crossfit_result_hash": self.crossfit.result_hash,
            "crossfit_fold_lock_hash": self.fold_lock.lock_hash,
            "query_metric_row_count": len(self.query_metrics),
            "outer_metric_row_count": len(self.outer_metrics),
            "family_summary_row_count": len(self.family_summaries),
            "family_summary_row_hashes": [
                row.row_hash for row in self.family_summaries
            ],
            "proxy_information_gate_passed": self.proxy_information_gate_passed,
            "informative_family_ids": list(self.informative_family_ids),
            "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
            "response_row_count": EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
            "technical_seed_row_count": EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT,
            "technical_seed_rows_are_independent_observations": False,
            "outer_target_centers_are_inference_units": True,
            "consumed_validation_data": True,
            "diagnostic_only": True,
            "screening_gate_may_authorize_policy": False,
            "routing_quality_claimed": False,
            "policy_update_authorized": False,
            "promotion_eligible": False,
            "audit_result_hash": self.result_hash,
        }


def _center(value: object, name: str) -> str:
    if type(value) is not str or value not in CENTERS:
        raise ProtocolError(f"{name} must be one of the frozen MIDOG++ centers.")
    return value


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a canonical lowercase SHA-256.")
    return value


def _exact_nine_hashes(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ProtocolError(f"{name} must be an exact-nine sequence.")
    result = tuple(_hash(value, name) for value in values)
    if len(result) != 9 or len(set(result)) != 9:
        raise ProtocolError(f"{name} must contain nine unique hashes.")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ProtocolError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(result):
        raise ProtocolError(f"{name} must be a finite real number.")
    return result


def _bounded(value: object, name: str, lower: float, upper: float) -> float:
    result = _finite(value, name)
    if result < lower - _TOLERANCE or result > upper + _TOLERANCE:
        raise ProtocolError(f"{name} is outside its scientific range.")
    return min(max(result, lower), upper)


def family_specs_payload(specs: Mapping[str, ProxyFamilySpec]) -> dict[str, object]:
    values = {key: value for key, value in specs.items()}
    if tuple(values) != FAMILY_IDS:
        raise ProtocolError("Proxy family registry order/coverage drifted.")
    return {
        "schema_version": "midogpp_stage90_proxy_information_family_registry_v1",
        "families": [values[family].to_payload() for family in FAMILY_IDS],
        "ridge_alpha": RIDGE_ALPHA,
        "hyperparameter_selection": "none_fixed_predeclared",
        "maximum_predictors_per_family": 3,
        "cyclic_permutation_seed": CYCLIC_PERMUTATION_SEED,
        "cyclic_permutation_shift": CYCLIC_PERMUTATION_SHIFT,
    }


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "CrossfitFoldAudit",
    "CrossfitFoldLock",
    "CrossfitPredictionRow",
    "FamilySummaryRow",
    "OuterMetricRow",
    "ProxyCrossfitResult",
    "ProxyFamilyDesign",
    "ProxyFamilySpec",
    "ProxyFeatureRow",
    "ProxyFeatureSurface",
    "ProxyInformationAuditResult",
    "ProxyUtilityRow",
    "ProxyUtilitySurface",
    "QueryMetricRow",
    "candidate_sources",
    "family_specs_payload",
)
