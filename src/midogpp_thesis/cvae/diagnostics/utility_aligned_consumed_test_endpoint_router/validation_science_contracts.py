"""Shared schemas and immutable results for reconstructive science validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .features import SourceInnerFeatureSurfaceSet


RESPONSE_FIELDS = {
    "schema_version", "outer_target_id", "query_id", "candidate_source",
    "candidate_source_count", "support_partition_hash",
    "evaluation_partition_hash", "prediction_seal_hash",
    "evaluation_row_identity_hash", "evaluation_label_hash",
    "base_endpoint_hash", "tail_endpoint_hash",
    "base_probability_cell_hashes_hash", "tail_probability_cell_hashes_hash",
    "base_ensemble_probability_hash", "tail_ensemble_probability_hash",
    "base_ensemble_prediction_hash", "tail_ensemble_prediction_hash",
    "source_response_hash", "source_endpoint_row_hash",
    "base_component_vector_hashes", "tail_component_vector_hashes",
    "base_bacc", "tail_bacc", "utility_delta", "support_eval_disjoint",
    "predictions_sealed_before_labels", "source_expert_frozen",
    "target_labels_used_for_routing", "utility_semantics", "row_hash",
}
FEATURE_FIELDS = {
    "schema_version", "role", "outer_target_id", "query_id",
    "candidate_source", "candidate_source_count", "support_partition_hash",
    "support_case_count", "seed_pair_count", "seed_row_hashes",
    "feature_mean_by_name", "feature_seed_standard_deviation_by_name",
    "target_local_scalar", "target_local_scalar_name",
    "target_local_scalar_semantics",
    "target_local_scalar_seed_standard_deviation",
    "target_local_scalar_provenance_hash",
    "seed_rows_are_independent_observations", "row_hash",
}
MODEL_FIELDS = {
    "schema_version", "outer_target_id", "model_hashes_by_role",
    "cardinality_transfer_hash", "source_feature_surface_hash",
    "development_response_set_hash", "training_response_count",
    "response_unit", "alpha_tuning_endpoint", "strict_H_q_e_exclusion",
    "same_outer_H_evaluation_labels_used_for_fit",
    "support_labels_used_for_fit", "target_features_used_for_fit",
    "seed_rows_are_independent_observations", "model_hash",
}
POLICY_FIELDS = {
    "schema_version", "target_id", "core_policy_hash", "model_hash",
    "target_feature_hash", "support_partition_lock_hash",
    "target_policy_seal_hash", "routed_candidate_source",
    "executed_routed_source", "selected_action_id", "selected_action_role",
    "exact_B_fallback", "fallback_reason", "source_inner_transfer_authorized",
    "target_static", "case_router_used", "support_labels_used",
    "same_outer_H_evaluation_labels_used", "target_utility_used",
    "may_update_from_terminal_scores", "diagnostic_only", "policy_hash",
}
ACTION_FIELDS = {
    "schema_version", "outer_target_id", "query_id", "action_id",
    "action_role", "effective_action_id", "selected_source", "geometry",
    "topup_counts_by_source", "realized_total_per_class", "core_action_hash",
    "policy_hash", "diagnostic_control", "target_static", "case_router_used",
    "labels_used_to_build", "terminal_scores_used_to_build", "diagnostic_only",
    "action_hash",
}
SCORE_FIELDS = {
    "schema_version", "target_center", "action_id", "action_hash",
    "policy_hash", "support_partition_lock_hash", "evaluation_partition_hash",
    "global_target_prediction_seal_hash", "global_prelabel_seal_hash",
    "evaluation_row_identity_hash", "evaluation_label_hash", "endpoint_hash",
    "evaluation_case_count", "evaluation_row_count",
    "observed_class_0_row_count", "observed_class_1_row_count",
    "balanced_accuracy", "primary_endpoint",
    "same_outer_H_evaluation_labels_opened_after_plan_and_global_seal",
    "terminal_scores_may_update_plan", "inference_unit",
    "technical_seed_cells_are_independent_units",
    "consumed_test_diagnostic_only", "score_hash",
}


@dataclass(frozen=True)
class ScientificPartitionContext:
    support_case_ids_by_center: Mapping[str, tuple[str, ...]]
    support_row_identity_hash_by_center: Mapping[str, str]
    support_feature_hash_by_center: Mapping[str, str]
    evaluation_identity_hash_by_center: Mapping[str, str]
    partition_hash_by_center: Mapping[str, str]
    support_partition_lock_hash: str


@dataclass(frozen=True)
class DevelopmentScienceValidation:
    response_count: int
    response_set_hash: str
    prediction_seal_hash: str
    binding_hash_by_target: Mapping[str, str]
    response_set: object


@dataclass(frozen=True)
class FeatureScienceValidation:
    source_feature_count: int
    target_feature_count: int
    source_surface_set_hash: str
    source_surface_hash_by_target: Mapping[str, str]
    target_feature_hash_by_target: Mapping[str, str]
    source_surface_set: SourceInnerFeatureSurfaceSet


@dataclass(frozen=True)
class PrelabelScienceValidation:
    model_set_hash: str
    policy_set_hash: str
    action_library_hash: str
    policy_hash_by_target: Mapping[str, str]
    action_hash_by_key: Mapping[tuple[str, str], str]
    effective_action_by_key: Mapping[tuple[str, str], str]
    routed_candidate_by_target: Mapping[str, str]
    routed_executed_source_by_target: Mapping[str, str | None]
    selected_action_by_target: Mapping[str, str]
    routed_prediction_by_target: Mapping[str, Mapping[str, float]]
    global_target_prediction_seal_hash: str
    global_prelabel_seal_hash: str


@dataclass(frozen=True)
class TerminalScienceValidation:
    score_count: int
    score_set_hash: str
    contrast_count: int
    inference_hash: str


__all__ = (
    "ACTION_FIELDS", "DevelopmentScienceValidation", "FEATURE_FIELDS",
    "FeatureScienceValidation", "MODEL_FIELDS", "POLICY_FIELDS",
    "PrelabelScienceValidation", "RESPONSE_FIELDS", "SCORE_FIELDS",
    "ScientificPartitionContext", "TerminalScienceValidation",
)
