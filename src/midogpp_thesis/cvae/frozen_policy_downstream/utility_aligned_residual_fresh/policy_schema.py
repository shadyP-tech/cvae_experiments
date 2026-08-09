"""Versioned Stage-60 policy and action-library schemas admitted by Stage 70."""

ACTION_LIBRARY_SCHEMA = "midogpp_utility_aligned_residual_action_library_v2"
POLICY_LOCK_SCHEMA = "midogpp_utility_aligned_residual_policy_lock_v2"
TARGET_POLICY_LOCK_SCHEMA = "midogpp_utility_aligned_target_policy_lock_v2"
POLICY_EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_aligned_residual_policy_lock.v1"
)
ENSEMBLE_POLICY_FAMILY = (
    "utility_aligned_candidate_level_exact_nine_ensemble_m0_m1_v1"
)
ENSEMBLE_TARGET_POLICY_SCHEMA = "midogpp_utility_aligned_ensemble_policy_v1"
ENSEMBLE_ENDPOINT_BINDING_KEYS = frozenset(
    {
        "ensemble_endpoint_id", "ensemble_endpoint_lock_hash",
        "ensemble_endpoint_table_sha256", "ensemble_endpoint_response_hash",
        "ensemble_endpoint_row_hashes_hash",
        "ensemble_probability_cell_surface_hash",
        "ensemble_prediction_arrays_sha256", "ensemble_seed_pair_count",
        "ensemble_threshold", "ensemble_aggregation_semantics",
        "ensemble_response_semantics", "ensemble_endpoint_role",
    }
)
SOURCE_INNER_ACTION_SHIFT_BINDING_KEYS = frozenset(
    {
        "source_inner_action_shift_lock_hash",
        "source_inner_action_shift_table_sha256",
        "source_inner_action_shift_row_hashes_hash",
        "source_inner_action_shift_row_count",
        "source_inner_action_shift_scalar_name",
        "source_inner_action_shift_row_semantics",
        "source_inner_action_shift_aggregate_semantics",
        "source_inner_action_shift_descriptive_seed_values_may_feed_model",
    }
)
MODEL_CAPACITY_BINDING_KEYS = frozenset(
    {"model_capacity_reports_by_target", "model_capacity_reports_hash"}
)
TARGET_ACTION_SHIFT_BINDING_KEYS = frozenset(
    {
        "target_local_scalar_name", "target_local_scalar_semantics",
        "target_local_scalar_row_semantics",
        "target_support_action_shift_lock_hash",
        "target_support_action_shift_table_sha256",
        "target_support_action_shift_row_hashes_hash",
        "target_support_action_shift_row_count",
        "target_support_action_shift_case_ensemble_group_count",
        "target_support_action_shift_descriptive_seed_values_may_feed_model",
    }
)

ACTION_KEYS = frozenset(
    {
        "target_center", "action_id", "action_role", "selected_source",
        "abstained_to_base", "fallback_reason", "source_order",
        "counts_per_class", "total_per_class", "topup_action_hash",
        "decision_hash", "target_labels_used", "support_labels_used",
    }
)
LIBRARY_KEYS = frozenset(
    {
        "schema_version", "experiment_id", "output_artifact_id",
        "exact_tail_surface_lock_hash", "equal_union_policy_lock_hash",
        "metadata_profile_sha256", "development_reservation_artifact_id",
        "development_reservation_hash", "development_case_manifest_hash",
        "development_support_case_ids_by_query",
        "development_evaluation_case_ids_by_query",
        "development_target_evaluation_case_ids_by_target",
        "development_partition_hashes_by_query",
        "target_support_surface_artifact_id",
        "target_support_surface_hash",
        "target_support_parent_reservation_artifact_id",
        "target_support_parent_reservation_hash", "target_reservation_artifact_id",
        "target_reservation_hash", "target_support_case_ids_by_target",
        "target_evaluation_case_ids_by_target", "target_evaluation_binding_hash",
        "feature_surface_hash", "feature_schema_hash", "model_lock_hash",
        "global_ablation_lock_hash", "cardinality_transfer_lock_hash",
        "target_policy_lock_hash", "action_ids", "actions", "action_count",
        "action_library_hash",
    }
) | ENSEMBLE_ENDPOINT_BINDING_KEYS | SOURCE_INNER_ACTION_SHIFT_BINDING_KEYS | MODEL_CAPACITY_BINDING_KEYS | TARGET_ACTION_SHIFT_BINDING_KEYS
POLICY_KEYS = frozenset(
    {
        "schema_version", "experiment_id", "output_artifact_id",
        "exact_tail_surface_lock_hash", "equal_union_policy_lock_hash",
        "metadata_profile_sha256", "development_reservation_artifact_id",
        "development_reservation_hash", "development_case_manifest_hash",
        "development_support_case_ids_by_query",
        "development_evaluation_case_ids_by_query",
        "development_target_evaluation_case_ids_by_target",
        "development_partition_hashes_by_query",
        "target_support_surface_artifact_id",
        "target_support_surface_hash",
        "target_support_parent_reservation_artifact_id",
        "target_support_parent_reservation_hash", "target_reservation_artifact_id",
        "target_reservation_hash", "target_support_case_ids_by_target",
        "target_evaluation_case_ids_by_target", "target_evaluation_binding_hash",
        "feature_surface_hash", "feature_schema_hash", "model_lock_hash",
        "global_ablation_lock_hash", "cardinality_transfer_lock_hash",
        "target_policy_lock_hash", "action_library_hash", "candidate_centers",
        "primary_contrasts", "permutation_contrast",
        "success_requires_positive_one_sided_lcb", "policy_family",
        "fallback_policy",
        "outer_target_excluded_from_fit", "target_support_labels_used",
        "target_evaluation_labels_used", "seed_selection_performed",
        "minimum_independent_support_cases_per_target", "support_bootstrap_count",
        "policy_lock_hash",
    }
) | ENSEMBLE_ENDPOINT_BINDING_KEYS | SOURCE_INNER_ACTION_SHIFT_BINDING_KEYS | MODEL_CAPACITY_BINDING_KEYS | TARGET_ACTION_SHIFT_BINDING_KEYS
SHARED_BINDING_KEYS = tuple(
    key
    for key in POLICY_KEYS.intersection(LIBRARY_KEYS)
    if key not in {"schema_version", "action_library_hash"}
)
TARGET_POLICY_KEYS = frozenset(
    {
        "schema_version", "experiment_id", "output_artifact_id",
        "exact_tail_surface_lock_hash", "development_case_manifest_hash",
        "development_support_case_ids_by_query",
        "development_evaluation_case_ids_by_query",
        "development_target_evaluation_case_ids_by_target",
        "development_partition_hashes_by_query",
        "target_support_surface_artifact_id",
        "target_support_surface_hash",
        "target_support_parent_reservation_artifact_id",
        "target_support_parent_reservation_hash", "target_reservation_artifact_id",
        "target_reservation_hash", "target_support_case_ids_by_target",
        "target_evaluation_case_ids_by_target", "target_evaluation_binding_hash",
        "metadata_profile_sha256", "target_feature_locks", "policies",
        "target_policy_lock_hash",
    }
) | ENSEMBLE_ENDPOINT_BINDING_KEYS | SOURCE_INNER_ACTION_SHIFT_BINDING_KEYS | TARGET_ACTION_SHIFT_BINDING_KEYS
TARGET_FEATURE_LOCK_KEYS = frozenset(
    {
        "target_id", "case_bootstrap_plan", "target_feature_surface_hash",
        "target_feature_row_count", "bootstrap_surface_hashes",
        "bootstrap_surface_hashes_hash", "candidate_sources", "training_seeds",
        "generation_seeds", "case_level_resampling", "labels_used",
        "support_case_count",
        "target_feature_lock_hash",
    }
)
TARGET_POLICY_SHARED_KEYS = (
    "experiment_id", "output_artifact_id", "exact_tail_surface_lock_hash",
    "ensemble_endpoint_id", "ensemble_endpoint_lock_hash",
    "ensemble_endpoint_table_sha256", "ensemble_endpoint_response_hash",
    "ensemble_endpoint_row_hashes_hash", "ensemble_probability_cell_surface_hash",
    "ensemble_prediction_arrays_sha256", "ensemble_seed_pair_count",
    "ensemble_threshold", "ensemble_aggregation_semantics",
    "ensemble_response_semantics", "ensemble_endpoint_role",
    "source_inner_action_shift_lock_hash",
    "source_inner_action_shift_table_sha256",
    "source_inner_action_shift_row_hashes_hash",
    "source_inner_action_shift_row_count",
    "source_inner_action_shift_scalar_name",
    "source_inner_action_shift_row_semantics",
    "source_inner_action_shift_aggregate_semantics",
    "source_inner_action_shift_descriptive_seed_values_may_feed_model",
    "target_local_scalar_name", "target_local_scalar_semantics",
    "target_local_scalar_row_semantics",
    "target_support_action_shift_lock_hash",
    "target_support_action_shift_table_sha256",
    "target_support_action_shift_row_hashes_hash",
    "target_support_action_shift_row_count",
    "target_support_action_shift_case_ensemble_group_count",
    "target_support_action_shift_descriptive_seed_values_may_feed_model",
    "development_case_manifest_hash",
    "development_support_case_ids_by_query",
    "development_evaluation_case_ids_by_query",
    "development_target_evaluation_case_ids_by_target",
    "development_partition_hashes_by_query",
    "target_support_surface_artifact_id", "target_support_surface_hash",
    "target_support_parent_reservation_artifact_id",
    "target_support_parent_reservation_hash", "target_reservation_artifact_id",
    "target_reservation_hash", "target_support_case_ids_by_target",
    "target_evaluation_case_ids_by_target", "target_evaluation_binding_hash",
    "metadata_profile_sha256", "target_policy_lock_hash",
)
UPSTREAM_ARTIFACT_HASH_KEYS = frozenset(
    {
        "exact_tail_surface_lock_hash", "equal_union_policy_lock_hash",
        "development_reservation_hash", "target_support_parent_reservation_hash",
        "target_reservation_hash", "ensemble_endpoint_lock_hash",
        "ensemble_endpoint_row_hashes_hash",
        "ensemble_probability_cell_surface_hash",
        "source_inner_action_shift_lock_hash",
        "source_inner_action_shift_row_hashes_hash",
    }
)


__all__ = tuple(name for name in globals() if name.isupper())
