"""Versioned Stage-60 policy and action-library schemas admitted by Stage 70."""

ACTION_LIBRARY_SCHEMA = "midogpp_utility_aligned_residual_action_library_v1"
POLICY_LOCK_SCHEMA = "midogpp_utility_aligned_residual_policy_lock_v1"
TARGET_POLICY_LOCK_SCHEMA = "midogpp_utility_aligned_target_policy_lock_v1"
POLICY_EXPERIMENT_ID = (
    "midogpp.routing_and_composition."
    "uniform_b_v2_utility_aligned_residual_policy_lock.v1"
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
)
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
)
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
)
TARGET_FEATURE_LOCK_KEYS = frozenset(
    {
        "target_id", "case_bootstrap_plan", "target_feature_surface_hash",
        "target_feature_row_count", "bootstrap_surface_hashes",
        "bootstrap_surface_hashes_hash", "candidate_sources", "training_seeds",
        "generation_seeds", "case_level_resampling", "labels_used",
        "target_feature_lock_hash",
    }
)
UTILITY_POLICY_KEYS = frozenset(
    {
        "schema_version", "target_id", "candidate_sources", "router_kind",
        "proposed_action_id", "action_id", "proposed_source", "selected_source",
        "predicted_gain", "standard_error", "lower_confidence_bound",
        "confidence_multiplier", "minimum_gain", "support_case_count",
        "minimum_support_case_count", "seed_pair_count",
        "replicate_standard_deviation", "support_bootstrap_replicates",
        "minimum_support_bootstrap_replicates",
        "support_bootstrap_standard_deviation", "support_bootstrap_surface_hashes",
        "case_bootstrap_replicate_hashes", "used_exact_base_fallback",
        "fallback_reason", "global_only", "permutation_seed", "model_hash",
        "feature_surface_hash", "cardinality_eligibility_hash",
        "case_bootstrap_plan_hash", "target_support_labels_used",
        "target_evaluation_used", "seed_selection_performed",
        "abstention_semantics", "policy_hash",
    }
)
TARGET_POLICY_SHARED_KEYS = (
    "experiment_id", "output_artifact_id", "exact_tail_surface_lock_hash",
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
        "target_reservation_hash",
    }
)


__all__ = tuple(name for name in globals() if name.isupper())
