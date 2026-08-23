"""Frozen planned protocol for the P-DCAPS v3 mechanical repair."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .nullable_statistics import (
    CONSTANT_RANK_UNDEFINED_REASON,
    DENOMINATOR_UNDEFINED_REASON,
    NULLABLE_STATISTIC_SCHEMA,
)
from .identity import (
    ACTION_FAMILIES,
    ACTION_STRATA,
    DIRECT_INPUT_ROLES,
    EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256,
    EXPERIMENT_ID,
    METHOD_MENU,
    METRICS,
    POLICY_ONLY_METHOD_ID,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    P_METHOD_ID,
    RIDGE_ALPHA,
    TERMINAL_DECISION,
    TIE_TOLERANCE,
    V2_EXECUTION_STATUS,
    V2_EXPERIMENT_ID,
    V2_OUTPUT_ARTIFACT_ID,
    V2_PATH_INDEPENDENT_CONFIG_SHA256,
    V2_PROTOCOL_CONTRACT_SHA256,
    V2_SCIENTIFIC_MECHANICS_SCHEMA,
    ACTION_ONLY_METHOD_ID,
    CYCLIC_METHOD_ID,
    LEGACY_METHOD_ID,
    canonical_hash,
)
from .source_seal import (
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    V2_SOURCE_SNAPSHOT_SCHEMA,
    V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
)


PROTOCOL_SCHEMA = "pdcaps_v3_terminal_protocol_v1"
POSTERIOR_CONTROL_IDS = ("IDENTITY", "WITHIN_CASE_CYCLIC_SHIFT")

# Every v2 protocol field is intentionally classified.  The first tuple is the
# scientific/lifecycle mechanics copied into the v3 binding.  The second tuple
# contains only predecessor identity, authorization, provenance, and claim
# metadata; none of those historical authorization values is carried into v3.
V2_MECHANICS_PROTOCOL_KEYS = (
    "response_denominators",
    "endpoint_donor_prior_policy",
    "minimum_effective_sample_size_per_class",
    "physical_probability_surface_recomputed_from_original_inputs",
    "held_unit",
    "outer_center_excluded_from_every_scientific_fit",
    "pseudo_center_excluded_from_own_prediction",
    "held_case_d_role",
    "action_strata",
    "ridge_alpha",
    "hyperparameter_selection_used",
    "fit_only_standardization",
    "hierarchical_weighting",
    "minimum_reliability_center_count",
    "all_action_surface_sealed_before_pseudo_response_access",
    "posterior_control_ids",
    "identity_and_cyclic_action_surfaces_jointly_sealed_before_pseudo_response_access",
    "pseudo_label_capability_opened_once_per_route_for_all_posterior_controls",
    "all_prefix_cells_sealed_before_policy_response_access",
    "per_outer_center_admission",
    "pooled_admission_can_affect_routes",
    "descriptive_lower_envelope_only",
    "method_menu",
    "tie_tolerance",
    "exact_p_fallback_required",
    "target_labels_open_only_after_preterminal_attestation",
    "all_fixed_method_decisions_and_compositions_sealed_before_target_labels",
    "raw_labels_may_be_persisted",
)
V2_NON_MECHANICS_PROTOCOL_KEYS = (
    "schema_version",
    "experiment_id",
    "dataset_family",
    "split",
    "split_previously_consumed",
    "fresh_evidence",
    "publication_status",
    "terminal_decision",
    "execution_authorized",
    "authorization_basis",
    "authorization_scope",
    "single_use_execution_identity",
    "authorization_exhausted",
    "scientific_protocol_unchanged_from_v1",
    "scientific_method_changed_from_v1",
    "methodological_delta_role",
    "methodological_deltas",
    "methodological_deltas_are_terminal_consumed_test_only",
    "methodological_deltas_create_fresh_evidence",
    "methodological_deltas_are_promotable",
    "v1_output_used",
    "v1_amendment_used",
    "v1_label_capability_history_used",
    "v1_scratch_or_checkpoint_used",
    "prior_v1_execution_authorization_reused",
    "source_snapshot_binding_required",
    "source_snapshot_schema",
    "source_snapshot_scope",
    "external_neutral_module_source_policy",
    "source_snapshot_manifest_sha256",
    "source_snapshot_tree_sha256",
    "source_snapshot_member_count",
    "source_snapshot_excludes_pyc_and_cache",
    "authorization_is_separate_from_implementation_request",
    "source_code_or_implementation_request_alone_authorizes_execution",
    "execution_requires_external_authorized_config_and_ledger",
    "input_roles",
    "input_count",
    "previous_stage90_outputs_used",
    "previous_stage90_amendments_used",
    "previous_stage90_scratch_or_checkpoints_used",
    "finite_sample_coverage_claimed",
    "may_feed_stage50",
    "may_feed_stage60",
    "may_feed_stage70",
    "may_feed_another_stage90",
    "may_feed_another_experiment",
    "routing_success_claimed",
    "downstream_utility_claimed",
    "nelbo_compatibility_claimed",
    "deployment_claimed",
)


def _v2_protocol_mechanics_payload() -> dict[str, object]:
    return {
        "response_denominators": (
            "derived_inside_lifecycle_from_support_plus_held"
        ),
        "endpoint_donor_prior_policy": "ZERO_VECTOR_NO_FITTED_PRIOR",
        "minimum_effective_sample_size_per_class": 5.0,
        "physical_probability_surface_recomputed_from_original_inputs": True,
        "held_unit": "whole_case_or_group",
        "outer_center_excluded_from_every_scientific_fit": True,
        "pseudo_center_excluded_from_own_prediction": True,
        "held_case_d_role": "scored_response_only_after_surface_seal",
        "action_strata": [list(row) for row in ACTION_STRATA],
        "ridge_alpha": RIDGE_ALPHA,
        "hyperparameter_selection_used": False,
        "fit_only_standardization": True,
        "hierarchical_weighting": "equal_center_then_route_then_cell",
        "minimum_reliability_center_count": 6,
        "all_action_surface_sealed_before_pseudo_response_access": True,
        "posterior_control_ids": list(POSTERIOR_CONTROL_IDS),
        "identity_and_cyclic_action_surfaces_jointly_sealed_before_pseudo_response_access": True,
        "pseudo_label_capability_opened_once_per_route_for_all_posterior_controls": True,
        "all_prefix_cells_sealed_before_policy_response_access": True,
        "per_outer_center_admission": True,
        "pooled_admission_can_affect_routes": False,
        "descriptive_lower_envelope_only": True,
        "method_menu": list(METHOD_MENU),
        "tie_tolerance": TIE_TOLERANCE,
        "exact_p_fallback_required": True,
        "target_labels_open_only_after_preterminal_attestation": True,
        "all_fixed_method_decisions_and_compositions_sealed_before_target_labels": True,
        "raw_labels_may_be_persisted": False,
    }


def _v2_action_library_payload() -> dict[str, object]:
    return {
        "schema_version": "pdcaps_action_library_v1",
        "centers": list(CENTERS),
        "excluded_center": "4",
        "physical_actions_per_target": 10,
        "physical_action_ids": "B_U_and_eight_A1_source_actions",
        "candidate_action_families": list(ACTION_FAMILIES),
        "candidate_action_strata": [list(row) for row in ACTION_STRATA],
        "B_rows_per_source_class": 128,
        "U_rows_per_source_class": 144,
        "A1_selected_rows_per_class": 256,
        "A1_other_rows_per_class": 128,
        "A1_selected_row_weight": 1.4375,
        "A1_other_row_weight": 0.875,
        "target_expert_excluded": True,
        "probabilities_averaged_exact_nine_before_routing": True,
        "physical_probability_storage_dtype": "float32",
        "all_crossing_actions_retained_until_action_surface_seal": True,
        "previous_probability_surface_used": False,
        "previous_stage90_output_used": False,
        "endpoint_donor_prior_policy": "ZERO_VECTOR_NO_FITTED_PRIOR",
        "minimum_effective_sample_size_per_class": 5.0,
    }


def _v2_method_controls_payload() -> dict[str, object]:
    return {
        "schema_version": "pdcaps_fixed_method_controls_v1",
        "fixed_menu": {
            P_METHOD_ID: {
                "source_layer": "protected_P_probability_surface",
                "selected_actions": "none_exact_P_byte_identity",
                "admission_H": "not_applied",
                "expected_posterior_control_id": "IDENTITY",
                "same_physical_surface_as_identity_required": True,
                "role": "protected_reference",
            },
            PRIMARY_METHOD_ID: {
                "source_layer": (
                    "identity_action_selection_then_selected_policy_prefix"
                ),
                "selected_actions": (
                    "selected_policy_prefix_only_when_Admission_H_passes"
                ),
                "admission_H": "required_pseudo_only_same_run",
                "expected_posterior_control_id": "IDENTITY",
                "same_physical_surface_as_identity_required": True,
                "role": "primary_terminal_diagnostic",
            },
            ACTION_ONLY_METHOD_ID: {
                "source_layer": "identity_target_action_selections",
                "selected_actions": "all_nonfallback_target_action_selections",
                "admission_H": "bypassed_terminal_ablation",
                "expected_posterior_control_id": "IDENTITY",
                "same_physical_surface_as_identity_required": True,
                "role": "terminal_action_layer_ablation",
            },
            POLICY_ONLY_METHOD_ID: {
                "source_layer": "identity_selected_target_policy_prefix",
                "selected_actions": "selected_policy_prefix",
                "admission_H": "bypassed_terminal_ablation",
                "expected_posterior_control_id": "IDENTITY",
                "same_physical_surface_as_identity_required": True,
                "role": "terminal_admission_layer_ablation",
            },
            LEGACY_METHOD_ID: {
                "source_layer": (
                    "typed_same_run_legacy_center_pooled_target_decision"
                ),
                "selected_actions": "sealed_legacy_target_prefix",
                "admission_H": "not_applied",
                "expected_posterior_control_id": "IDENTITY",
                "same_physical_surface_as_identity_required": True,
                "role": "terminal_legacy_control",
            },
            CYCLIC_METHOD_ID: {
                "source_layer": (
                    "distinct_cyclic_posterior_action_and_policy_result"
                ),
                "selected_actions": (
                    "cyclic_selected_policy_prefix_only_when_cyclic_"
                    "Admission_H_passes"
                ),
                "admission_H": "required_cyclic_pseudo_only_same_run",
                "expected_posterior_control_id": "WITHIN_CASE_CYCLIC_SHIFT",
                "same_physical_surface_as_identity_required": True,
                "distinct_action_surface_seal_from_identity_required": True,
                "role": "terminal_cyclic_poison_control",
            },
        },
        "caller_selected_action_hashes_permitted": False,
        "typed_source_and_seal_binding_required": True,
        "terminal_diagnostic_only": True,
        "routing_authorized": False,
        "promotion_allowed": False,
        "target_labels_used": False,
    }


def _v2_policy_menu_payload() -> dict[str, object]:
    return {
        "schema_version": "pdcaps_policy_menu_v1",
        "method_ids": list(METHOD_MENU),
        "protected_fallback": P_METHOD_ID,
        "primary_method_id": PRIMARY_METHOD_ID,
        "exact_P_fallback_required": True,
        "exact_P_fallback_storage_contract": "byte_for_byte_float32_identity",
        "action_response_model": {
            "family": "equal_center_weighted_ridge",
            "response_coordinates": list(METRICS),
            "alpha": RIDGE_ALPHA,
            "hyperparameter_selection_used": False,
            "standardization": "fit_rows_only",
            "solve_dtype": "float64",
            "features": [
                "predicted_favorable_metric",
                "crossing_fraction",
                "six_action_family_direction_indicators",
                "predicted_metric_x_action_stratum_interactions",
            ],
            "weighting": "equal_center_then_route_then_action_cell",
            "target_fit_scope": "all_J_not_equal_H",
            "pseudo_fit_scope": "all_K_not_in_H_or_J",
            "held_case_role": "SCORED_RESPONSE_ONLY_AFTER_SURFACE_SEAL",
        },
        "stratum_reliability_gate": {
            "minimum_represented_donor_centers": 6,
            "bacc_spearman_strictly_positive": True,
            "equal_center_mean_realized_bacc_strictly_positive": True,
            "strict_majority_positive_donor_center_bacc_means": True,
            "equal_center_mean_realized_brier_nonnegative": True,
            "equal_center_mean_realized_log_nonnegative": True,
            "class_domain_support_required": True,
            "nonfinite_or_unsupported_action": P_METHOD_ID,
            "applied_before_within_case_argmax": True,
        },
        "action_selection": {
            "feasibility": (
                "calibrated_BACC_positive_and_calibrated_favorable_Brier_and_"
                "log_nonnegative"
            ),
            "ranking": (
                "maximum_calibrated_BACC_then_B_I_R_then_zero_to_one_"
                "then_one_to_zero_then_action_hash"
            ),
            "tie_tolerance": TIE_TOLERANCE,
            "tie_or_failure_fallback": P_METHOD_ID,
        },
        "policy_surface_model": {
            "family": "equal_center_weighted_ridge",
            "response_coordinates": list(METRICS),
            "alpha": RIDGE_ALPHA,
            "hyperparameter_selection_used": False,
            "standardization": "fit_rows_only",
            "solve_dtype": "float64",
            "features": [
                "predicted_favorable_metric",
                "normalized_prefix_depth",
                "maximum_positive_candidate_share",
                "six_action_stratum_proportions",
            ],
            "weighting": "equal_center_then_complete_policy_cell",
            "target_fit_scope": "all_J_not_equal_H",
            "pseudo_fit_scope": "all_K_not_in_H_or_J",
            "all_complete_prefix_cells_scored": True,
        },
        "lower_envelope": {
            "formula": (
                "policy_prediction_minus_equal_center_mean_positive_OOF_"
                "overprediction_minus_maximum_delete_donor_change"
            ),
            "out_of_fold_only": True,
            "finite_sample_coverage_claimed": False,
            "confidence_bound_claimed": False,
        },
        "prefix_selection": {
            "feasibility": (
                "lower_envelope_BACC_positive_and_lower_envelope_favorable_"
                "Brier_and_log_nonnegative"
            ),
            "ranking": "maximum_BACC_then_smaller_K_then_prefix_hash",
            "tie_tolerance": TIE_TOLERANCE,
            "empty_or_failed_surface_fallback": P_METHOD_ID,
        },
        "legacy_control_recomputed_same_run": True,
        "legacy_control": {
            "method_id": LEGACY_METHOD_ID,
            "implementation_scope": "same_run_package_local_only",
            "response_scope": "donor_pseudo_responses_only",
            "pooling": "equal_center",
            "selection_grid": "normalized_target_prefix_depth_grid",
            "donor_prefix_match": (
                "nearest_normalized_depth_then_smaller_k_then_prefix_hash"
            ),
            "feasibility": (
                "pooled_BACC_strictly_positive_and_pooled_favorable_Brier_"
                "and_log_nonnegative"
            ),
            "ranking": (
                "maximum_pooled_BACC_then_smaller_target_k_then_target_"
                "prefix_hash"
            ),
            "target_labels_used": False,
            "exact_P_permitted_at_k0": True,
        },
        "method_controls": _v2_method_controls_payload(),
        "cyclic_poison_control_predeclared": True,
        "terminal_labels_may_change_same_surface_routes": False,
        "response_denominators": (
            "derived_inside_lifecycle_from_support_plus_held"
        ),
    }


def _v2_classifier_payload() -> dict[str, object]:
    return {
        "family": "sklearn_logistic_regression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 3000,
        "class_weight": None,
        "random_state": 23,
        "l1_ratio": None,
        "threshold_policy": "predict",
        "scaler_fit": "synthetic_train_only",
    }


def _v2_evaluation_payload() -> dict[str, object]:
    return {
        "primary_metric": "equal_center_BACC",
        "secondary_metrics": [
            "sample_pooled_BACC",
            "global_Brier",
            "equal_center_Brier",
            "global_log_loss",
            "equal_center_log_loss",
        ],
        "log_loss_clip_epsilon": 1.0e-12,
        "center_contrasts_against": P_METHOD_ID,
        "primary_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "fixed_method_menu": list(METHOD_MENU),
        "selection_control": (
            "exact_2_power_9_center_sign_flip_max_over_fixed_menu"
        ),
        "route_pipeline_refit_inside_null_replicate": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "terminal_diagnostics": [
            "action_expected_vs_realized_midrank_spearman_by_stratum",
            "policy_expected_vs_realized_midrank_spearman",
            "joint_safe_routed_policy_rate",
            "normalized_endpoint_oracle_gap",
            "case_harm_rate",
            "center_action_frequencies",
        ],
        "preterminal_admission_uses_pseudo_donor_responses_only": True,
        "target_terminal_labels_may_change_routes": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "raw_labels_persisted": False,
        "diagnostic_rows_are_not_independent_inference_units": True,
        "nonzero_route_count_is_not_success": True,
    }


def frozen_v2_scientific_mechanics_payload() -> dict[str, object]:
    """Return the complete authorization-free v2 scientific method binding."""

    return {
        "schema_version": V2_SCIENTIFIC_MECHANICS_SCHEMA,
        "v2_protocol_contract_sha256": V2_PROTOCOL_CONTRACT_SHA256,
        "v2_path_independent_config_sha256": (
            V2_PATH_INDEPENDENT_CONFIG_SHA256
        ),
        "protocol_controls": _v2_protocol_mechanics_payload(),
        "action_library": _v2_action_library_payload(),
        "policy_menu": _v2_policy_menu_payload(),
        "classifier": _v2_classifier_payload(),
        "evaluation": _v2_evaluation_payload(),
    }


def validate_v2_scientific_mechanics_payload(
    payload: Mapping[str, object],
) -> None:
    expected = frozen_v2_scientific_mechanics_payload()
    if dict(payload) != expected:
        raise ProtocolError("P-DCAPS v3 frozen v2 mechanics payload drifted.")
    if canonical_hash(expected) != EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256:
        raise ProtocolError("P-DCAPS v3 frozen v2 mechanics hash drifted.")


def frozen_protocol_payload() -> dict[str, object]:
    inherited_mechanics = frozen_v2_scientific_mechanics_payload()
    validate_v2_scientific_mechanics_payload(inherited_mechanics)
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "separate_future_run_authorization_required": True,
        "v2_experiment_id": V2_EXPERIMENT_ID,
        "v2_output_artifact_id": V2_OUTPUT_ARTIFACT_ID,
        "v2_execution_status": V2_EXECUTION_STATUS,
        "v2_authorization_exhausted": True,
        "v2_retry_forbidden": True,
        "v2_output_used": False,
        "v2_scratch_or_checkpoint_used": False,
        "v2_probability_or_capability_history_used": False,
        "repair_class": "MECHANICAL_NULLABLE_ADMISSION_STATISTICS_ONLY",
        "scientific_thresholds_changed_from_v2": False,
        "scientific_ordering_changed_from_v2": False,
        "donor_inventory_rule_changed_from_v2": False,
        "physical_surface_or_outer_fit_changed_from_v2": False,
        "method_menu_changed_from_v2": False,
        "scientific_method_changed_from_v2": False,
        "v2_scientific_mechanics_schema": V2_SCIENTIFIC_MECHANICS_SCHEMA,
        "v2_protocol_contract_sha256": V2_PROTOCOL_CONTRACT_SHA256,
        "v2_path_independent_config_sha256": (
            V2_PATH_INDEPENDENT_CONFIG_SHA256
        ),
        "v2_scientific_mechanics_sha256": (
            EXPECTED_V2_SCIENTIFIC_MECHANICS_SHA256
        ),
        "nullable_statistic_schema": NULLABLE_STATISTIC_SCHEMA,
        "nullable_statistic_fields": [
            "name",
            "value",
            "defined",
            "undefined_reason",
        ],
        "nullable_undefined_reasons": [
            CONSTANT_RANK_UNDEFINED_REASON,
            DENOMINATOR_UNDEFINED_REASON,
        ],
        "constant_rank_correlation_persisted_as_null": True,
        "invalid_or_zero_denominator_gap_persisted_as_null": True,
        "undefined_statistic_fails_outer_admission": True,
        "undefined_statistic_selects_byte_exact_p": True,
        "caller_injected_nan_or_infinity_rejected": True,
        "canonical_json_allow_nan": False,
        "input_roles": list(DIRECT_INPUT_ROLES),
        "input_count": len(DIRECT_INPUT_ROLES),
        "exact_six_direct_inputs_required": True,
        "held_unit": "whole_case_or_group",
        "outer_center_excluded_from_every_scientific_fit": True,
        "pseudo_center_excluded_from_own_prediction": True,
        "held_case_d_role": "SCORED_RESPONSE_ONLY_AFTER_SURFACE_SEAL",
        "posterior_control_ids": list(POSTERIOR_CONTROL_IDS),
        "method_menu": list(METHOD_MENU),
        "exact_p_fallback_required": True,
        "target_labels_open_only_after_durable_preterminal_attestation": True,
        "raw_labels_may_be_persisted": False,
        "inherited_v2_base_source_snapshot_schema": V2_SOURCE_SNAPSHOT_SCHEMA,
        "inherited_v2_base_source_manifest_sha256": (
            EXPECTED_V2_SOURCE_MANIFEST_SHA256
        ),
        "inherited_v2_base_source_tree_sha256": (
            EXPECTED_V2_SOURCE_TREE_SHA256
        ),
        "inherited_v2_base_source_member_count": (
            EXPECTED_V2_SOURCE_MEMBER_COUNT
        ),
        "v3_repair_source_snapshot_schema": V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
        "v3_repair_source_manifest_sha256": (
            EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
        ),
        "v3_repair_source_tree_sha256": EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
        "v3_repair_source_member_count": EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
        "source_scopes_are_disjoint": True,
        "forbidden_repaired_path_modules": [
            "base.admission",
            "base.routing",
            "base.method_controls",
            "v2.method_runtime",
            "v2.outer_runtime",
        ],
        "allowed_inherited_surface": (
            "source_sealed_base_DTOs_and_pure_scientific_kernels_only"
        ),
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "deployment_claimed": False,
        "promotion_allowed": False,
    }
    return {**payload, "protocol_hash": canonical_hash(payload)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or dict(payload) != frozen_protocol_payload():
        raise ProtocolError("P-DCAPS v3 frozen protocol drifted.")


def validate_nested_exclusions(
    *,
    outer_center: object,
    scored_center: object,
    excluded_centers: Sequence[object],
) -> tuple[str, str]:
    outer = str(outer_center)
    scored = str(scored_center)
    excluded = tuple(sorted({str(value) for value in excluded_centers}))
    if outer not in CENTERS or scored not in CENTERS or outer == scored:
        raise ProtocolError("P-DCAPS v3 nested H/J identity drifted.")
    if set(excluded) != {outer, scored}:
        raise ProtocolError(
            "P-DCAPS v3 nested fit must exclude exactly outer H and scored J."
        )
    return outer, scored


def validate_held_case_role(
    held_case_id: object,
    *,
    fit_case_ids: Sequence[object],
    role: str,
) -> str:
    held = str(held_case_id)
    if not held or held in {str(value) for value in fit_case_ids}:
        raise ProtocolError("P-DCAPS v3 held case d entered a fitted role.")
    if role != "SCORED_RESPONSE_ONLY_AFTER_SURFACE_SEAL":
        raise ProtocolError("P-DCAPS v3 held case d role drifted.")
    return held


__all__ = (
    "POSTERIOR_CONTROL_IDS",
    "PROTOCOL_SCHEMA",
    "V2_MECHANICS_PROTOCOL_KEYS",
    "V2_NON_MECHANICS_PROTOCOL_KEYS",
    "frozen_protocol_payload",
    "frozen_v2_scientific_mechanics_payload",
    "validate_held_case_role",
    "validate_nested_exclusions",
    "validate_protocol_payload",
    "validate_v2_scientific_mechanics_payload",
)
