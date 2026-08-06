"""Closed-world file contract for the utility/regret policy artifact."""

from __future__ import annotations

from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .config import UtilityRegretPolicyConfig
from .contracts import (
    CLAIM_SCOPE,
    CONSUMPTION_RULE_HASH,
    EXPERIMENT_ID,
    POLICY_FAMILY,
    POLICY_NAMESPACE,
    PolicySelection,
    UtilityRegretPolicyLock,
)


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_lock.json",
    "manifests/utility_regret_policy_plan.json",
    "manifests/content_index.json",
    "reports/policy_decision.json",
    "reports/leakage_report.json",
    "reports/source_inner_training_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/outer_regret_cells.csv",
    "tables/candidate_regret_summary.csv",
    "tables/bootstrap_results.csv",
    "tables/policy_selections.csv",
    "tables/policy_assignments.csv",
)

CONTENT_INDEX_MEMBERS = tuple(
    relative
    for relative in REQUIRED_FILES
    if relative
    not in {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)


def protocol_manifest_payload(
    config: UtilityRegretPolicyConfig,
    *,
    generation_lock_hash: str,
    utility_lock_hash: str,
    utility_content_hash: str,
    utility_table_hash: str,
    case_confusion_table_hash: str,
    regret_table_hash: str,
    summary_table_hash: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(config.input_artifact_ids),
        "bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "equal_union_policy_plan_hash": config.expected_equal_union_policy_plan_hash,
        "equal_union_assignment_table_hash": (
            config.expected_equal_union_assignment_table_hash
        ),
        "utility_lock_hash": utility_lock_hash,
        "utility_content_hash": utility_content_hash,
        "utility_table_hash": utility_table_hash,
        "case_confusion_table_hash": case_confusion_table_hash,
        "regret_table_hash": regret_table_hash,
        "summary_table_hash": summary_table_hash,
        "policy_consumption_lock_hash": CONSUMPTION_RULE_HASH,
        "policy_family": POLICY_FAMILY,
        "policy_namespace": POLICY_NAMESPACE,
        "outer_filter": "remove_q_equal_H_and_e_equal_H_before_policy_computation",
        "outer_target_query_excluded_before_transform": True,
        "outer_target_candidate_excluded_before_transform": True,
        "replicate_policy": "all_3x3_training_generation_seed_cells",
        "bootstrap_levels": [
            "pseudo_target_centers",
            "cases_within_pseudo_target",
            "paired_training_generation_seed_cells",
        ],
        "macro_f1_role": "secondary_descriptive_only",
        "uncertain_action": "exact_frozen_equal_union_fallback",
        "target_data_used": False,
        "target_support_used": False,
        "target_labels_used": False,
        "raw_labels_opened_by_policy": False,
        "seed_selection_performed": False,
        "policy_frozen_before_stage70": True,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
        "may_feed_deployable_selection": True,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def policy_decision_payload(
    lock: UtilityRegretPolicyLock,
    selections: Sequence[PolicySelection],
) -> dict[str, object]:
    selected = {
        row.target_center: (row.selected_source or None) for row in selections
    }
    actions = {row.target_center: row.action for row in selections}
    return {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_decision_v1",
        "decision": (
            "FROZEN_AS_SOURCE_INNER_UTILITY_REGRET_POLICY_WITH_EXACT_"
            "EQUAL_UNION_FALLBACK"
        ),
        "publication_state": "POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION",
        "claim_scope": CLAIM_SCOPE,
        "policy_lock_hash": lock.policy_lock_hash,
        "policy_consumption_lock_hash": CONSUMPTION_RULE_HASH,
        "action_by_target": actions,
        "selected_source_by_target": selected,
        "single_source_selection_count": sum(
            row.action == "single_source_full_budget" for row in selections
        ),
        "equal_union_fallback_count": sum(
            row.action == "fallback_equal_union" for row in selections
        ),
        "deployable_selection_input": True,
        "routing_quality_claimed": False,
        "downstream_utility_claimed": False,
        "next_required_evidence": (
            "run_fresh_matched_stage70_downstream_scoring_against_the_frozen_"
            "equal_union_control"
        ),
    }


def leakage_report_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_leakage_v1",
        "status": "PASS",
        "source_inner_validation_utility_consumed": True,
        "source_inner_raw_labels_reopened_by_policy": False,
        "dataset_manifest_accessed_by_policy": False,
        "feature_cache_accessed_by_policy": False,
        "target_identity_role": "outer_fold_exclusion_and_composition_binding_only",
        "target_identity_as_predictive_feature": False,
        "target_samples_used": False,
        "target_support_used": False,
        "target_labels_used": False,
        "target_evaluation_labels_used": False,
        "outer_target_query_rows_removed_before_transform": True,
        "outer_target_candidate_rows_removed_before_transform": True,
        "target_expert_excluded_in_every_assignment": True,
        "stage20_consumed_metrics_reused": False,
        "stage50_artifacts_used": False,
        "stage90_artifacts_used": False,
        "nelbo_computed": False,
        "macro_f1_used_for_selection": False,
        "seed_selection_performed": False,
        "equal_union_fallback_reestimated": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
    }


def source_inner_training_summary_payload(
    selections: Sequence[PolicySelection],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_training_summary_v1",
        "status": "SOURCE_INNER_POLICY_TRAINING_COMPLETE",
        "outer_fold_count": len(selections),
        "queries_per_outer_before_candidate_identity_exclusion": 8,
        "queries_per_candidate_summary": 7,
        "candidate_sources_per_outer": 8,
        "legal_candidates_per_query": 7,
        "paired_training_generation_seed_cell_count": 9,
        "bootstrap_valid_replicates_per_outer": 2000,
        "bootstrap_level_count": 3,
        "selection_gate_pass_count": sum(row.bootstrap.gate_passed for row in selections),
        "selection_gate_fail_count": sum(
            not row.bootstrap.gate_passed for row in selections
        ),
        "validation_labels_consumed_in_upstream_utility_only": True,
        "raw_labels_opened_by_policy": False,
        "macro_f1_role": "secondary_descriptive_only",
        "training_seeds_are_replications_not_candidates": True,
        "generation_seeds_are_replications_not_candidates": True,
        "routing_quality_claimed": False,
    }


def run_state_payload(status: str) -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_utility_regret_run_state_v1",
        "status": status,
        "claim_scope": CLAIM_SCOPE,
    }


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "leakage_report_payload",
    "policy_decision_payload",
    "protocol_manifest_payload",
    "run_state_payload",
    "source_inner_training_summary_payload",
)
