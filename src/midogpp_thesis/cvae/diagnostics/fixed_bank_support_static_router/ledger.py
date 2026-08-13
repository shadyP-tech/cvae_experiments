"""Direct immutable-parent ledger admission for the support-static S4 run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    AUTHORIZATION_SCOPE,
    CLAIM_ROLE,
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    METHOD_IDS,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    PRE_EVALUATION_METHOD_IDS,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    TERMINAL_ORACLE_IDS,
)


class LedgerInputConfig(Protocol):
    experiment_id: str
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path


@dataclass(frozen=True)
class ValidatedLedgerChain:
    parent: Mapping[str, object]
    amendment: Mapping[str, object]


def load_validated_ledger_chain(config: LedgerInputConfig) -> ValidatedLedgerChain:
    """Validate the byte-exact original ledger and direct single-use amendment."""

    parent = _json(config.test_consumption_ledger_path)
    parent_sha = sha256_file(config.test_consumption_ledger_path)
    if (
        config.experiment_id != EXPERIMENT_ID
        or parent_sha != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("schema_version")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or parent.get(
            "may_be_reused_as_fresh_representation_selection_evidence"
        )
        is not False
        or _descriptive_reuse_permission(parent) is not True
    ):
        raise ProtocolError("Support-static S4 parent ledger drifted.")

    amendment = _json(config.ledger_amendment_path)
    required_true = (
        "split_previously_consumed",
        "method_development_informed_by_consumed_test_results",
        "action_geometry_frozen_before_label_access",
        "technical_seed_cells_are_not_independent_units",
        "probabilities_averaged_exact_nine_before_routing_or_scoring",
        "each_case_evaluated_exactly_once",
        "support_evaluation_whole_case_disjoint",
        "heldout_evaluation_fold_absent_from_support_scoring_selection_tie_breaking_fallback_and_decision",
        "cross_role_case_reuse_only_in_other_folds",
        "all_action_probabilities_globally_sealed_before_any_label_access",
        "role_scoped_label_capabilities_enforced",
        "each_H_f_decision_and_seal_precedes_opening_same_H_f_evaluation_role_labels",
        "target_support_labels_used",
        "U_is_internal_control_not_support_selection_candidate",
        "single_class_support_falls_back_to_B",
        "permutation_preserves_class_denominators_tp_tn_and_candidate_multiset",
        "permutation_keeps_B_fixed",
        "null_selection_plan_sealed_before_corresponding_evaluation_capability",
        "single_class_cases_retained_as_sufficient_statistics",
    )
    required_false = (
        "fresh_evidence",
        "unseen_expert_transfer_claim",
        "action_strength_sweep_used",
        "class_conditional_action_variant_used",
        "source_pair_action_used",
        "geometry_selection_used",
        "previous_prediction_surfaces_used",
        "previous_stage90_outputs_used",
        "previous_stage90_amendments_used",
        "previous_stage90_scratch_or_checkpoints_used",
        "stage50_stage60_or_stage70_result_used",
        "target_support_labels_may_update_shared_model",
        "target_support_may_select_features_hyperparameters_thresholds_action_geometry_or_strength",
        "G_static_target_support_labels_used",
        "case_features_used",
        "donor_model_used",
        "target_local_calibration_used",
        "shared_model_fit_used",
        "hyperparameter_search_used",
        "confirmatory_gate_used",
        "confirmatory_p_value_computed",
        "terminal_oracles_available_before_evaluation_labels",
        "terminal_oracles_admitted_as_pre_evaluation_methods",
        "per_case_bacc_stored_or_used",
        "permutation_changes_labels",
        "source_expert_updated",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "routing_success_claimed",
        "routing_quality_claimed",
        "target_performance_claimed",
        "action_selection_authorized",
        "action_geometry_update_authorized",
        "policy_update_authorized",
        "model_update_authorized",
        "expert_update_authorized",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
        "may_feed_recipe_selection",
        "may_feed_deployable_selection",
        "generic_consumer_authorized",
    )
    if (
        sha256_file(config.ledger_amendment_path)
        != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment.get("schema_version")
        != "midogpp_test_consumption_ledger_amendment_v3"
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_artifact_id")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or amendment.get("parent_member")
        != "reports/test_consumption_ledger.json"
        or amendment.get("parent_sha256") != parent_sha
        or amendment.get("authorized_consumer_experiment_ids")
        != [config.experiment_id]
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("publication_status") != PUBLICATION_STATUS
        or amendment.get("terminal_decision") != TERMINAL_DECISION
        or amendment.get("claim_role") != CLAIM_ROLE
        or amendment.get("action_geometry_ids") != ["A1"]
        or amendment.get("technical_seed_pair_count") != 9
        or amendment.get("target_probability_cell_count") != 810
        or amendment.get("whole_case_oof_fold_count") != 5
        or amendment.get("partition_seed") != OOF_FOLD_SEED
        or amendment.get("partition_namespace") != OOF_PARTITION_NAMESPACE
        or amendment.get("support_scope")
        != "other_four_same_H_whole_case_folds"
        or amendment.get("support_candidate_set")
        != "eight_frozen_A1_source_actions_vs_B_only"
        or amendment.get("support_selection_objective")
        != "pooled_exact_bacc_gain_vs_B"
        or amendment.get("support_selection_rule")
        != "highest_strictly_positive_A1_gain_then_numeric_source_tie_else_B"
        or amendment.get("single_class_support_fallback_action") != "B"
        or amendment.get("support_static_method_id") != "S4"
        or amendment.get("G_static_definition")
        != "equal_center_mean_exact_gain_over_q_not_in_H_or_e"
        or amendment.get("G_static_donor_query_scope") != "q_not_in_H_or_e"
        or amendment.get("G_static_candidate_gain_aggregation")
        != "equal_center_mean"
        or amendment.get("G_static_selection_rule")
        != "highest_strictly_positive_gain_then_numeric_source_tie_else_B"
        or amendment.get("diagnostic_method_ids")
        != list(PRE_EVALUATION_METHOD_IDS)
        or amendment.get("terminal_oracle_ids") != list(TERMINAL_ORACLE_IDS)
        or [*PRE_EVALUATION_METHOD_IDS, *TERMINAL_ORACLE_IDS]
        != list(METHOD_IDS)
        or amendment.get("primary_endpoint")
        != "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        or amendment.get("outer_inference_unit") != "target_center"
        or amendment.get("outer_inference_unit_count") != 9
        or amendment.get("descriptive_interval")
        != "two_sided_t8_interval_over_nine_center_contrasts"
        or amendment.get("permutation_null_count") != PERMUTATION_COUNT
        or amendment.get("permutation_seed") != PERMUTATION_SEED
        or amendment.get("permutation_algorithm_id")
        != (
            "case_sha256_candidate_order_counter_splitmix64_nonzero_"
            "cyclic_shift_1_to_7_v1"
        )
        or amendment.get("null_selection_plan_row_count") != 450_000
        or amendment.get("permutation_output")
        != "descriptive_null_exceedance_count_and_fraction_only"
        or any(amendment.get(key) is not True for key in required_true)
        or any(amendment.get(key) is not False for key in required_false)
    ):
        raise ProtocolError(
            "Support-static S4 ledger amendment chain or whitelist drifted."
        )
    return ValidatedLedgerChain(
        parent=MappingProxyType(dict(parent)),
        amendment=MappingProxyType(dict(amendment)),
    )


def _descriptive_reuse_permission(parent: Mapping[str, object]) -> object:
    canonical = "may_be_reused_for_descriptive_locked_model_scoring"
    published = "may_be_reused_for_descriptive_locked-model_scoring"
    if canonical not in parent and published not in parent:
        raise ProtocolError(
            "Support-static S4 parent ledger descriptive-reuse field is absent."
        )
    if (
        canonical in parent
        and published in parent
        and parent[canonical] != parent[published]
    ):
        raise ProtocolError(
            "Support-static S4 parent ledger has conflicting reuse aliases."
        )
    return parent[published] if published in parent else parent[canonical]


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read support-static S4 ledger: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Support-static S4 ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
