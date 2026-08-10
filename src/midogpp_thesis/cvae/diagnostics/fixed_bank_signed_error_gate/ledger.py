"""Direct immutable-parent ledger chain for the signed-error diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    CLAIM_ROLE,
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    LEDGER_AMENDMENT_ARTIFACT_ID,
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
        raise ProtocolError("Signed-error parent test-consumption ledger drifted.")

    amendment = _json(config.ledger_amendment_path)
    required_true = (
        "split_previously_consumed",
        "method_development_informed_by_consumed_test_results",
        "each_case_evaluated_exactly_once",
        "heldout_fold_absent_from_its_support_and_decision_fit",
        "cross_role_case_reuse_only_in_other_folds",
        "all_target_probabilities_globally_sealed_before_any_label_access",
        "signed_feature_context_frozen_before_labels",
        "target_support_labels_used",
        "support_exact_bacc_lcb_safety_gate",
        "calibrated_B_control_required",
        "intercept_only_global_control_required",
        "feature_permutation_control_required",
        "feature_permutation_applied_before_donor_fit",
        "feature_permutation_applied_before_target_inference",
        "feature_permutation_refits_same_capacity_model",
        "R_raw_and_R_safe_separately_sealed",
        "full_lambda_path_threshold_crossings_and_fallback_persisted",
        "single_class_cases_retained_as_sufficient_statistics",
        "architecture_and_hyperparameters_frozen_before_label_access",
    )
    required_false = (
        "fresh_evidence",
        "previous_prediction_surfaces_used",
        "previous_stage90_outputs_used",
        "previous_stage90_scratch_or_checkpoints_used",
        "hierarchical_residual_stacker_output_or_amendment_used",
        "stage50_stage60_or_stage70_result_used",
        "baseline_predicted_class_branch_used",
        "target_labels_may_update_shared_model",
        "target_support_may_select_features_or_ridge_alpha",
        "support_exact_bacc_lcb_relaxed",
        "feature_permutation_changes_labels_or_gradient_targets",
        "per_case_bacc_stored_or_used",
        "source_expert_updated",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "action_selection_authorized",
        "policy_update_authorized",
        "routing_quality_claimed",
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
        or amendment.get("authorization_scope")
        != "one_terminal_consumed_test_fixed_bank_signed_error_gate_v1"
        or amendment.get("claim_role") != CLAIM_ROLE
        or amendment.get("whole_case_oof_fold_count") != 5
        or amendment.get("shared_model_fit_scope")
        != "strict_outer_H_and_nested_query_q_exclusion"
        or amendment.get("global_control_predictors") != "intercept_only"
        or amendment.get("target_support_parameter_scope")
        != "B_cal_intercept_then_common_lambda_only"
        or amendment.get("support_selection_surrogate")
        != "fixed_class_balanced_log_loss_only"
        or amendment.get("diagnostic_method_ids")
        != ["B", "B_cal", "G", "R_raw", "R_safe", "P"]
        or amendment.get("primary_endpoint")
        != "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        or amendment.get("primary_contrasts")
        != ["R_safe-B_cal", "R_safe-G", "R_safe-P"]
        or amendment.get("uncertainty_unit") != "paired_whole_case_cluster"
        or any(amendment.get(key) is not True for key in required_true)
        or any(amendment.get(key) is not False for key in required_false)
    ):
        raise ProtocolError("Signed-error ledger amendment chain or whitelist drifted.")
    return ValidatedLedgerChain(
        parent=MappingProxyType(dict(parent)),
        amendment=MappingProxyType(dict(amendment)),
    )


def _descriptive_reuse_permission(parent: Mapping[str, object]) -> object:
    canonical = "may_be_reused_for_descriptive_locked_model_scoring"
    published = "may_be_reused_for_descriptive_locked-model_scoring"
    if canonical in parent and published in parent and parent[canonical] != parent[published]:
        raise ProtocolError("Signed-error parent ledger has conflicting reuse aliases.")
    return parent[published] if published in parent else parent.get(canonical)


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read signed-error ledger JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Signed-error ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
