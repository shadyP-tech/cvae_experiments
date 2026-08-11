"""Direct immutable-parent ledger chain for prediction-only test reuse."""

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
    MODEL_FAMILY_IDS,
    SURFACE_IDS,
)


class LedgerInputConfig(Protocol):
    experiment_id: str
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path


@dataclass(frozen=True)
class ValidatedLedgerChain:
    parent: Mapping[str, object]
    amendment: Mapping[str, object]


def load_validated_ledger_chain(
    config: LedgerInputConfig,
) -> ValidatedLedgerChain:
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
        raise ProtocolError(
            "Prediction-only parent test-consumption ledger drifted."
        )

    amendment = _json(config.ledger_amendment_path)
    required_true = (
        "A1_reuses_exact_A0_rows",
        "architecture_actions_and_hyperparameters_frozen_before_target_cache_admission",
        "candidate_source_e_response_query_excluded",
        "method_development_informed_by_consumed_test_results",
        "model_hashes_sealed_before_target_cache_admission",
        "no_target_label_capability_created",
        "source_labels_previously_available",
        "source_oof_is_posthoc",
        "source_oof_predictions_sealed_before_source_label_open",
        "strict_outer_target_H_exclusion",
        "target_cache_is_label_free",
        "target_prediction_rows_sealed",
        "whole_consumed_test_prediction_only",
    )
    required_false = (
        "action_strength_sweep_used",
        "class_conditional_action_variant_used",
        "fresh_evidence",
        "generic_consumer_authorized",
        "geometry_selection_used",
        "may_authorize_action_selection",
        "may_authorize_deployment",
        "may_authorize_expert_update",
        "may_authorize_model_update",
        "may_authorize_policy_update",
        "may_authorize_promotion",
        "may_authorize_routing",
        "may_feed_another_experiment",
        "may_feed_another_stage90",
        "may_feed_deployable_selection",
        "may_feed_recipe_selection",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "nested_query_q_models_used",
        "output_is_policy",
        "previous_prediction_surfaces_used",
        "previous_stage90_outputs_used",
        "previous_stage90_scratch_or_checkpoints_used",
        "promotion_eligible",
        "routing_success_claimed",
        "source_labels_are_fresh_or_unused",
        "source_pair_action_used",
        "stage50_stage60_or_stage70_result_used",
        "target_bacc_accuracy_regret_utility_or_oracle_computed",
        "target_expert_used_on_target",
        "target_labels_available",
        "target_labels_opened_read_or_persisted",
        "target_nelbo_or_downstream_metric_computed",
        "test_rows_dropped_or_subsampled",
    )
    if (
        sha256_file(config.ledger_amendment_path)
        != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment.get("schema_version")
        != "midogpp_test_consumption_ledger_amendment_v4"
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_artifact_id")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or amendment.get("parent_member")
        != "reports/test_consumption_ledger.json"
        or amendment.get("parent_sha256") != parent_sha
        or amendment.get("authorized_consumer_experiment_ids")
        != [config.experiment_id]
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("claim_role") != CLAIM_ROLE
        or amendment.get("source_label_split") != "train"
        or amendment.get("target_cache_split") != "test"
        or amendment.get("consumed_test_row_count") != 9928
        or amendment.get("action_geometry_ids") != ["A0", "A1"]
        or amendment.get("model_family_ids") != list(MODEL_FAMILY_IDS)
        or amendment.get("prediction_only_surface_ids") != list(SURFACE_IDS)
        or amendment.get("candidate_generalization") != "known_fixed_bank_reuse"
        or amendment.get("A1_selected_row_weight") != 1.4375
        or amendment.get("A1_other_row_weight") != 0.875
        or any(amendment.get(key) is not True for key in required_true)
        or any(amendment.get(key) is not False for key in required_false)
    ):
        raise ProtocolError(
            "Prediction-only ledger amendment chain or whitelist drifted."
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
            "Prediction-only parent ledger descriptive-reuse field is absent."
        )
    if (
        canonical in parent
        and published in parent
        and parent[canonical] != parent[published]
    ):
        raise ProtocolError(
            "Prediction-only parent ledger has conflicting reuse aliases."
        )
    return parent[published] if published in parent else parent[canonical]


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read prediction-only ledger JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Prediction-only ledger must be a JSON object.")
    return value


__all__ = ("ValidatedLedgerChain", "load_validated_ledger_chain")
