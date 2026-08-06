"""Closed-world bundle schemas and deterministic payload builders."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import SourceInnerUtilityConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPECTED_CASE_CONFUSION_ROW_COUNT,
    EXPECTED_EVAL_CASES,
    EXPECTED_EVAL_ROWS,
    EXPECTED_FIT_COUNT,
    EXPECTED_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    POLICY_CONSUMPTION_LOCK_HASH,
    UTILITY_POLICY_FAMILY,
    policy_consumption_lock_payload,
)
from .prediction_io import (
    EVALUATION_ROW_COLUMNS,
    EVALUATION_ROW_MEMBER,
    FIT_TABLE_MEMBER,
    PREDICTION_ARRAY_MEMBER,
    evaluation_row_table,
    prediction_index_payload,
    read_prediction_arrays,
    write_prediction_arrays,
)


UTILITY_TABLE_MEMBER = "tables/candidate_utility.csv"
CASE_CONFUSION_TABLE_MEMBER = "tables/case_confusions.csv"

REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_consumption_lock.json",
    "manifests/prediction_index.json",
    "manifests/utility_lock.json",
    "manifests/content_index.json",
    "reports/utility_decision.json",
    "reports/label_consumption_report.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    EVALUATION_ROW_MEMBER,
    FIT_TABLE_MEMBER,
    UTILITY_TABLE_MEMBER,
    CASE_CONFUSION_TABLE_MEMBER,
    PREDICTION_ARRAY_MEMBER,
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


def policy_consumption_manifest_payload(config: SourceInnerUtilityConfig) -> dict[str, object]:
    rule = policy_consumption_lock_payload()
    if dict(config.policy_consumption_lock) != rule:
        raise ProtocolError("Config policy consumption rule drifted from canonical source.")
    return {
        "schema_version": "midogpp_uniform_b_v2_policy_consumption_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "config_contract_hash": config.contract_hash,
        "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
        "rule": rule,
        "locked_before_validation_labels_opened": True,
        "label_consumption_authorizes_only_this_rule": True,
    }


def protocol_manifest_payload(
    config: SourceInnerUtilityConfig,
    *,
    generation_lock_hash: str,
    bank_lock_hash: str,
    cache_binding: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_source_inner_utility_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "generation_lock_hash": generation_lock_hash,
        "bank_lock_hash": bank_lock_hash,
        "cache_binding": dict(cache_binding),
        "manifest_sha256": config.expected_manifest_sha256,
        "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
        "candidate_unit": "source_center",
        "full_source_budget_per_class": 1024,
        "classifier_fit_count": EXPECTED_FIT_COUNT,
        "candidate_utility_row_count": EXPECTED_UTILITY_ROW_COUNT,
        "case_confusion_row_count": EXPECTED_CASE_CONFUSION_ROW_COUNT,
        "eval_row_count": EXPECTED_EVAL_ROWS,
        "eval_case_count": EXPECTED_EVAL_CASES,
        "prediction_pass_label_free": True,
        "labels_opened_after_prediction_arrays_materialized": True,
        "q_must_differ_from_e": True,
        "outer_target_instantiated": False,
        "selection_performed": False,
        "seed_selection_performed": False,
        "nelbo_computed": False,
        "may_feed_deployable_selection": True,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def utility_lock_payload(
    config: SourceInnerUtilityConfig,
    *,
    protocol: Mapping[str, object],
    prediction_index: Mapping[str, object],
    member_sha256: Mapping[str, str],
    case_confusion_row_count: int,
) -> dict[str, object]:
    expected_members = {
        EVALUATION_ROW_MEMBER,
        FIT_TABLE_MEMBER,
        UTILITY_TABLE_MEMBER,
        CASE_CONFUSION_TABLE_MEMBER,
        PREDICTION_ARRAY_MEMBER,
    }
    if set(member_sha256) != expected_members:
        raise ProtocolError("Utility-lock member hash coverage drifted.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_source_inner_utility_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "protocol_hash": protocol.get("protocol_hash"),
        "prediction_index_hash": prediction_index.get("prediction_index_hash"),
        "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
        "policy_family": UTILITY_POLICY_FAMILY,
        "member_sha256": dict(member_sha256),
        "eval_row_count": EXPECTED_EVAL_ROWS,
        "eval_case_count": EXPECTED_EVAL_CASES,
        "classifier_fit_count": EXPECTED_FIT_COUNT,
        "candidate_utility_row_count": EXPECTED_UTILITY_ROW_COUNT,
        "case_confusion_row_count": int(case_confusion_row_count),
        "labels_persisted": False,
        "labels_consumed_for_scoring_only": True,
        "selection_performed": False,
        "alternative_router_tuning_authorized": False,
        "seed_selection_performed": False,
    }
    payload["utility_lock_hash"] = stable_hash(payload)
    return payload


def run_state_payload(status: str) -> dict[str, object]:
    if status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise ProtocolError(f"Invalid source-inner utility run state: {status!r}.")
    return {
        "schema_version": "midogpp_uniform_b_v2_source_inner_utility_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "status": status,
        "selection_performed": False,
    }


def leakage_report_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_source_inner_utility_leakage_v1",
        "status": "PASS",
        "source_inner_validation_labels_consumed": True,
        "labels_available_to_generation": False,
        "labels_available_to_classifier_fit": False,
        "labels_available_to_prediction": False,
        "labels_used_for_scoring_only": True,
        "train_labels_consumed": False,
        "test_labels_consumed": False,
        "center_4_labels_consumed": False,
        "target_support_used": False,
        "target_metadata_used": False,
        "outer_target_instantiated": False,
        "outer_target_expert_used": False,
        "q_equals_e_rows_emitted": False,
        "stage20_metrics_used": False,
        "stage50_artifacts_used": False,
        "stage90_artifacts_used": False,
        "nelbo_computed": False,
        "candidate_ranking_performed": False,
        "policy_selection_performed": False,
        "seed_selection_performed": False,
        "alternative_router_tuning_authorized": False,
    }


def label_consumption_report_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_source_inner_label_consumption_v1",
        "status": "CONSUMED_FOR_PREDECLARED_POLICY_FAMILY_ONLY",
        "policy_family": UTILITY_POLICY_FAMILY,
        "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
        "manifest_split_consumed": "val",
        "eligible_centers": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
        "consumed_row_count": EXPECTED_EVAL_ROWS,
        "consumed_case_count": EXPECTED_EVAL_CASES,
        "labels_opened_after_all_predictions_materialized": True,
        "case_confusions_are_future_label_free_bootstrap_input": True,
        "train_labels_consumed": False,
        "test_labels_consumed": False,
        "center_4_labels_consumed": False,
        "may_authorize_alternative_router_tuning": False,
        "may_authorize_alternative_policy_family": False,
    }


def utility_decision_payload(utility_lock: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_source_inner_utility_decision_v1",
        "status": "SOURCE_INNER_UTILITY_READY_FOR_LOCKED_POLICY_CONSUMER",
        "utility_lock_hash": utility_lock.get("utility_lock_hash"),
        "policy_consumption_lock_hash": POLICY_CONSUMPTION_LOCK_HASH,
        "selection_performed": False,
        "candidate_ranking_performed": False,
        "seed_selection_performed": False,
        "outer_target_instantiated": False,
        "may_feed_deployable_selection": True,
        "authorized_consumer_policy_family": UTILITY_POLICY_FAMILY,
        "alternative_router_tuning_authorized": False,
        "routing_quality_claimed": False,
        "outer_target_downstream_utility_claimed": False,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "CASE_CONFUSION_TABLE_MEMBER",
    "CONTENT_INDEX_MEMBERS",
    "EVALUATION_ROW_COLUMNS",
    "EVALUATION_ROW_MEMBER",
    "FIT_TABLE_MEMBER",
    "PREDICTION_ARRAY_MEMBER",
    "REQUIRED_FILES",
    "UTILITY_TABLE_MEMBER",
    "evaluation_row_table",
    "label_consumption_report_payload",
    "leakage_report_payload",
    "policy_consumption_manifest_payload",
    "prediction_index_payload",
    "protocol_manifest_payload",
    "read_prediction_arrays",
    "run_state_payload",
    "sha256_file",
    "utility_decision_payload",
    "utility_lock_payload",
    "write_prediction_arrays",
)
