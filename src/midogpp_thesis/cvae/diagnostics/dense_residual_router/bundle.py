"""Closed-world bundle contract for the Stage-90 residual-router diagnostic."""

from __future__ import annotations

from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import DenseResidualDiagnosticConfig, canonical_claim_boundary_payload
from .contracts import (
    ACTION_LIBRARY_HASH,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    action_library,
)
from .seals import (
    AllActionTargetPredictionSeal,
    DevelopmentPredictionSeal,
    DiagnosticDecisionSeal,
    TargetPredictionSeal,
)


DEVELOPMENT_PREDICTION_ARRAY_MEMBER = "arrays/development_predictions.npz"
TARGET_PREDICTION_ARRAY_MEMBER = "arrays/target_predictions.npz"

REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/support_partition_lock.json",
    "manifests/compatibility_index.json",
    "manifests/development_prediction_seals.json",
    "manifests/all_action_target_prediction_seal.json",
    "manifests/diagnostic_decision_seals.json",
    "manifests/target_prediction_seals.json",
    "manifests/content_index.json",
    DEVELOPMENT_PREDICTION_ARRAY_MEMBER,
    TARGET_PREDICTION_ARRAY_MEMBER,
    "tables/support_partitions.csv",
    "tables/compatibility_case_energy.csv",
    "tables/compatibility_scores.csv",
    "tables/development_prediction_index.csv",
    "tables/development_metrics.csv",
    "tables/action_summaries.csv",
    "tables/diagnostic_selections.csv",
    "tables/target_weight_plans.csv",
    "tables/target_assignments.csv",
    "tables/target_prediction_index.csv",
    "tables/target_metrics.csv",
    "tables/paired_deltas.csv",
    "reports/phase_01_support_and_compatibility_complete.json",
    "reports/phase_02_development_complete.json",
    "reports/phase_03_target_predictions_complete.json",
    "reports/phase_04_scoring_complete.json",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

CONTENT_INDEX_MEMBERS = tuple(
    member
    for member in REQUIRED_FILES
    if member
    not in {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)

_NON_ADOPTIVE_FLAGS = {
    "claim_scope": CLAIM_SCOPE,
    **canonical_claim_boundary_payload(),
}


def protocol_manifest_payload(
    config: DenseResidualDiagnosticConfig,
    *,
    input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    hashes = {str(key): str(value) for key, value in input_artifact_hashes.items()}
    if tuple(hashes) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Dense residual protocol input identity/order drifted.")
    for artifact_id, digest in hashes.items():
        _require_hash(digest, f"input artifact {artifact_id}")
    _require_hash(validation_cache_binding_hash, "validation-cache binding hash")
    payload: dict[str, object] = {
        "schema_version": "midogpp_dense_residual_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "input_artifact_hashes": hashes,
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "validation_manifest_sha256": config.expected_manifest_sha256,
        "protocol": dict(config.protocol),
        "compatibility": dict(config.compatibility),
        "router": dict(config.router),
        "classifier": config.classifier.to_payload(),
        "selection": dict(config.selection),
        "runtime_budget": dict(config.runtime),
        **_NON_ADOPTIVE_FLAGS,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def action_library_payload(config: DenseResidualDiagnosticConfig) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    actions = [action.to_payload() for action in action_library()]
    if config.router.get("actions") != actions:
        raise ProtocolError("Dense residual config action library drifted.")
    return {
        "schema_version": "midogpp_dense_residual_action_library_v1",
        "experiment_id": EXPERIMENT_ID,
        "config_contract_hash": config.contract_hash,
        "action_library_hash": ACTION_LIBRARY_HASH,
        "actions": actions,
        "library_frozen_before_any_development_label_access": True,
        **_NON_ADOPTIVE_FLAGS,
    }


def phase_01_support_and_compatibility_payload(
    config: DenseResidualDiagnosticConfig,
    *,
    support_partition_lock_hash: str,
    compatibility_index_hash: str,
    support_partition_row_count: int,
    compatibility_score_row_count: int,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    _require_hash(support_partition_lock_hash, "support partition-lock hash")
    _require_hash(compatibility_index_hash, "compatibility-index hash")
    _positive_count(support_partition_row_count, "support partition row count")
    _positive_count(compatibility_score_row_count, "compatibility score row count")
    return _phase_payload(
        phase="PHASE_01_SUPPORT_AND_COMPATIBILITY_COMPLETE",
        extra={
            "support_partition_lock_hash": support_partition_lock_hash,
            "compatibility_index_hash": compatibility_index_hash,
            "support_partition_row_count": support_partition_row_count,
            "compatibility_score_row_count": compatibility_score_row_count,
            "manifest_labels_opened": False,
            "support_labels_opened": False,
            "target_labels_opened": False,
        },
    )


def phase_02_development_complete_payload(
    config: DenseResidualDiagnosticConfig,
    *,
    development_seals: Sequence[DevelopmentPredictionSeal],
    all_action_target_seal: AllActionTargetPredictionSeal,
    decision_seals: Sequence[DiagnosticDecisionSeal],
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    development = tuple(development_seals)
    decisions = tuple(decision_seals)
    _require_outer_target_coverage(development, "development seals")
    _require_outer_target_coverage(decisions, "decision seals")
    all_action_target_seal.verify_complete()
    development_hash = stable_hash([seal.to_payload() for seal in development])
    decision_hash = stable_hash([seal.to_payload() for seal in decisions])
    return _phase_payload(
        phase="PHASE_02_DEVELOPMENT_COMPLETE",
        extra={
            "development_prediction_seals_hash": development_hash,
            "prelabel_all_action_target_prediction_seal_hash": (
                all_action_target_seal.seal_hash
            ),
            "diagnostic_decision_seals_hash": decision_hash,
            "development_prediction_cell_count": sum(
                seal.cell_count for seal in development
            ),
            "all_action_predictions_sealed_before_development_labels": True,
            "all_action_target_predictions_materialized_before_any_label_access": True,
            "development_labels_opened_for_q_not_H_only": True,
            "target_labels_opened": False,
        },
    )


def phase_03_target_predictions_complete_payload(
    config: DenseResidualDiagnosticConfig,
    *,
    target_seals: Sequence[TargetPredictionSeal],
    all_action_target_seal: AllActionTargetPredictionSeal,
    all_action_prediction_cell_count: int,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    seals = tuple(target_seals)
    _require_outer_target_coverage(seals, "target seals")
    all_action_target_seal.verify_complete()
    _positive_count(
        all_action_prediction_cell_count,
        "all-action target prediction cell count",
    )
    return _phase_payload(
        phase="PHASE_03_TARGET_PREDICTIONS_COMPLETE",
        extra={
            "target_prediction_seals_hash": stable_hash(
                [seal.to_payload() for seal in seals]
            ),
            "prelabel_all_action_target_prediction_seal_hash": (
                all_action_target_seal.seal_hash
            ),
            "sealed_selected_and_control_prediction_cell_count": sum(
                seal.cell_count for seal in seals
            ),
            "prelabel_all_action_prediction_cell_count": (
                all_action_prediction_cell_count
            ),
            "selected_and_control_predictions_sealed": True,
            "selected_and_control_are_views_of_prelabel_all_action_predictions": True,
            "target_labels_opened": False,
        },
    )


def phase_04_scoring_complete_payload(
    config: DenseResidualDiagnosticConfig,
    *,
    target_metrics_sha256: str,
    paired_deltas_sha256: str,
    target_metric_row_count: int,
    paired_delta_row_count: int,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    _require_hash(target_metrics_sha256, "target-metrics SHA-256")
    _require_hash(paired_deltas_sha256, "paired-deltas SHA-256")
    _positive_count(target_metric_row_count, "target metric row count")
    _positive_count(paired_delta_row_count, "paired delta row count")
    return _phase_payload(
        phase="PHASE_04_SCORING_COMPLETE",
        extra={
            "target_metrics_sha256": target_metrics_sha256,
            "paired_deltas_sha256": paired_deltas_sha256,
            "target_metric_row_count": target_metric_row_count,
            "paired_delta_row_count": paired_delta_row_count,
            "target_labels_opened_after_selected_and_control_prediction_seals": True,
            "labels_persisted": False,
        },
    )


def label_access_report_payload(
    *,
    development_label_vector_hash_by_outer_target: Mapping[str, str],
    target_label_vector_hash_by_outer_target: Mapping[str, str],
    consumed_row_count: int,
    consumed_case_count: int,
) -> dict[str, object]:
    development = {
        str(key): str(value)
        for key, value in development_label_vector_hash_by_outer_target.items()
    }
    target = {
        str(key): str(value)
        for key, value in target_label_vector_hash_by_outer_target.items()
    }
    _require_center_hashes(development, "development label vectors")
    _require_center_hashes(target, "target label vectors")
    _positive_count(consumed_row_count, "consumed row count")
    _positive_count(consumed_case_count, "consumed case count")
    return {
        "schema_version": "midogpp_dense_residual_label_access_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "CONSUMED_FOR_STAGE90_DIAGNOSTIC_ROUTER_PROTOTYPING",
        "development_label_vector_hash_by_outer_target": development,
        "target_label_vector_hash_by_outer_target": target,
        "unique_consumed_validation_row_count": consumed_row_count,
        "unique_consumed_validation_case_count": consumed_case_count,
        "whole_label_column_loaded": False,
        "nonrequested_rows_skipped_before_label_access": True,
        "support_labels_opened": False,
        "development_labels_opened_after_all_action_prediction_seals": True,
        "all_action_target_predictions_materialized_before_development_labels": True,
        "development_labels_restricted_to_q_not_H": True,
        "target_labels_opened_after_selected_and_control_prediction_seals": True,
        **_NON_ADOPTIVE_FLAGS,
    }


def leakage_report_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_dense_residual_leakage_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "source_experts_trained_source_only": True,
        "outer_target_expert_used_in_development": False,
        "pseudo_target_expert_used_for_own_development_query": False,
        "support_evaluation_case_overlap": 0,
        "support_evaluation_sample_overlap": 0,
        "support_labels_used": False,
        "development_labels_available_to_prediction": False,
        "target_labels_available_to_selection": False,
        "target_labels_available_to_prediction": False,
        "all_action_target_predictions_materialized_before_any_label_access": True,
        "seed_selection_performed": False,
        "action_library_modified_after_labels": False,
        "stage60_policy_modified": False,
        "stage70_policy_or_scores_modified": False,
        **_NON_ADOPTIVE_FLAGS,
    }


def publication_decision_payload(
    *,
    descriptive_summary_hash: str,
) -> dict[str, object]:
    _require_hash(descriptive_summary_hash, "descriptive summary hash")
    return {
        "schema_version": "midogpp_dense_residual_publication_decision_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "decision": PUBLICATION_STATUS,
        "descriptive_summary_hash": descriptive_summary_hash,
        "allowed_interpretation": (
            "consumed_validation_mechanism_diagnostic_for_future_data_collection"
        ),
        "forbidden_interpretations": [
            "fresh_confirmation",
            "routing_quality_established",
            "equal_union_beaten",
            "stage60_policy_authorization",
            "stage70_evaluation_authorization",
            "promotion",
            "deployment",
        ],
        **_NON_ADOPTIVE_FLAGS,
    }


def run_state_payload(status: str) -> dict[str, object]:
    if status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise ProtocolError(f"Invalid dense residual run state: {status!r}.")
    return {
        "schema_version": "midogpp_dense_residual_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "status": status,
        **_NON_ADOPTIVE_FLAGS,
    }


def assert_non_adoptive_payload(payload: Mapping[str, object]) -> None:
    """Reject any bundle payload that relaxes the Stage-90 claim firewall."""

    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in _NON_ADOPTIVE_FLAGS.items()
        if (
            payload.get(key) is not expected
            if isinstance(expected, bool)
            else payload.get(key) != expected
        )
    }
    if mismatches:
        raise ProtocolError(
            "Dense residual bundle contains an adoptive or fresh-evidence claim: "
            f"{mismatches!r}."
        )


def _phase_payload(*, phase: str, extra: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_dense_residual_phase_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "COMPLETE",
        **dict(extra),
        **_NON_ADOPTIVE_FLAGS,
    }
    payload["phase_hash"] = stable_hash(payload)
    return payload


def _assert_config_non_adoptive(config: DenseResidualDiagnosticConfig) -> None:
    if dict(config.claim_boundary) != canonical_claim_boundary_payload():
        raise ProtocolError("Dense residual config claim boundary is not canonical.")


def _require_outer_target_coverage(values: Sequence[object], role: str) -> None:
    centers = [str(getattr(value, "outer_target", "")) for value in values]
    from .contracts import CENTERS

    if tuple(centers) != CENTERS:
        raise ProtocolError(f"Dense residual {role} outer-target coverage drifted.")


def _require_center_hashes(values: Mapping[str, str], role: str) -> None:
    from .contracts import CENTERS

    if tuple(values) != CENTERS:
        raise ProtocolError(f"Dense residual {role} center coverage drifted.")
    for digest in values.values():
        _require_hash(digest, role)


def _require_hash(value: str, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in {16, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Dense residual {role} is malformed.")


def _positive_count(value: int, role: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"Dense residual {role} must be a positive integer.")


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "DEVELOPMENT_PREDICTION_ARRAY_MEMBER",
    "REQUIRED_FILES",
    "TARGET_PREDICTION_ARRAY_MEMBER",
    "action_library_payload",
    "assert_non_adoptive_payload",
    "label_access_report_payload",
    "leakage_report_payload",
    "phase_01_support_and_compatibility_payload",
    "phase_02_development_complete_payload",
    "phase_03_target_predictions_complete_payload",
    "phase_04_scoring_complete_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
