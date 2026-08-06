"""Closed-world artifact contract for the Stage-90 local-utility diagnostic."""

from __future__ import annotations

from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import (
    LocalMarginalUtilityRouterConfig,
    canonical_claim_boundary_payload,
)
from .contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_MARGINAL_UTILITY_ROW_COUNT,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PERTURBATION_LIBRARY_HASH,
    PUBLICATION_STATUS,
    perturbation_library_payloads,
)
from .prediction_io import DEVELOPMENT_ARRAY_MEMBER
from .seals import GlobalDevelopmentPredictionSeal


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/perturbation_library.json",
    "manifests/support_partition_lock.json",
    "manifests/compatibility_index.json",
    "manifests/global_development_prediction_seal.json",
    "manifests/content_index.json",
    DEVELOPMENT_ARRAY_MEMBER,
    "tables/support_partitions.csv",
    "tables/compatibility_case_energy.csv",
    "tables/compatibility_scores.csv",
    "tables/development_prediction_index.csv",
    "tables/development_metrics.csv",
    "tables/marginal_utilities.csv",
    "tables/loqdo_predictions.csv",
    "tables/loqdo_summary.csv",
    "tables/model_fits.csv",
    "tables/target_plans.csv",
    "reports/phase_01_support_and_compatibility_complete.json",
    "reports/phase_02_global_predictions_sealed.json",
    "reports/phase_03_utility_surface_complete.json",
    "reports/phase_04_model_and_plans_complete.json",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/learnability_report.json",
    "reports/optimizer_report.json",
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
    config: LocalMarginalUtilityRouterConfig,
    *,
    input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    hashes = {str(key): str(value) for key, value in input_artifact_hashes.items()}
    if tuple(hashes) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Local-utility protocol input identity/order drifted.")
    for artifact_id, digest in hashes.items():
        _require_hash(digest, f"input artifact {artifact_id}")
    _require_hash(validation_cache_binding_hash, "validation-cache binding hash")
    payload: dict[str, object] = {
        "schema_version": "midogpp_local_marginal_utility_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "input_artifact_hashes": hashes,
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "validation_manifest_sha256": config.expected_manifest_sha256,
        "protocol": dict(config.protocol),
        "compatibility": dict(config.compatibility),
        "perturbations": dict(config.perturbations),
        "classifier": config.classifier.to_payload(),
        "model": dict(config.model),
        "optimizer": dict(config.optimizer),
        "runtime_budget": dict(config.runtime),
        **_NON_ADOPTIVE_FLAGS,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def perturbation_library_payload(
    config: LocalMarginalUtilityRouterConfig,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    actions = list(perturbation_library_payloads())
    if config.perturbations.get("perturbation_library_hash") != (
        PERTURBATION_LIBRARY_HASH
    ):
        raise ProtocolError("Local-utility config perturbation library drifted.")
    return {
        "schema_version": "midogpp_local_marginal_utility_library_v1",
        "experiment_id": EXPERIMENT_ID,
        "config_contract_hash": config.contract_hash,
        "perturbation_library_hash": PERTURBATION_LIBRARY_HASH,
        "action_count": len(actions),
        "actions": actions,
        "library_frozen_before_prediction_generation": True,
        "library_frozen_before_any_development_label_access": True,
        **_NON_ADOPTIVE_FLAGS,
    }


def phase_01_support_and_compatibility_payload(
    config: LocalMarginalUtilityRouterConfig,
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


def phase_02_global_predictions_sealed_payload(
    config: LocalMarginalUtilityRouterConfig,
    *,
    seal: GlobalDevelopmentPredictionSeal,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    seal.verify_complete()
    if seal.config_contract_hash != config.contract_hash:
        raise ProtocolError("Local-utility seal/config binding drifted.")
    if seal.cell_count != EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT:
        raise ProtocolError("Local-utility global prediction count drifted.")
    return _phase_payload(
        phase="PHASE_02_GLOBAL_PREDICTIONS_SEALED",
        extra={
            "global_development_prediction_seal_hash": seal.seal_hash,
            "development_prediction_cell_count": seal.cell_count,
            "all_global_development_predictions_materialized": True,
            "global_predictions_sealed_before_any_development_labels": True,
            "development_labels_opened": False,
            "target_labels_opened": False,
        },
    )


def phase_03_utility_surface_complete_payload(
    config: LocalMarginalUtilityRouterConfig,
    *,
    global_prediction_seal_hash: str,
    development_metrics_sha256: str,
    marginal_utilities_sha256: str,
    development_metric_row_count: int,
    marginal_utility_row_count: int,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    for role, value in (
        ("global prediction seal hash", global_prediction_seal_hash),
        ("development metrics SHA-256", development_metrics_sha256),
        ("marginal utilities SHA-256", marginal_utilities_sha256),
    ):
        _require_hash(value, role)
    _positive_count(development_metric_row_count, "development metric row count")
    if marginal_utility_row_count != EXPECTED_MARGINAL_UTILITY_ROW_COUNT:
        raise ProtocolError("Local-utility marginal row count drifted.")
    return _phase_payload(
        phase="PHASE_03_UTILITY_SURFACE_COMPLETE",
        extra={
            "global_development_prediction_seal_hash": global_prediction_seal_hash,
            "development_metrics_sha256": development_metrics_sha256,
            "marginal_utilities_sha256": marginal_utilities_sha256,
            "development_metric_row_count": development_metric_row_count,
            "marginal_utility_row_count": marginal_utility_row_count,
            "response_is_paired_boost_minus_control_divided_by_epsilon": True,
            "labels_persisted": False,
            "target_labels_opened_for_target_scoring": False,
        },
    )


def phase_04_model_and_plans_complete_payload(
    config: LocalMarginalUtilityRouterConfig,
    *,
    loqdo_predictions_sha256: str,
    loqdo_summary_sha256: str,
    model_fits_sha256: str,
    target_plans_sha256: str,
    learnability_report_sha256: str,
    optimizer_report_sha256: str,
    loqdo_prediction_row_count: int,
    target_plan_row_count: int,
) -> dict[str, object]:
    _assert_config_non_adoptive(config)
    for role, value in (
        ("LOQDO predictions SHA-256", loqdo_predictions_sha256),
        ("LOQDO summary SHA-256", loqdo_summary_sha256),
        ("model fits SHA-256", model_fits_sha256),
        ("target plans SHA-256", target_plans_sha256),
        ("learnability report SHA-256", learnability_report_sha256),
        ("optimizer report SHA-256", optimizer_report_sha256),
    ):
        _require_hash(value, role)
    _positive_count(loqdo_prediction_row_count, "LOQDO prediction row count")
    if target_plan_row_count != len(CENTERS):
        raise ProtocolError("Local-utility target-plan coverage drifted.")
    return _phase_payload(
        phase="PHASE_04_MODEL_AND_PLANS_COMPLETE",
        extra={
            "loqdo_predictions_sha256": loqdo_predictions_sha256,
            "loqdo_summary_sha256": loqdo_summary_sha256,
            "model_fits_sha256": model_fits_sha256,
            "target_plans_sha256": target_plans_sha256,
            "learnability_report_sha256": learnability_report_sha256,
            "optimizer_report_sha256": optimizer_report_sha256,
            "loqdo_prediction_row_count": loqdo_prediction_row_count,
            "target_plan_row_count": target_plan_row_count,
            "target_H_labels_used_for_target_plans": False,
            "target_predictions_materialized": False,
            "target_performance_scored": False,
        },
    )


def label_access_report_payload(
    *,
    label_vector_hash_by_query_center: Mapping[str, str],
    consumed_row_count: int,
    consumed_case_count: int,
) -> dict[str, object]:
    hashes = {
        str(query): str(value)
        for query, value in label_vector_hash_by_query_center.items()
    }
    _require_center_hashes(hashes, "query label vectors")
    _positive_count(consumed_row_count, "consumed row count")
    _positive_count(consumed_case_count, "consumed case count")
    return {
        "schema_version": "midogpp_local_marginal_utility_label_access_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "CONSUMED_FOR_STAGE90_LOCAL_UTILITY_DIAGNOSTIC",
        "label_vector_hash_by_query_center": hashes,
        "unique_consumed_validation_row_count": consumed_row_count,
        "unique_consumed_validation_case_count": consumed_case_count,
        "whole_label_column_loaded": False,
        "nonrequested_rows_skipped_before_label_access": True,
        "support_labels_opened": False,
        "development_labels_opened_after_global_prediction_seal": True,
        "target_H_label_vector_excluded_from_each_target_plan_fit": True,
        "target_labels_opened_for_target_scoring": False,
        **_NON_ADOPTIVE_FLAGS,
    }


def leakage_report_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_local_marginal_utility_leakage_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "source_experts_trained_source_only": True,
        "outer_target_expert_used_in_development": False,
        "pseudo_target_expert_used_for_own_development_query": False,
        "support_evaluation_case_overlap": 0,
        "support_evaluation_sample_overlap": 0,
        "support_labels_used": False,
        "development_labels_available_to_prediction": False,
        "all_global_development_predictions_sealed_before_label_access": True,
        "target_H_labels_used_for_target_plan": False,
        "target_labels_used_for_target_scoring": False,
        "seed_selection_performed": False,
        "perturbation_library_modified_after_labels": False,
        "stage60_policy_modified": False,
        "stage70_policy_or_scores_modified": False,
        **_NON_ADOPTIVE_FLAGS,
    }


def publication_decision_payload(*, descriptive_summary_hash: str) -> dict[str, object]:
    _require_hash(descriptive_summary_hash, "descriptive summary hash")
    return {
        "schema_version": "midogpp_local_marginal_utility_publication_decision_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "decision": PUBLICATION_STATUS,
        "descriptive_summary_hash": descriptive_summary_hash,
        "allowed_interpretation": (
            "consumed_validation_local_utility_learnability_diagnostic_and_"
            "unscored_target_plan_prototype"
        ),
        "forbidden_interpretations": [
            "fresh_confirmation",
            "routing_quality_established",
            "target_performance_established",
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
        raise ProtocolError(f"Invalid local-utility run state: {status!r}.")
    return {
        "schema_version": "midogpp_local_marginal_utility_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "status": status,
        **_NON_ADOPTIVE_FLAGS,
    }


def assert_non_adoptive_payload(payload: Mapping[str, object]) -> None:
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
            "Local-utility bundle contains an adoptive or fresh-evidence claim: "
            f"{mismatches!r}."
        )


def _phase_payload(*, phase: str, extra: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_local_marginal_utility_phase_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "COMPLETE",
        **dict(extra),
        **_NON_ADOPTIVE_FLAGS,
    }
    payload["phase_hash"] = stable_hash(payload)
    return payload


def _assert_config_non_adoptive(config: LocalMarginalUtilityRouterConfig) -> None:
    if dict(config.claim_boundary) != canonical_claim_boundary_payload():
        raise ProtocolError("Local-utility config claim boundary is not canonical.")


def _require_center_hashes(values: Mapping[str, str], role: str) -> None:
    if tuple(values) != CENTERS:
        raise ProtocolError(f"Local-utility {role} center coverage drifted.")
    for digest in values.values():
        _require_hash(digest, role)


def _require_hash(value: str, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in {16, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Local marginal-utility {role} is malformed.")


def _positive_count(value: int, role: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(
            f"Local marginal-utility {role} must be a positive integer."
        )


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_non_adoptive_payload",
    "label_access_report_payload",
    "leakage_report_payload",
    "perturbation_library_payload",
    "phase_01_support_and_compatibility_payload",
    "phase_02_global_predictions_sealed_payload",
    "phase_03_utility_surface_complete_payload",
    "phase_04_model_and_plans_complete_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
