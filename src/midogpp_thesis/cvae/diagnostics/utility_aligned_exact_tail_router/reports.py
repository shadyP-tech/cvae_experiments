"""Canonical reports for the terminal consumed-data diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .contracts import (
    CLAIM_SCOPE,
    DATASET_FAMILY,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    ROUTING_STATUS,
    STAGE_ID,
)


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "consumed_validation_data": True,
        "terminal_diagnostic": True,
        "routing_status": ROUTING_STATUS,
        "support_case_count": 2,
        "fresh_policy_minimum_support_case_count": 8,
        "strict_H_q_e_exclusion": True,
        "development_predictions_sealed_before_development_labels": True,
        "development_crossfit_labels_opened_before_target_action_lock": True,
        "outer_H_development_rows_excluded_from_plan_H": True,
        "target_predictions_sealed_before_terminal_target_scoring": True,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    return {**unhashed, "protocol_manifest_hash": stable_hash(unhashed)}


def phase_completion_payload(
    phase: str,
    *,
    config_contract_hash: str,
    bindings: Mapping[str, object],
    counts: Mapping[str, int],
    development_labels_opened: bool,
    terminal_target_scoring_opened: bool,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_phase_completion_v1",
        "phase": phase,
        "status": "COMPLETE",
        "config_contract_hash": config_contract_hash,
        "bindings": dict(bindings),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "development_labels_opened": development_labels_opened,
        "terminal_target_scoring_opened": terminal_target_scoring_opened,
        "diagnostic_only": True,
    }
    return {**unhashed, "phase_hash": stable_hash(unhashed)}


def leakage_report_payload(
    *,
    support_partition_lock_hash: str,
    case_fold_lock_hash: str,
    development_prediction_seal_hash: str,
    model_set_hash: str,
    plan_set_hash: str,
    action_library_hash: str,
    target_prediction_seal_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_leakage_report_v1",
        "status": "PASS",
        "support_partition_lock_hash": support_partition_lock_hash,
        "case_fold_lock_hash": case_fold_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "model_set_hash": model_set_hash,
        "plan_set_hash": plan_set_hash,
        "action_library_hash": action_library_hash,
        "target_prediction_seal_hash": target_prediction_seal_hash,
        "pre_gpu_firewall_status": firewall.get("status"),
        "support_evaluation_case_disjoint": True,
        "strict_H_q_e_exclusion": True,
        "heldout_H_labels_used_for_H_model_fit": False,
        "support_labels_used_for_route": False,
        "other_evaluation_embeddings_used_for_route": False,
        "development_labels_opened_only_after_global_development_seal": True,
        "development_crossfit_labels_previously_opened": True,
        "outer_H_development_rows_excluded_from_plan_H": True,
        "terminal_target_scoring_capability_opened_after_target_seal": True,
        "prediction_or_action_update_after_target_labels": False,
        "previous_stage90_outputs_used": False,
        "stage60_or_stage70_policy_outputs_used": False,
        "diagnostic_only": True,
    }


def development_label_access_payload(labels: object) -> dict[str, object]:
    """Persist only hashes/capability facts from the sealed cross-fit label phase."""

    return {
        "schema_version": "midogpp_utility_aligned_stage90_development_label_access_v1",
        "status": "PASS",
        "prediction_seal_hash": str(getattr(labels, "prediction_seal_hash")),
        "manifest_sha256": str(getattr(labels, "manifest_sha256")),
        "capability_hash": str(getattr(labels, "capability_hash")),
        "evaluation_row_hash_by_center": dict(
            getattr(labels, "evaluation_row_hash_by_center")
        ),
        "label_hash_by_center": dict(getattr(labels, "label_hash_by_center")),
        "labels_by_center_persisted": False,
        "support_labels_opened": False,
        "development_predictions_globally_sealed_first": True,
        "development_crossfit_labels_opened_before_target_action_lock": True,
        "outer_H_rows_excluded_from_model_and_plan_H": True,
        "diagnostic_only": True,
    }


def scoring_summary_payload(
    ensemble_rows: Sequence[Mapping[str, object]],
    inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = {
        str(row["contrast_id"]): {
            "mean_paired_bacc_delta": float(row["mean_paired_bacc_delta"]),
            "ci95_lower": float(row["ci95_lower"]),
            "ci95_upper": float(row["ci95_upper"]),
        }
        for row in inference_rows
        if row.get("contrast_role") == "primary_predeclared"
    }
    return {
        "schema_version": "midogpp_utility_aligned_stage90_scoring_summary_v1",
        "ensemble_metric_row_count": len(ensemble_rows),
        "contrast_inference_row_count": len(inference_rows),
        "oracle_target_count": len(oracle_rows),
        "primary_contrasts": primary,
        "R2_exact_top1_agreement_count": sum(
            bool(row["R2_top1_exact_agreement"]) for row in oracle_rows
        ),
        "R2_tie_top1_agreement_count": sum(
            bool(row["R2_top1_tie_agreement"]) for row in oracle_rows
        ),
        "mean_R2_normalized_oracle_gap": (
            sum(float(row["normalized_oracle_gap"]) for row in oracle_rows)
            / float(len(oracle_rows))
            if oracle_rows
            else None
        ),
        "inference_unit": "target_center",
        "center_count": 9,
        "technical_seed_cells_are_independent_units": False,
        "consumed_data_diagnostic_only": True,
    }


def publication_decision_payload(
    scoring_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_publication_decision_v1",
        "decision": "DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": PUBLICATION_STATUS,
        "routing_status": ROUTING_STATUS,
        "reason": "consumed_validation_and_two_support_cases_below_fresh_policy_minimum",
        "scoring_summary_hash": stable_hash(dict(scoring_summary)),
        "promising_signal_may_be_reported_descriptively": True,
        "routing_quality_claimed": False,
        "target_specific_router_success_claimed": False,
        "downstream_utility_claimed": False,
        "fresh_confirmation_claimed": False,
        "promotion_eligible": False,
        "policy_update_authorized": False,
        "fallback_authorized": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def runtime_summary_payload(
    preflight: Mapping[str, object],
    *,
    counts: Mapping[str, int],
    source_cache_staging: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_runtime_summary_v1",
        "workstation_preflight": dict(preflight),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "source_cache_staging": dict(source_cache_staging or {
            "attempted": False,
            "used": False,
            "status": "NOT_REQUESTED",
        }),
        "two_persistent_gpu_workers": True,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "float32_memmap_source_cache": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "hash_validated_resume": True,
        "tf32_enabled": False,
        "amp_enabled": False,
    }


def run_state_payload(
    status: str,
    phase: str,
    *,
    error: str | None = None,
) -> dict[str, object]:
    if status not in {"RUNNING", "FAILED", "COMPLETE"}:
        raise ValueError("Utility-aligned run state is invalid.")
    return {
        "schema_version": "midogpp_utility_aligned_stage90_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "promotion_eligible": False,
    }


__all__ = (
    "development_label_access_payload",
    "leakage_report_payload",
    "phase_completion_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "runtime_summary_payload",
    "scoring_summary_payload",
)
