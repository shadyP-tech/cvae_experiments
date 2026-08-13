"""Claim-bound manifests and reports for the terminal DCSE diagnostic."""

from __future__ import annotations

from typing import Mapping

from .constants import ARM_IDS, METHOD_IDS, TERMINAL_DECISION
from .experiment_contracts import CLAIM_ROLE, PUBLICATION_STATUS
from .hashing import canonical_hash, json_native


def protocol_manifest_payload(
    config: object,
    *,
    protocol: object,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_dcse_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol": json_native(getattr(protocol, "to_payload")()),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": json_native(firewall),
        "action_library": json_native(getattr(config, "action_library")),
        "directional_ensemble": json_native(
            getattr(config, "directional_ensemble")
        ),
        "controls": json_native(getattr(config, "controls")),
        "evaluation": json_native(getattr(config, "evaluation")),
        "nulls": json_native(getattr(config, "nulls")),
        "runtime": json_native(getattr(config, "runtime")),
        "claim_boundary": json_native(getattr(config, "claim_boundary")),
        "original_six_inputs_only": True,
        "all_810_physical_cells_rematerialized": True,
        "all_physical_probabilities_sealed_before_labels": True,
        "all_plans_and_decisions_sealed_before_terminal_labels": True,
        "previous_stage90_artifact_checkpoint_or_scratch_consumed": False,
        "fresh_evidence": False,
        "terminal_diagnostic_only": True,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    physical_prelabel_seal_hash: str,
    aggregate_plan_decision_seal_hash: str,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_dcse_leakage_report_v1",
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
        "aggregate_plan_decision_seal_hash": aggregate_plan_decision_seal_hash,
        "all_810_physical_cells_sealed_before_label_access": True,
        "whole_case_loo_plan_count": 218,
        "each_held_case_absent_from_its_support": True,
        "donor_prior_excludes_target_and_source": True,
        "all_218_plans_and_endpoint_decisions_durable_before_terminal_labels": True,
        "terminal_evaluation_labels_opened": capability_report.get("terminal_opened")
        is True,
        "terminal_labels_used_to_train_tune_rank_or_select": False,
        "target_expert_used": False,
        "source_experts_updated": False,
        "shared_model_updated_with_target_labels": False,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "per_case_bacc_persisted_or_used": False,
        "previous_stage90_artifact_checkpoint_or_scratch_consumed": False,
    }


def publication_decision_payload(
    terminal_seal_hash: str,
    *,
    descriptive_success_rubric: Mapping[str, object],
) -> dict[str, object]:
    rubric = json_native(descriptive_success_rubric)
    return {
        "schema_version": "fixed_bank_dcse_publication_decision_v1",
        "decision": TERMINAL_DECISION,
        "publication_status": PUBLICATION_STATUS,
        "claim_role": CLAIM_ROLE,
        "terminal_seal_hash": terminal_seal_hash,
        "method_ids": list(METHOD_IDS),
        "arm_ids": list(ARM_IDS),
        "primary_method_id": "DCSE_LOO",
        "descriptive_success_rubric": rubric,
        "full_DCSE_LOO_minus_B_strictly_positive_required": True,
        "full_DCSE_LOO_minus_U_strictly_positive_required": True,
        "all_nine_delete_one_center_B_and_U_contrasts_positive_required": True,
        "at_least_eight_of_nine_center_B_deltas_nonnegative_required": True,
        "every_leave_one_arm_DCSE_LOO_minus_B_positive_required": True,
        "matched_G_contrast_is_gate": False,
        "nominal_t_interval_is_gate": False,
        "jackknife_interval_is_gate": False,
        "null_summary_is_gate": False,
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "policy_model_expert_or_deployment_update_authorized": False,
        "may_feed_another_experiment": False,
    }


def run_state_payload(
    status: str,
    phase: str,
    *,
    error: str | None = None,
    error_class: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "fixed_bank_dcse_run_state_v1",
        "status": status,
        "phase": phase,
        "terminal_diagnostic_only": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "owned_task_checkpoint_replay_allowed": False,
        "task_checkpoints_are_intra_launch_atomicity_only": True,
    }
    if error is not None:
        if error_class is None:
            raise ValueError("FAILED state requires error_class.")
        payload.update({"error": error, "error_class": error_class})
    elif error_class is not None:
        raise ValueError("error_class requires an error.")
    return payload


def seal_payload(
    schema_version: str,
    *,
    bindings: Mapping[str, object],
    **facts: object,
) -> dict[str, object]:
    unhashed = {
        "schema_version": schema_version,
        "bindings": json_native(bindings),
        **{key: json_native(value) for key, value in facts.items()},
    }
    return {**unhashed, "seal_hash": canonical_hash(unhashed)}


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "seal_payload",
)
