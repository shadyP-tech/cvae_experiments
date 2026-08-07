"""Small report builders for the residual top-up diagnostic."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ....common.hashing import stable_hash
from .contracts import (
    BASE_ONLY_ACTION_ID,
    CENTERS,
    ENERGY_TOPUP_ACTION_ID,
    EXPECTED_PREDICTION_CELL_COUNT,
    PUBLICATION_STATUS,
    UNIFORM_TOPUP_ACTION_ID,
)


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str,
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_residual_topup_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "protocol": dict(getattr(config, "protocol")),
        "actions": dict(getattr(config, "actions")),
        "selection": dict(getattr(config, "selection")),
        "classifier": getattr(config, "classifier").to_payload(),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "previous_stage90_router_or_utility_inputs_used": False,
    }
    return {**payload, "protocol_manifest_hash": stable_hash(payload)}


def action_library_payload(config: object) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_residual_topup_action_library_v1",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "action_ids": [BASE_ONLY_ACTION_ID, UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID],
        "development_action_ids": [UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID],
        "primary_matched_control": UNIFORM_TOPUP_ACTION_ID,
        "primary_routed_action": ENERGY_TOPUP_ACTION_ID,
        "base_only_role": "separate_budget_reference",
        "action_contract": dict(getattr(config, "actions")),
        "finite_predeclared_menu": True,
        "labels_used": False,
        "diagnostic_only": True,
    }
    return {**payload, "action_library_hash": stable_hash(payload)}


def scoring_summary_payload(
    target_metrics: Sequence[Mapping[str, object]],
    target_deltas: Sequence[Mapping[str, object]],
    ensemble_metrics: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [row for row in target_metrics if str(row["phase"]) == "target"]
    action_means: dict[str, float] = {}
    f1_means: dict[str, float] = {}
    for action in (BASE_ONLY_ACTION_ID, UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID):
        center_bacc = [
            float(np.mean([float(row["bacc"]) for row in rows if row["outer_target"] == center and row["action_id"] == action]))
            for center in CENTERS
        ]
        center_f1 = [
            float(np.mean([float(row["macro_f1"]) for row in rows if row["outer_target"] == center and row["action_id"] == action]))
            for center in CENTERS
        ]
        action_means[action] = float(np.mean(center_bacc))
        f1_means[action] = float(np.mean(center_f1))
    raw_by_center = np.asarray(
        [
            np.mean(
                [
                    float(row["raw_energy_vs_uniform_bacc_delta"])
                    for row in target_deltas
                    if row["target_center"] == center
                ]
            )
            for center in CENTERS
        ],
        dtype=np.float64,
    )
    selected_by_center = np.asarray(
        [
            np.mean(
                [
                    float(row["selected_vs_uniform_bacc_delta"])
                    for row in target_deltas
                    if row["target_center"] == center
                ]
            )
            for center in CENTERS
        ],
        dtype=np.float64,
    )
    budget_by_center = np.asarray(
        [
            np.mean(
                [
                    float(row["uniform_topup_vs_base_budget_delta"])
                    for row in target_deltas
                    if row["target_center"] == center
                ]
            )
            for center in CENTERS
        ],
        dtype=np.float64,
    )
    interval = _cluster_interval(raw_by_center)
    ensemble_by_action = {
        action: float(
            np.mean(
                [float(row["bacc"]) for row in ensemble_metrics if row["action_id"] == action]
            )
        )
        for action in (BASE_ONLY_ACTION_ID, UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID)
    }
    selected_count = sum(
        str(row["selected_action_id"]) == ENERGY_TOPUP_ACTION_ID
        for row in selections
    )
    return {
        "schema_version": "midogpp_residual_topup_scoring_summary_v1",
        "target_count": len(CENTERS),
        "target_metric_row_count": len(rows),
        "mean_bacc_center_equal_by_action": action_means,
        "mean_macro_f1_center_equal_by_action": f1_means,
        "mean_raw_energy_vs_uniform_bacc_delta_center_equal": float(np.mean(raw_by_center)),
        "raw_energy_vs_uniform_two_sided_95_interval": [interval[0], interval[1]],
        "raw_energy_vs_uniform_one_sided_95_lcb": interval[2],
        "mean_selected_vs_uniform_bacc_delta_center_equal": float(np.mean(selected_by_center)),
        "mean_uniform_topup_vs_base_budget_delta_center_equal": float(np.mean(budget_by_center)),
        "raw_target_center_wins": int(np.sum(raw_by_center > 0.0)),
        "raw_target_center_losses": int(np.sum(raw_by_center < 0.0)),
        "raw_target_center_ties": int(np.sum(raw_by_center == 0.0)),
        "energy_action_selected_target_count": selected_count,
        "uniform_fallback_target_count": len(CENTERS) - selected_count,
        "probability_ensemble_mean_bacc_center_equal_by_action": ensemble_by_action,
        "primary_comparison": "energy_directed_topup_minus_uniform_topup_matched_budget",
        "base_only_role": "separate_budget_reference",
        "all_seed_cells_retained": True,
        "fresh_evidence": False,
        "diagnostic_only": True,
    }


def leakage_report_payload(
    *,
    support_partition_lock_hash: str,
    source_cache_lock_hash: str,
    router_plan_lock_hash: str,
    global_prediction_seal_hash: str,
    calibration_lock_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_leakage_report_v1",
        "status": "PASS",
        "support_partition_lock_hash": support_partition_lock_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "router_plan_lock_hash": router_plan_lock_hash,
        "global_prediction_seal_hash": global_prediction_seal_hash,
        "calibration_lock_hash": calibration_lock_hash,
        "support_evaluation_case_disjoint": True,
        "support_evaluation_sample_disjoint": True,
        "support_labels_used": False,
        "target_expert_excluded": True,
        "outer_H_and_inner_q_experts_excluded": True,
        "target_H_labels_used_for_own_selection": False,
        "all_actions_sealed_before_any_label_access": True,
        "seed_selection_performed": False,
        "previous_stage90_outputs_used_as_inputs": False,
        "diagnostic_only": True,
    }


def publication_decision_payload(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_publication_decision_v1",
        "decision": "PUBLISH_AS_EXPLORATORY_CONSUMED_DATA_DIAGNOSTIC_ONLY",
        "publication_status": PUBLICATION_STATUS,
        "primary_result": dict(summary),
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "oracle_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def phase_completion_payload(
    phase: str,
    *,
    config_contract_hash: str,
    bindings: Mapping[str, object],
    counts: Mapping[str, int],
    labels_opened: bool,
) -> dict[str, object]:
    """Build one deterministic, hash-bound phase completion report."""

    if phase not in {
        "phase_01_source_cache_complete",
        "phase_02_all_actions_sealed",
        "phase_03_calibration_complete",
        "phase_04_scoring_complete",
    }:
        raise ValueError("Unknown residual top-up completion phase.")
    payload = {
        "schema_version": "midogpp_residual_topup_phase_completion_v1",
        "phase": phase,
        "status": "COMPLETE",
        "config_contract_hash": config_contract_hash,
        "bindings": dict(bindings),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "labels_opened": bool(labels_opened),
        "diagnostic_only": True,
    }
    return {**payload, "phase_report_hash": stable_hash(payload)}


def runtime_summary_payload(
    preflight: Mapping[str, object],
    *,
    source_task_count: int,
    source_block_count: int,
    prediction_task_count: int,
    prediction_cell_count: int,
    unique_classifier_fit_count: int,
) -> dict[str, object]:
    """Report the frozen workstation schedule without timing-dependent claims."""

    return {
        "schema_version": "midogpp_residual_topup_runtime_summary_v1",
        "status": "PASS",
        "workstation_preflight": dict(preflight),
        "source_task_count": int(source_task_count),
        "source_block_count": int(source_block_count),
        "prediction_task_count": int(prediction_task_count),
        "prediction_cell_count": int(prediction_cell_count),
        "unique_classifier_fit_count": int(unique_classifier_fit_count),
        "gpu_worker_count": 2,
        "classifier_worker_count": 4,
        "classifier_threads_per_worker": 3,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "resume_policy": (
            "hash_validated_source_and_prediction_task_checkpoints_"
            "with_completed_product_reuse"
        ),
        "dependency_versions_are_report_only": True,
    }


def run_state_payload(status: str, phase: str, *, error: str | None = None) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_run_state_v1",
        "status": status,
        "phase": phase,
        "resumable": status != "COMPLETE",
        "error": error,
    }


def _cluster_interval(values: np.ndarray) -> tuple[float, float, float]:
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    two = float(student_t.ppf(0.975, df=n - 1))
    one = float(student_t.ppf(0.95, df=n - 1))
    return mean - two * se, mean + two * se, mean - one * se


__all__ = (
    "action_library_payload",
    "leakage_report_payload",
    "phase_completion_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "runtime_summary_payload",
    "scoring_summary_payload",
)
