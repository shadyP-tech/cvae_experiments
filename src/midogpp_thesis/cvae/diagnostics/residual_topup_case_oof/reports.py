"""Deterministic table serialization and report payloads for case-OOF."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    PUBLICATION_STATUS,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
)
from .inference import mechanism_interpretation


PROXY_BALLOT_COLUMNS = (
    "schema_version",
    "outer_target",
    "query_role",
    "query_center",
    "case_id",
    "candidate_sources_json",
    "mean_proxy_energy_by_source_json",
    "normalized_midrank_by_source_json",
    "ballot_hash",
    "training_replicas_averaged_before_ballot",
    "labels_used",
    "evaluation_embeddings_used",
)

PROXY_RANK_COLUMNS = (
    "schema_version",
    "outer_target",
    "query_role",
    "candidate_sources_json",
    "mean_normalized_midrank_by_source_json",
    "priority_by_source_json",
    "ballot_count_by_source_json",
    "rank_hash",
    "fixed_support_only",
    "global_excludes_H_and_q",
    "labels_used",
    "evaluation_embeddings_used",
)

ACTION_PLAN_COLUMNS = (
    "schema_version",
    "plan_ordinal",
    "target_center",
    "action_id",
    "policy_id",
    "action_kind",
    "action_semantics",
    "source_order_json",
    "base_per_source_per_class",
    "topup_total_per_class",
    "final_total_per_class",
    "mean_normalized_midrank_by_source_json",
    "source_identity_permutation_json",
    "selected_source",
    "direction_weights_by_source_json",
    "topup_counts_by_source_json",
    "final_counts_by_class_json",
    "core_action_hash",
    "diagnostic_control",
    "action_hash",
    "labels_used",
    "selector_or_fallback_used",
)

ACTION_ASSIGNMENT_COLUMNS = (
    "schema_version",
    "plan_ordinal",
    "target_center",
    "action_id",
    "class_label",
    "source_center",
    "base_start",
    "base_stop",
    "topup_start",
    "topup_stop",
    "base_count",
    "topup_count",
    "final_count",
    "target_expert_excluded",
    "diagnostic_control",
)


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str,
    pre_gpu_firewall: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "pre_gpu_firewall": dict(pre_gpu_firewall),
        "protocol": dict(getattr(config, "protocol")),
        "actions": dict(getattr(config, "actions")),
        "evaluation": dict(getattr(config, "evaluation")),
        "classifier": getattr(config, "classifier").to_payload(),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "previous_stage90_outputs_used": False,
        "cross_fitted_transductive_diagnostic": False,
        "cross_fitted_fixed_support_diagnostic": True,
    }
    return {**payload, "protocol_manifest_hash": stable_hash(payload)}


def proxy_ballot_rows(
    rank_surface: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for target in CENTERS:
        surface = rank_surface[target]
        for summary in (surface.global_summary, surface.support_summary):
            for ballot in summary.ballots:
                rows.append(
                    {
                        "schema_version": "midogpp_residual_topup_case_oof_proxy_ballot_row_v1",
                        "outer_target": target,
                        "query_role": ballot.query_role,
                        "query_center": ballot.query_center,
                        "case_id": ballot.case_id,
                        "candidate_sources_json": _compact(
                            list(ballot.candidate_sources)
                        ),
                        "mean_proxy_energy_by_source_json": _compact(
                            dict(ballot.mean_proxy_energy_by_source)
                        ),
                        "normalized_midrank_by_source_json": _compact(
                            dict(ballot.normalized_midrank_by_source)
                        ),
                        "ballot_hash": ballot.ballot_hash,
                        "training_replicas_averaged_before_ballot": True,
                        "labels_used": False,
                        "evaluation_embeddings_used": False,
                    }
                )
    return tuple(rows)


def proxy_rank_rows(
    rank_surface: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for target in CENTERS:
        surface = rank_surface[target]
        for summary in (surface.global_summary, surface.support_summary):
            rows.append(
                {
                    "schema_version": "midogpp_residual_topup_case_oof_proxy_rank_row_v1",
                    "outer_target": target,
                    "query_role": summary.query_role,
                    "candidate_sources_json": _compact(
                        list(summary.candidate_sources)
                    ),
                    "mean_normalized_midrank_by_source_json": _compact(
                        dict(summary.mean_normalized_midrank_by_source)
                    ),
                    "priority_by_source_json": _compact(
                        dict(summary.priority_by_source)
                    ),
                    "ballot_count_by_source_json": _compact(
                        dict(summary.ballot_count_by_source)
                    ),
                    "rank_hash": summary.rank_hash,
                    "fixed_support_only": True,
                    "global_excludes_H_and_q": summary.query_role
                    == "global_fixed_support",
                    "labels_used": False,
                    "evaluation_embeddings_used": False,
                }
            )
    return tuple(rows)


def action_plan_rows(
    plan: object,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for target in CENTERS:
        for action in plan.actions_for_target(target):
            payload = action.to_payload()
            rows.append(
                {
                    "schema_version": "midogpp_residual_topup_case_oof_action_plan_row_v1",
                    "plan_ordinal": len(rows),
                    "target_center": target,
                    "action_id": action.action_id,
                    "policy_id": action.policy_id,
                    "action_kind": action.action_kind,
                    "action_semantics": action.action_semantics,
                    "source_order_json": _compact(list(action.source_order)),
                    "base_per_source_per_class": action.base_per_source_per_class,
                    "topup_total_per_class": action.topup_total_per_class,
                    "final_total_per_class": action.final_total_per_class,
                    "mean_normalized_midrank_by_source_json": _compact(
                        dict(action.mean_normalized_midrank_by_source)
                    ),
                    "source_identity_permutation_json": _compact(
                        dict(action.source_identity_permutation)
                    ),
                    "selected_source": action.selected_source or "",
                    "direction_weights_by_source_json": _compact(
                        dict(action.direction_weights_by_source)
                    ),
                    "topup_counts_by_source_json": _compact(
                        dict(action.topup_counts_by_source)
                    ),
                    "final_counts_by_class_json": _compact(
                        payload["final_counts_by_class"]
                    ),
                    "core_action_hash": action.core_action_hash or "",
                    "diagnostic_control": action.diagnostic_control,
                    "action_hash": action.action_hash,
                    "labels_used": False,
                    "selector_or_fallback_used": False,
                }
            )
    return tuple(rows)


def action_assignment_rows(
    plan: object,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for target in CENTERS:
        for action in plan.actions_for_target(target):
            core = action.core_action
            for label in (0, 1):
                for source in action.source_order:
                    topup = int(action.topup_counts_by_source[source])
                    rows.append(
                        {
                            "schema_version": "midogpp_residual_topup_case_oof_action_assignment_v1",
                            "plan_ordinal": ordinal,
                            "target_center": target,
                            "action_id": action.action_id,
                            "class_label": label,
                            "source_center": source,
                            "base_start": 0,
                            "base_stop": 128,
                            "topup_start": 128,
                            "topup_stop": 128 + topup,
                            "base_count": 128,
                            "topup_count": topup,
                            "final_count": int(
                                action.final_counts_by_class[label][source]
                            ),
                            "target_expert_excluded": source != target,
                            "diagnostic_control": action.diagnostic_control,
                        }
                    )
            ordinal += 1
    return tuple(rows)


def scoring_summary_payload(
    ensemble_rows: Sequence[Mapping[str, object]],
    inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_means: dict[str, float] = {}
    for action in (
        BASE_ACTION_ID,
        UNIFORM_ACTION_ID,
        GLOBAL_ACTION_ID,
        SUPPORT_ACTION_ID,
    ):
        values = [
            float(row["bacc"])
            for row in ensemble_rows
            if str(row["action_id"]) == action
        ]
        if len(values) != len(CENTERS):
            raise ProtocolError("Case-OOF primary action summary is incomplete.")
        action_means[action] = float(np.mean(values))
    inference = {
        str(row["contrast_id"]): {
            "mean_bacc_delta": float(row["mean_bacc_delta"]),
            "two_sided_95_ci": [
                float(row["two_sided_95_ci_low"]),
                float(row["two_sided_95_ci_high"]),
            ],
            "one_sided_95_lcb": float(row["one_sided_95_lcb"]),
            "wins": int(row["center_wins"]),
            "ties": int(row["center_ties"]),
            "losses": int(row["center_losses"]),
        }
        for row in inference_rows
    }
    return {
        "schema_version": "midogpp_residual_topup_case_oof_scoring_summary_v1",
        "target_center_count": len(CENTERS),
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "mean_center_equal_bacc_by_primary_action": action_means,
        "contrast_inference": inference,
        "mechanism_interpretation": mechanism_interpretation(inference_rows),
        "mean_support_utility_spearman_defined_only": _defined_mean(
            oracle_rows,
            value_key="support_score_utility_spearman",
            defined_key="spearman_defined",
        ),
        "oracle_top1_agreement_count": sum(
            _truthy(row["top1_agreement"]) for row in oracle_rows
        ),
        "mean_normalized_oracle_gap": float(
            np.mean([float(row["normalized_oracle_gap"]) for row in oracle_rows])
        ),
        "all_nine_seed_cells_retained": True,
        "no_selector_or_fallback": True,
        "fresh_evidence": False,
        "diagnostic_only": True,
    }


def leakage_report_payload(
    *,
    support_partition_lock_hash: str,
    crossfit_fold_lock_hash: str,
    source_cache_lock_hash: str,
    router_plan_lock_hash: str,
    global_prediction_seal_hash: str,
    pre_gpu_firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_case_oof_leakage_report_v1",
        "status": "PASS",
        "support_partition_lock_hash": support_partition_lock_hash,
        "crossfit_fold_lock_hash": crossfit_fold_lock_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "router_plan_lock_hash": router_plan_lock_hash,
        "global_prediction_seal_hash": global_prediction_seal_hash,
        "pre_gpu_firewall": dict(pre_gpu_firewall),
        "fixed_support_evaluation_case_disjoint": True,
        "each_evaluation_case_held_out_exactly_once": True,
        "fixed_support_only_routes": True,
        "other_evaluation_embeddings_used_for_route": False,
        "support_labels_used": False,
        "evaluation_labels_used_before_seal": False,
        "target_expert_excluded": True,
        "global_H_and_q_exclusions_enforced": True,
        "all_B_U_G_S_P_Hxe_predictions_sealed": True,
        "seed_selection_performed": False,
        "selector_or_fallback_performed": False,
        "oracle_Hxe_policy_update_performed": False,
        "previous_stage90_outputs_used_as_inputs": False,
        "cross_fitted_transductive_diagnostic": False,
        "diagnostic_only": True,
    }


def publication_decision_payload(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_case_oof_publication_decision_v1",
        "decision": "PUBLISH_AS_EXPLORATORY_CONSUMED_DATA_CASE_OOF_DIAGNOSTIC_ONLY",
        "publication_status": PUBLICATION_STATUS,
        "primary_result": dict(summary),
        "target_specific_router_success_claimed": False,
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
    if phase not in {
        "phase_01_source_cache_complete",
        "phase_02_all_predictions_sealed",
        "phase_03_terminal_scoring_complete",
    }:
        raise ProtocolError("Unknown case-OOF completion phase.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_phase_completion_v1",
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
    return {
        "schema_version": "midogpp_residual_topup_case_oof_runtime_summary_v1",
        "status": "PASS",
        "workstation_preflight": dict(preflight),
        "source_task_count": source_task_count,
        "source_block_count": source_block_count,
        "prediction_task_count": prediction_task_count,
        "prediction_cell_count": prediction_cell_count,
        "unique_classifier_fit_count": unique_classifier_fit_count,
        "gpu_worker_count": 2,
        "persistent_gpu_worker_per_device": True,
        "classifier_worker_count": 4,
        "classifier_threads_per_worker": 3,
        "tf32_disabled": True,
        "float32_memmaps": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "resume_policy": "hash_validated_source_and_prediction_task_checkpoints",
        "dependency_versions_are_report_only": True,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_case_oof_run_state_v1",
        "status": status,
        "phase": phase,
        "resumable": status != "COMPLETE",
        "error": error,
    }


def _defined_mean(
    rows: Sequence[Mapping[str, object]], *, value_key: str, defined_key: str
) -> float | None:
    values = [
        float(row[value_key]) for row in rows if _truthy(row[defined_key])
    ]
    return None if not values else float(np.mean(values))


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = (
    "ACTION_ASSIGNMENT_COLUMNS",
    "ACTION_PLAN_COLUMNS",
    "PROXY_BALLOT_COLUMNS",
    "PROXY_RANK_COLUMNS",
    "action_assignment_rows",
    "action_plan_rows",
    "leakage_report_payload",
    "phase_completion_payload",
    "protocol_manifest_payload",
    "proxy_ballot_rows",
    "proxy_rank_rows",
    "publication_decision_payload",
    "run_state_payload",
    "runtime_summary_payload",
    "scoring_summary_payload",
)
