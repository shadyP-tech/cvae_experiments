"""Stable table and report schemas shared by the Stage-60 writer and validator."""

from __future__ import annotations

from typing import Mapping

from ..residual_topup.hashing import canonical_sha256
from .config import CLAIM_SCOPE, EXPERIMENT_ID, ResidualTopupPolicyLockConfig
from .io import ValidatedFreshProxyInputs
from .products import PolicyProducts


POLICY_DECISION = "POLICY_FROZEN_FOR_MATCHED_FRESH_STAGE70_EVALUATION"
PUBLICATION_STATE = "POLICY_LOCK_ONLY_NO_ROUTING_QUALITY_RESULT"


def ballot_table_rows(products: PolicyProducts) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in products.action_library.centers:
        summary = products.summaries_by_target[target]
        for rank_summary in (summary.global_summary, summary.support_summary):
            for ballot in rank_summary.ballots:
                for source in ballot.candidate_sources:
                    rows.append(
                        {
                            "schema_version": "midogpp_residual_topup_proxy_ballot_row_v1",
                            "outer_target": target,
                            "policy_id": rank_summary.policy_id,
                            "query_role": ballot.query_role,
                            "query_center": ballot.query_center,
                            "case_id": ballot.case_id,
                            "candidate_source": source,
                            "training_seeds": "|".join(
                                str(seed) for seed in ballot.training_seeds
                            ),
                            "mean_proxy_energy": ballot.mean_proxy_energy_by_source[source],
                            "normalized_midrank": ballot.normalized_midrank_by_source[source],
                            "labels_consumed": False,
                            "evaluation_overlap": False,
                            "source_expert_updated": False,
                            "proxy_energy_semantics": ballot.proxy_energy_semantics,
                            "replica_aggregation_semantics": (
                                ballot.replica_aggregation_semantics
                            ),
                            "normalized_midrank_semantics": (
                                ballot.normalized_midrank_semantics
                            ),
                        }
                    )
    return rows


def rank_table_rows(products: PolicyProducts) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in products.action_library.centers:
        summary = products.summaries_by_target[target]
        for rank_summary in (summary.global_summary, summary.support_summary):
            for source in rank_summary.candidate_sources:
                rows.append(
                    {
                        "schema_version": "midogpp_residual_topup_proxy_rank_row_v1",
                        "outer_target": target,
                        "policy_id": rank_summary.policy_id,
                        "candidate_source": source,
                        "mean_normalized_midrank": (
                            rank_summary.mean_normalized_midrank_by_source[source]
                        ),
                        "priority": rank_summary.priority_by_source[source],
                        "ballot_count": rank_summary.ballot_count_by_source[source],
                        "query_centers": "|".join(rank_summary.query_centers),
                        "aggregation_semantics": rank_summary.aggregation_semantics,
                        "priority_semantics": rank_summary.priority_semantics,
                        "labels_consumed": False,
                        "evaluation_overlap": False,
                        "source_expert_updated": False,
                    }
                )
    return rows


def action_table_rows(products: PolicyProducts) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in products.action_library.centers:
        for action in products.action_library.actions_by_target[target]:
            for class_label in (0, 1):
                for source in action.source_order:
                    rows.append(
                        {
                            "schema_version": "midogpp_residual_topup_policy_action_row_v1",
                            "outer_target": target,
                            "action_id": action.action_id,
                            "policy_id": action.policy_id,
                            "action_kind": action.action_kind,
                            "candidate_source": source,
                            "class_label": class_label,
                            "base_count": action.base_per_source_per_class,
                            "topup_count": action.topup_counts_by_source[source],
                            "final_count": action.final_counts_by_class[class_label][source],
                            "direction_weight": action.direction_weights_by_source.get(
                                source, 0.0
                            ),
                            "mean_normalized_midrank": (
                                action.mean_normalized_midrank_by_source.get(source, "")
                            ),
                            "permuted_source_identity": (
                                action.source_identity_permutation.get(source, "")
                            ),
                            "selected_source": action.selected_source or "",
                            "diagnostic_control": action.diagnostic_control,
                            "core_action_hash": action.core_action_hash or "",
                            "action_hash": action.action_hash,
                        }
                    )
    return rows


def build_protocol_manifest(
    config: ResidualTopupPolicyLockConfig,
    inputs: ValidatedFreshProxyInputs,
    products: PolicyProducts,
    policy_lock: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_b_u_g_s_stage60_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "policy_lock_hash": policy_lock["policy_lock_hash"],
        "rank_summary_hash": products.rank_summary_hash,
        "action_library_hash": products.action_library.action_library_hash,
        "fresh_surface_attestation_hash": inputs.attestation.attestation_hash,
        "proxy_score_table_sha256": inputs.proxy_score_table_sha256,
        "dataset_family": "MIDOG++",
        "representation_id": "annotation_jpeg_fixed_center_b_v3",
        "training_seeds": list(config.training_seeds),
        "replicas_averaged_before_case_ballot": True,
        "normalized_true_midranks": True,
        "global_excludes_H_and_q": True,
        "support_uses_target_support_only": True,
        "pseudoquery_support_evaluation_case_disjoint": True,
        "all_actions_frozen_before_stage70": True,
        "proxy_only": True,
        "labels_consumed": False,
        "target_evaluation_used": False,
        "source_experts_updated": False,
        "downstream_outcome_computed": False,
    }
    payload["protocol_hash"] = canonical_sha256(payload)
    return payload


def build_protocol_report(
    policy_lock: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_b_u_g_s_protocol_report_v1",
        "status": "PASS",
        "policy_lock_hash": policy_lock["policy_lock_hash"],
        "protocol_hash": protocol["protocol_hash"],
        "fresh_surface_attestation_validated": True,
        "all_actions_frozen_before_stage70": True,
        "proxy_only": True,
        "labels_accessed": False,
        "target_evaluation_accessed": False,
        "source_experts_updated": False,
    }


def build_leakage_report() -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_b_u_g_s_leakage_v1",
        "status": "PASS",
        "fresh_unconsumed_surface": True,
        "pseudoquery_support_evaluation_case_disjoint": True,
        "outer_target_excluded_from_every_candidate_pool": True,
        "pseudoquery_excluded_from_every_global_ballot": True,
        "support_evaluation_overlap_count": 0,
        "labels_present_in_proxy_table": False,
        "labels_consumed": False,
        "target_evaluation_used": False,
        "source_experts_updated": False,
        "seed_selection_performed": False,
        "hyperparameters_tuned": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "downstream_outcome_computed": False,
    }


def build_policy_decision(
    products: PolicyProducts,
    policy_lock: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_residual_topup_b_u_g_s_policy_decision_v1",
        "decision": POLICY_DECISION,
        "publication_state": PUBLICATION_STATE,
        "claim_scope": CLAIM_SCOPE,
        "policy_lock_hash": policy_lock["policy_lock_hash"],
        "action_library_hash": products.action_library.action_library_hash,
        "routing_quality_claimed": False,
        "downstream_outcome_claimed": False,
        "authorized_next_stage": "fresh_matched_stage70_only_after_validation_pass",
    }


__all__ = (
    "POLICY_DECISION",
    "PUBLICATION_STATE",
    "action_table_rows",
    "ballot_table_rows",
    "build_leakage_report",
    "build_policy_decision",
    "build_protocol_manifest",
    "build_protocol_report",
    "rank_table_rows",
)
