"""Pure construction of frozen Stage-60 policy products and lock payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..residual_topup.hashing import canonical_sha256
from .actions import FrozenActionLibrary, build_frozen_action_library
from .config import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    ResidualTopupPolicyLockConfig,
)
from .contracts import TargetProxyPolicySummary
from .io import ValidatedFreshProxyInputs
from .policy import build_proxy_policies_by_target


@dataclass(frozen=True)
class PolicyProducts:
    summaries_by_target: Mapping[str, TargetProxyPolicySummary]
    action_library: FrozenActionLibrary
    rank_summary_hash: str


def build_policy_products(
    config: ResidualTopupPolicyLockConfig,
    inputs: ValidatedFreshProxyInputs,
) -> PolicyProducts:
    summaries = build_proxy_policies_by_target(
        inputs.rows,
        source_centers=config.centers,
        permutation_index_by_target={
            target: config.permutation_index for target in config.centers
        },
    )
    library = build_frozen_action_library(summaries)
    rank_payload = {
        target: summaries[target].to_payload() for target in config.centers
    }
    return PolicyProducts(
        summaries_by_target=summaries,
        action_library=library,
        rank_summary_hash=canonical_sha256(rank_payload),
    )


def build_policy_lock_payload(
    config: ResidualTopupPolicyLockConfig,
    inputs: ValidatedFreshProxyInputs,
    products: PolicyProducts,
    *,
    workspace_binding: Mapping[str, object],
) -> dict[str, object]:
    library_payload = products.action_library.to_payload()
    payload: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_b_u_g_s_policy_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "bank_lock_hash": config.expected_bank_lock_hash,
        "generation_lock_hash": config.expected_generation_lock_hash,
        "equal_union_policy_lock_hash": config.expected_equal_union_policy_lock_hash,
        "proxy_surface_artifact_id": config.proxy_surface_artifact_id,
        "fresh_surface_reservation_id": inputs.attestation.reservation_id,
        "fresh_surface_attestation_hash": inputs.attestation.attestation_hash,
        "fresh_surface_attestation_file_sha256": inputs.attestation_file_sha256,
        "proxy_score_table_sha256": inputs.proxy_score_table_sha256,
        "pseudoquery_case_ids_by_center": {
            center: list(inputs.attestation.pseudoquery_case_ids_by_center[center])
            for center in config.centers
        },
        "support_case_ids_by_target": {
            target: list(inputs.attestation.support_case_ids_by_target[target])
            for target in config.centers
        },
        "evaluation_case_ids_by_target": {
            target: list(inputs.attestation.evaluation_case_ids_by_target[target])
            for target in config.centers
        },
        "workspace_binding": dict(workspace_binding),
        "rank_summary_hash": products.rank_summary_hash,
        "action_library_hash": products.action_library.action_library_hash,
        "action_count": products.action_library.action_count,
        "actions_by_target": library_payload["actions_by_target"],
        "policy_frozen_before_stage70": True,
        "all_main_and_control_actions_frozen": True,
        "all_H_by_e_actions_frozen": True,
        "proxy_only": True,
        "labels_consumed": False,
        "target_evaluation_used": False,
        "source_experts_updated": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "hyperparameters_tuned": False,
        "routing_quality_claimed": False,
        "downstream_outcome_computed": False,
        "may_feed_stage70_only_after_validation_pass": True,
    }
    payload["policy_lock_hash"] = canonical_sha256(payload)
    return payload


__all__ = (
    "PolicyProducts",
    "build_policy_lock_payload",
    "build_policy_products",
)
