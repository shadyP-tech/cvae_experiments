"""Exact workspace binding for the Uniform-B v2 equal-union policy lock."""

from __future__ import annotations

from pathlib import Path

from ...workspace.runtime import MidogppWorkspace
from ..protocol import ProtocolError
from .config import UniformBV2EqualUnionPolicyConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPECTED_ASSIGNMENT_TABLE_HASH,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_POLICY_LOCK_HASH,
    EXPECTED_POLICY_PLAN_HASH,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
)


INPUT_IDS = (EXPERT_BANK_ARTIFACT_ID, GENERATION_LOCK_ARTIFACT_ID)


def validate_production_workspace_binding(
    config: UniformBV2EqualUnionPolicyConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    bank = workspace.artifacts[EXPERT_BANK_ARTIFACT_ID]
    generation = workspace.artifacts[GENERATION_LOCK_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]
    stage70 = workspace.stages["70_frozen_policy_downstream"]
    expected_semantic_identities = {
        "policy_lock_contract": "midogpp_uniform_b_v2_equal_union_policy_lock_v1",
        "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
        "policy_lock_hash": EXPECTED_POLICY_LOCK_HASH,
        "policy_plan_hash": EXPECTED_POLICY_PLAN_HASH,
        "assignment_table_hash": EXPECTED_ASSIGNMENT_TABLE_HASH,
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
    }
    if (
        experiment.status != "active"
        or experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or bank.may_feed_deployable_selection is not True
        or generation.may_feed_deployable_selection is not True
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
        or dict(output.semantic_identities) != expected_semantic_identities
        or CLAIM_SCOPE not in stage60.get("allowed_claim_scopes", ())
        or CLAIM_SCOPE not in stage70.get("allowed_input_claim_scopes", ())
    ):
        raise ProtocolError("Uniform-B v2 equal-union policy workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(GENERATION_LOCK_ARTIFACT_ID),
    }
    configured = {
        "artifact_root": config.artifact_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Equal-union policy workspace path drifted: {key}.")


__all__ = ("INPUT_IDS", "validate_production_workspace_binding")
