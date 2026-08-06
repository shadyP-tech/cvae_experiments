"""Exact workspace binding for the metadata tie-union comparison policy."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import UniformBV2MetadataTieUnionPolicyConfig
from .contracts import (
    CLAIM_SCOPE,
    COMPATIBILITY_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    OUTPUT_SEMANTIC_IDENTITIES,
)


INPUT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    COMPATIBILITY_ARTIFACT_ID,
)


def validate_production_workspace_binding(
    config: UniformBV2MetadataTieUnionPolicyConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    inputs = [workspace.artifacts[artifact_id] for artifact_id in INPUT_IDS]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]
    stage70 = workspace.stages["70_frozen_policy_downstream"]
    expected_semantic_identities = OUTPUT_SEMANTIC_IDENTITIES
    if (
        experiment.status != "active"
        or experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or any(item.may_feed_deployable_selection is not True for item in inputs)
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
        or dict(output.semantic_identities) != expected_semantic_identities
        or CLAIM_SCOPE not in stage60.get("allowed_claim_scopes", ())
        or CLAIM_SCOPE not in stage70.get("allowed_input_claim_scopes", ())
    ):
        raise ProtocolError("Metadata tie-union policy workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(GENERATION_LOCK_ARTIFACT_ID),
        "equal_union_policy_root": workspace.resolve_artifact(
            EQUAL_UNION_POLICY_ARTIFACT_ID
        ),
        "metadata_compatibility_root": workspace.resolve_artifact(
            COMPATIBILITY_ARTIFACT_ID
        ),
    }
    configured = {
        "artifact_root": config.artifact_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
        "equal_union_policy_root": config.equal_union_policy_root,
        "metadata_compatibility_root": config.metadata_compatibility_root,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Metadata tie-union workspace path drifted: {key}.")


__all__ = ("INPUT_IDS", "validate_production_workspace_binding")
