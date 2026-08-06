"""Exact workspace binding for the source-inner utility/regret policy."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from ..source_inner_utility.contracts import (
    OUTPUT_SEMANTIC_IDENTITIES as UTILITY_SEMANTIC_IDENTITIES,
)
from .config import UtilityRegretPolicyConfig
from .contracts import (
    CLAIM_SCOPE,
    EQUAL_UNION_ARTIFACT_ID,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    OUTPUT_SEMANTIC_IDENTITIES,
    UTILITY_ARTIFACT_ID,
)


INPUT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_ARTIFACT_ID,
    UTILITY_ARTIFACT_ID,
)


def validate_production_workspace_binding(config: UtilityRegretPolicyConfig) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    inputs = [workspace.artifacts[artifact_id] for artifact_id in INPUT_IDS]
    utility = workspace.artifacts[UTILITY_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]
    stage70 = workspace.stages["70_frozen_policy_downstream"]
    if (
        experiment.status != "active"
        or experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or any(item.may_feed_deployable_selection is not True for item in inputs)
        or dict(utility.semantic_identities) != UTILITY_SEMANTIC_IDENTITIES
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
        or dict(output.semantic_identities) != OUTPUT_SEMANTIC_IDENTITIES
        or CLAIM_SCOPE not in stage60.get("allowed_claim_scopes", ())
        or CLAIM_SCOPE not in stage70.get("allowed_input_claim_scopes", ())
    ):
        raise ProtocolError("Utility/regret policy workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID,
            for_output=True,
            require_exists=False,
        ),
        "bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(
            GENERATION_LOCK_ARTIFACT_ID
        ),
        "equal_union_root": workspace.resolve_artifact(EQUAL_UNION_ARTIFACT_ID),
        "utility_root": workspace.resolve_artifact(UTILITY_ARTIFACT_ID),
    }
    configured = {
        "artifact_root": config.artifact_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
        "equal_union_root": config.equal_union_root,
        "utility_root": config.utility_root,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Utility/regret workspace path drifted: {key}.")


__all__ = ("INPUT_IDS", "validate_production_workspace_binding")
