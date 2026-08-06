"""Exact workspace binding for the source-inner candidate utility output."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import SourceInnerUtilityConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    MANIFEST_MEMBER,
    OUTPUT_ARTIFACT_ID,
    OUTPUT_SEMANTIC_IDENTITIES,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)


INPUT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)


def validate_production_workspace_binding(config: SourceInnerUtilityConfig) -> None:
    """Require the one narrowly authorized source-inner label-consumption graph."""

    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    inputs = [workspace.artifacts[artifact_id] for artifact_id in INPUT_IDS]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]
    if (
        experiment.status != "active"
        or experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or any(item.may_feed_deployable_selection is not True for item in inputs)
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
        or dict(output.semantic_identities) != OUTPUT_SEMANTIC_IDENTITIES
        or CLAIM_SCOPE not in stage60.get("allowed_claim_scopes", ())
        or CLAIM_SCOPE not in stage60.get("allowed_input_claim_scopes", ())
    ):
        raise ProtocolError("Source-inner candidate utility workspace binding drifted.")

    manifest_root = workspace.resolve_artifact(VALIDATION_MANIFEST_ARTIFACT_ID)
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
        "validation_cache_root": workspace.resolve_artifact(
            VALIDATION_CACHE_ARTIFACT_ID
        ),
        "manifest_path": manifest_root / MANIFEST_MEMBER,
    }
    configured = {
        "artifact_root": config.artifact_root,
        "bank_root": config.bank_root,
        "generation_lock_root": config.generation_lock_root,
        "validation_cache_root": config.validation_cache_root,
        "manifest_path": config.manifest_path,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(
                f"Source-inner candidate utility workspace path drifted: {key}."
            )


__all__ = ("INPUT_IDS", "validate_production_workspace_binding")
