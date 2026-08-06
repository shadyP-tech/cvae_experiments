"""Exact workspace binding for the metadata compatibility output."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import MetadataCompatibilityConfig
from .contracts import (
    CLAIM_SCOPE,
    DOMAIN_MAPPING_MEMBER,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    OUTPUT_SEMANTIC_IDENTITIES,
)


INPUT_IDS = (INPUT_ARTIFACT_ID,)


def validate_production_workspace_binding(config: MetadataCompatibilityConfig) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    metadata_input = workspace.artifacts[INPUT_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]
    if (
        experiment.status != "active"
        or experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or metadata_input.may_feed_deployable_selection is not True
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
        or dict(output.semantic_identities) != OUTPUT_SEMANTIC_IDENTITIES
        or CLAIM_SCOPE not in stage60.get("allowed_claim_scopes", ())
    ):
        raise ProtocolError("Metadata compatibility workspace binding drifted.")

    input_root = workspace.resolve_artifact(INPUT_ARTIFACT_ID)
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID,
            for_output=True,
            require_exists=False,
        ),
        "metadata_mapping_path": input_root / DOMAIN_MAPPING_MEMBER,
    }
    configured = {
        "artifact_root": config.artifact_root,
        "metadata_mapping_path": config.metadata_mapping_path,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Metadata compatibility workspace path drifted: {key}.")


__all__ = ("INPUT_IDS", "validate_production_workspace_binding")
