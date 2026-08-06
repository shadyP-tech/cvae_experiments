"""Exact workspace binding for the Uniform-B v2 GenerationLock."""

from __future__ import annotations

from pathlib import Path

from ...workspace.runtime import MidogppWorkspace
from ..protocol import ProtocolError
from .config import UniformBV2GenerationLockConfig
from .contracts import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
)


INPUT_IDS = (EXPERT_BANK_ARTIFACT_ID,)


def validate_production_workspace_binding(config: UniformBV2GenerationLockConfig) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    source = workspace.artifacts[EXPERT_BANK_ARTIFACT_ID]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage60 = workspace.stages["60_routing_and_composition"]
    stage70 = workspace.stages["70_frozen_policy_downstream"]
    if (
        experiment.status != "active"
        or experiment.stage != "40_prior_and_generation"
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_IDS
        or source.may_feed_deployable_selection is not True
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
        or "routing_evidence" in output.forbidden_reuse
        or "expert_selection_evidence" in output.forbidden_reuse
        or "nelbo_compatibility_evidence" in output.forbidden_reuse
        or CLAIM_SCOPE not in stage60.get("allowed_input_claim_scopes", ())
        or CLAIM_SCOPE not in stage70.get("allowed_input_claim_scopes", ())
    ):
        raise ProtocolError("Uniform-B v2 GenerationLock workspace binding drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
    }
    configured = {
        "artifact_root": config.artifact_root,
        "bank_root": config.bank_root,
    }
    for key, value in expected.items():
        if configured[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Uniform-B v2 GenerationLock input binding drifted: {key}.")


__all__ = ("INPUT_IDS", "validate_production_workspace_binding")
