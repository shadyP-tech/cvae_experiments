"""Active workspace binding for the utility-aligned Stage-70 evaluator."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    UtilityAlignedResidualFreshConfig,
)


STAGE_ID = "70_frozen_policy_downstream"
CLAIM_SCOPE = "synthetic_downstream_utility"


def validate_utility_aligned_residual_fresh_workspace_binding(
    config: UtilityAlignedResidualFreshConfig,
) -> dict[str, object]:
    """Require an active registry entry and exact canonical artifact paths."""

    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(config.experiment_id)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage = workspace.stages[STAGE_ID]
    if (
        experiment.status != "active"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
        or output.stage != STAGE_ID
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not False
        or output.may_feed_recipe_selection is not False
        or CLAIM_SCOPE not in stage.get("allowed_claim_scopes", ())
    ):
        raise ProtocolError("Utility-aligned Stage-70 workspace binding is not active.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "expert_bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(GENERATION_LOCK_ARTIFACT_ID),
        "policy_root": workspace.resolve_artifact(POLICY_ARTIFACT_ID),
        "fresh_reservation_path": workspace.resolve_artifact(RESERVATION_ARTIFACT_ID)
        / "manifests/reservation.json",
        "fresh_target_cache_root": workspace.resolve_artifact(TARGET_CACHE_ARTIFACT_ID),
        "fresh_scoring_manifest_path": workspace.resolve_artifact(
            SCORING_MANIFEST_ARTIFACT_ID
        )
        / "manifest.csv",
    }
    configured = {
        "artifact_root": config.artifact_root,
        "expert_bank_root": config.expert_bank_root,
        "generation_lock_root": config.generation_lock_root,
        "policy_root": config.policy_root,
        "fresh_reservation_path": config.fresh_reservation_path,
        "fresh_target_cache_root": config.fresh_target_cache_root,
        "fresh_scoring_manifest_path": config.fresh_scoring_manifest_path,
    }
    mismatch = [
        key
        for key in expected
        if configured[key].resolve() != Path(expected[key]).resolve()
    ]
    if mismatch:
        raise ProtocolError(
            f"Utility-aligned Stage-70 workspace paths drifted: {mismatch}."
        )
    return {
        "status": "PASS",
        "experiment_id": config.experiment_id,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "workspace_experiment_active": True,
        "consumed_stage70_artifact_used": False,
        "consumed_stage90_artifact_used": False,
    }


__all__ = (
    "CLAIM_SCOPE",
    "STAGE_ID",
    "validate_utility_aligned_residual_fresh_workspace_binding",
)
