"""Workspace authorization gate for the exact-tail Stage-60 producer."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import (
    INPUT_ARTIFACT_IDS,
    STAGE_ID,
    ExactTailUtilitySurfaceConfig,
)
from .contracts import CLAIM_SCOPE, EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from ..metadata_compatibility.contracts import DOMAIN_MAPPING_MEMBER


DEVELOPMENT_MANIFEST_MEMBER = "manifest.csv"
FRESH_ATTESTATION_MEMBER = "manifests/fresh_surface_attestation.json"


def validate_production_workspace_binding(
    config: ExactTailUtilitySurfaceConfig,
    *,
    _workspace: MidogppWorkspace | None = None,
) -> None:
    """Require explicit registry activation before any GPU or label work."""

    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if experiment.status != "active":
        raise ProtocolError(
            "Exact-tail Stage-60 experiment remains status='planned'; fresh "
            "reservation promotion and registry activation are required."
        )
    if (
        experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(experiment.input_artifact_ids) != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Exact-tail production workspace binding drifted.")
    inputs = [workspace.artifacts[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS]
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    if (
        any(item.may_feed_deployable_selection is not True for item in inputs)
        or output.claim_scope != CLAIM_SCOPE
        or output.may_feed_deployable_selection is not True
    ):
        raise ProtocolError("Exact-tail workspace selection authorization drifted.")

    roots = {
        artifact_id: workspace.resolve_artifact(artifact_id)
        for artifact_id in INPUT_ARTIFACT_IDS
    }
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "expert_bank_root": roots[INPUT_ARTIFACT_IDS[0]],
        "generation_lock_root": roots[INPUT_ARTIFACT_IDS[1]],
        "development_reservation_root": roots[INPUT_ARTIFACT_IDS[2]],
        "development_cache_root": roots[INPUT_ARTIFACT_IDS[3]],
        "development_manifest_path": roots[INPUT_ARTIFACT_IDS[4]]
        / DEVELOPMENT_MANIFEST_MEMBER,
        "metadata_profile_root": roots[INPUT_ARTIFACT_IDS[5]],
        "reservation_attestation_path": roots[INPUT_ARTIFACT_IDS[2]]
        / FRESH_ATTESTATION_MEMBER,
    }
    for field, expected_path in expected.items():
        if Path(getattr(config, field)).resolve() != Path(expected_path).resolve():
            raise ProtocolError(f"Exact-tail workspace path drifted: {field}.")


__all__ = (
    "DEVELOPMENT_MANIFEST_MEMBER",
    "FRESH_ATTESTATION_MEMBER",
    "validate_production_workspace_binding",
)
