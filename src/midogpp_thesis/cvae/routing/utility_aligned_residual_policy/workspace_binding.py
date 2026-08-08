"""Workspace activation gate for the utility-aligned Stage-60 policy."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import UtilityAlignedResidualPolicyConfig
from .contracts import EXPERIMENT_ID, INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID


def validate_production_workspace_binding(
    config: UtilityAlignedResidualPolicyConfig,
    *,
    _workspace: MidogppWorkspace | None = None,
) -> None:
    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if experiment.status != "active":
        raise ProtocolError(
            "Utility-aligned policy remains planned pending active fresh target inputs."
        )
    if (
        experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != "routing_and_composition"
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(experiment.input_artifact_ids) != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Utility-aligned policy workspace graph drifted.")
    roots = {
        artifact_id: workspace.resolve_artifact(artifact_id)
        for artifact_id in INPUT_ARTIFACT_IDS
    }
    expected = {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "exact_tail_surface_root": roots[INPUT_ARTIFACT_IDS[0]],
        "equal_union_policy_root": roots[INPUT_ARTIFACT_IDS[1]],
        "target_support_surface_root": roots[INPUT_ARTIFACT_IDS[2]],
        "target_support_parent_reservation_root": roots[INPUT_ARTIFACT_IDS[3]],
        "target_reservation_root": roots[INPUT_ARTIFACT_IDS[4]],
        "metadata_profile_root": roots[INPUT_ARTIFACT_IDS[5]],
    }
    for field, expected_path in expected.items():
        if Path(getattr(config, field)).resolve() != Path(expected_path).resolve():
            raise ProtocolError(f"Utility-aligned policy workspace path drifted: {field}.")


__all__ = ("validate_production_workspace_binding",)
