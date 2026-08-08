"""Registry activation and path binding for target-support production."""

from __future__ import annotations

from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import TargetSupportSurfaceConfig
from .contracts import EXPERIMENT_ID, INPUT_ARTIFACT_IDS, OUTPUT_ARTIFACT_ID


def validate_production_workspace_binding(
    config: TargetSupportSurfaceConfig, *, _workspace: MidogppWorkspace | None = None
) -> None:
    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    if experiment.status != "active":
        raise ProtocolError("Target-support producer remains planned pending fresh inputs.")
    if (
        experiment.stage != "60_routing_and_composition"
        or experiment.claim_scope != "routing_compatibility_only"
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(experiment.input_artifact_ids) != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Target-support workspace graph drifted.")
    roots = {value: workspace.resolve_artifact(value) for value in INPUT_ARTIFACT_IDS}
    expected = {
        "artifact_root": workspace.resolve_artifact(OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False),
        "expert_bank_root": roots[INPUT_ARTIFACT_IDS[0]],
        "generation_lock_root": roots[INPUT_ARTIFACT_IDS[1]],
        "reservation_root": roots[INPUT_ARTIFACT_IDS[2]],
        "support_cache_root": roots[INPUT_ARTIFACT_IDS[3]],
        "metadata_profile_root": roots[INPUT_ARTIFACT_IDS[4]],
    }
    for name, value in expected.items():
        if Path(getattr(config, name)).resolve() != Path(value).resolve():
            raise ProtocolError(f"Target-support workspace path drifted: {name}.")


__all__ = ("validate_production_workspace_binding",)
