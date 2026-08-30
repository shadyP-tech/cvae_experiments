"""Fail-closed workspace authorization for HARP Stage-60 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .config import HarpStage60Config
from .constants import (
    ACTION_SURFACE,
    DEVELOPMENT_MANIFEST_ARTIFACT_ID,
    DEVELOPMENT_RESERVATION_ARTIFACT_ID,
    FRESH_TARGET_RESERVATION_ARTIFACT_ID,
    POLICY_LOCK,
    STAGE_ID,
    TARGET_SUPPORT_RESERVATION_ARTIFACT_ID,
    TARGET_SUPPORT_SURFACE,
)


def validate_harp_production_workspace_binding(
    config: HarpStage60Config,
    *,
    _workspace: MidogppWorkspace | None = None,
) -> None:
    """Require an exact active graph before completed-path reads or mutation."""

    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(config.experiment_id)
    if experiment.status != "active":
        raise ProtocolError(
            f"HARP surface {config.contract.surface} remains status={experiment.status!r}; "
            "fresh input promotion and explicit registry activation are required."
        )
    if (
        experiment.stage != STAGE_ID
        or experiment.claim_scope != config.contract.claim_scope
        or experiment.output_artifact_id != config.output_artifact_id
        or tuple(experiment.input_artifact_ids) != config.input_artifact_ids
        or experiment.preparation_authority_gate is not None
    ):
        raise ProtocolError("HARP production workspace graph drifted.")
    inputs = [workspace.artifacts[artifact_id] for artifact_id in config.input_artifact_ids]
    output = workspace.artifacts[config.output_artifact_id]
    if (
        any(item.may_feed_deployable_selection is not True for item in inputs)
        or output.may_feed_deployable_selection is not True
        or output.claim_scope != config.contract.claim_scope
    ):
        raise ProtocolError("HARP workspace selection eligibility drifted.")
    for artifact in (*inputs, output):
        authorized = {
            value.strip()
            for value in artifact.semantic_identities.get(
                "authorized_consumer_experiment_ids", ""
            ).split("|")
            if value.strip()
        }
        registered = {
            value.strip()
            for value in artifact.semantic_identities.get(
                "registered_consumer_experiment_ids", ""
            ).split("|")
            if value.strip()
        }
        if artifact is output:
            continue
        if authorized and config.experiment_id not in authorized:
            raise ProtocolError("HARP input is fenced to another authorized consumer.")
        if registered and config.experiment_id not in registered:
            raise ProtocolError("HARP planned input is fenced to another consumer.")
    expected = _expected_paths(workspace, config)
    if Path(config.artifact_root).resolve() != expected.pop("artifact_root").resolve():
        raise ProtocolError("HARP canonical output path drifted.")
    for key, expected_path in expected.items():
        if Path(config.input_paths[key]).resolve() != expected_path.resolve():
            raise ProtocolError(f"HARP workspace input path drifted: {key}.")


def _expected_paths(
    workspace: MidogppWorkspace, config: HarpStage60Config
) -> dict[str, Path]:
    roots = {
        artifact_id: workspace.resolve_artifact(artifact_id)
        for artifact_id in config.input_artifact_ids
    }
    expected: dict[str, Path] = {
        "artifact_root": workspace.resolve_artifact(
            config.output_artifact_id, for_output=True, require_exists=False
        )
    }
    if config.contract == ACTION_SURFACE:
        ids = config.input_artifact_ids
        expected.update(
            {
                "expert_bank_root": roots[ids[0]],
                "generation_lock_root": roots[ids[1]],
                "development_reservation_root": roots[ids[2]],
                "development_cache_root": roots[ids[3]],
                "development_manifest_path": roots[DEVELOPMENT_MANIFEST_ARTIFACT_ID]
                / "manifest.csv",
                "readiness_attestation_path": roots[DEVELOPMENT_RESERVATION_ARTIFACT_ID]
                / ACTION_SURFACE.readiness_member,
            }
        )
    elif config.contract == TARGET_SUPPORT_SURFACE:
        ids = config.input_artifact_ids
        expected.update(
            {
                "expert_bank_root": roots[ids[0]],
                "generation_lock_root": roots[ids[1]],
                "target_support_reservation_root": roots[ids[2]],
                "target_support_cache_root": roots[ids[3]],
                "readiness_attestation_path": roots[
                    TARGET_SUPPORT_RESERVATION_ARTIFACT_ID
                ]
                / TARGET_SUPPORT_SURFACE.readiness_member,
            }
        )
    elif config.contract == POLICY_LOCK:
        ids = config.input_artifact_ids
        expected.update(
            {
                "action_surface_root": roots[ids[0]],
                "exact_b_policy_root": roots[ids[1]],
                "target_support_surface_root": roots[ids[2]],
                "target_support_reservation_root": roots[ids[3]],
                "fresh_target_reservation_root": roots[ids[4]],
                "readiness_attestation_path": roots[FRESH_TARGET_RESERVATION_ARTIFACT_ID]
                / POLICY_LOCK.readiness_member,
            }
        )
    else:  # pragma: no cover - config construction already closes this world.
        raise ProtocolError("HARP workspace received an unknown surface contract.")
    return expected


__all__ = ("validate_harp_production_workspace_binding",)
