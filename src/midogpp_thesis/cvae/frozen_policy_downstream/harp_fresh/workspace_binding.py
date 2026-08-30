"""Canonical workspace resolution for fresh HARP Stage-70 execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu.hashing import canonical_sha256
from .config import (
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    POLICY_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    SCORING_MANIFEST_ARTIFACT_ID,
    TARGET_CACHE_ARTIFACT_ID,
    HarpFreshStage70Config,
)


STAGE_ID = "70_frozen_policy_downstream"
CLAIM_SCOPE = "synthetic_downstream_utility"


@dataclass(frozen=True, kw_only=True)
class HarpFreshWorkspaceBinding:
    artifact_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    policy_root: Path
    reservation_root: Path
    target_cache_root: Path
    scoring_manifest_path: Path
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        paths = {
            "artifact_root": self.artifact_root,
            "expert_bank_root": self.expert_bank_root,
            "generation_lock_root": self.generation_lock_root,
            "policy_root": self.policy_root,
            "reservation_root": self.reservation_root,
            "target_cache_root": self.target_cache_root,
            "scoring_manifest_path": self.scoring_manifest_path,
        }
        normalized: dict[str, Path] = {}
        for name, path in paths.items():
            value = Path(path).resolve()
            if value == Path(value.anchor):
                raise ProtocolError("Fresh HARP workspace resolved an unsafe root path.")
            normalized[name] = value
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_sha256(
                {
                    "schema_version": "midogpp_harp_fresh_workspace_binding_v1",
                    "experiment_id": EXPERIMENT_ID,
                    "output_artifact_id": OUTPUT_ARTIFACT_ID,
                    "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
                    # Paths are an execution receipt, not part of the scientific
                    # config identity.  They are nevertheless sealed here.
                    "resolved_paths": {
                        key: value.as_posix() for key, value in normalized.items()
                    },
                }
            ),
        )


def validate_harp_fresh_workspace_binding(
    config: HarpFreshStage70Config,
    *,
    _workspace: MidogppWorkspace | None = None,
) -> HarpFreshWorkspaceBinding:
    """Resolve only the exact active registry graph and artifact locations."""

    if not isinstance(config, HarpFreshStage70Config):
        raise ProtocolError("Fresh HARP workspace validation requires typed config.")
    workspace = _workspace or MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    output = workspace.artifacts[OUTPUT_ARTIFACT_ID]
    stage = workspace.stages[STAGE_ID]
    if (
        experiment.status != "active"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(experiment.input_artifact_ids) != INPUT_ARTIFACT_IDS
        or output.stage != STAGE_ID
        or output.claim_scope != CLAIM_SCOPE
        or CLAIM_SCOPE not in stage.get("allowed_claim_scopes", ())
    ):
        raise ProtocolError("Fresh HARP Stage-70 workspace graph drifted.")
    fresh_requirements = {
        RESERVATION_ARTIFACT_ID: {
            "architecture_family": "HARP_V1",
            "split_role": "fresh_case_disjoint_target_support_and_evaluation_reservation",
            "reservation_precedes_policy_and_cache": "true",
            "labels_opened": "false",
            "consumed_test_rows_used": "false",
        },
        TARGET_CACHE_ARTIFACT_ID: {
            "architecture_family": "HARP_V1",
            "split_role": "fresh_case_disjoint_target_evaluation_only",
            "reservation_artifact_id": RESERVATION_ARTIFACT_ID,
            "reservation_precedes_cache": "true",
            "policy_frozen_before_cache": "true",
            "labels_persisted": "false",
            "closed_world_content_indexed": "true",
        },
        SCORING_MANIFEST_ARTIFACT_ID: {
            "architecture_family": "HARP_V1",
            "split_role": "fresh_target_evaluation_scoring_only",
            "reservation_artifact_id": RESERVATION_ARTIFACT_ID,
            "contains_exactly_reserved_evaluation_rows": "true",
            "labels_present": "true",
            "labels_available_before_global_route_seal": "false",
            "raw_labels_may_be_persisted_to_output": "false",
        },
    }
    forbidden_true = (
        "consumed_test_used",
        "consumed_test_rows_used",
        "consumed_validation_used",
        "consumed_validation_rows_used",
        "consumed_stage90_used",
        "stage50_or_stage90_artifacts_used",
    )
    for artifact_id, required in fresh_requirements.items():
        identities = workspace.artifacts[artifact_id].semantic_identities
        if any(identities.get(key) == "true" for key in forbidden_true) or any(
            identities.get(key) != value for key, value in required.items()
        ):
            raise ProtocolError(
                "Fresh HARP workspace lacks explicit fresh/case-disjoint/label-sealed identities."
            )
    output_identities = output.semantic_identities
    output_required = {
        "architecture_family": "HARP_V1",
        "inference_unit": "target_center",
        "effective_center_count": "9",
        "case_equal_within_center": "true",
        "complete_action_menu_before_routing": "true",
        "complete_routes_and_vectors_before_labels": "true",
        "target_labels_used_for_scoring_only": "true",
        "target_labels_may_update_policy": "false",
        "consumed_test_rows_used": "false",
        "stage50_or_stage90_artifacts_used": "false",
    }
    if any(output_identities.get(key) == "true" for key in forbidden_true) or any(
        output_identities.get(key) != value for key, value in output_required.items()
    ):
        raise ProtocolError("Fresh HARP output artifact claim boundary drifted.")
    return HarpFreshWorkspaceBinding(
        artifact_root=workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        expert_bank_root=workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        generation_lock_root=workspace.resolve_artifact(GENERATION_LOCK_ARTIFACT_ID),
        policy_root=workspace.resolve_artifact(POLICY_ARTIFACT_ID),
        reservation_root=workspace.resolve_artifact(RESERVATION_ARTIFACT_ID),
        target_cache_root=workspace.resolve_artifact(TARGET_CACHE_ARTIFACT_ID),
        scoring_manifest_path=(
            workspace.resolve_artifact(SCORING_MANIFEST_ARTIFACT_ID) / "manifest.csv"
        ),
    )


__all__ = (
    "CLAIM_SCOPE",
    "HarpFreshWorkspaceBinding",
    "STAGE_ID",
    "validate_harp_fresh_workspace_binding",
)
