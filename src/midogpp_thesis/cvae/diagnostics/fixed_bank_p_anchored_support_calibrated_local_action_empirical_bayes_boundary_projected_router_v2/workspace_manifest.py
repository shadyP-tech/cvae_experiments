"""Pure workspace-manifest contract for SCALE-BP v2 launch provenance."""

from __future__ import annotations

from typing import Mapping, Sequence

from .experiment_contracts import validate_exact_input_fence
from .identity import (
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    GovernanceError,
)


WORKSPACE_INPUT_MANIFEST_SCHEMA = "midogpp_input_artifacts_v2"
WORKSPACE_DATASET_ID = "midogpp"
WORKSPACE_STAGE_ID = "90_oracles_and_diagnostics"
WORKSPACE_REPLAY_FIELDS = (
    "schema_version",
    "dataset_id",
    "experiment_id",
    "stage",
    "claim_scope",
    "selection_used_target_eval_artifacts",
    "repository_revision",
    "repository_dirty",
    "repository_status_hash",
)


def validate_workspace_manifest(payload: Mapping[str, object]) -> None:
    """Validate transport identity and exact-six provenance rows."""

    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise GovernanceError("SCALE-BP v2 workspace input rows are malformed.")
    artifact_ids: Sequence[object] = tuple(
        row.get("artifact_id") for row in rows if isinstance(row, Mapping)
    )
    resolved_paths = tuple(
        str(row.get("resolved_path", "")) for row in rows if isinstance(row, Mapping)
    )
    validate_exact_input_fence(DIRECT_INPUT_ARTIFACT_IDS, resolved_paths=resolved_paths)
    if (
        tuple(artifact_ids) != tuple(sorted(DIRECT_INPUT_ARTIFACT_IDS))
        or len(set(str(value) for value in artifact_ids)) != len(artifact_ids)
        or payload.get("schema_version") != WORKSPACE_INPUT_MANIFEST_SCHEMA
        or payload.get("dataset_id") != WORKSPACE_DATASET_ID
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != WORKSPACE_STAGE_ID
        or payload.get("claim_scope") != CLAIM_SCOPE
        or payload.get("selection_used_target_eval_artifacts") is not False
        or not isinstance(payload.get("repository_revision"), str)
        or not payload.get("repository_revision")
        or payload.get("repository_dirty") not in {True, False, None}
        or not isinstance(payload.get("repository_status_hash"), str)
        or not payload.get("repository_status_hash")
    ):
        raise GovernanceError("SCALE-BP v2 workspace provenance header drifted.")
    for row in rows:
        assert isinstance(row, Mapping)
        if (
            row.get("exists") is not True
            or not isinstance(row.get("resolved_path"), str)
            or not row.get("resolved_path")
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise GovernanceError("SCALE-BP v2 workspace provenance row drifted.")


__all__ = (
    "WORKSPACE_DATASET_ID",
    "WORKSPACE_INPUT_MANIFEST_SCHEMA",
    "WORKSPACE_REPLAY_FIELDS",
    "WORKSPACE_STAGE_ID",
    "validate_workspace_manifest",
)
