"""Canonical workspace transport header for SCEPTRE v3."""

from __future__ import annotations

from typing import Mapping

from ...protocol import ProtocolError
from .identity import EXPERIMENT_ID


WORKSPACE_INPUT_MANIFEST_SCHEMA = "midogpp_input_artifacts_v2"
WORKSPACE_DATASET_ID = "midogpp"
WORKSPACE_STAGE_ID = "90_oracles_and_diagnostics"
WORKSPACE_CLAIM_SCOPE = "diagnostic_only"
WORKSPACE_HEADER_FIELDS = (
    "schema_version",
    "dataset_id",
    "experiment_id",
    "stage",
    "claim_scope",
    "selection_used_target_eval_artifacts",
)
WORKSPACE_REPLAY_FIELDS = WORKSPACE_HEADER_FIELDS + (
    "repository_revision",
    "repository_dirty",
    "repository_status_hash",
)


def validate_workspace_manifest_header(payload: Mapping[str, object]) -> None:
    revision = payload.get("repository_revision")
    dirty = payload.get("repository_dirty")
    status_hash = payload.get("repository_status_hash")
    if (
        payload.get("schema_version") != WORKSPACE_INPUT_MANIFEST_SCHEMA
        or payload.get("dataset_id") != WORKSPACE_DATASET_ID
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != WORKSPACE_STAGE_ID
        or payload.get("claim_scope") != WORKSPACE_CLAIM_SCOPE
        or payload.get("selection_used_target_eval_artifacts") is not False
        or not isinstance(revision, str)
        or not revision
        or (dirty is not None and not isinstance(dirty, bool))
        or not isinstance(status_hash, str)
        or not status_hash
    ):
        raise ProtocolError("SCEPTRE v3 workspace provenance header drifted.")


__all__ = (
    "WORKSPACE_CLAIM_SCOPE",
    "WORKSPACE_DATASET_ID",
    "WORKSPACE_HEADER_FIELDS",
    "WORKSPACE_INPUT_MANIFEST_SCHEMA",
    "WORKSPACE_REPLAY_FIELDS",
    "WORKSPACE_STAGE_ID",
    "validate_workspace_manifest_header",
)
