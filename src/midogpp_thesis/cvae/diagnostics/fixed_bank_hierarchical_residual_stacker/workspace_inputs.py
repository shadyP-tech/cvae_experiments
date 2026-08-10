"""Workspace and provenance binding for the isolated residual stacker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .experiment_contracts import (
    CLAIM_SCOPE,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    STAGE_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)


class WorkspaceInputConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: Sequence[str]
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path


def validate_workspace_provenance(
    root: Path, config: WorkspaceInputConfig
) -> dict[str, Mapping[str, object]]:
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("stage") != STAGE_ID
        or payload.get("claim_scope") != CLAIM_SCOPE
    ):
        raise ProtocolError("Residual-stacker workspace provenance header drifted.")
    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Residual-stacker workspace provenance rows are malformed.")
    actual_ids = tuple(str(row.get("artifact_id")) for row in rows)
    if actual_ids != tuple(sorted(config.input_artifact_ids)) or len(set(actual_ids)) != len(actual_ids):
        raise ProtocolError("Residual-stacker workspace provenance order drifted.")
    by_id = {str(row.get("artifact_id")): row for row in rows}
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: config.expert_bank_root,
        GENERATION_LOCK_ARTIFACT_ID: config.generation_lock_root,
        TEST_CACHE_ARTIFACT_ID: config.test_cache_root,
        TEST_MANIFEST_ARTIFACT_ID: config.test_manifest_path.parent,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: config.test_consumption_ledger_path.parent.parent,
        LEDGER_AMENDMENT_ARTIFACT_ID: config.ledger_amendment_path.parent,
    }
    for artifact_id in config.input_artifact_ids:
        row = by_id.get(artifact_id)
        if (
            row is None
            or Path(str(row.get("resolved_path", ""))).resolve()
            != expected_paths[artifact_id].resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"Residual-stacker provenance drifted: {artifact_id}.")
    return {artifact_id: by_id[artifact_id] for artifact_id in config.input_artifact_ids}


def validate_active_diagnostic_workspace_binding(
    config: WorkspaceInputConfig,
) -> Mapping[str, object]:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(config.experiment_id)
        output = workspace.artifacts[config.output_artifact_id]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Residual-stacker canonical workspace binding failed.") from exc
    if (
        experiment.status != "diagnostic"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.output_artifact_id != config.output_artifact_id
        or experiment.input_artifact_ids != tuple(config.input_artifact_ids)
        or output.stage != STAGE_ID
        or output.claim_scope != CLAIM_SCOPE
    ):
        raise ProtocolError("Residual-stacker experiment binding drifted.")
    return {
        "status": "PASS",
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
        "stage": experiment.stage,
        "claim_scope": experiment.claim_scope,
    }


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read residual-stacker provenance JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Residual-stacker provenance input must be a JSON object.")
    return value


__all__ = ("validate_active_diagnostic_workspace_binding", "validate_workspace_provenance")
