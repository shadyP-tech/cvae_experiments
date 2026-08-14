"""Canonical workspace and exact-six provenance admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .experiment_contracts import (
    CLAIM_SCOPE,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_FILENAME,
    STAGE_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)


def validate_workspace_provenance(
    root: Path, config: object
) -> dict[str, Mapping[str, object]]:
    payload = _read_object(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != getattr(config, "experiment_id")
        or payload.get("stage") != STAGE_ID
        or payload.get("claim_scope") != CLAIM_SCOPE
    ):
        raise ProtocolError("Dual-endpoint provenance header drifted.")
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("Dual-endpoint provenance rows are malformed.")
    expected_ids = tuple(sorted(getattr(config, "input_artifact_ids")))
    actual_ids = tuple(str(row.get("artifact_id")) for row in raw_rows)
    if actual_ids != expected_ids or len(actual_ids) != 6 or len(set(actual_ids)) != 6:
        raise ProtocolError("Dual-endpoint provenance coverage drifted.")
    rows = {str(row["artifact_id"]): row for row in raw_rows}
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: getattr(config, "expert_bank_root"),
        GENERATION_LOCK_ARTIFACT_ID: getattr(config, "generation_lock_root"),
        TEST_CACHE_ARTIFACT_ID: getattr(config, "test_cache_root"),
        TEST_MANIFEST_ARTIFACT_ID: Path(getattr(config, "test_manifest_path")).parent,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: Path(
            getattr(config, "test_consumption_ledger_path")
        ).parent.parent,
        LEDGER_AMENDMENT_ARTIFACT_ID: Path(
            getattr(config, "ledger_amendment_path")
        ).parent,
    }
    for artifact_id, expected_path in expected_paths.items():
        row = rows.get(artifact_id)
        if (
            not isinstance(row, Mapping)
            or Path(str(row.get("resolved_path", ""))).resolve()
            != Path(expected_path).resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"Dual-endpoint provenance drifted: {artifact_id}.")
    _replay_workspace_manifest(payload, config)
    return {
        artifact_id: rows[artifact_id]
        for artifact_id in getattr(config, "input_artifact_ids")
    }


def validate_active_diagnostic_workspace_binding(config: object) -> Mapping[str, object]:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(getattr(config, "experiment_id"))
        output = workspace.artifacts[getattr(config, "output_artifact_id")]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Dual-endpoint workspace binding failed.") from exc
    if (
        experiment.status != "diagnostic"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.input_artifact_ids != tuple(getattr(config, "input_artifact_ids"))
        or output.stage != STAGE_ID
        or output.claim_scope != CLAIM_SCOPE
    ):
        raise ProtocolError("Dual-endpoint workspace catalog drifted.")
    expected = {
        "artifact_root": workspace.resolve_artifact(
            getattr(config, "output_artifact_id"), for_output=True, require_exists=False
        ),
        "expert_bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(
            GENERATION_LOCK_ARTIFACT_ID
        ),
        "test_cache_root": workspace.resolve_artifact(TEST_CACHE_ARTIFACT_ID),
        "test_manifest_path": workspace.resolve_artifact(TEST_MANIFEST_ARTIFACT_ID)
        / "manifest.csv",
        "test_consumption_ledger_path": workspace.resolve_artifact(
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
        )
        / "reports/test_consumption_ledger.json",
        "ledger_amendment_path": workspace.resolve_artifact(
            LEDGER_AMENDMENT_ARTIFACT_ID
        )
        / LEDGER_AMENDMENT_FILENAME,
    }
    if any(
        Path(getattr(config, role)).resolve() != Path(path).resolve()
        for role, path in expected.items()
    ):
        raise ProtocolError("Dual-endpoint resolved input paths drifted.")
    return {
        "status": "PASS",
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
    }


def _replay_workspace_manifest(payload: Mapping[str, object], config: object) -> None:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        rendered = workspace._render_run(  # noqa: SLF001 - audit seam
            str(getattr(config, "experiment_id")),
            require_inputs=True,
            validate_workspace=False,
            include_all_declared_inputs=True,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Dual-endpoint provenance replay failed.") from exc
    expected = rendered.input_manifest
    header = (
        "schema_version",
        "dataset_id",
        "experiment_id",
        "stage",
        "claim_scope",
        "selection_used_target_eval_artifacts",
    )
    if any(payload.get(key) != expected.get(key) for key in header) or payload.get(
        "input_artifacts"
    ) != expected.get("input_artifacts"):
        raise ProtocolError("Dual-endpoint provenance replay differs.")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read dual-endpoint JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Dual-endpoint JSON must be an object.")
    return value


__all__ = (
    "validate_active_diagnostic_workspace_binding",
    "validate_workspace_provenance",
)
