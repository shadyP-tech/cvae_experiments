"""Exact-eight active-workspace and rendered-provenance binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .experiment_contracts import (
    EXECUTION_AMENDMENT_ARTIFACT_ID,
    EXECUTION_AMENDMENT_FILENAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_FILENAME,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    V1_OUTPUT_ARTIFACT_ID,
    V2_OUTPUT_ARTIFACT_ID,
)
from .workspace_manifest import (
    WORKSPACE_CLAIM_SCOPE,
    WORKSPACE_REPLAY_FIELDS,
    WORKSPACE_STAGE_ID,
    validate_workspace_manifest_header,
)


def validate_active_workspace_binding(config: object) -> Mapping[str, object]:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(str(getattr(config, "experiment_id")))
        output = workspace.artifacts[str(getattr(config, "output_artifact_id"))]
    except (KeyError, ValueError, OSError, AttributeError) as exc:
        raise ProtocolError("SCEPTRE v3 workspace binding failed.") from exc
    input_ids = tuple(getattr(config, "input_artifact_ids", ()))
    if (
        experiment.experiment_id != EXPERIMENT_ID
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.status != "diagnostic"
        or experiment.stage != WORKSPACE_STAGE_ID
        or experiment.claim_scope != WORKSPACE_CLAIM_SCOPE
        or experiment.run_recovery_strategy is not None
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
        or input_ids != INPUT_ARTIFACT_IDS
        or len(input_ids) != 8
        or len(set(input_ids)) != 8
        or V1_OUTPUT_ARTIFACT_ID in input_ids
        or V2_OUTPUT_ARTIFACT_ID in input_ids
        or output.artifact_id != OUTPUT_ARTIFACT_ID
        or output.stage != WORKSPACE_STAGE_ID
        or output.claim_scope != WORKSPACE_CLAIM_SCOPE
        or output.availability != "generated_on_run"
        or output.migration != "canonical_output"
    ):
        raise ProtocolError("SCEPTRE v3 workspace catalog drifted.")
    semantic = dict(output.semantic_identities)
    expected = {
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": str(getattr(config, "protocol")["protocol_hash"]),
        "expected_execution_amendment_sha256": str(
            getattr(config, "expected_execution_amendment_sha256")
        ),
        "execution_authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "source_snapshot_manifest_sha256": str(
            getattr(config, "expected_source_snapshot_manifest_sha256")
        ),
        "source_snapshot_tree_sha256": str(
            getattr(config, "expected_source_snapshot_tree_sha256")
        ),
        "source_snapshot_member_count": str(
            getattr(config, "expected_source_snapshot_member_count")
        ),
    }
    if (
        any(semantic.get(key) != value for key, value in expected.items())
        or semantic.get("execution_authorized") != "true"
        or semantic.get("consumed_test_reuse_authorized") != "true"
        or semantic.get("single_use_execution_identity") != "true"
        or semantic.get("authorization_exhausted") != "false"
        or semantic.get("fresh_evidence") != "false"
        or semantic.get("routing_success_claimed") != "false"
        or semantic.get("nelbo_compatibility_claimed") != "false"
        or semantic.get("cross_run_recovery_allowed") != "false"
        or any(artifact_id not in workspace.artifacts for artifact_id in INPUT_ARTIFACT_IDS)
    ):
        raise ProtocolError("SCEPTRE v3 workspace authorization binding drifted.")
    paths = _workspace_paths(workspace)
    if any(
        Path(getattr(config, role)).resolve() != path.resolve()
        for role, path in paths.items()
    ):
        raise ProtocolError("SCEPTRE v3 resolved workspace paths drifted.")
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "input_artifact_count": 8,
        "source_snapshot_bound": True,
    }


def validate_workspace_provenance(
    root: Path, config: object
) -> dict[str, Mapping[str, object]]:
    payload = _read_object(Path(root) / "provenance/input_artifacts.json")
    validate_workspace_manifest_header(payload)
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("SCEPTRE v3 provenance rows are malformed.")
    actual_ids = tuple(str(row.get("artifact_id")) for row in raw_rows)
    if (
        actual_ids != tuple(sorted(INPUT_ARTIFACT_IDS))
        or len(actual_ids) != 8
        or len(set(actual_ids)) != 8
        or V1_OUTPUT_ARTIFACT_ID in actual_ids
        or V2_OUTPUT_ARTIFACT_ID in actual_ids
    ):
        raise ProtocolError("SCEPTRE v3 provenance coverage drifted.")
    rows = {str(row["artifact_id"]): row for row in raw_rows}
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: Path(getattr(config, "expert_bank_root")),
        GENERATION_LOCK_ARTIFACT_ID: Path(getattr(config, "generation_lock_root")),
        SOURCE_INNER_ALIAS_ARTIFACT_ID: Path(getattr(config, "source_inner_root")),
        SOURCE_INNER_AMENDMENT_ARTIFACT_ID: Path(
            getattr(config, "source_inner_amendment_path")
        ).parent,
        TEST_CACHE_ARTIFACT_ID: Path(getattr(config, "test_cache_root")),
        TEST_MANIFEST_ARTIFACT_ID: Path(getattr(config, "test_manifest_path")).parent,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: Path(
            getattr(config, "test_consumption_ledger_path")
        ).parent.parent,
        EXECUTION_AMENDMENT_ARTIFACT_ID: Path(
            getattr(config, "execution_amendment_path")
        ).parent,
    }
    for artifact_id, expected_path in expected_paths.items():
        row = rows.get(artifact_id)
        if (
            not isinstance(row, Mapping)
            or Path(str(row.get("resolved_path", ""))).resolve()
            != expected_path.resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"SCEPTRE v3 provenance drifted: {artifact_id}.")
    _replay_workspace_manifest(payload)
    return {artifact_id: rows[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS}


def _workspace_paths(workspace: MidogppWorkspace) -> dict[str, Path]:
    return {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
        ),
        "expert_bank_root": workspace.resolve_artifact(EXPERT_BANK_ARTIFACT_ID),
        "generation_lock_root": workspace.resolve_artifact(GENERATION_LOCK_ARTIFACT_ID),
        "source_inner_root": workspace.resolve_artifact(SOURCE_INNER_ALIAS_ARTIFACT_ID),
        "source_inner_amendment_path": workspace.resolve_artifact(
            SOURCE_INNER_AMENDMENT_ARTIFACT_ID
        )
        / SOURCE_INNER_AMENDMENT_FILENAME,
        "test_cache_root": workspace.resolve_artifact(TEST_CACHE_ARTIFACT_ID),
        "test_manifest_path": workspace.resolve_artifact(TEST_MANIFEST_ARTIFACT_ID)
        / "manifest.csv",
        "test_consumption_ledger_path": workspace.resolve_artifact(
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
        )
        / "reports/test_consumption_ledger.json",
        "execution_amendment_path": workspace.resolve_artifact(
            EXECUTION_AMENDMENT_ARTIFACT_ID
        )
        / EXECUTION_AMENDMENT_FILENAME,
    }


def _replay_workspace_manifest(payload: Mapping[str, object]) -> None:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        rendered = workspace._render_run(  # noqa: SLF001 - audit replay seam
            EXPERIMENT_ID,
            require_inputs=True,
            validate_workspace=False,
            include_all_declared_inputs=True,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("SCEPTRE v3 provenance replay failed.") from exc
    expected = rendered.input_manifest
    if any(
        payload.get(key) != expected.get(key) for key in WORKSPACE_REPLAY_FIELDS
    ) or payload.get("input_artifacts") != expected.get("input_artifacts"):
        raise ProtocolError("SCEPTRE v3 provenance replay differs.")


def _read_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("SCEPTRE v3 provenance member is absent or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read SCEPTRE v3 workspace provenance.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("SCEPTRE v3 workspace provenance must be an object.")
    return value


__all__ = (
    "validate_active_workspace_binding",
    "validate_workspace_provenance",
)
