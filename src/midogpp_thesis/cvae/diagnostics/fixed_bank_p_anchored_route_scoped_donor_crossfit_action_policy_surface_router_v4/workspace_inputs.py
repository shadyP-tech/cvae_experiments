"""Exact-six workspace and provenance binding for P-DCAPS v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from .experiment_contracts import (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_FILENAME,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    V1_OUTPUT_ARTIFACT_ID,
    V2_OUTPUT_ARTIFACT_ID,
    V3_OUTPUT_ARTIFACT_ID,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from .workspace_manifest import (
    WORKSPACE_CLAIM_SCOPE as CLAIM_SCOPE,
    WORKSPACE_REPLAY_FIELDS,
    WORKSPACE_STAGE_ID as STAGE_ID,
    validate_workspace_manifest_header,
)


def validate_active_workspace_binding(config: object) -> Mapping[str, object]:
    """Replay the active catalog without writing output state."""

    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(str(getattr(config, "experiment_id")))
        output = workspace.artifacts[str(getattr(config, "output_artifact_id"))]
    except (KeyError, ValueError, OSError, AttributeError) as exc:
        raise ProtocolError("P-DCAPS v4 workspace binding failed.") from exc
    input_ids = tuple(getattr(config, "input_artifact_ids", ()))
    if (
        experiment.experiment_id != EXPERIMENT_ID
        or experiment.output_artifact_id != OUTPUT_ARTIFACT_ID
        or experiment.status != "diagnostic"
        or experiment.stage != STAGE_ID
        or experiment.claim_scope != CLAIM_SCOPE
        or experiment.run_recovery_strategy is not None
        or experiment.input_artifact_ids != INPUT_ARTIFACT_IDS
        or input_ids != INPUT_ARTIFACT_IDS
        or len(input_ids) != 6
        or len(set(input_ids)) != 6
        or any(
            predecessor in input_ids
            for predecessor in (
                V1_OUTPUT_ARTIFACT_ID,
                V2_OUTPUT_ARTIFACT_ID,
                V3_OUTPUT_ARTIFACT_ID,
            )
        )
        or output.artifact_id != OUTPUT_ARTIFACT_ID
        or output.stage != STAGE_ID
        or output.claim_scope != CLAIM_SCOPE
        or output.availability != "generated_on_run"
        or output.migration != "canonical_output"
    ):
        raise ProtocolError("P-DCAPS v4 workspace catalog drifted.")

    protocol = dict(getattr(config, "protocol"))
    semantic = dict(output.semantic_identities)
    expected_semantic = {
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": str(protocol.get("protocol_hash", "")),
        "expected_ledger_amendment_sha256": str(
            getattr(config, "expected_ledger_amendment_sha256")
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
        "v2_source_snapshot_manifest_sha256": str(
            getattr(config, "expected_v2_source_snapshot_manifest_sha256")
        ),
        "v2_source_snapshot_tree_sha256": str(
            getattr(config, "expected_v2_source_snapshot_tree_sha256")
        ),
        "v3_repair_source_manifest_sha256": str(
            getattr(config, "expected_v3_repair_source_manifest_sha256")
        ),
        "v3_repair_source_tree_sha256": str(
            getattr(config, "expected_v3_repair_source_tree_sha256")
        ),
        "combined_source_seal_sha256": str(
            getattr(config, "expected_combined_source_seal_sha256")
        ),
    }
    if (
        any(semantic.get(key) != value for key, value in expected_semantic.items())
        or semantic.get("execution_authorized") != "true"
        or semantic.get("consumed_test_reuse_authorized") != "true"
        or semantic.get("single_use_execution_identity") != "true"
        or semantic.get("authorization_exhausted") != "false"
        or semantic.get("fresh_evidence") != "false"
        or semantic.get("v1_output_used") != "false"
        or semantic.get("v2_output_used") != "false"
        or semantic.get("v3_output_used") != "false"
        or semantic.get("previous_stage90_amendments_used") != "false"
        or semantic.get("previous_stage90_run_state_used") != "false"
        or semantic.get("previous_stage90_scratch_used") != "false"
        or semantic.get("previous_probability_or_capability_history_used") != "false"
        or semantic.get("cross_run_recovery_allowed") != "false"
        or any(artifact_id not in workspace.artifacts for artifact_id in INPUT_ARTIFACT_IDS)
    ):
        raise ProtocolError("P-DCAPS v4 workspace authorization binding drifted.")

    expected_paths = _workspace_paths(workspace, config)
    if any(
        Path(getattr(config, role)).resolve() != path.resolve()
        for role, path in expected_paths.items()
    ):
        raise ProtocolError("P-DCAPS v4 resolved input paths drifted.")
    return {
        "status": "PASS",
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
        "input_artifact_count": 6,
        "source_snapshot_bound": True,
    }


def validate_workspace_provenance(
    root: Path, config: object
) -> dict[str, Mapping[str, object]]:
    """Require the exact workspace-rendered six-input manifest."""

    payload = _read_object(Path(root) / "provenance/input_artifacts.json")
    validate_workspace_manifest_header(payload)
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("P-DCAPS v4 provenance rows are malformed.")
    actual_ids = tuple(str(row.get("artifact_id")) for row in raw_rows)
    expected_ids = tuple(sorted(INPUT_ARTIFACT_IDS))
    if (
        actual_ids != expected_ids
        or len(actual_ids) != 6
        or len(set(actual_ids)) != 6
        or any(
            predecessor in actual_ids
            for predecessor in (
                V1_OUTPUT_ARTIFACT_ID,
                V2_OUTPUT_ARTIFACT_ID,
                V3_OUTPUT_ARTIFACT_ID,
            )
        )
    ):
        raise ProtocolError("P-DCAPS v4 provenance coverage drifted.")
    rows = {str(row["artifact_id"]): row for row in raw_rows}
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: Path(getattr(config, "expert_bank_root")),
        GENERATION_LOCK_ARTIFACT_ID: Path(getattr(config, "generation_lock_root")),
        TEST_CACHE_ARTIFACT_ID: Path(getattr(config, "test_cache_root")),
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
            != expected_path.resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(
                f"P-DCAPS v4 provenance drifted: {artifact_id}."
            )
    _replay_workspace_manifest(payload, config)
    return {artifact_id: rows[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS}


def _workspace_paths(
    workspace: MidogppWorkspace, config: object
) -> dict[str, Path]:
    return {
        "artifact_root": workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID, for_output=True, require_exists=False
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


def _replay_workspace_manifest(payload: Mapping[str, object], config: object) -> None:
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        rendered = workspace._render_run(  # noqa: SLF001 - deliberate audit seam
            EXPERIMENT_ID,
            require_inputs=True,
            validate_workspace=False,
            include_all_declared_inputs=True,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("P-DCAPS v4 provenance replay failed.") from exc
    expected = rendered.input_manifest
    if any(
        payload.get(key) != expected.get(key) for key in WORKSPACE_REPLAY_FIELDS
    ) or payload.get("input_artifacts") != expected.get("input_artifacts"):
        raise ProtocolError("P-DCAPS v4 provenance replay differs.")


def _read_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("P-DCAPS v4 provenance member is absent or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v4 workspace provenance.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("P-DCAPS v4 workspace provenance must be an object.")
    return value


__all__ = (
    "CLAIM_SCOPE",
    "STAGE_ID",
    "validate_active_workspace_binding",
    "validate_workspace_provenance",
)
