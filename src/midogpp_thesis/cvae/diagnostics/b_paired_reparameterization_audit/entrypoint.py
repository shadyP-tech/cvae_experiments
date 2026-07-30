"""Workspace-only execution guard for registered Stage-90 diagnostic outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from midogpp_thesis.cvae.protocol import ProtocolError

from .artifacts import file_sha256


SNAPSHOT_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1"
)
AUDIT_EXPERIMENT_ID = "midogpp.oracle.uniform_b_paired_reparameterization_audit.v1"
SNAPSHOT_CANONICAL_RELATIVE = Path(
    "artifacts/midogpp/90_oracles_and_diagnostics/inputs/"
    "uniform_b_paired_reparameterization_snapshot_v1"
)
AUDIT_CANONICAL_RELATIVE = Path(
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_paired_reparameterization_audit/v1"
)


def assert_workspace_prepared_entrypoint(
    *,
    resolved_config_path: str | Path,
    artifact_root: str | Path,
    experiment_id: str,
    canonical_relative: Path,
    input_artifact_ids: Iterable[str],
    expected_input_members: Mapping[str, str | Path],
) -> None:
    """Reject direct/unregistered execution before reading inputs or using GPUs."""

    root = Path(artifact_root).resolve()
    config_path = Path(resolved_config_path).resolve()
    if config_path != root / "config.resolved.yaml" or not config_path.is_file():
        raise ProtocolError(
            "Stage-90 diagnostics require the workspace-prepared "
            "<artifact_root>/config.resolved.yaml."
        )
    repo_root = _discover_registered_repo(root)
    expected_root = (repo_root / canonical_relative).resolve()
    if root != expected_root:
        raise ProtocolError(
            "Stage-90 diagnostic artifact_root is not its registered canonical output."
        )
    provenance_path = root / "provenance/input_artifacts.json"
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Stage-90 diagnostics require workspace-prepared input provenance."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Workspace input provenance root must be a mapping.")
    if payload.get("schema_version") != "midogpp_input_artifacts_v2":
        raise ProtocolError("Workspace input provenance schema is not canonical.")
    if str(payload.get("experiment_id", "")) != experiment_id:
        raise ProtocolError("Workspace input provenance binds the wrong experiment.")
    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Workspace input provenance lacks artifact records.")
    observed_ids = {
        str(row.get("artifact_id", ""))
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_ids = {str(value) for value in input_artifact_ids}
    if observed_ids != expected_ids:
        raise ProtocolError(
            "Workspace input provenance does not match the registered artifact set."
        )
    if payload.get("selection_used_target_eval_artifacts") is not False:
        raise ProtocolError("Workspace provenance crossed the target-evaluation firewall.")
    if set(expected_input_members) != expected_ids:
        raise ProtocolError("Entrypoint input bindings do not cover the registered inputs.")
    by_id = {
        str(row["artifact_id"]): row
        for row in rows
        if isinstance(row, Mapping)
    }
    for artifact_id, raw_member in expected_input_members.items():
        row = by_id[artifact_id]
        artifact_path = Path(str(row.get("resolved_path", ""))).resolve()
        expected_member = Path(raw_member).resolve()
        if not artifact_path.is_absolute() or (
            expected_member != artifact_path
            and not expected_member.is_relative_to(artifact_path)
        ):
            raise ProtocolError(
                f"Resolved config input escapes workspace artifact {artifact_id}."
            )
        if row.get("exists") is not True:
            raise ProtocolError(f"Workspace input artifact is unavailable: {artifact_id}.")
        integrity = row.get("file_integrity")
        if not isinstance(integrity, Mapping):
            raise ProtocolError(f"Workspace input lacks integrity evidence: {artifact_id}.")
        files = integrity.get("files")
        if not isinstance(files, list):
            raise ProtocolError(f"Workspace input file inventory is malformed: {artifact_id}.")
        if expected_member != artifact_path:
            matching = [
                item
                for item in files
                if isinstance(item, Mapping)
                and Path(str(item.get("resolved_path", ""))).resolve()
                == expected_member
            ]
            if len(matching) != 1:
                raise ProtocolError(
                    f"Workspace provenance does not bind resolved input {expected_member}."
                )
            computed = matching[0].get("computed")
            if (
                not isinstance(computed, Mapping)
                or computed.get("sha256") != file_sha256(expected_member)
            ):
                raise ProtocolError(
                    f"Workspace input hash does not match current bytes: {expected_member}."
                )


def _discover_registered_repo(artifact_root: Path) -> Path:
    for parent in (artifact_root, *artifact_root.parents):
        if (parent / "experiments/midogpp/registry.yaml").is_file():
            return parent
    raise ProtocolError(
        "Stage-90 diagnostic output is not inside a registered MIDOG++ checkout."
    )


__all__ = (
    "AUDIT_CANONICAL_RELATIVE",
    "AUDIT_EXPERIMENT_ID",
    "SNAPSHOT_CANONICAL_RELATIVE",
    "SNAPSHOT_EXPERIMENT_ID",
    "assert_workspace_prepared_entrypoint",
)
