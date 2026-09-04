"""Authenticate pristine workspace-admin output for v16 supersession."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import cast

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_bytes, canonical_hash
from .. import authorization
from ..activation_paths import RepositoryBoundary
from ..activation_transaction import ActivationJournal
from ..config import INPUT_ARTIFACT_IDS, HarpStage90V16Config, load_config
from ..identity import EXPERIMENT_ID
from .contracts import (
    ARCHIVED_ADMIN_CONTENT,
    ARCHIVED_ADMIN_MANIFEST,
    RETIRED_ADMIN_OUTPUT,
    sha256_bytes,
)


ADMIN_SNAPSHOT_SCHEMA = "midogpp_harp_v16_workspace_admin_snapshot_v1"
ADMIN_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
)
ADMIN_DIRECTORIES = (
    "manifests",
    "provenance",
    "reports",
    "tables",
)


def inspect_workspace_admin_output(
    boundary: RepositoryBoundary,
    *,
    journal: ActivationJournal,
    active_config: HarpStage90V16Config,
) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    """Capture only the exact non-scientific workspace scaffold."""

    relative = authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    lexical = boundary.lexical_root / relative
    if not os.path.lexists(lexical):
        output_root = boundary.member(relative, label="output identity", kind="future")
        body: dict[str, object] = {
            "schema_version": ADMIN_SNAPSHOT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "state": "ABSENT",
            "output_root": relative,
            "directories": [],
            "files": [],
            "scientific_files_present": False,
            "labels_opened": False,
            "routes_sealed": False,
        }
        return output_root, {**body, "snapshot_hash": canonical_hash(body)}, {}

    output_root = boundary.member(relative, label="output identity", kind="directory")
    directories: list[str] = []
    files: dict[str, bytes] = {}
    for path in sorted(output_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ProtocolError("HARP v16 workspace-admin output contains a symlink.")
        member = path.relative_to(output_root).as_posix()
        if path.is_dir():
            directories.append(member)
        elif path.is_file():
            files[member] = path.read_bytes()
        else:
            raise ProtocolError(
                "HARP v16 workspace-admin output contains a non-regular member."
            )
    if tuple(directories) != tuple(sorted(ADMIN_DIRECTORIES)) or set(files) != set(
        ADMIN_FILES
    ):
        raise ProtocolError(
            "HARP v16 output is not an exact workspace-admin pristine snapshot."
        )
    validate_admin_config(
        output_root / "config.resolved.yaml",
        output_root=output_root,
        active_config=active_config,
        journal=journal,
    )
    validate_admin_input_manifest(files["provenance/input_artifacts.json"])
    file_rows = [
        {
            "path": member,
            "sha256": sha256_bytes(files[member]),
            "size_bytes": len(files[member]),
        }
        for member in sorted(files)
    ]
    body = {
        "schema_version": ADMIN_SNAPSHOT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "state": "WORKSPACE_ADMIN_PRISTINE",
        "output_root": relative,
        "directories": sorted(directories),
        "files": file_rows,
        "scientific_files_present": False,
        "labels_opened": False,
        "routes_sealed": False,
    }
    return output_root, {**body, "snapshot_hash": canonical_hash(body)}, files


def load_recovery_admin_snapshot(
    boundary: RepositoryBoundary,
    *,
    archive_root: Path,
) -> tuple[Path, dict[str, object], dict[str, bytes], str]:
    """Authenticate an archived snapshot and locate its live or retired tree."""

    manifest = read_canonical_json(
        archive_root / ARCHIVED_ADMIN_MANIFEST,
        label="archived admin snapshot manifest",
    )
    snapshot_hash = manifest.get("snapshot_hash")
    body = {key: value for key, value in manifest.items() if key != "snapshot_hash"}
    if (
        snapshot_hash != canonical_hash(body)
        or manifest.get("schema_version") != ADMIN_SNAPSHOT_SCHEMA
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("scientific_files_present") is not False
        or manifest.get("labels_opened") is not False
        or manifest.get("routes_sealed") is not False
        or manifest.get("output_root")
        != authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    ):
        raise ProtocolError("HARP v16 archived admin snapshot manifest drifted.")
    directories = manifest.get("directories")
    rows = manifest.get("files")
    if (
        not isinstance(directories, list)
        or any(type(item) is not str for item in directories)
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ProtocolError("HARP v16 archived admin snapshot inventory is malformed.")
    state = manifest.get("state")
    live_lexical = (
        boundary.lexical_root / authorization.WORKSPACE_OUTPUT_CANONICAL_PATH
    )
    retired = archive_root / RETIRED_ADMIN_OUTPUT
    content = archive_root / ARCHIVED_ADMIN_CONTENT
    if state == "ABSENT":
        if directories or rows or any(
            os.path.lexists(path) for path in (live_lexical, retired, content)
        ):
            raise ProtocolError("HARP v16 absent admin snapshot recovery drifted.")
        output = boundary.member(
            authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
            label="output identity",
            kind="future",
        )
        return output, manifest, {}, "absent"
    if (
        state != "WORKSPACE_ADMIN_PRISTINE"
        or tuple(sorted(directories)) != tuple(sorted(ADMIN_DIRECTORIES))
    ):
        raise ProtocolError("HARP v16 archived admin snapshot state drifted.")
    files: dict[str, bytes] = {}
    if not content.is_dir() or content.is_symlink():
        raise ProtocolError("HARP v16 archived admin snapshot content is absent.")
    row_paths = [row.get("path") for row in rows]
    if (
        any(type(relative) is not str for relative in row_paths)
        or len(row_paths) != len(set(row_paths))
        or set(row_paths) != set(ADMIN_FILES)
    ):
        raise ProtocolError("HARP v16 archived admin snapshot file inventory drifted.")
    for row in rows:
        relative = cast(str, row.get("path"))
        path = content / relative
        if not path.is_file() or path.is_symlink():
            raise ProtocolError("HARP v16 archived admin snapshot file is absent.")
        raw = path.read_bytes()
        if (
            type(row.get("sha256")) is not str
            or row.get("sha256") != sha256_bytes(raw)
            or type(row.get("size_bytes")) is not int
            or row.get("size_bytes") != len(raw)
        ):
            raise ProtocolError("HARP v16 archived admin snapshot hash drifted.")
        files[relative] = raw
    require_exact_snapshot_tree(
        content,
        directories=tuple(directories),
        files=files,
    )
    live_exists = os.path.lexists(live_lexical)
    retired_exists = os.path.lexists(retired)
    if live_exists == retired_exists:
        raise ProtocolError("HARP v16 admin output recovery location is ambiguous.")
    if live_exists:
        output = boundary.member(
            authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
            label="output identity",
            kind="directory",
        )
        require_exact_snapshot_tree(
            output,
            directories=tuple(directories),
            files=files,
        )
        return output, manifest, files, "live"
    if not retired.is_dir() or retired.is_symlink():
        raise ProtocolError("HARP v16 retired admin output is unsafe.")
    require_exact_snapshot_tree(
        retired,
        directories=tuple(directories),
        files=files,
    )
    output = boundary.member(
        authorization.WORKSPACE_OUTPUT_CANONICAL_PATH,
        label="output identity",
        kind="future",
    )
    return output, manifest, files, "retired"


def require_exact_snapshot_tree(
    root: Path,
    *,
    directories: tuple[str, ...],
    files: Mapping[str, bytes],
) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v16 admin snapshot root is unsafe.")
    actual_directories: list[str] = []
    actual_files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ProtocolError("HARP v16 admin snapshot contains a symlink.")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            actual_directories.append(relative)
        elif path.is_file():
            actual_files[relative] = path.read_bytes()
        else:
            raise ProtocolError("HARP v16 admin snapshot has a special member.")
    if tuple(actual_directories) != tuple(sorted(directories)) or actual_files != dict(
        files
    ):
        raise ProtocolError("HARP v16 admin snapshot tree drifted.")


def validate_partial_snapshot_tree(
    root: Path,
    *,
    directories: tuple[str, ...],
    files: Mapping[str, bytes],
) -> bool:
    """Validate a crash-interrupted prefix of the archived admin snapshot."""

    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v16 partial admin snapshot root is unsafe.")
    expected_directories = set(directories)
    actual_directories: set[str] = set()
    actual_files: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("HARP v16 partial admin snapshot has a symlink.")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files[relative] = path.read_bytes()
        else:
            raise ProtocolError("HARP v16 partial admin snapshot has a special member.")
    if not actual_directories.issubset(expected_directories) or any(
        relative not in files or raw != files[relative]
        for relative, raw in actual_files.items()
    ):
        raise ProtocolError("HARP v16 partial admin snapshot content drifted.")
    return actual_directories == expected_directories and actual_files == dict(files)


def read_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"HARP v16 {label} is absent or unsafe.")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"HARP v16 {label} is unreadable.") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise ProtocolError(f"HARP v16 {label} is not canonical.")
    return value


def validate_admin_config(
    path: Path,
    *,
    output_root: Path,
    active_config: HarpStage90V16Config,
    journal: ActivationJournal,
) -> None:
    resolved = load_config(path)
    if (
        not resolved.execution_authorized
        or resolved.expected_execution_amendment_sha256 != journal.amendment_sha256
        or Path(resolved.artifact_root).resolve() != output_root
        or resolved.expected_hashes != active_config.expected_hashes
        or resolved.protocol != active_config.protocol
        or resolved.model != active_config.model
        or resolved.runtime != active_config.runtime
        or resolved.claim_boundary != active_config.claim_boundary
        or resolved.input_artifact_ids != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("HARP v16 workspace-admin config binding drifted.")


def validate_admin_input_manifest(raw: bytes) -> None:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "HARP v16 workspace-admin input manifest is unreadable."
        ) from exc
    if not isinstance(value, dict):
        raise ProtocolError("HARP v16 workspace-admin input manifest is malformed.")
    rows = value.get("input_artifacts")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ProtocolError("HARP v16 workspace-admin input inventory is malformed.")
    identities = [row.get("artifact_id") for row in rows]
    if any(type(identity) is not str for identity in identities):
        raise ProtocolError("HARP v16 workspace-admin artifact identity is malformed.")
    if (
        value.get("schema_version") != "midogpp_input_artifacts_v2"
        or value.get("dataset_id") != "midogpp"
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("stage") != "90_oracles_and_diagnostics"
        or value.get("selection_used_target_eval_artifacts") is not False
        or len(identities) != len(set(identities))
        or set(identities) != set(INPUT_ARTIFACT_IDS)
        or any(row.get("exists") is not True for row in rows)
    ):
        raise ProtocolError("HARP v16 workspace-admin input binding drifted.")


__all__ = (
    "ADMIN_DIRECTORIES",
    "ADMIN_FILES",
    "inspect_workspace_admin_output",
    "load_recovery_admin_snapshot",
    "require_exact_snapshot_tree",
    "validate_admin_config",
    "validate_admin_input_manifest",
    "validate_partial_snapshot_tree",
)
