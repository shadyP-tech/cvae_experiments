"""Closed-world source snapshot for the isolated executable v2 package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from .hashing import canonical_hash, require_sha256
from .identity import GovernanceError


SOURCE_SNAPSHOT_SCHEMA = "scale_bp_v2_execution_source_snapshot_v1"
SOURCE_TREE_SCHEMA = "scale_bp_v2_execution_source_tree_v1"
SOURCE_ROOT_ROLE = "scale_bp_v2_executable_python"


@dataclass(frozen=True, slots=True)
class SourceSnapshotReceipt:
    manifest_sha256: str
    tree_sha256: str
    member_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "source_snapshot_schema": SOURCE_SNAPSHOT_SCHEMA,
            "source_snapshot_manifest_sha256": self.manifest_sha256,
            "source_snapshot_tree_sha256": self.tree_sha256,
            "source_snapshot_member_count": self.member_count,
            "source_snapshot_excludes_bytecode_and_cache": True,
        }


def package_source_root() -> Path:
    return Path(__file__).resolve().parent


def build_source_snapshot_members(
    package_root: Path | None = None,
) -> tuple[dict[str, object], ...]:
    root = package_source_root() if package_root is None else Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise GovernanceError("SCALE-BP v2 source root is absent or unsafe.")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        _validate_member_name(relative)
        if path.is_symlink() or not path.is_file():
            raise GovernanceError("SCALE-BP v2 source member is unsafe.")
        payload = path.read_bytes()
        rows.append(
            {
                "member": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not rows:
        raise GovernanceError("SCALE-BP v2 source snapshot is empty.")
    return tuple(rows)


def build_source_snapshot_payload(package_root: Path | None = None) -> dict[str, object]:
    members = build_source_snapshot_members(package_root)
    tree = {"schema_version": SOURCE_TREE_SCHEMA, "members": list(members)}
    manifest = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "hash_algorithm": "sha256",
        "source_root_role": SOURCE_ROOT_ROLE,
        "member_pattern": "**/*.py",
        "member_count": len(members),
        "members": list(members),
        "tree_sha256": canonical_hash(tree),
    }
    return {**manifest, "manifest_sha256": canonical_hash(manifest)}


def source_snapshot_identity(package_root: Path | None = None) -> Mapping[str, object]:
    payload = build_source_snapshot_payload(package_root)
    return MappingProxyType(
        SourceSnapshotReceipt(
            str(payload["manifest_sha256"]),
            str(payload["tree_sha256"]),
            int(payload["member_count"]),
        ).to_payload()
    )


def validate_source_snapshot(
    *,
    expected_manifest_sha256: object,
    expected_tree_sha256: object,
    expected_member_count: object,
    package_root: Path | None = None,
) -> SourceSnapshotReceipt:
    expected_manifest = require_sha256(expected_manifest_sha256, "source manifest hash")
    expected_tree = require_sha256(expected_tree_sha256, "source tree hash")
    if (
        isinstance(expected_member_count, bool)
        or not isinstance(expected_member_count, int)
        or expected_member_count <= 0
    ):
        raise GovernanceError("SCALE-BP v2 source member count is invalid.")
    identity = source_snapshot_identity(package_root)
    receipt = SourceSnapshotReceipt(
        str(identity["source_snapshot_manifest_sha256"]),
        str(identity["source_snapshot_tree_sha256"]),
        int(identity["source_snapshot_member_count"]),
    )
    if (
        receipt.manifest_sha256 != expected_manifest
        or receipt.tree_sha256 != expected_tree
        or receipt.member_count != expected_member_count
    ):
        raise GovernanceError("SCALE-BP v2 source snapshot drifted.")
    return receipt


def _validate_member_name(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", "..", "__pycache__"} for part in path.parts)
        or path.suffix != ".py"
    ):
        raise GovernanceError("SCALE-BP v2 source member name is unsafe.")


__all__ = (
    "SOURCE_ROOT_ROLE",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SOURCE_TREE_SCHEMA",
    "SourceSnapshotReceipt",
    "build_source_snapshot_members",
    "build_source_snapshot_payload",
    "package_source_root",
    "source_snapshot_identity",
    "validate_source_snapshot",
)
