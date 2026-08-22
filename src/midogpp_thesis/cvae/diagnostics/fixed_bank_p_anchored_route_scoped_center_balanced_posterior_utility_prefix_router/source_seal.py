"""Deterministic byte seal for the complete CBPUPR v2 Python source tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


SOURCE_MANIFEST_FILENAME = "repair_source_manifest_v2.json"
SOURCE_MANIFEST_SCHEMA_VERSION = "fixed_bank_cbpupr_repair_source_manifest_v1"
SOURCE_TREE_SCHEMA_VERSION = "fixed_bank_cbpupr_repair_source_tree_v1"
SOURCE_MEMBER_PATTERN = "**/*.py"


def package_source_root() -> Path:
    """Return the package directory whose Python bytes are authorized."""

    return Path(__file__).resolve().parent


def source_manifest_path() -> Path:
    return package_source_root() / SOURCE_MANIFEST_FILENAME


def build_source_members(package_root: Path) -> list[dict[str, object]]:
    """Build the exact, sorted inventory of regular Python source files."""

    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("CBPUPR repair source root is not a regular directory.")
    paths = sorted(root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix())
    if not paths:
        raise ProtocolError("CBPUPR repair source tree is empty.")
    rows: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _validate_member_name(relative)
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("CBPUPR repair source member is not a regular file.")
        payload = path.read_bytes()
        rows.append(
            {
                "member": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def build_source_manifest_payload(package_root: Path) -> dict[str, object]:
    """Build the canonical manifest payload used by the offline sealing step."""

    members = build_source_members(package_root)
    tree_payload = {
        "schema_version": SOURCE_TREE_SCHEMA_VERSION,
        "members": members,
    }
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "source_root_role": "cbpupr_v2_router_python_package",
        "member_pattern": SOURCE_MEMBER_PATTERN,
        "member_count": len(members),
        "members": members,
        "tree_sha256": canonical_hash(tree_payload),
    }


def load_source_manifest(
    manifest_path: Path | None = None,
) -> Mapping[str, object]:
    """Read and structurally validate the checked-in repair-source manifest."""

    path = Path(manifest_path) if manifest_path is not None else source_manifest_path()
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("CBPUPR repair source manifest is not a regular file.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read CBPUPR repair source manifest.") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "hash_algorithm",
        "source_root_role",
        "member_pattern",
        "member_count",
        "members",
        "tree_sha256",
    }:
        raise ProtocolError("CBPUPR repair source manifest schema drifted.")
    members = payload.get("members")
    if (
        payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION
        or payload.get("hash_algorithm") != "sha256"
        or payload.get("source_root_role") != "cbpupr_v2_router_python_package"
        or payload.get("member_pattern") != SOURCE_MEMBER_PATTERN
        or not isinstance(members, list)
        or payload.get("member_count") != len(members)
        or not members
    ):
        raise ProtocolError("CBPUPR repair source manifest identity drifted.")
    normalized: list[dict[str, object]] = []
    previous = ""
    for row in members:
        if not isinstance(row, dict) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("CBPUPR repair source manifest member drifted.")
        member = str(row.get("member"))
        _validate_member_name(member)
        if member <= previous:
            raise ProtocolError("CBPUPR repair source manifest order drifted.")
        previous = member
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProtocolError("CBPUPR repair source member size drifted.")
        normalized.append(
            {
                "member": member,
                "size_bytes": size,
                "sha256": require_sha256(
                    row.get("sha256"), "CBPUPR repair source member hash"
                ),
            }
        )
    tree_payload = {
        "schema_version": SOURCE_TREE_SCHEMA_VERSION,
        "members": normalized,
    }
    tree_hash = require_sha256(
        payload.get("tree_sha256"), "CBPUPR repair source tree hash"
    )
    if canonical_hash(tree_payload) != tree_hash:
        raise ProtocolError("CBPUPR repair source tree hash drifted.")
    return MappingProxyType({**payload, "members": tuple(normalized)})


def validate_repair_source_seal(
    *,
    package_root: Path | None = None,
    manifest_path: Path | None = None,
    expected_manifest_sha256: object | None = None,
    expected_tree_sha256: object | None = None,
) -> Mapping[str, object]:
    """Verify exact package membership, bytes, and external hash anchors."""

    root = Path(package_root) if package_root is not None else package_source_root()
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else root / SOURCE_MANIFEST_FILENAME
    )
    manifest = load_source_manifest(manifest_file)
    observed = build_source_members(root)
    expected_members = [dict(row) for row in manifest["members"]]
    if observed != expected_members:
        raise ProtocolError("CBPUPR repair source bytes or membership drifted.")
    manifest_hash = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    tree_hash = str(manifest["tree_sha256"])
    if (
        expected_manifest_sha256 is not None
        and manifest_hash
        != require_sha256(
            expected_manifest_sha256, "CBPUPR expected repair source manifest hash"
        )
    ):
        raise ProtocolError("CBPUPR repair source manifest hash drifted.")
    if (
        expected_tree_sha256 is not None
        and tree_hash
        != require_sha256(
            expected_tree_sha256, "CBPUPR expected repair source tree hash"
        )
    ):
        raise ProtocolError("CBPUPR repair source tree identity drifted.")
    return MappingProxyType(
        {
            "status": "PASS",
            "repair_source_manifest_validated": True,
            "repair_source_manifest_member": SOURCE_MANIFEST_FILENAME,
            "repair_source_manifest_sha256": manifest_hash,
            "repair_source_tree_sha256": tree_hash,
            "repair_source_member_count": len(observed),
        }
    )


def source_seal_identity() -> Mapping[str, object]:
    """Return the self-validated identity embedded into external contracts."""

    return validate_repair_source_seal()


def _validate_member_name(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".py"
    ):
        raise ProtocolError("CBPUPR repair source member name is unsafe.")


__all__ = (
    "SOURCE_MANIFEST_FILENAME",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "SOURCE_MEMBER_PATTERN",
    "SOURCE_TREE_SCHEMA_VERSION",
    "build_source_manifest_payload",
    "build_source_members",
    "load_source_manifest",
    "package_source_root",
    "source_manifest_path",
    "source_seal_identity",
    "validate_repair_source_seal",
)
