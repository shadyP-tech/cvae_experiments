"""Nofollow hashing and exact file-inventory checks for complete v3 artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .output_persistence import (
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    COMPLETE_CATALOG_MEMBERS,
    COMPLETE_INTERNAL_MEMBERS,
    _sha256_file,
)


RUN_STATE_MEMBER = "reports/run_state.json"


def safe_artifact_root(value: str | Path) -> Path:
    root = Path(os.path.abspath(Path(value)))
    current = root
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 complete artifact path contains a symlink.")
        if current == current.parent:
            break
        current = current.parent
    if (
        not root.is_absolute()
        or root == Path(root.anchor)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v3 complete artifact root is unsafe.")
    return root


def validate_exact_artifact_inventory(
    root: Path,
    *,
    index_present: bool,
) -> None:
    """Reject unsafe members and extras before any semantic loader runs."""

    expected_catalog = tuple(
        member
        for member in COMPLETE_CATALOG_MEMBERS
        if index_present or member != COMPLETE_ARTIFACT_INDEX_MEMBER
    )
    expected = tuple(sorted((*expected_catalog, *COMPLETE_INTERNAL_MEMBERS)))
    observed: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("OE-PPUR v3 complete artifact contains a symlink.")
        if path.is_dir():
            continue
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProtocolError("OE-PPUR v3 complete artifact member is unsafe.")
        observed.append(path.relative_to(root).as_posix())
    if tuple(sorted(observed)) != expected:
        raise ProtocolError("OE-PPUR v3 complete artifact inventory drifted.")


def catalog_member_hashes(
    root: Path,
    *,
    expected_complete_file_sha256: str | None = None,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for member in COMPLETE_CATALOG_MEMBERS:
        if member == COMPLETE_ARTIFACT_INDEX_MEMBER:
            continue
        if member == RUN_STATE_MEMBER and expected_complete_file_sha256 is not None:
            hashes[member] = require_sha256(
                expected_complete_file_sha256,
                "prepared COMPLETE file hash",
            )
        else:
            hashes[member] = _sha256_file(root / member)
    return hashes


def internal_member_hashes(root: Path) -> dict[str, str]:
    return {member: _sha256_file(root / member) for member in COMPLETE_INTERNAL_MEMBERS}


def artifact_inventory_hash(
    member_hashes: Mapping[str, str],
    internal_hashes: Mapping[str, str],
) -> str:
    expected_hashed = tuple(
        member
        for member in COMPLETE_CATALOG_MEMBERS
        if member != COMPLETE_ARTIFACT_INDEX_MEMBER
    )
    if (
        set(member_hashes) != set(expected_hashed)
        or set(internal_hashes) != set(COMPLETE_INTERNAL_MEMBERS)
    ):
        raise ProtocolError("OE-PPUR v3 complete artifact hash inventory drifted.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_complete_artifact_content_inventory_v1",
            "catalog_members": list(COMPLETE_CATALOG_MEMBERS),
            "catalog_member_sha256": dict(member_hashes),
            "self_index_member": COMPLETE_ARTIFACT_INDEX_MEMBER,
            "internal_member_sha256": dict(internal_hashes),
        }
    )


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [thaw_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ProtocolError("OE-PPUR v3 complete artifact payload is not canonical JSON.")


__all__ = ()
