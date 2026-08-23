"""Closed-world bundle indexing for fresh-process validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json, sha256_file
from ..identity import canonical_hash


CONTENT_INDEX_MEMBER = "manifests/content_index.json"


def build_content_index(
    root: Path,
    *,
    required_members: Sequence[str],
    phase: str,
) -> dict[str, object]:
    canonical = tuple(sorted(set(str(value) for value in required_members)))
    if len(canonical) != len(tuple(required_members)) or CONTENT_INDEX_MEMBER in canonical:
        raise ProtocolError("P-DCAPS content inventory is duplicate or recursive.")
    rows = []
    for member in canonical:
        path = Path(root) / member
        if not path.is_file() or path.is_symlink():
            raise ProtocolError(f"P-DCAPS required bundle member is absent: {member}.")
        rows.append({"member": member, "sha256": sha256_file(path), "size": path.stat().st_size})
    base = {
        "schema_version": "pdcaps_content_index_v1",
        "phase": str(phase),
        "members": rows,
        "member_count": len(rows),
    }
    payload = {**base, "content_index_hash": canonical_hash(base)}
    atomic_json(Path(root) / CONTENT_INDEX_MEMBER, payload)
    return payload


def verify_content_index(root: Path) -> dict[str, object]:
    payload = read_json(Path(root) / CONTENT_INDEX_MEMBER)
    rows = payload.get("members")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ProtocolError("P-DCAPS content index is malformed.")
    unhashed = {key: value for key, value in payload.items() if key != "content_index_hash"}
    if (
        payload.get("member_count") != len(rows)
        or payload.get("content_index_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("P-DCAPS content index hash drifted.")
    for row in rows:
        path = Path(root) / str(row["member"])
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("sha256") != sha256_file(path)
            or row.get("size") != path.stat().st_size
        ):
            raise ProtocolError("P-DCAPS indexed member drifted.")
    return payload


__all__ = ("CONTENT_INDEX_MEMBER", "build_content_index", "verify_content_index")
