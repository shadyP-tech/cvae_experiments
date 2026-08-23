"""Disjoint source seals for P-DCAPS v2, v3 repair, and v4 execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.source_seal import (
    build_v2_base_source_snapshot_payload,
    build_v3_repair_source_snapshot_payload,
    v2_base_source_root,
    v3_repair_source_root,
)
from .experiment_contracts import (
    EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256,
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256,
    EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT,
    EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256,
    V2_SOURCE_SNAPSHOT_SCHEMA,
    V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
    V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA,
)
from .identity import canonical_hash, require_sha256


V4_EXECUTION_SOURCE_TREE_SCHEMA = "pdcaps_v4_execution_source_tree_v1"
V4_EXECUTION_SOURCE_ROOT_ROLE = "pdcaps_v4_executable_python"
SOURCE_MEMBER_PATTERN = "**/*.py"
_NORMALIZED_MEMBERS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "experiment_contracts.py": MappingProxyType(
            {
                "EXPECTED_LEDGER_AMENDMENT_SHA256": (
                    "__PDCAPS_V4_AMENDMENT_SHA256__"
                ),
                "EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256": (
                    "__PDCAPS_V4_EXECUTION_SOURCE_MANIFEST_SHA256__"
                ),
                "EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256": (
                    "__PDCAPS_V4_EXECUTION_SOURCE_TREE_SHA256__"
                ),
                "EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT": -1,
                "EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256": (
                    "__PDCAPS_V4_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256__"
                ),
            }
        )
    }
)


@dataclass(frozen=True)
class CombinedSourceSeal:
    """Plain spawn-safe receipt for all three disjoint source scopes."""

    v2_manifest_sha256: str
    v2_tree_sha256: str
    v2_member_count: int
    v3_manifest_sha256: str
    v3_tree_sha256: str
    v3_member_count: int
    v4_manifest_sha256: str
    v4_tree_sha256: str
    v4_member_count: int
    combined_source_seal_sha256: str

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


def v4_execution_source_root() -> Path:
    return Path(__file__).resolve().parent


def build_v4_execution_source_snapshot_members(
    package_root: Path | None = None,
) -> tuple[dict[str, object], ...]:
    root = v4_execution_source_root() if package_root is None else Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("P-DCAPS v4 execution source root is absent or unsafe.")
    paths = sorted(
        root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix()
    )
    if not paths:
        raise ProtocolError("P-DCAPS v4 execution source snapshot is empty.")
    rows: list[dict[str, object]] = []
    seen_normalized: set[str] = set()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _validate_member_name(relative)
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("P-DCAPS v4 execution source member is unsafe.")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProtocolError("Cannot read P-DCAPS v4 execution source member.") from exc
        assignments = _NORMALIZED_MEMBERS.get(relative)
        if assignments is not None:
            payload = _normalize_assignments(payload, assignments, member=relative)
            seen_normalized.add(relative)
        rows.append(
            {
                "member": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if seen_normalized != set(_NORMALIZED_MEMBERS):
        raise ProtocolError("P-DCAPS v4 normalized source inventory drifted.")
    return tuple(rows)


def build_v4_execution_source_snapshot_payload(
    package_root: Path | None = None,
) -> dict[str, object]:
    members = build_v4_execution_source_snapshot_members(package_root)
    tree_payload = {
        "schema_version": V4_EXECUTION_SOURCE_TREE_SCHEMA,
        "members": list(members),
    }
    tree_sha256 = canonical_hash(tree_payload)
    manifest = {
        "schema_version": V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA,
        "hash_algorithm": "sha256",
        "source_root_role": V4_EXECUTION_SOURCE_ROOT_ROLE,
        "member_pattern": SOURCE_MEMBER_PATTERN,
        "normalized_external_anchor_members": {
            member: sorted(assignments)
            for member, assignments in sorted(_NORMALIZED_MEMBERS.items())
        },
        "member_count": len(members),
        "members": list(members),
        "tree_sha256": tree_sha256,
    }
    return {**manifest, "manifest_sha256": canonical_hash(manifest)}


def build_combined_source_seal_payload(
    *,
    v2_root: Path | None = None,
    v3_root: Path | None = None,
    v4_root: Path | None = None,
) -> dict[str, object]:
    v2 = build_v2_base_source_snapshot_payload(v2_root)
    v3 = build_v3_repair_source_snapshot_payload(v3_root)
    v4 = build_v4_execution_source_snapshot_payload(v4_root)
    payload = {
        "schema_version": "pdcaps_v4_combined_three_scope_source_seal_v1",
        "source_scopes_are_disjoint": True,
        "v2_base": _scope_identity(v2),
        "v3_nullable_admission_repair": _scope_identity(v3),
        "v4_executable_orchestration": _scope_identity(v4),
    }
    return {**payload, "combined_source_seal_sha256": canonical_hash(payload)}


def source_snapshot_identity(
    package_root: Path | None = None,
) -> Mapping[str, object]:
    payload = build_v4_execution_source_snapshot_payload(package_root)
    return MappingProxyType(
        {
            "source_snapshot_schema": payload["schema_version"],
            "source_snapshot_manifest_sha256": payload["manifest_sha256"],
            "source_snapshot_tree_sha256": payload["tree_sha256"],
            "source_snapshot_member_count": payload["member_count"],
            "source_snapshot_member_pattern": SOURCE_MEMBER_PATTERN,
            "source_snapshot_excludes_bytecode_and_cache": True,
        }
    )


def validate_source_snapshot(
    *,
    expected_manifest_sha256: object,
    expected_tree_sha256: object,
    expected_member_count: object,
    package_root: Path | None = None,
) -> Mapping[str, object]:
    expected_manifest = require_sha256(
        expected_manifest_sha256, "v4 source snapshot manifest hash"
    )
    expected_tree = require_sha256(
        expected_tree_sha256, "v4 source snapshot tree hash"
    )
    if (
        isinstance(expected_member_count, bool)
        or not isinstance(expected_member_count, int)
        or expected_member_count <= 0
    ):
        raise ProtocolError("P-DCAPS v4 source snapshot member count drifted.")
    identity = source_snapshot_identity(package_root)
    if (
        identity["source_snapshot_manifest_sha256"] != expected_manifest
        or identity["source_snapshot_tree_sha256"] != expected_tree
        or identity["source_snapshot_member_count"] != expected_member_count
    ):
        raise ProtocolError("P-DCAPS v4 source snapshot bytes or inventory drifted.")
    return MappingProxyType({"status": "PASS", **dict(identity)})


def validate_combined_source_seal(
    *,
    v2_root: Path | None = None,
    v3_root: Path | None = None,
    v4_root: Path | None = None,
) -> CombinedSourceSeal:
    payload = build_combined_source_seal_payload(
        v2_root=v2_root, v3_root=v3_root, v4_root=v4_root
    )
    v2 = payload["v2_base"]
    v3 = payload["v3_nullable_admission_repair"]
    v4 = payload["v4_executable_orchestration"]
    if not all(isinstance(row, Mapping) for row in (v2, v3, v4)):
        raise ProtocolError("P-DCAPS v4 source-seal payload drifted.")
    expected = (
        (v2, V2_SOURCE_SNAPSHOT_SCHEMA, EXPECTED_V2_SOURCE_MANIFEST_SHA256,
         EXPECTED_V2_SOURCE_TREE_SHA256, EXPECTED_V2_SOURCE_MEMBER_COUNT),
        (v3, V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
         EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
         EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
         EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT),
        (v4, V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA,
         EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256,
         EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256,
         EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT),
    )
    if any(
        row.get("schema_version") != schema
        or row.get("manifest_sha256") != manifest
        or row.get("tree_sha256") != tree
        or row.get("member_count") != count
        for row, schema, manifest, tree, count in expected
    ):
        raise ProtocolError("P-DCAPS v4 source scope drifted.")
    combined = require_sha256(
        payload["combined_source_seal_sha256"], "combined source seal"
    )
    if combined != EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256:
        raise ProtocolError("P-DCAPS v4 combined source seal drifted.")
    return CombinedSourceSeal(
        str(v2["manifest_sha256"]), str(v2["tree_sha256"]), int(v2["member_count"]),
        str(v3["manifest_sha256"]), str(v3["tree_sha256"]), int(v3["member_count"]),
        str(v4["manifest_sha256"]), str(v4["tree_sha256"]), int(v4["member_count"]),
        combined,
    )


def _scope_identity(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": payload["schema_version"],
        "manifest_sha256": payload["manifest_sha256"],
        "tree_sha256": payload["tree_sha256"],
        "member_count": payload["member_count"],
    }


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
        raise ProtocolError("P-DCAPS v4 source member name is unsafe.")


def _normalize_assignments(
    payload: bytes,
    assignments: Mapping[str, object],
    *,
    member: str,
) -> bytes:
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProtocolError(f"P-DCAPS v4 source anchor malformed: {member}.") from exc
    replacements: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in assignments:
            continue
        if node.end_lineno is None:
            raise ProtocolError("P-DCAPS v4 source-anchor location unavailable.")
        seen.add(target.id)
        replacements.append(
            (
                node.lineno - 1,
                node.end_lineno,
                f"{target.id} = {assignments[target.id]!r}\n",
            )
        )
    if seen != set(assignments):
        raise ProtocolError(f"P-DCAPS v4 source-anchor assignments drifted: {member}.")
    lines = text.splitlines(keepends=True)
    for start, stop, replacement in sorted(replacements, reverse=True):
        lines[start:stop] = [replacement]
    return "".join(lines).encode("utf-8")


__all__ = (
    "CombinedSourceSeal",
    "SOURCE_MEMBER_PATTERN",
    "V4_EXECUTION_SOURCE_ROOT_ROLE",
    "V4_EXECUTION_SOURCE_TREE_SCHEMA",
    "build_combined_source_seal_payload",
    "build_v4_execution_source_snapshot_members",
    "build_v4_execution_source_snapshot_payload",
    "source_snapshot_identity",
    "validate_combined_source_seal",
    "validate_source_snapshot",
    "v2_base_source_root",
    "v3_repair_source_root",
    "v4_execution_source_root",
)
