"""Source seal for SCEPTRE-owned executable and inherited scientific code."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .identity import canonical_hash, require_sha256


SOURCE_TREE_SCHEMA = "sceptre_v3_execution_source_tree_v1"
SOURCE_SNAPSHOT_SCHEMA = "sceptre_v3_execution_source_snapshot_v1"
SOURCE_ROOT_ROLE = (
    "sceptre_v3_executable_neutral_worker_runtime_and_inherited_scientific_python"
)
SOURCE_NAMESPACES = (
    "midogpp_thesis/cvae/diagnostics/fixed_bank_sceptre_router_v3",
    "midogpp_thesis/cvae/diagnostics/sceptre_runtime",
    "midogpp_thesis/cvae/diagnostics/fixed_bank_sceptre_router",
    "midogpp_thesis/cvae/routing/sceptre",
)
SOURCE_MEMBER_PATTERN = "|".join(f"{namespace}/**/*.py" for namespace in SOURCE_NAMESPACES)
_NORMALIZED_MEMBERS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        (
            "midogpp_thesis/cvae/diagnostics/"
            "fixed_bank_sceptre_router_v3/experiment_contracts.py"
        ): MappingProxyType(
            {
                "EXPECTED_EXECUTION_AMENDMENT_SHA256": (
                    "__SCEPTRE_V3_EXECUTION_AMENDMENT_SHA256__"
                )
            }
        )
    }
)


def package_source_root() -> Path:
    return Path(__file__).resolve().parent


def default_source_namespace_roots() -> Mapping[str, Path]:
    """Return the four production roots without exposing paths in the seal."""

    v3_root = package_source_root()
    diagnostics_root = v3_root.parent
    cvae_root = diagnostics_root.parent
    return MappingProxyType(
        {
            SOURCE_NAMESPACES[0]: v3_root,
            SOURCE_NAMESPACES[1]: diagnostics_root / "sceptre_runtime",
            SOURCE_NAMESPACES[2]: diagnostics_root / "fixed_bank_sceptre_router",
            SOURCE_NAMESPACES[3]: cvae_root / "routing/sceptre",
        }
    )


def build_source_snapshot_members(
    package_root: Path | None = None,
    *,
    namespace_roots: Mapping[str, Path] | None = None,
) -> tuple[dict[str, object], ...]:
    roots = _source_namespace_roots(package_root, namespace_roots)
    rows: list[dict[str, object]] = []
    normalized: set[str] = set()
    for namespace, root in roots:
        unsafe = tuple(path for path in root.rglob("*") if path.is_symlink())
        if unsafe:
            raise ProtocolError("SCEPTRE v3 source namespace contains a symlink.")
        paths = sorted(
            root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
        )
        if not paths:
            raise ProtocolError("SCEPTRE v3 source namespace is empty.")
        for path in paths:
            relative = path.relative_to(root).as_posix()
            member = f"{namespace}/{relative}"
            _validate_member_name(member)
            if not path.is_file():
                raise ProtocolError("SCEPTRE v3 source member is unsafe.")
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise ProtocolError("Cannot read SCEPTRE v3 source member.") from exc
            assignments = _NORMALIZED_MEMBERS.get(member)
            if assignments is not None:
                payload = _normalize_assignments(payload, assignments, member=member)
                normalized.add(member)
            rows.append(
                {
                    "member": member,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    if normalized != set(_NORMALIZED_MEMBERS):
        raise ProtocolError("SCEPTRE v3 normalized source inventory drifted.")
    if len(rows) != len({str(row["member"]) for row in rows}):
        raise ProtocolError("SCEPTRE v3 source member inventory is not unique.")
    rows.sort(key=lambda row: str(row["member"]))
    return tuple(rows)


def build_source_snapshot_payload(
    package_root: Path | None = None,
    *,
    namespace_roots: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    members = build_source_snapshot_members(
        package_root, namespace_roots=namespace_roots
    )
    tree_payload = {"schema_version": SOURCE_TREE_SCHEMA, "members": list(members)}
    tree_sha256 = canonical_hash(tree_payload)
    manifest = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "hash_algorithm": "sha256",
        "source_root_role": SOURCE_ROOT_ROLE,
        "member_pattern": SOURCE_MEMBER_PATTERN,
        "source_namespaces": list(SOURCE_NAMESPACES),
        "source_namespace_count": len(SOURCE_NAMESPACES),
        "normalized_external_anchor_members": {
            member: sorted(assignments)
            for member, assignments in sorted(_NORMALIZED_MEMBERS.items())
        },
        "member_count": len(members),
        "members": list(members),
        "tree_sha256": tree_sha256,
    }
    return {**manifest, "manifest_sha256": canonical_hash(manifest)}


def source_snapshot_identity(
    package_root: Path | None = None,
    *,
    namespace_roots: Mapping[str, Path] | None = None,
) -> Mapping[str, object]:
    payload = build_source_snapshot_payload(
        package_root, namespace_roots=namespace_roots
    )
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
    namespace_roots: Mapping[str, Path] | None = None,
) -> Mapping[str, object]:
    expected_manifest = require_sha256(expected_manifest_sha256, "source manifest")
    expected_tree = require_sha256(expected_tree_sha256, "source tree")
    if (
        isinstance(expected_member_count, bool)
        or not isinstance(expected_member_count, int)
        or expected_member_count <= 0
    ):
        raise ProtocolError("SCEPTRE v3 source member count drifted.")
    identity = source_snapshot_identity(
        package_root, namespace_roots=namespace_roots
    )
    if (
        identity["source_snapshot_manifest_sha256"] != expected_manifest
        or identity["source_snapshot_tree_sha256"] != expected_tree
        or identity["source_snapshot_member_count"] != expected_member_count
    ):
        raise ProtocolError("SCEPTRE v3 source bytes or inventory drifted.")
    return MappingProxyType({"status": "PASS", **dict(identity)})


def _source_namespace_roots(
    package_root: Path | None,
    namespace_roots: Mapping[str, Path] | None,
) -> tuple[tuple[str, Path], ...]:
    if package_root is not None and namespace_roots is not None:
        raise ProtocolError("SCEPTRE v3 source-root overrides are ambiguous.")
    if namespace_roots is not None:
        if set(namespace_roots) != set(SOURCE_NAMESPACES):
            raise ProtocolError("SCEPTRE v3 source namespace inventory drifted.")
        raw = {name: Path(namespace_roots[name]) for name in SOURCE_NAMESPACES}
    else:
        if package_root is None:
            raw = dict(default_source_namespace_roots())
        else:
            v3_root = Path(package_root)
            diagnostics_root = v3_root.parent
            cvae_root = diagnostics_root.parent
            raw = {
                SOURCE_NAMESPACES[0]: v3_root,
                SOURCE_NAMESPACES[1]: diagnostics_root / "sceptre_runtime",
                SOURCE_NAMESPACES[2]: diagnostics_root
                / "fixed_bank_sceptre_router",
                SOURCE_NAMESPACES[3]: cvae_root / "routing/sceptre",
            }
    rows: list[tuple[str, Path]] = []
    resolved: set[Path] = set()
    for namespace in SOURCE_NAMESPACES:
        root = raw[namespace]
        if root.is_symlink() or not root.is_dir():
            raise ProtocolError("SCEPTRE v3 source root is absent or unsafe.")
        try:
            physical = root.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError("Cannot resolve SCEPTRE v3 source root.") from exc
        if physical in resolved:
            raise ProtocolError("SCEPTRE v3 source namespaces share a root.")
        resolved.add(physical)
        rows.append((namespace, root))
    return tuple(rows)


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
        raise ProtocolError("SCEPTRE v3 source member name is unsafe.")


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
        raise ProtocolError(f"SCEPTRE v3 source anchor malformed: {member}.") from exc
    replacements: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in assignments:
            continue
        if node.end_lineno is None:
            raise ProtocolError("SCEPTRE v3 source anchor location unavailable.")
        seen.add(target.id)
        replacements.append(
            (
                node.lineno - 1,
                node.end_lineno,
                f"{target.id} = {assignments[target.id]!r}\n",
            )
        )
    if seen != set(assignments):
        raise ProtocolError(f"SCEPTRE v3 source anchors drifted: {member}.")
    lines = text.splitlines(keepends=True)
    for start, stop, replacement in sorted(replacements, reverse=True):
        lines[start:stop] = [replacement]
    return "".join(lines).encode("utf-8")


__all__ = (
    "SOURCE_MEMBER_PATTERN",
    "SOURCE_NAMESPACES",
    "SOURCE_ROOT_ROLE",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SOURCE_TREE_SCHEMA",
    "build_source_snapshot_members",
    "build_source_snapshot_payload",
    "default_source_namespace_roots",
    "package_source_root",
    "source_snapshot_identity",
    "validate_source_snapshot",
)
