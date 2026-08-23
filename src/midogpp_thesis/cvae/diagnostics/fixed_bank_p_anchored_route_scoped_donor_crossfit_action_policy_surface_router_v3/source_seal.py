"""Independent source seals for the exhausted v2 base and v3 repair sibling."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
from pathlib import Path, PurePosixPath
from typing import Mapping

from ...protocol import ProtocolError
from .identity import canonical_hash, require_sha256


V2_SOURCE_SNAPSHOT_SCHEMA = "pdcaps_v2_source_snapshot_v1"
V2_SOURCE_TREE_SCHEMA = "pdcaps_v2_source_snapshot_tree_v1"
V2_SOURCE_ROOT_ROLE = "pdcaps_v2_and_package_local_scientific_python"
V2_CONTRACT_MEMBER = "v2/experiment_contracts.py"
EXPECTED_V2_SOURCE_MANIFEST_SHA256 = (
    "3dc6d096ad607fe550eac47b114332fd6ac9ebec5d9cfb59e80897e9a982addc"
)
EXPECTED_V2_SOURCE_TREE_SHA256 = (
    "f457d8678eb93fe51520c9fcc188c8d44f8331aec6f37c88124a677bfcc2d5cb"
)
EXPECTED_V2_SOURCE_MEMBER_COUNT = 105

V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA = "pdcaps_v3_repair_source_snapshot_v1"
V3_REPAIR_SOURCE_TREE_SCHEMA = "pdcaps_v3_repair_source_tree_v1"
V3_REPAIR_SOURCE_ROOT_ROLE = "pdcaps_v3_nullable_admission_repair_python"
SOURCE_MEMBER_PATTERN = "**/*.py"

# These three values are the only normalized assignments in the v3 sibling.
# Replacing them at final seal therefore cannot change the source identity they
# bind.  No executable authority or artifact hash is normalized.
EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256 = (
    "37b8e51f8d0900212ec4bfc8bd68b14ddbde1ed783eaacf86e8301dd9295b4a7"
)
EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256 = (
    "df35265e8d27aa602c3ae6c3fcebdc2a3c4838effa4d9f1200e9deb12e7e0a3e"
)
EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT = 13
EXPECTED_COMBINED_SOURCE_SEAL_SHA256 = (
    "98252133ec58838851093e3a55434306b531280e57edbe70efbb7fe1d14a3994"
)

_V2_NORMALIZED_ASSIGNMENTS: Mapping[str, object] = {
    "EXPECTED_LEDGER_AMENDMENT_SHA256": "__PDCAPS_V2_AMENDMENT_SHA256__",
    "EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256": (
        "__PDCAPS_V2_SOURCE_SNAPSHOT_MANIFEST_SHA256__"
    ),
    "EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256": (
        "__PDCAPS_V2_SOURCE_SNAPSHOT_TREE_SHA256__"
    ),
    "EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT": -1,
}
_V3_NORMALIZED_ASSIGNMENTS: Mapping[str, object] = {
    "EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256": (
        "__PDCAPS_V3_REPAIR_SOURCE_MANIFEST_SHA256__"
    ),
    "EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256": (
        "__PDCAPS_V3_REPAIR_SOURCE_TREE_SHA256__"
    ),
    "EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT": -1,
    "EXPECTED_COMBINED_SOURCE_SEAL_SHA256": (
        "__PDCAPS_V3_COMBINED_SOURCE_SEAL_SHA256__"
    ),
}


def v2_base_source_root() -> Path:
    return Path(__file__).resolve().parent.parent / (
        "fixed_bank_p_anchored_route_scoped_"
        "donor_crossfit_action_policy_surface_router"
    )


def v3_repair_source_root() -> Path:
    return Path(__file__).resolve().parent


def _validate_member_name(value: str, role: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", "..", "__pycache__"} for part in path.parts)
        or path.suffix != ".py"
    ):
        raise ProtocolError(f"P-DCAPS v3 {role} source member name is unsafe.")


def _normalize_assignments(
    payload: bytes,
    assignments: Mapping[str, object],
    *,
    role: str,
) -> bytes:
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProtocolError(
            f"P-DCAPS v3 {role} source-anchor module is malformed."
        ) from exc
    replacements: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in assignments:
            continue
        if node.end_lineno is None:
            raise ProtocolError(
                f"P-DCAPS v3 {role} source-anchor location is unavailable."
            )
        seen.add(target.id)
        replacements.append(
            (
                node.lineno - 1,
                node.end_lineno,
                f"{target.id} = {assignments[target.id]!r}\n",
            )
        )
    if seen != set(assignments):
        raise ProtocolError(
            f"P-DCAPS v3 {role} source-anchor assignments drifted."
        )
    lines = text.splitlines(keepends=True)
    for start, stop, replacement in sorted(replacements, reverse=True):
        lines[start:stop] = [replacement]
    return "".join(lines).encode("utf-8")


def _source_members(
    root: Path,
    *,
    role: str,
    normalized_member: str,
    normalized_assignments: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError(f"P-DCAPS v3 {role} source root is absent or unsafe.")
    paths = sorted(
        root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix()
    )
    if not paths:
        raise ProtocolError(f"P-DCAPS v3 {role} source snapshot is empty.")
    rows: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _validate_member_name(relative, role)
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(
                f"P-DCAPS v3 {role} source member is absent or unsafe."
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(
                f"Cannot read P-DCAPS v3 {role} source member."
            ) from exc
        if relative == normalized_member:
            payload = _normalize_assignments(
                payload, normalized_assignments, role=role
            )
        rows.append(
            {
                "member": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if normalized_member not in {str(row["member"]) for row in rows}:
        raise ProtocolError(
            f"P-DCAPS v3 {role} normalized source member is absent."
        )
    return tuple(rows)


def _source_payload(
    *,
    members: tuple[dict[str, object], ...],
    snapshot_schema: str,
    tree_schema: str,
    root_role: str,
    normalized_member: str,
    normalized_assignments: Mapping[str, object],
) -> dict[str, object]:
    tree_payload = {"schema_version": tree_schema, "members": list(members)}
    tree_hash = canonical_hash(tree_payload)
    manifest = {
        "schema_version": snapshot_schema,
        "hash_algorithm": "sha256",
        "source_root_role": root_role,
        "member_pattern": SOURCE_MEMBER_PATTERN,
        "normalized_external_anchor_member": normalized_member,
        "normalized_external_anchor_names": sorted(normalized_assignments),
        "member_count": len(members),
        "members": list(members),
        "tree_sha256": tree_hash,
    }
    return {**manifest, "manifest_sha256": canonical_hash(manifest)}


def build_v2_base_source_snapshot_payload(
    package_root: Path | None = None,
) -> dict[str, object]:
    root = v2_base_source_root() if package_root is None else Path(package_root)
    members = _source_members(
        root,
        role="v2 base",
        normalized_member=V2_CONTRACT_MEMBER,
        normalized_assignments=_V2_NORMALIZED_ASSIGNMENTS,
    )
    return _source_payload(
        members=members,
        snapshot_schema=V2_SOURCE_SNAPSHOT_SCHEMA,
        tree_schema=V2_SOURCE_TREE_SCHEMA,
        root_role=V2_SOURCE_ROOT_ROLE,
        normalized_member=V2_CONTRACT_MEMBER,
        normalized_assignments=_V2_NORMALIZED_ASSIGNMENTS,
    )


def build_v3_repair_source_snapshot_payload(
    package_root: Path | None = None,
) -> dict[str, object]:
    root = v3_repair_source_root() if package_root is None else Path(package_root)
    members = _source_members(
        root,
        role="v3 repair",
        normalized_member="source_seal.py",
        normalized_assignments=_V3_NORMALIZED_ASSIGNMENTS,
    )
    return _source_payload(
        members=members,
        snapshot_schema=V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
        tree_schema=V3_REPAIR_SOURCE_TREE_SCHEMA,
        root_role=V3_REPAIR_SOURCE_ROOT_ROLE,
        normalized_member="source_seal.py",
        normalized_assignments=_V3_NORMALIZED_ASSIGNMENTS,
    )


def _identity(payload: Mapping[str, object], prefix: str) -> dict[str, object]:
    """Return a plain, process-safe seal DTO.

    Seal receipts may cross a ``spawn`` process boundary.  A mapping proxy is
    useful for in-process immutability but is not pickleable, so the canonical
    public DTO is deliberately a detached plain dictionary of scalar values.
    """

    return {
        f"{prefix}_schema": payload["schema_version"],
        f"{prefix}_manifest_sha256": payload["manifest_sha256"],
        f"{prefix}_tree_sha256": payload["tree_sha256"],
        f"{prefix}_member_count": payload["member_count"],
        f"{prefix}_member_pattern": SOURCE_MEMBER_PATTERN,
        f"{prefix}_excludes_bytecode_and_cache": True,
    }


def v2_base_source_snapshot_identity(
    package_root: Path | None = None,
) -> dict[str, object]:
    return _identity(
        build_v2_base_source_snapshot_payload(package_root),
        "v2_base_source_snapshot",
    )


def v3_repair_source_snapshot_identity(
    package_root: Path | None = None,
) -> dict[str, object]:
    return _identity(
        build_v3_repair_source_snapshot_payload(package_root),
        "v3_repair_source_snapshot",
    )


def validate_v2_base_source_seal(
    package_root: Path | None = None,
) -> dict[str, object]:
    identity = v2_base_source_snapshot_identity(package_root)
    if (
        identity["v2_base_source_snapshot_manifest_sha256"]
        != EXPECTED_V2_SOURCE_MANIFEST_SHA256
        or identity["v2_base_source_snapshot_tree_sha256"]
        != EXPECTED_V2_SOURCE_TREE_SHA256
        or identity["v2_base_source_snapshot_member_count"]
        != EXPECTED_V2_SOURCE_MEMBER_COUNT
    ):
        raise ProtocolError(
            "P-DCAPS v3 inherited v2/base source bytes or inventory drifted."
        )
    return {"status": "PASS", **identity}


def validate_v3_repair_source_seal(
    *,
    expected_manifest_sha256: object = EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    expected_tree_sha256: object = EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    expected_member_count: object = EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    package_root: Path | None = None,
) -> dict[str, object]:
    expected_manifest = require_sha256(
        expected_manifest_sha256, "repair source manifest hash"
    )
    expected_tree = require_sha256(
        expected_tree_sha256, "repair source tree hash"
    )
    if (
        isinstance(expected_member_count, bool)
        or not isinstance(expected_member_count, int)
        or expected_member_count <= 0
    ):
        raise ProtocolError("P-DCAPS v3 repair source member count drifted.")
    identity = v3_repair_source_snapshot_identity(package_root)
    if (
        identity["v3_repair_source_snapshot_manifest_sha256"]
        != expected_manifest
        or identity["v3_repair_source_snapshot_tree_sha256"] != expected_tree
        or identity["v3_repair_source_snapshot_member_count"]
        != expected_member_count
    ):
        raise ProtocolError(
            "P-DCAPS v3 repair source bytes or inventory drifted."
        )
    return {"status": "PASS", **identity}


@dataclass(frozen=True)
class CombinedSourceSeal:
    """Pickle-safe receipt joining the independently validated source scopes."""

    v2_base: dict[str, object]
    v3_repair: dict[str, object]
    combined_source_seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        base = dict(self.v2_base)
        repair = dict(self.v3_repair)
        if base.get("status") != "PASS" or repair.get("status") != "PASS":
            raise ProtocolError("P-DCAPS v3 combined source seal is incomplete.")
        object.__setattr__(self, "v2_base", base)
        object.__setattr__(self, "v3_repair", repair)
        object.__setattr__(
            self,
            "combined_source_seal_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v3_combined_source_seal_v1",
                    "v2_base": base,
                    "v3_repair": repair,
                    "source_scopes_are_disjoint": True,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v3_combined_source_seal_v1",
            "v2_base": dict(self.v2_base),
            "v3_repair": dict(self.v3_repair),
            "source_scopes_are_disjoint": True,
            "combined_source_seal_hash": self.combined_source_seal_hash,
        }


def build_combined_source_seal(
    *,
    v2_package_root: Path | None = None,
    v3_package_root: Path | None = None,
) -> CombinedSourceSeal:
    return CombinedSourceSeal(
        validate_v2_base_source_seal(v2_package_root),
        validate_v3_repair_source_seal(package_root=v3_package_root),
    )


def validate_combined_source_seal(
    *,
    expected_combined_source_seal_hash: object = (
        EXPECTED_COMBINED_SOURCE_SEAL_SHA256
    ),
    v2_package_root: Path | None = None,
    v3_package_root: Path | None = None,
) -> CombinedSourceSeal:
    """Validate both disjoint scopes and their pinned combined receipt."""

    expected = require_sha256(
        expected_combined_source_seal_hash,
        "combined source seal hash",
    )
    seal = build_combined_source_seal(
        v2_package_root=v2_package_root,
        v3_package_root=v3_package_root,
    )
    if seal.combined_source_seal_hash != expected:
        raise ProtocolError("P-DCAPS v3 combined source seal drifted.")
    return seal


__all__ = (
    "CombinedSourceSeal",
    "EXPECTED_COMBINED_SOURCE_SEAL_SHA256",
    "EXPECTED_V2_SOURCE_MANIFEST_SHA256",
    "EXPECTED_V2_SOURCE_MEMBER_COUNT",
    "EXPECTED_V2_SOURCE_TREE_SHA256",
    "EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256",
    "EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT",
    "EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256",
    "V2_SOURCE_SNAPSHOT_SCHEMA",
    "V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA",
    "build_combined_source_seal",
    "build_v2_base_source_snapshot_payload",
    "build_v3_repair_source_snapshot_payload",
    "v2_base_source_snapshot_identity",
    "v3_repair_source_snapshot_identity",
    "validate_v2_base_source_seal",
    "validate_v3_repair_source_seal",
    "validate_combined_source_seal",
)
