"""Deterministic byte seal for the complete SCALE-BP Python source tree."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


SOURCE_MANIFEST_FILENAME = "source_manifest_v1.json"
SOURCE_MANIFEST_SCHEMA_VERSION = "scale_bp_v1_source_manifest_v1"
SOURCE_TREE_SCHEMA_VERSION = "scale_bp_v1_source_tree_v1"
SOURCE_MEMBER_PATTERN = "**/*.py"
SOURCE_ROOT_ROLE = "scale_bp_v1_router_python_package"


@dataclass(frozen=True, slots=True)
class SourceSealReceipt:
    """Primitive receipt safe to retain without open files or mapping proxies."""

    manifest_member: str
    manifest_sha256: str
    tree_sha256: str
    member_count: int

    def __post_init__(self) -> None:
        if self.manifest_member != SOURCE_MANIFEST_FILENAME or self.member_count <= 0:
            raise ProtocolError("SCALE-BP source seal receipt drifted.")
        require_sha256(self.manifest_sha256, "source manifest hash")
        require_sha256(self.tree_sha256, "source tree hash")


def package_source_root() -> Path:
    return Path(__file__).resolve().parent


def source_manifest_path() -> Path:
    return package_source_root() / SOURCE_MANIFEST_FILENAME


def build_source_members(package_root: Path) -> tuple[dict[str, object], ...]:
    """Hash every regular Python member in stable path order."""

    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("SCALE-BP source root is not a regular directory.")
    paths = sorted(
        root.rglob("*.py"),
        key=lambda value: value.relative_to(root).as_posix(),
    )
    if not paths:
        raise ProtocolError("SCALE-BP source tree is empty.")
    rows: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _validate_member_name(relative)
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("SCALE-BP source member is not a regular file.")
        payload = path.read_bytes()
        rows.append(
            {
                "member": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return tuple(rows)


def build_source_manifest_payload(package_root: Path) -> dict[str, object]:
    members = build_source_members(package_root)
    tree_payload = {
        "schema_version": SOURCE_TREE_SCHEMA_VERSION,
        "members": members,
    }
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "source_root_role": SOURCE_ROOT_ROLE,
        "member_pattern": SOURCE_MEMBER_PATTERN,
        "member_count": len(members),
        "members": members,
        "tree_sha256": canonical_hash(tree_payload),
    }


def load_source_manifest(manifest_path: Path | None = None) -> dict[str, object]:
    """Read and validate the closed manifest schema and canonical tree hash."""

    path = Path(manifest_path) if manifest_path is not None else source_manifest_path()
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("SCALE-BP source manifest is not a regular file.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read SCALE-BP source manifest.") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "hash_algorithm",
        "source_root_role",
        "member_pattern",
        "member_count",
        "members",
        "tree_sha256",
    }:
        raise ProtocolError("SCALE-BP source manifest schema drifted.")
    members = payload.get("members")
    if (
        payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION
        or payload.get("hash_algorithm") != "sha256"
        or payload.get("source_root_role") != SOURCE_ROOT_ROLE
        or payload.get("member_pattern") != SOURCE_MEMBER_PATTERN
        or not isinstance(members, list)
        or not members
        or payload.get("member_count") != len(members)
    ):
        raise ProtocolError("SCALE-BP source manifest identity drifted.")
    normalized: list[dict[str, object]] = []
    previous = ""
    for row in members:
        if not isinstance(row, dict) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("SCALE-BP source manifest member drifted.")
        member = str(row.get("member"))
        _validate_member_name(member)
        if member <= previous:
            raise ProtocolError("SCALE-BP source manifest order drifted.")
        previous = member
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProtocolError("SCALE-BP source member size drifted.")
        normalized.append(
            {
                "member": member,
                "size_bytes": size,
                "sha256": require_sha256(row.get("sha256"), "source member hash"),
            }
        )
    tree_hash = require_sha256(payload.get("tree_sha256"), "source tree hash")
    if canonical_hash(
        {
            "schema_version": SOURCE_TREE_SCHEMA_VERSION,
            "members": tuple(normalized),
        }
    ) != tree_hash:
        raise ProtocolError("SCALE-BP source tree hash drifted.")
    return {**payload, "members": tuple(normalized), "tree_sha256": tree_hash}


def validate_source_seal(
    *,
    package_root: Path | None = None,
    manifest_path: Path | None = None,
    expected_manifest_sha256: object | None = None,
    expected_tree_sha256: object | None = None,
) -> SourceSealReceipt:
    """Verify exact source membership, bytes, and optional external anchors."""

    root = Path(package_root) if package_root is not None else package_source_root()
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else root / SOURCE_MANIFEST_FILENAME
    )
    manifest = load_source_manifest(manifest_file)
    observed = build_source_members(root)
    if observed != manifest["members"]:
        raise ProtocolError("SCALE-BP source bytes or membership drifted.")
    manifest_hash = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    tree_hash = str(manifest["tree_sha256"])
    if expected_manifest_sha256 is not None and manifest_hash != require_sha256(
        expected_manifest_sha256,
        "expected source manifest hash",
    ):
        raise ProtocolError("SCALE-BP source manifest hash drifted.")
    if expected_tree_sha256 is not None and tree_hash != require_sha256(
        expected_tree_sha256,
        "expected source tree hash",
    ):
        raise ProtocolError("SCALE-BP source tree identity drifted.")
    return SourceSealReceipt(
        manifest_member=SOURCE_MANIFEST_FILENAME,
        manifest_sha256=manifest_hash,
        tree_sha256=tree_hash,
        member_count=len(observed),
    )


def source_seal_identity() -> SourceSealReceipt:
    return validate_source_seal()


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
        raise ProtocolError("SCALE-BP source member name is unsafe.")


__all__ = (
    "SOURCE_MANIFEST_FILENAME",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "SOURCE_MEMBER_PATTERN",
    "SOURCE_ROOT_ROLE",
    "SOURCE_TREE_SCHEMA_VERSION",
    "SourceSealReceipt",
    "build_source_manifest_payload",
    "build_source_members",
    "load_source_manifest",
    "package_source_root",
    "source_manifest_path",
    "source_seal_identity",
    "validate_source_seal",
)
