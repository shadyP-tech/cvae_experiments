"""Label-free input identities and the exact P-DCAPS v2 source snapshot."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....generation.contracts import COMMON_OUTPUT_DIM
from ....protocol import ProtocolError
from ..identity import canonical_hash, require_sha256
from .experiment_contracts import SOURCE_SNAPSHOT_SCHEMA


SOURCE_TREE_SCHEMA = "pdcaps_v2_source_snapshot_tree_v1"
SOURCE_ROOT_ROLE = "pdcaps_v2_and_package_local_scientific_python"
SOURCE_MEMBER_PATTERN = "**/*.py"
_CONTRACT_MEMBER = "v2/experiment_contracts.py"
_NORMALIZED_SOURCE_ASSIGNMENTS: Mapping[str, object] = MappingProxyType(
    {
        "EXPECTED_LEDGER_AMENDMENT_SHA256": "__PDCAPS_V2_AMENDMENT_SHA256__",
        "EXPECTED_SOURCE_SNAPSHOT_MANIFEST_SHA256": (
            "__PDCAPS_V2_SOURCE_SNAPSHOT_MANIFEST_SHA256__"
        ),
        "EXPECTED_SOURCE_SNAPSHOT_TREE_SHA256": (
            "__PDCAPS_V2_SOURCE_SNAPSHOT_TREE_SHA256__"
        ),
        "EXPECTED_SOURCE_SNAPSHOT_MEMBER_COUNT": -1,
    }
)


@dataclass(frozen=True, order=True)
class TestRowIdentity:
    """One stable consumed-test row without a path or target label."""

    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or int(self.row_ordinal) < 0
            or isinstance(self.manifest_row_index, bool)
            or int(self.manifest_row_index) < 0
            or str(self.center) not in CENTERS
            or self.split != "test"
            or not str(self.sample_id)
            or not str(self.case_id)
        ):
            raise ProtocolError("P-DCAPS v2 test-row identity drifted.")
        object.__setattr__(self, "row_ordinal", int(self.row_ordinal))
        object.__setattr__(self, "manifest_row_index", int(self.manifest_row_index))
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "center", str(self.center))

    @property
    def evaluation_row_id(self) -> str:
        return self.sample_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LabelFreeTestFrame:
    """Immutable float32 embeddings with no label or sample-path capability."""

    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {
            str(center): tuple(center_rows)
            for center, center_rows in self.rows_by_center.items()
        }
        if (
            values.shape != (len(rows), COMMON_OUTPUT_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != tuple(CENTERS)
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or tuple(row.row_ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.sample_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
        ):
            raise ProtocolError("P-DCAPS v2 label-free frame drifted.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(
            self, "cache_binding", MappingProxyType(dict(self.cache_binding))
        )

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "pdcaps_v2_test_cache_binding_v1",
                "cache_binding": dict(self.cache_binding),
            }
        )

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].sample_id for index in ordinals)
            != tuple(row.sample_id for row in rows)
        ):
            raise ProtocolError("P-DCAPS v2 embedding identity drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


def package_source_root() -> Path:
    """Return the complete package-local P-DCAPS source root."""

    return Path(__file__).resolve().parent.parent


def build_source_snapshot_members(
    package_root: Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Hash every regular package Python member in deterministic order.

    Bytecode and cache directories are deliberately outside the member pattern.
    The four external artifact/source-anchor assignments are normalized so the
    amendment/source binding cycle cannot make the identity self-referential.
    """

    root = package_source_root() if package_root is None else Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("P-DCAPS v2 source root is absent or unsafe.")
    paths = sorted(
        root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix()
    )
    if not paths:
        raise ProtocolError("P-DCAPS v2 source snapshot is empty.")
    rows: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        _validate_source_member(relative)
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("P-DCAPS v2 source member is absent or unsafe.")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProtocolError("Cannot read P-DCAPS v2 source member.") from exc
        if relative == _CONTRACT_MEMBER:
            payload = _normalize_source_contract(payload)
        rows.append(
            {
                "member": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return tuple(rows)


def build_source_snapshot_payload(
    package_root: Path | None = None,
) -> dict[str, object]:
    members = build_source_snapshot_members(package_root)
    tree_payload = {
        "schema_version": SOURCE_TREE_SCHEMA,
        "members": list(members),
    }
    tree_hash = canonical_hash(tree_payload)
    manifest = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "hash_algorithm": "sha256",
        "source_root_role": SOURCE_ROOT_ROLE,
        "member_pattern": SOURCE_MEMBER_PATTERN,
        "normalized_external_anchor_member": _CONTRACT_MEMBER,
        "normalized_external_anchor_names": sorted(_NORMALIZED_SOURCE_ASSIGNMENTS),
        "member_count": len(members),
        "members": list(members),
        "tree_sha256": tree_hash,
    }
    return {
        **manifest,
        "manifest_sha256": canonical_hash(manifest),
    }


def source_snapshot_identity(
    package_root: Path | None = None,
) -> Mapping[str, object]:
    payload = build_source_snapshot_payload(package_root)
    return MappingProxyType(
        {
            "source_snapshot_schema": SOURCE_SNAPSHOT_SCHEMA,
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
        expected_manifest_sha256, "v2 source snapshot manifest hash"
    )
    expected_tree = require_sha256(
        expected_tree_sha256, "v2 source snapshot tree hash"
    )
    if (
        isinstance(expected_member_count, bool)
        or not isinstance(expected_member_count, int)
        or expected_member_count <= 0
    ):
        raise ProtocolError("P-DCAPS v2 source snapshot member count drifted.")
    identity = source_snapshot_identity(package_root)
    if (
        identity["source_snapshot_manifest_sha256"] != expected_manifest
        or identity["source_snapshot_tree_sha256"] != expected_tree
        or identity["source_snapshot_member_count"] != expected_member_count
    ):
        raise ProtocolError("P-DCAPS v2 source snapshot bytes or inventory drifted.")
    return MappingProxyType({"status": "PASS", **dict(identity)})


def _normalize_source_contract(payload: bytes) -> bytes:
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProtocolError("P-DCAPS v2 source-anchor contract is malformed.") from exc
    replacements: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in _NORMALIZED_SOURCE_ASSIGNMENTS:
            continue
        if node.end_lineno is None:
            raise ProtocolError("P-DCAPS v2 source-anchor location is unavailable.")
        seen.add(target.id)
        replacement = f"{target.id} = {_NORMALIZED_SOURCE_ASSIGNMENTS[target.id]!r}\n"
        replacements.append((node.lineno - 1, node.end_lineno, replacement))
    if seen != set(_NORMALIZED_SOURCE_ASSIGNMENTS):
        raise ProtocolError("P-DCAPS v2 source-anchor assignments drifted.")
    lines = text.splitlines(keepends=True)
    for start, stop, replacement in sorted(replacements, reverse=True):
        lines[start:stop] = [replacement]
    return "".join(lines).encode("utf-8")


def _validate_source_member(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", "..", "__pycache__"} for part in path.parts)
        or path.suffix != ".py"
    ):
        raise ProtocolError("P-DCAPS v2 source member name is unsafe.")


__all__ = (
    "LabelFreeTestFrame",
    "SOURCE_MEMBER_PATTERN",
    "SOURCE_ROOT_ROLE",
    "SOURCE_TREE_SCHEMA",
    "TestRowIdentity",
    "build_source_snapshot_members",
    "build_source_snapshot_payload",
    "package_source_root",
    "source_snapshot_identity",
    "validate_source_snapshot",
)
