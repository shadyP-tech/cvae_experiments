"""Read-only exact-six filesystem binding for OE-PPUR v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_INPUT_KINDS,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_TEST_ROW_COUNT,
    FORBIDDEN_INPUT_PATH_FRAGMENTS,
    V1_EXPERIMENT_ID,
    V1_OUTPUT_ARTIFACT_ID,
)
from .run_paths import (
    assert_no_symlink_chain,
    paths_overlap,
    validate_absolute_path,
)


@dataclass(frozen=True, slots=True)
class WorkspaceInputBinding:
    """One workspace-resolved direct input in declared role order."""

    role: str
    artifact_id: str
    path: Path
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "artifact_id", str(self.artifact_id))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "kind", str(self.kind))

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "path": self.path.as_posix(),
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class InputFileEvidence:
    role: str
    artifact_id: str
    path: str
    kind: str
    content_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.content_sha256, f"{self.role} content hash")

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "kind": self.kind,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class ValidatedWorkspaceInputs:
    evidence: tuple[InputFileEvidence, ...]
    artifact_root: str
    scratch_root: str
    bank_content_index_sha256: str
    generation_content_index_sha256: str
    cache_content_sha256: str
    cache_row_order_sha256: str
    manifest_sha256: str
    parent_ledger_sha256: str
    amendment_sha256: str
    input_binding_hash: str
    input_location_binding_sha256: str

    def __post_init__(self) -> None:
        if (
            self.bank_content_index_sha256
            != EXPECTED_BANK_CONTENT_INDEX_SHA256
            or self.generation_content_index_sha256
            != EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ):
            raise ProtocolError(
                "OE-PPUR v2 upstream content-index file identity drifted."
            )
        for role, digest in (
            ("bank content-index hash", self.bank_content_index_sha256),
            (
                "GenerationLock content-index hash",
                self.generation_content_index_sha256,
            ),
            ("cache content hash", self.cache_content_sha256),
            ("cache row-order hash", self.cache_row_order_sha256),
            ("manifest hash", self.manifest_sha256),
            ("parent ledger hash", self.parent_ledger_sha256),
            ("amendment hash", self.amendment_sha256),
            ("input binding hash", self.input_binding_hash),
            ("input location binding hash", self.input_location_binding_sha256),
        ):
            require_sha256(digest, role)

    @property
    def amendment_path(self) -> Path:
        return Path(self.evidence[-1].path)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_validated_workspace_inputs_v1",
            "evidence": [row.to_payload() for row in self.evidence],
            "artifact_root": self.artifact_root,
            "scratch_root": self.scratch_root,
            "bank_content_index_sha256": self.bank_content_index_sha256,
            "generation_content_index_sha256": (
                self.generation_content_index_sha256
            ),
            "cache_content_sha256": self.cache_content_sha256,
            "cache_row_order_sha256": self.cache_row_order_sha256,
            "manifest_sha256": self.manifest_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "amendment_sha256": self.amendment_sha256,
            "input_binding_hash": self.input_binding_hash,
            "input_location_binding_sha256": (
                self.input_location_binding_sha256
            ),
        }


def validate_input_topology(
    bindings: Sequence[WorkspaceInputBinding],
    *,
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> tuple[tuple[WorkspaceInputBinding, ...], Path, Path]:
    """Validate ordering, uniqueness, kind, and path separation only."""

    rows = tuple(bindings)
    if (
        len(rows) != 6
        or tuple(row.role for row in rows) != DIRECT_INPUT_ROLES
        or tuple(row.artifact_id for row in rows) != DIRECT_INPUT_ARTIFACT_IDS
        or tuple(row.kind for row in rows) != EXPECTED_INPUT_KINDS
        or len({row.role for row in rows}) != 6
        or len({row.artifact_id for row in rows}) != 6
    ):
        raise ProtocolError("OE-PPUR v2 requires exactly six ordered direct inputs.")

    artifact = _safe_requested_root(artifact_root, role="artifact root")
    scratch = _safe_requested_root(scratch_root, role="scratch root")
    if paths_overlap(artifact, scratch):
        raise ProtocolError("OE-PPUR v2 output and scratch roots overlap.")

    canonical_rows: list[WorkspaceInputBinding] = []
    observed: list[Path] = []
    for expected_role, row in zip(DIRECT_INPUT_ROLES, rows, strict=True):
        path = _safe_existing_input(row.path, role=expected_role, kind=row.kind)
        folded = path.as_posix().casefold()
        if (
            V1_EXPERIMENT_ID.casefold() in folded
            or V1_OUTPUT_ARTIFACT_ID.casefold() in folded
            or any(
                fragment.casefold() in folded
                for fragment in FORBIDDEN_INPUT_PATH_FRAGMENTS
            )
        ):
            raise ProtocolError("OE-PPUR v2 rejected predecessor/quarantine input.")
        if paths_overlap(path, artifact) or paths_overlap(path, scratch):
            raise ProtocolError("OE-PPUR v2 direct input overlaps run state.")
        if any(paths_overlap(path, previous) for previous in observed):
            raise ProtocolError("OE-PPUR v2 direct input paths are duplicated/overlapping.")
        observed.append(path)
        canonical_rows.append(
            WorkspaceInputBinding(row.role, row.artifact_id, path, row.kind)
        )
    return tuple(canonical_rows), artifact, scratch


def validate_workspace_inputs(
    bindings: Sequence[WorkspaceInputBinding],
    *,
    artifact_root: str | Path,
    scratch_root: str | Path,
    expected_amendment_sha256: str,
) -> ValidatedWorkspaceInputs:
    """Revalidate every immutable direct-input boundary without writing state."""

    rows, artifact, scratch = validate_input_topology(
        bindings, artifact_root=artifact_root, scratch_root=scratch_root
    )
    expected_amendment = require_sha256(
        expected_amendment_sha256, "expected amendment hash"
    )
    by_role = {row.role: row for row in rows}

    bank_member = by_role[DIRECT_INPUT_ROLES[0]].path / (
        "manifests/expert_bank_index.json"
    )
    bank_payload, _ = _read_json_regular(bank_member)
    if (
        bank_payload.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        or bank_payload.get("routing_authorized") is not True
    ):
        raise ProtocolError("OE-PPUR v2 promoted expert-bank identity drifted.")
    bank_content_index_sha256 = _validate_indexed_directory(
        by_role[DIRECT_INPUT_ROLES[0]].path,
        expected_schema="midogpp_uniform_b_v2_expert_bank_content_index_v1",
        expected_index_sha256=EXPECTED_BANK_CONTENT_INDEX_SHA256,
        role="promoted expert bank",
    )

    generation_member = by_role[DIRECT_INPUT_ROLES[1]].path / (
        "manifests/generation_lock.json"
    )
    generation_payload, _ = _read_json_regular(
        generation_member
    )
    nested_bank = generation_payload.get("bank")
    if (
        generation_payload.get("generation_lock_hash")
        != EXPECTED_GENERATION_LOCK_HASH
        or not isinstance(nested_bank, Mapping)
        or nested_bank.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
    ):
        raise ProtocolError("OE-PPUR v2 GenerationLock identity drifted.")
    generation_content_index_sha256 = _validate_indexed_directory(
        by_role[DIRECT_INPUT_ROLES[1]].path,
        expected_schema="midogpp_uniform_b_v2_generation_lock_content_v1",
        expected_index_sha256=EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
        role="GenerationLock",
    )

    cache_root = by_role[DIRECT_INPUT_ROLES[2]].path
    _reject_tree_symlinks(cache_root)
    try:
        content = _validate_cache_content_index(cache_root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ProtocolError("OE-PPUR v2 test-cache content index drifted.") from exc
    alignment, _ = _read_json_regular(
        cache_root / "manifests/row_alignment.json"
    )
    cache_content_sha256 = str(content.get("content_hash", ""))
    cache_row_order_sha256 = str(alignment.get("row_order_hash", ""))
    if (
        cache_content_sha256 != EXPECTED_TEST_CACHE_CONTENT_HASH
        or cache_row_order_sha256 != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or alignment.get("manifest_sha256") != EXPECTED_TEST_MANIFEST_SHA256
        or alignment.get("row_count") != EXPECTED_TEST_ROW_COUNT
        or alignment.get("split") != "test"
    ):
        raise ProtocolError("OE-PPUR v2 immutable test-cache identity drifted.")

    manifest_path = by_role[DIRECT_INPUT_ROLES[3]].path
    manifest_sha256 = hash_regular_file(manifest_path)
    if manifest_sha256 != EXPECTED_TEST_MANIFEST_SHA256:
        raise ProtocolError("OE-PPUR v2 canonical manifest bytes drifted.")

    parent_path = by_role[DIRECT_INPUT_ROLES[4]].path
    parent_payload, parent_ledger_sha256 = _read_json_regular(parent_path)
    if (
        parent_ledger_sha256 != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256
        or parent_payload.get("status")
        != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent_payload.get("split") != "test"
    ):
        raise ProtocolError("OE-PPUR v2 original parent ledger drifted.")

    amendment_path = by_role[DIRECT_INPUT_ROLES[5]].path
    amendment_sha256 = hash_regular_file(amendment_path)
    if amendment_sha256 != expected_amendment:
        raise ProtocolError("OE-PPUR v2 amendment file hash drifted.")

    content_hashes = (
        bank_content_index_sha256,
        generation_content_index_sha256,
        cache_content_sha256,
        manifest_sha256,
        parent_ledger_sha256,
        amendment_sha256,
    )
    evidence = tuple(
        InputFileEvidence(
            row.role,
            row.artifact_id,
            row.path.as_posix(),
            row.kind,
            digest,
        )
        for row, digest in zip(rows, content_hashes, strict=True)
    )
    binding_body = {
        "schema_version": "oe_ppur_v2_six_input_binding_v1",
        "evidence": [row.to_payload() for row in evidence],
        "artifact_root": artifact.as_posix(),
        "scratch_root": scratch.as_posix(),
        "cache_row_order_sha256": cache_row_order_sha256,
        "predecessor_state_used": False,
        "mutation_performed": False,
    }
    input_binding_hash = canonical_hash(binding_body)
    input_location_binding_sha256 = hash_ordered_input_locations(rows)
    return ValidatedWorkspaceInputs(
        evidence=evidence,
        artifact_root=artifact.as_posix(),
        scratch_root=scratch.as_posix(),
        bank_content_index_sha256=bank_content_index_sha256,
        generation_content_index_sha256=generation_content_index_sha256,
        cache_content_sha256=cache_content_sha256,
        cache_row_order_sha256=cache_row_order_sha256,
        manifest_sha256=manifest_sha256,
        parent_ledger_sha256=parent_ledger_sha256,
        amendment_sha256=amendment_sha256,
        input_binding_hash=input_binding_hash,
        input_location_binding_sha256=input_location_binding_sha256,
    )


def hash_ordered_input_locations(
    bindings: Sequence[WorkspaceInputBinding],
) -> str:
    """Hash the exact ordered role/artifact/path/kind resolved-input tuple.

    This intentionally excludes content evidence and run roots: those remain in
    ``input_binding_hash``.  The separate location binding lets a later service
    factory prove that it received the same six canonical paths admitted by the
    read-only filesystem validation before it discards the three label-bearing
    paths.
    """

    rows = tuple(bindings)
    if (
        len(rows) != 6
        or any(type(row) is not WorkspaceInputBinding for row in rows)
        or tuple(row.role for row in rows) != DIRECT_INPUT_ROLES
        or tuple(row.artifact_id for row in rows) != DIRECT_INPUT_ARTIFACT_IDS
        or tuple(row.kind for row in rows) != EXPECTED_INPUT_KINDS
    ):
        raise ProtocolError(
            "OE-PPUR v2 input-location binding requires six exact rows."
        )
    payload_rows: list[dict[str, str]] = []
    for row in rows:
        path = Path(row.path)
        if (
            not path.is_absolute()
            or path == Path(path.anchor)
            or ".." in path.parts
            or str(path).startswith(("artifact://", "output://", "file://"))
        ):
            raise ProtocolError("OE-PPUR v2 input-location path is unsafe.")
        payload_rows.append(
            {
                "role": row.role,
                "artifact_id": row.artifact_id,
                "path": path.as_posix(),
                "kind": row.kind,
            }
        )
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v2_ordered_input_locations_v1",
            "ordered_input_locations": payload_rows,
            "direct_input_count": 6,
            "order_is_semantic": True,
        }
    )


def hash_regular_file(path: str | Path) -> str:
    """Hash one no-follow regular file and reject concurrent replacement."""

    source = Path(path)
    descriptor = _open_regular_file(source)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ProtocolError("OE-PPUR v2 input changed while hashing.")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_cache_content_index(root: Path) -> dict[str, object]:
    """Rehash the complete label-free cache tree inside the v2 source seal."""

    index_path = root / "manifests/content_index.json"
    payload, _ = _read_json_regular(index_path)
    if (
        set(payload) != {"schema_version", "files", "content_hash"}
        or payload.get("schema_version")
        != "midogpp_stage70_descriptive_test_cache_content_index_v1"
    ):
        raise ProtocolError("OE-PPUR v2 cache content-index schema drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if payload.get("content_hash") != canonical_hash(unhashed):
        raise ProtocolError("OE-PPUR v2 cache semantic content hash drifted.")
    raw_files = payload.get("files")
    if isinstance(raw_files, (str, bytes)) or not isinstance(raw_files, Sequence):
        raise ProtocolError("OE-PPUR v2 cache content-index files are invalid.")
    indexed: dict[str, str] = {}
    for record in raw_files:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ProtocolError("OE-PPUR v2 cache content-index member drifted.")
        relative = str(record["path"])
        member_path = Path(relative)
        if (
            not relative
            or relative in indexed
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise ProtocolError("OE-PPUR v2 cache content-index path is unsafe.")
        indexed[relative] = require_sha256(
            record["sha256"], "cache content-index member hash"
        )
    try:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != index_path
        }
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 cache tree cannot be indexed.") from exc
    if set(indexed) != actual:
        raise ProtocolError("OE-PPUR v2 cache indexed member set drifted.")
    for relative, expected_sha256 in indexed.items():
        if hash_regular_file(root / relative) != expected_sha256:
            raise ProtocolError(
                f"OE-PPUR v2 cache indexed member drifted: {relative}."
            )
    return payload


def _validate_indexed_directory(
    root: Path,
    *,
    expected_schema: str,
    expected_index_sha256: str,
    role: str,
) -> str:
    """Validate every immutable member named by a bank/GenerationLock index."""

    expected_file_hash = require_sha256(
        expected_index_sha256, f"{role} expected content-index file hash"
    )
    index_path = root / "manifests/content_index.json"
    payload, index_sha256 = _read_json_regular(index_path)
    if (
        set(payload) != {"schema_version", "records", "content_hash"}
        or payload.get("schema_version") != expected_schema
    ):
        raise ProtocolError(f"OE-PPUR v2 {role} content-index schema drifted.")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if payload.get("content_hash") != canonical_hash(unhashed)[:16]:
        raise ProtocolError(f"OE-PPUR v2 {role} semantic content hash drifted.")
    raw_records = payload.get("records")
    if isinstance(raw_records, (str, bytes)) or not isinstance(
        raw_records, Sequence
    ):
        raise ProtocolError(f"OE-PPUR v2 {role} content-index is invalid.")
    indexed: dict[str, tuple[str, int]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ProtocolError(f"OE-PPUR v2 {role} content-index row drifted.")
        relative = str(record["relative_path"])
        member_path = Path(relative)
        size = record["size_bytes"]
        if (
            not relative
            or relative in indexed
            or member_path.is_absolute()
            or ".." in member_path.parts
            or type(size) is not int
            or size < 0
        ):
            raise ProtocolError(f"OE-PPUR v2 {role} content-index path drifted.")
        indexed[relative] = (
            require_sha256(record["sha256"], f"{role} indexed member hash"),
            size,
        )
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    try:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() not in excluded
        }
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v2 {role} tree cannot be indexed.") from exc
    if set(indexed) != actual:
        raise ProtocolError(f"OE-PPUR v2 {role} indexed member set drifted.")
    for relative, (expected_sha256, expected_size) in indexed.items():
        member = root / relative
        try:
            observed_size = member.stat(follow_symlinks=False).st_size
        except OSError as exc:
            raise ProtocolError(
                f"OE-PPUR v2 {role} indexed member is absent."
            ) from exc
        if (
            observed_size != expected_size
            or hash_regular_file(member) != expected_sha256
        ):
            raise ProtocolError(
                f"OE-PPUR v2 {role} indexed member drifted: {relative}."
            )
    if index_sha256 != expected_file_hash:
        raise ProtocolError(f"OE-PPUR v2 {role} content-index file bytes drifted.")
    return index_sha256


def _read_json_regular(path: Path) -> tuple[dict[str, object], str]:
    descriptor = _open_regular_file(path)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 64 * 1024 * 1024:
                raise ProtocolError("OE-PPUR v2 JSON input is oversized.")
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ProtocolError("OE-PPUR v2 JSON input changed while reading.")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot parse OE-PPUR v2 JSON input: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("OE-PPUR v2 JSON input is not an object.")
    return value, digest.hexdigest()


def _open_regular_file(path: Path) -> int:
    assert_no_symlink_chain(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 input file is absent or unsafe.") from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        raise ProtocolError("OE-PPUR v2 input is not a regular file.")
    return descriptor


def _safe_existing_input(path: Path, *, role: str, kind: str) -> Path:
    if not path.is_absolute() or path == Path(path.anchor) or ".." in path.parts:
        raise ProtocolError(f"OE-PPUR v2 {role} path is unsafe.")
    assert_no_symlink_chain(path)
    try:
        resolved = path.resolve(strict=True)
        mode = os.stat(resolved, follow_symlinks=False).st_mode
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v2 {role} input is absent.") from exc
    if (kind == "file" and not stat.S_ISREG(mode)) or (
        kind == "directory" and not stat.S_ISDIR(mode)
    ):
        raise ProtocolError(f"OE-PPUR v2 {role} input kind drifted.")
    return resolved


def _safe_requested_root(path: str | Path, *, role: str) -> Path:
    value = validate_absolute_path(path, role=role)
    assert_no_symlink_chain(value, allow_missing_leaf=True)
    resolved = value.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ProtocolError(f"OE-PPUR v2 {role} is not a directory.")
    folded = resolved.as_posix().casefold()
    if (
        V1_EXPERIMENT_ID.casefold() in folded
        or V1_OUTPUT_ARTIFACT_ID.casefold() in folded
        or ".quarantine" in folded
        or "/quarantine/" in folded
    ):
        raise ProtocolError(f"OE-PPUR v2 {role} reuses unsafe history.")
    return resolved


def _reject_tree_symlinks(root: Path) -> None:
    try:
        if any(member.is_symlink() for member in root.rglob("*")):
            raise ProtocolError("OE-PPUR v2 cache tree contains a symlink.")
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 cache tree cannot be inspected.") from exc


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


__all__ = (
    "InputFileEvidence",
    "ValidatedWorkspaceInputs",
    "WorkspaceInputBinding",
    "hash_ordered_input_locations",
    "hash_regular_file",
    "validate_input_topology",
    "validate_workspace_inputs",
)
