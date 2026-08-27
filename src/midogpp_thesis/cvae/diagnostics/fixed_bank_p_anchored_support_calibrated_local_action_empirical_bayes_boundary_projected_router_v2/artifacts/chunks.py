"""Atomic, content-addressed per-center chunks and manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping, Sequence

from ..protocol import GovernanceError
from .hashing import canonical_hash, json_native, require_sha256
from .io import atomic_json, indexed_file_row, member_path, read_json_object, safe_member


_SLUG = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_CENTER = re.compile(r"[0-9]+")
_RAW_LABEL_KEYS = frozenset(
    {
        "labels",
        "raw_labels",
        "sample_labels",
        "target_labels",
        "label_values",
        "truth",
        "ground_truth",
        "y_true",
    }
)


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """Primitive immutable reference returned by an outer-center worker."""

    target_center: str
    phase_id: str
    member: str
    size_bytes: int
    sha256: str
    record_count: int
    payload_hash: str
    chunk_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.target_center)
        phase = str(self.phase_id)
        member = safe_member(self.member)
        if (
            _CENTER.fullmatch(center) is None
            or _SLUG.fullmatch(phase) is None
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or type(self.record_count) is not int
            or self.record_count < 0
        ):
            raise GovernanceError("SCALE-BP v2 chunk reference is malformed.")
        require_sha256(self.sha256, "chunk file hash")
        require_sha256(self.payload_hash, "chunk payload hash")
        object.__setattr__(self, "target_center", center)
        object.__setattr__(self, "phase_id", phase)
        object.__setattr__(self, "member", member)
        base = self.to_payload(include_hash=False)
        object.__setattr__(self, "chunk_hash", canonical_hash(base))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_chunk_ref_v1",
            "target_center": self.target_center,
            "phase_id": self.phase_id,
            "member": self.member,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "record_count": self.record_count,
            "payload_hash": self.payload_hash,
        }
        if include_hash:
            payload["chunk_hash"] = self.chunk_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChunkRef":
        expected = {
            "schema_version",
            "target_center",
            "phase_id",
            "member",
            "size_bytes",
            "sha256",
            "record_count",
            "payload_hash",
            "chunk_hash",
        }
        if set(payload) != expected or payload.get("schema_version") != "scale_bp_v2_chunk_ref_v1":
            raise GovernanceError("SCALE-BP v2 chunk-ref schema drifted.")
        row = cls(
            target_center=str(payload["target_center"]),
            phase_id=str(payload["phase_id"]),
            member=str(payload["member"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=str(payload["sha256"]),
            record_count=int(payload["record_count"]),
            payload_hash=str(payload["payload_hash"]),
        )
        if row.chunk_hash != payload.get("chunk_hash"):
            raise GovernanceError("SCALE-BP v2 chunk-ref hash drifted.")
        return row


@dataclass(frozen=True, slots=True)
class CenterManifestRef:
    target_center: str
    member: str
    size_bytes: int
    sha256: str
    task_hash: str
    result_hash: str
    manifest_hash: str

    def __post_init__(self) -> None:
        if _CENTER.fullmatch(str(self.target_center)) is None:
            raise GovernanceError("SCALE-BP v2 center-manifest identity drifted.")
        safe_member(self.member)
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise GovernanceError("SCALE-BP v2 center-manifest size drifted.")
        for value, role in (
            (self.sha256, "center-manifest file hash"),
            (self.task_hash, "outer task hash"),
            (self.result_hash, "outer result hash"),
            (self.manifest_hash, "center-manifest hash"),
        ):
            require_sha256(value, role)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_v2_center_manifest_ref_v1",
            "target_center": self.target_center,
            "member": self.member,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "task_hash": self.task_hash,
            "result_hash": self.result_hash,
            "manifest_hash": self.manifest_hash,
        }


def write_center_chunk(
    root: str | Path,
    *,
    target_center: str,
    phase_id: str,
    payload: Mapping[str, object],
    record_count: int,
    bindings: Mapping[str, str] | None = None,
) -> ChunkRef:
    """Create one immutable JSON chunk; an existing mismatch is fatal."""

    center = str(target_center)
    phase = str(phase_id)
    if _CENTER.fullmatch(center) is None or _SLUG.fullmatch(phase) is None:
        raise GovernanceError("SCALE-BP v2 center chunk identity drifted.")
    if type(record_count) is not int or record_count < 0:
        raise GovernanceError("SCALE-BP v2 center chunk count drifted.")
    body = json_native(dict(payload))
    _reject_raw_label_fields(body)
    binding_payload = {
        str(key): require_sha256(value, str(key))
        for key, value in sorted((bindings or {}).items())
    }
    payload_hash = canonical_hash(body)
    unhashed = {
        "schema_version": "scale_bp_v2_center_chunk_v1",
        "target_center": center,
        "phase_id": phase,
        "record_count": record_count,
        "bindings": binding_payload,
        "payload": body,
        "payload_hash": payload_hash,
        "raw_labels_persisted": False,
    }
    document = {**unhashed, "document_hash": canonical_hash(unhashed)}
    member = f"chunks/{phase}/center_{center}.json"
    path = member_path(root, member)
    atomic_json(path, document)
    row = indexed_file_row(root, member)
    return ChunkRef(
        target_center=center,
        phase_id=phase,
        member=member,
        size_bytes=int(row["size_bytes"]),
        sha256=str(row["sha256"]),
        record_count=record_count,
        payload_hash=payload_hash,
    )


def validate_chunk(root: str | Path, reference: ChunkRef) -> dict[str, object]:
    row = indexed_file_row(root, reference.member)
    if row["size_bytes"] != reference.size_bytes or row["sha256"] != reference.sha256:
        raise GovernanceError("SCALE-BP v2 chunk bytes drifted.")
    payload = read_json_object(member_path(root, reference.member))
    unhashed = {key: value for key, value in payload.items() if key != "document_hash"}
    if (
        payload.get("schema_version") != "scale_bp_v2_center_chunk_v1"
        or payload.get("target_center") != reference.target_center
        or payload.get("phase_id") != reference.phase_id
        or payload.get("record_count") != reference.record_count
        or payload.get("payload_hash") != reference.payload_hash
        or payload.get("raw_labels_persisted") is not False
        or payload.get("document_hash") != canonical_hash(unhashed)
        or canonical_hash(payload.get("payload")) != reference.payload_hash
    ):
        raise GovernanceError("SCALE-BP v2 center chunk contract drifted.")
    _reject_raw_label_fields(payload.get("payload"))
    return payload


def write_center_manifest(
    root: str | Path,
    *,
    target_center: str,
    task_hash: str,
    result_hash: str,
    chunks: Sequence[ChunkRef],
    completed_support_fold_ids: Sequence[int],
    outer_result: Mapping[str, object] | object | None = None,
) -> CenterManifestRef:
    center = str(target_center)
    require_sha256(task_hash, "outer task hash")
    require_sha256(result_hash, "outer result hash")
    rows = tuple(chunks)
    fold_ids = tuple(int(value) for value in completed_support_fold_ids)
    if (
        _CENTER.fullmatch(center) is None
        or not rows
        or any(row.target_center != center for row in rows)
        or len({row.member for row in rows}) != len(rows)
        or tuple(sorted(rows, key=lambda row: (row.phase_id, row.member))) != rows
        or not fold_ids
        or fold_ids != tuple(range(len(fold_ids)))
    ):
        raise GovernanceError("SCALE-BP v2 center manifest topology drifted.")
    for row in rows:
        validate_chunk(root, row)
    result_payload: dict[str, object] | None = None
    if outer_result is not None:
        if isinstance(outer_result, Mapping):
            result_payload = dict(outer_result)
        else:
            to_payload = getattr(outer_result, "to_payload", None)
            if not callable(to_payload) or not isinstance(to_payload(), Mapping):
                raise GovernanceError("SCALE-BP v2 outer result payload is malformed.")
            result_payload = dict(to_payload())
        result_body = {
            key: value for key, value in result_payload.items() if key != "result_hash"
        }
        if (
            result_payload.get("schema_version")
            != "scale_bp_v2_outer_center_result_v1"
            or result_payload.get("target_center") != center
            or result_payload.get("task_hash") != task_hash
            or result_payload.get("result_hash") != result_hash
            or canonical_hash(result_body) != result_hash
            or result_payload.get("chunks") != [row.to_payload() for row in rows]
        ):
            raise GovernanceError("SCALE-BP v2 outer result/manifest binding drifted.")
    unhashed = {
        "schema_version": "scale_bp_v2_center_manifest_v1",
        "target_center": center,
        "task_hash": task_hash,
        "result_hash": result_hash,
        "completed_support_fold_ids": list(fold_ids),
        "support_folds_sequential_inside_one_outer_worker": True,
        "nested_process_pools_used": False,
        "chunks": [row.to_payload() for row in rows],
        "outer_result": result_payload,
        "outer_result_persisted": result_payload is not None,
        "chunk_count": len(rows),
        "raw_labels_persisted": False,
    }
    manifest_hash = canonical_hash(unhashed)
    document = {**unhashed, "manifest_hash": manifest_hash}
    member = f"manifests/centers/center_{center}.json"
    atomic_json(member_path(root, member), document)
    index = indexed_file_row(root, member)
    return CenterManifestRef(
        target_center=center,
        member=member,
        size_bytes=int(index["size_bytes"]),
        sha256=str(index["sha256"]),
        task_hash=task_hash,
        result_hash=result_hash,
        manifest_hash=manifest_hash,
    )


def validate_center_manifest(
    root: str | Path, reference: CenterManifestRef
) -> dict[str, object]:
    row = indexed_file_row(root, reference.member)
    payload = read_json_object(member_path(root, reference.member))
    chunks = payload.get("chunks")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        row["size_bytes"] != reference.size_bytes
        or row["sha256"] != reference.sha256
        or payload.get("schema_version") != "scale_bp_v2_center_manifest_v1"
        or payload.get("target_center") != reference.target_center
        or payload.get("task_hash") != reference.task_hash
        or payload.get("result_hash") != reference.result_hash
        or payload.get("manifest_hash") != reference.manifest_hash
        or canonical_hash(unhashed) != reference.manifest_hash
        or not isinstance(chunks, list)
        or payload.get("chunk_count") != len(chunks)
        or payload.get("support_folds_sequential_inside_one_outer_worker") is not True
        or payload.get("nested_process_pools_used") is not False
        or payload.get("raw_labels_persisted") is not False
    ):
        raise GovernanceError("SCALE-BP v2 center manifest drifted.")
    result_payload = payload.get("outer_result")
    if payload.get("outer_result_persisted") is True:
        if not isinstance(result_payload, Mapping):
            raise GovernanceError("SCALE-BP v2 persisted outer result is malformed.")
        result_body = {
            key: value
            for key, value in result_payload.items()
            if key != "result_hash"
        }
        if (
            result_payload.get("result_hash") != reference.result_hash
            or result_payload.get("task_hash") != reference.task_hash
            or result_payload.get("target_center") != reference.target_center
            or result_payload.get("chunks") != chunks
            or canonical_hash(result_body) != reference.result_hash
        ):
            raise GovernanceError("SCALE-BP v2 persisted outer result hash drifted.")
    elif result_payload is not None:
        raise GovernanceError("SCALE-BP v2 outer-result persistence flag drifted.")
    for item in chunks:
        if not isinstance(item, Mapping):
            raise GovernanceError("SCALE-BP v2 center manifest chunk is malformed.")
        validate_chunk(root, ChunkRef.from_payload(item))
    return payload


def _reject_raw_label_fields(value: object) -> None:
    if isinstance(value, Mapping):
        if {str(key).casefold() for key in value} & _RAW_LABEL_KEYS:
            raise GovernanceError("SCALE-BP v2 refuses raw labels in an artifact chunk.")
        for item in value.values():
            _reject_raw_label_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_raw_label_fields(item)


__all__ = (
    "CenterManifestRef",
    "ChunkRef",
    "validate_center_manifest",
    "validate_chunk",
    "write_center_chunk",
    "write_center_manifest",
)
