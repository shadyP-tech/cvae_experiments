"""Parent-side immutable input bindings for HARP v16 classifier tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..artifact_io import read_json, sha256_file
from .gpu_surface import SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER
from .hash_contracts import require_sha256, require_stable_hash


class _SourceRecord(Protocol):
    def to_payload(self) -> dict[str, object]: ...


class _SourceCache(Protocol):
    root: Path
    source_array_path: Path
    records: Sequence[_SourceRecord]
    lock_payload: Mapping[str, object]

    @property
    def lock_hash(self) -> str: ...

    @property
    def index_hash(self) -> str: ...


class _Frames(Protocol):
    path: Path
    receipt_path: Path
    sha256: str
    receipt_hash: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceTaskBinding:
    array_path: Path
    array_sha256: str
    index_path: Path
    index_sha256: str
    index_hash: str
    lock_hash: str
    lock_sha256: str
    records: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class FrameTaskBinding:
    array_path: Path
    array_sha256: str
    receipt_hash: str
    receipt_sha256: str


def validate_source_task_binding(source_cache: _SourceCache) -> SourceTaskBinding:
    """Reconstruct both source semantic identities from their durable files."""

    records = tuple(record.to_payload() for record in source_cache.records)
    array_sha256 = require_sha256(
        source_cache.lock_payload.get("source_array_sha256"),
        name="source-array hash",
    )
    index_sha256 = require_sha256(
        source_cache.lock_payload.get("source_stream_index_sha256"),
        name="source-index hash",
    )
    index_hash = require_stable_hash(
        source_cache.lock_payload.get("source_stream_index_hash"),
        name="source-stream index hash",
    )
    lock_hash = require_stable_hash(
        source_cache.lock_payload.get("source_stream_lock_hash"),
        name="source-stream lock hash",
    )
    lock_unhashed = {
        key: value
        for key, value in source_cache.lock_payload.items()
        if key != "source_stream_lock_hash"
    }
    if source_cache.lock_hash != lock_hash or stable_hash(lock_unhashed) != lock_hash:
        raise ProtocolError("HARP v16 source-stream semantic lock binding drifted.")

    lock_path = _plain_file(
        source_cache.root / SOURCE_LOCK_MEMBER, name="source-stream lock"
    )
    if read_json(lock_path) != dict(source_cache.lock_payload):
        raise ProtocolError("HARP v16 source-stream lock bytes drifted.")
    lock_sha256 = require_sha256(
        sha256_file(lock_path), name="source-stream lock SHA-256"
    )

    index_path = _plain_file(
        source_cache.root / SOURCE_INDEX_MEMBER, name="source-stream index"
    )
    index_payload = read_json(index_path)
    index_unhashed = {
        key: value
        for key, value in index_payload.items()
        if key != "source_stream_index_hash"
    }
    if (
        sha256_file(index_path) != index_sha256
        or index_payload.get("source_stream_index_hash") != index_hash
        or stable_hash(index_unhashed) != index_hash
        or index_payload.get("records") != list(records)
    ):
        raise ProtocolError("HARP v16 source-stream index binding drifted.")
    array_path = _plain_file(source_cache.source_array_path, name="source array")
    if sha256_file(array_path) != array_sha256:
        raise ProtocolError("HARP v16 source-array binding drifted.")
    return SourceTaskBinding(
        array_path=array_path,
        array_sha256=array_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
        index_hash=index_hash,
        lock_hash=lock_hash,
        lock_sha256=lock_sha256,
        records=records,
    )


def validate_frame_task_binding(frames: _Frames) -> FrameTaskBinding:
    """Bind frame content to its independent durable semantic receipt."""

    array_sha256 = require_sha256(frames.sha256, name="frame-array hash")
    receipt_hash = require_stable_hash(
        frames.receipt_hash, name="frame-receipt hash"
    )
    receipt_sha256 = require_sha256(
        frames.receipt_sha256, name="frame-receipt SHA-256"
    )
    array_path = _plain_file(frames.path, name="frame array")
    receipt_path = _plain_file(frames.receipt_path, name="frame receipt")
    receipt = read_json(receipt_path)
    receipt_unhashed = {
        key: value for key, value in receipt.items() if key != "frame_receipt_hash"
    }
    if (
        sha256_file(array_path) != array_sha256
        or sha256_file(receipt_path) != receipt_sha256
        or receipt.get("array_sha256") != array_sha256
        or receipt.get("frame_receipt_hash") != receipt_hash
        or stable_hash(receipt_unhashed) != receipt_hash
    ):
        raise ProtocolError("HARP v16 frame receipt/content binding drifted.")
    return FrameTaskBinding(
        array_path=array_path,
        array_sha256=array_sha256,
        receipt_hash=receipt_hash,
        receipt_sha256=receipt_sha256,
    )


def _plain_file(path: Path, *, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProtocolError(f"HARP v16 {name} path is unsafe.") from exc
    if (
        not path.is_absolute()
        or path != resolved
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ProtocolError(f"HARP v16 {name} path is unsafe.")
    return path


__all__ = (
    "FrameTaskBinding",
    "SourceTaskBinding",
    "validate_frame_task_binding",
    "validate_source_task_binding",
)
