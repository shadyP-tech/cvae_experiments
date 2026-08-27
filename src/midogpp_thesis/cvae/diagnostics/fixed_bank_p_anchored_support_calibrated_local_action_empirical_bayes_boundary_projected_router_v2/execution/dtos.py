"""Primitive-only DTOs for SCALE-BP v2 process boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

from ..artifacts.chunks import ChunkRef
from ..artifacts.hashing import canonical_hash, canonical_json, require_sha256
from ..protocol import GovernanceError


CANONICAL_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
DEFAULT_SUPPORT_FOLD_IDS = (0, 1, 2, 3)
_MEMMAP_DTYPES = frozenset(
    {"bool", "uint8", "int32", "int64", "float32", "float64"}
)


@dataclass(frozen=True, slots=True)
class MemmapReference:
    """Immutable absolute reference to an exact sealed array byte slice."""

    path: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    byte_length: int
    sha256: str
    semantic_role: str
    row_index_hash: str
    cache_content_hash: str
    row_order_hash: str
    order: str = "C"
    reference_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.path)
        dtype = str(self.dtype)
        shape = tuple(int(value) for value in self.shape)
        role = str(self.semantic_role).strip()
        try:
            expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
        except TypeError as exc:
            raise GovernanceError("SCALE-BP v2 memmap dtype is unsupported.") from exc
        if (
            not path.is_absolute()
            or dtype not in _MEMMAP_DTYPES
            or not shape
            or any(value <= 0 for value in shape)
            or type(self.offset_bytes) is not int
            or self.offset_bytes < 0
            or type(self.byte_length) is not int
            or self.byte_length != expected_bytes
            or not role
            or self.order != "C"
        ):
            raise GovernanceError("SCALE-BP v2 memmap reference drifted.")
        for value, label in (
            (self.sha256, "memmap slice hash"),
            (self.row_index_hash, "memmap row-index hash"),
            (self.cache_content_hash, "memmap cache-content hash"),
            (self.row_order_hash, "memmap row-order hash"),
        ):
            require_sha256(value, label)
        object.__setattr__(self, "path", str(path))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(
            self,
            "reference_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_memmap_reference_v1",
            "path": self.path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset_bytes": self.offset_bytes,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "semantic_role": self.semantic_role,
            "row_index_hash": self.row_index_hash,
            "cache_content_hash": self.cache_content_hash,
            "row_order_hash": self.row_order_hash,
            "order": self.order,
            "read_only": True,
        }
        if include_hash:
            payload["reference_hash"] = self.reference_hash
        return payload


MemmapRef = MemmapReference


@dataclass(frozen=True, slots=True)
class OuterCenterTask:
    """One complete outer-H task; all support folds execute sequentially."""

    target_center: str
    case_ids: tuple[str, ...]
    memmaps: tuple[MemmapReference, ...]
    protocol_hash: str
    support_fold_ids: tuple[int, ...] = DEFAULT_SUPPORT_FOLD_IDS
    phase_id: str = "outer_center"
    payload_json: str = "{}"
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.target_center)
        cases = tuple(str(value) for value in self.case_ids)
        maps = tuple(self.memmaps)
        folds = tuple(int(value) for value in self.support_fold_ids)
        try:
            import json

            primitive_payload = json.loads(str(self.payload_json))
        except (TypeError, ValueError) as exc:
            raise GovernanceError("SCALE-BP v2 task payload is not JSON.") from exc
        payload_json = canonical_json(primitive_payload)
        if (
            center not in CANONICAL_CENTERS
            or not cases
            or cases != tuple(sorted(set(cases)))
            or not maps
            or len({row.semantic_role for row in maps}) != len(maps)
            or len({row.reference_hash for row in maps}) != len(maps)
            or folds != tuple(range(len(folds)))
            or not folds
            or self.phase_id != "outer_center"
            or not isinstance(primitive_payload, dict)
        ):
            raise GovernanceError("SCALE-BP v2 outer-center task topology drifted.")
        require_sha256(self.protocol_hash, "protocol hash")
        object.__setattr__(self, "target_center", center)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "memmaps", maps)
        object.__setattr__(self, "support_fold_ids", folds)
        object.__setattr__(self, "payload_json", payload_json)
        object.__setattr__(
            self,
            "task_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_outer_center_task_v1",
            "target_center": self.target_center,
            "case_ids": list(self.case_ids),
            "memmap_references": [row.to_payload() for row in self.memmaps],
            "protocol_hash": self.protocol_hash,
            "support_fold_ids": list(self.support_fold_ids),
            "phase_id": self.phase_id,
            "payload_json": self.payload_json,
            "task_unit": "one_complete_outer_H",
            "support_folds_sequential_inside_worker": True,
            "nested_process_pools_allowed": False,
        }
        if include_hash:
            payload["task_hash"] = self.task_hash
        return payload

    def primitive_payload(self) -> dict[str, object]:
        """Decode the JSON-only science/governance payload inside the worker."""

        import json

        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict):  # constructor already enforces this
            raise GovernanceError("SCALE-BP v2 task primitive payload drifted.")
        return payload


@dataclass(frozen=True, slots=True)
class OuterCenterResult:
    """Small sealed result returned from exactly one outer-H worker."""

    target_center: str
    task_hash: str
    completed_support_fold_ids: tuple[int, ...]
    route_hashes: tuple[str, ...]
    chunks: tuple[ChunkRef, ...]
    no_refit_reconstruction_hash: str
    summary_json: str = "{}"
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.target_center)
        folds = tuple(int(value) for value in self.completed_support_fold_ids)
        routes = tuple(str(value) for value in self.route_hashes)
        chunks = tuple(self.chunks)
        try:
            import json

            summary = json.loads(str(self.summary_json))
        except (TypeError, ValueError) as exc:
            raise GovernanceError("SCALE-BP v2 worker summary is not JSON.") from exc
        summary_json = canonical_json(summary)
        if (
            center not in CANONICAL_CENTERS
            or folds != tuple(range(len(folds)))
            or not folds
            or not routes
            or len(set(routes)) != len(routes)
            or not chunks
            or any(row.target_center != center for row in chunks)
            or tuple(sorted(chunks, key=lambda row: (row.phase_id, row.member))) != chunks
            or not isinstance(summary, dict)
        ):
            raise GovernanceError("SCALE-BP v2 outer-center result topology drifted.")
        require_sha256(self.task_hash, "outer task hash")
        require_sha256(self.no_refit_reconstruction_hash, "no-refit reconstruction hash")
        for digest in routes:
            require_sha256(digest, "route hash")
        object.__setattr__(self, "target_center", center)
        object.__setattr__(self, "completed_support_fold_ids", folds)
        object.__setattr__(self, "route_hashes", routes)
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "summary_json", summary_json)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "scale_bp_v2_outer_center_result_v1",
            "target_center": self.target_center,
            "task_hash": self.task_hash,
            "completed_support_fold_ids": list(self.completed_support_fold_ids),
            "route_hashes": list(self.route_hashes),
            "chunks": [row.to_payload() for row in self.chunks],
            "no_refit_reconstruction_hash": self.no_refit_reconstruction_hash,
            "summary_json": self.summary_json,
            "support_folds_executed_sequentially": True,
            "nested_process_pools_used": False,
        }
        if include_hash:
            payload["result_hash"] = self.result_hash
        return payload

    def worker_capability_audit_payload(self) -> dict[str, object]:
        """Return the compact hash-only audit carried in ``summary_json``."""

        import json

        summary = json.loads(self.summary_json)
        audit = summary.get("worker_capability_audit")
        if not isinstance(audit, Mapping):
            raise GovernanceError(
                "SCALE-BP v2 outer result lacks its worker capability audit."
            )
        return dict(audit)


__all__ = (
    "CANONICAL_CENTERS",
    "DEFAULT_SUPPORT_FOLD_IDS",
    "MemmapRef",
    "MemmapReference",
    "OuterCenterResult",
    "OuterCenterTask",
)
