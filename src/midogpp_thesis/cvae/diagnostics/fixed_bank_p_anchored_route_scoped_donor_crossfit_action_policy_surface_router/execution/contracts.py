"""Spawn-safe primitive/tuple/contiguous-array worker contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import io
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..identity import canonical_hash, require_sha256


_PRIMITIVE_TYPES = (type(None), bool, int, str)


def validate_plain_payload(value: object, *, path: str = "payload") -> None:
    """Recursively reject stateful or mutation-prone worker payload members."""

    if isinstance(value, _PRIMITIVE_TYPES):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"P-DCAPS {path} contains a nonfinite float.")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            validate_plain_payload(item, path=f"{path}[{index}]")
        return
    if isinstance(value, np.ndarray):
        _validate_array(value, path=path)
        return
    if isinstance(value, (MappingProxyType, Mapping)):
        raise ProtocolError(f"P-DCAPS {path} contains a prohibited mapping.")
    if isinstance(value, (list, set, bytearray)):
        raise ProtocolError(f"P-DCAPS {path} contains a mutable container.")
    if isinstance(value, io.IOBase) or (
        hasattr(value, "read") and hasattr(value, "close")
    ):
        raise ProtocolError(f"P-DCAPS {path} contains a file handle.")
    if callable(value):
        qualname = str(getattr(value, "__qualname__", ""))
        role = "closure" if "<locals>" in qualname else "callable"
        raise ProtocolError(f"P-DCAPS {path} contains a prohibited {role}.")
    if hasattr(value, "fit") or hasattr(value, "predict"):
        raise ProtocolError(f"P-DCAPS {path} contains an estimator instance.")
    raise ProtocolError(
        f"P-DCAPS {path} contains unsupported type {type(value).__name__}."
    )


@dataclass(frozen=True)
class ContiguousArray:
    name: str
    values: np.ndarray
    dtype: str = field(init=False)
    shape: tuple[int, ...] = field(init=False)
    array_hash: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ProtocolError("P-DCAPS worker array name is empty.")
        values = np.ascontiguousarray(self.values)
        _validate_array(values, path=f"array[{name}]")
        # Own the worker buffer so a caller cannot mutate it after request seal.
        values = np.array(values, order="C", copy=True)
        values.setflags(write=False)
        dtype = values.dtype.str
        shape = tuple(int(value) for value in values.shape)
        digest = hashlib.sha256()
        digest.update(b"pdcaps_contiguous_array_v1\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(values.tobytes(order="C"))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "array_hash", digest.hexdigest())

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    def manifest_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "array_hash": self.array_hash,
        }


@dataclass(frozen=True)
class WorkerRequest:
    outer_center: str
    ordinal: int
    operation: str
    payload_entries: tuple[tuple[str, object], ...]
    arrays: tuple[ContiguousArray, ...]
    threads_per_worker: int
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        operation = str(self.operation)
        entries = tuple((str(key), value) for key, value in self.payload_entries)
        arrays = tuple(self.arrays)
        if (
            outer not in CENTERS
            or isinstance(self.ordinal, bool)
            or int(self.ordinal) < 0
            or not operation
            or tuple(key for key, _ in entries)
            != tuple(sorted(key for key, _ in entries))
            or len({key for key, _ in entries}) != len(entries)
            or len({row.name for row in arrays}) != len(arrays)
            or tuple(row.name for row in arrays)
            != tuple(sorted(row.name for row in arrays))
            or isinstance(self.threads_per_worker, bool)
            or int(self.threads_per_worker) <= 0
        ):
            raise ProtocolError("P-DCAPS worker request topology drifted.")
        for key, value in entries:
            validate_plain_payload(value, path=f"payload_entries[{key}]")
            if _contains_array(value):
                raise ProtocolError(
                    "P-DCAPS request arrays must use the contiguous-array channel."
                )
        if any(not isinstance(row, ContiguousArray) for row in arrays):
            raise ProtocolError("P-DCAPS worker request array contract drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "payload_entries", entries)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "threads_per_worker", int(self.threads_per_worker))
        object.__setattr__(
            self,
            "request_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_worker_request_v1",
                    "outer_center": outer,
                    "ordinal": self.ordinal,
                    "operation": operation,
                    "payload_entries": entries,
                    "array_manifests": tuple(row.manifest_payload() for row in arrays),
                    "threads_per_worker": self.threads_per_worker,
                }
            ),
        )

    def payload_value(self, key: str) -> object:
        matches = tuple(value for name, value in self.payload_entries if name == key)
        if len(matches) != 1:
            raise ProtocolError(f"P-DCAPS worker payload key {key!r} is absent.")
        return matches[0]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_worker_request_v1",
            "outer_center": self.outer_center,
            "ordinal": self.ordinal,
            "operation": self.operation,
            "payload_entries": [list(row) for row in self.payload_entries],
            "array_manifests": [row.manifest_payload() for row in self.arrays],
            "threads_per_worker": self.threads_per_worker,
            "request_hash": self.request_hash,
        }


@dataclass(frozen=True)
class WorkerResult:
    outer_center: str
    ordinal: int
    request_hash: str
    operation: str
    manifest_entries: tuple[tuple[str, object], ...]
    array_hashes: tuple[tuple[str, str], ...]
    artifact_paths: tuple[str, ...]
    workload_count: int
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        entries = tuple((str(key), value) for key, value in self.manifest_entries)
        array_hashes = tuple((str(name), str(value)) for name, value in self.array_hashes)
        paths = tuple(str(value) for value in self.artifact_paths)
        require_sha256(self.request_hash, "worker request hash")
        if (
            outer not in CENTERS
            or int(self.ordinal) < 0
            or not str(self.operation)
            or tuple(key for key, _ in entries)
            != tuple(sorted(key for key, _ in entries))
            or len({key for key, _ in entries}) != len(entries)
            or tuple(name for name, _ in array_hashes)
            != tuple(sorted(name for name, _ in array_hashes))
            or len({name for name, _ in array_hashes}) != len(array_hashes)
            or len(set(paths)) != len(paths)
            or any(not path or "\x00" in path for path in paths)
            or isinstance(self.workload_count, bool)
            or int(self.workload_count) < 0
        ):
            raise ProtocolError("P-DCAPS worker result topology drifted.")
        for key, value in entries:
            validate_plain_payload(value, path=f"manifest_entries[{key}]")
            if _contains_array(value):
                raise ProtocolError("P-DCAPS worker results must remain compact.")
        for _, digest in array_hashes:
            require_sha256(digest, "worker result array hash")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "operation", str(self.operation))
        object.__setattr__(self, "manifest_entries", entries)
        object.__setattr__(self, "array_hashes", array_hashes)
        object.__setattr__(self, "artifact_paths", paths)
        object.__setattr__(self, "workload_count", int(self.workload_count))
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_worker_result_v1",
                    "outer_center": outer,
                    "ordinal": self.ordinal,
                    "request_hash": self.request_hash,
                    "operation": self.operation,
                    "manifest_entries": entries,
                    "array_hashes": array_hashes,
                    "artifact_paths": paths,
                    "workload_count": self.workload_count,
                    "large_scientific_arrays_returned": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_worker_result_v1",
            "outer_center": self.outer_center,
            "ordinal": self.ordinal,
            "request_hash": self.request_hash,
            "operation": self.operation,
            "manifest_entries": [list(row) for row in self.manifest_entries],
            "array_hashes": [list(row) for row in self.array_hashes],
            "artifact_paths": list(self.artifact_paths),
            "workload_count": self.workload_count,
            "large_scientific_arrays_returned": False,
            "result_hash": self.result_hash,
        }


def _validate_array(value: np.ndarray, *, path: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or not value.flags.c_contiguous
        or value.dtype.hasobject
        or value.dtype.kind not in "biuf"
        or (value.dtype.kind == "f" and not np.isfinite(value).all())
    ):
        raise ProtocolError(
            f"P-DCAPS {path} is not a finite contiguous primitive array."
        )


def _contains_array(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    return isinstance(value, tuple) and any(_contains_array(item) for item in value)


__all__ = (
    "ContiguousArray",
    "WorkerRequest",
    "WorkerResult",
    "validate_plain_payload",
)
