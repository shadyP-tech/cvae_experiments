"""Canonical full-length hashes for residual top-up contracts and arrays."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def array_bundle_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


__all__ = ("array_bundle_sha256", "array_sha256", "canonical_sha256")
