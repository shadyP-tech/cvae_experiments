"""Canonical hashes shared by SCEPTRE v5 source production modules."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def block_bundle_sha256(block: np.ndarray, rows_per_class: int) -> str:
    values = np.ascontiguousarray(block, dtype=np.float32)
    truth = np.concatenate(
        (
            np.zeros(rows_per_class, dtype=np.int64),
            np.ones(rows_per_class, dtype=np.int64),
        )
    )
    digest = hashlib.sha256()
    for array in (values, truth):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ("block_bundle_sha256", "canonical_sha256")
