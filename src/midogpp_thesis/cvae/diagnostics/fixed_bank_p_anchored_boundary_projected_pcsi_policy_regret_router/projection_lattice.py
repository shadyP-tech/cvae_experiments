"""Canonical binary32 threshold lattice for PCSI-PARC actions."""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError


BINARY32_LE = np.dtype("<f4")
THRESHOLD = np.float32(0.5)
THRESHOLD_PREDECESSOR = np.nextafter(
    THRESHOLD,
    np.float32(0.0),
    dtype=np.float32,
)


def as_binary32(values: object, *, name: str) -> np.ndarray:
    """Return a one-dimensional canonical little-endian binary32 array."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError(f"PCSI-PARC {name} is not a finite binary32 vector.")
    if np.any((array < np.float32(0.0)) | (array > np.float32(1.0))):
        raise ProtocolError(f"PCSI-PARC {name} escaped the probability lattice.")
    return np.ascontiguousarray(array.astype(BINARY32_LE, copy=False))


def hard_classes(values: object) -> np.ndarray:
    return as_binary32(values, name="hard-class input") >= THRESHOLD


def canonical_bytes(values: object) -> bytes:
    return as_binary32(values, name="persisted probability").tobytes(order="C")


__all__ = (
    "BINARY32_LE",
    "THRESHOLD",
    "THRESHOLD_PREDECESSOR",
    "as_binary32",
    "canonical_bytes",
    "hard_classes",
)
