"""Canonical exact-float64 hashes shared by opportunity and utility surfaces."""

from __future__ import annotations

import numpy as np

from .contracts import canonical_sha256


def probability_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return canonical_sha256(
        {
            "schema": "label_free_probability_surface_v1",
            "dtype": "float64",
            "shape": tuple(int(value) for value in array.shape),
            "values": tuple(float(value).hex() for value in array),
        }
    )


def crossing_hash(baseline: np.ndarray, candidate: np.ndarray) -> str:
    signed = (candidate >= 0.5).astype(np.int8) - (baseline >= 0.5).astype(np.int8)
    return canonical_sha256(
        {
            "schema": "label_free_signed_crossing_mask_v1",
            "length": len(signed),
            "signed_crossings": tuple(int(value) for value in signed),
        }
    )


__all__ = ()
