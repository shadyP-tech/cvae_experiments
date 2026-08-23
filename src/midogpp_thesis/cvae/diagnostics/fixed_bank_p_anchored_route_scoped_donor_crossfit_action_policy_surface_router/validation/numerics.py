"""Numerical content checks that never refit an optimizer."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ....protocol import ProtocolError
from ....runtime.artifact_io import sha256_array


def validate_finite_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    rows = {str(key): np.asarray(value) for key, value in arrays.items()}
    if not rows:
        raise ProtocolError("P-DCAPS numeric validation received no arrays.")
    hashes: dict[str, str] = {}
    for key, value in rows.items():
        if value.dtype == object or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ProtocolError(f"P-DCAPS numeric array is invalid: {key}.")
        hashes[key] = sha256_array(value)
    if expected_hashes is not None and hashes != dict(expected_hashes):
        raise ProtocolError("P-DCAPS numeric array hashes drifted.")
    return {"status": "PASS", "array_count": len(rows), "array_hashes": hashes, "optimizer_refit_count": 0}


__all__ = ("validate_finite_arrays",)
