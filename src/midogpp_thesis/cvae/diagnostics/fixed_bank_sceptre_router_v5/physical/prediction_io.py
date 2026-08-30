"""Canonical, refusal-safe persistence primitives for SCEPTRE v5 predictions."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_array,
)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persist_exact_npy(path: Path, values: np.ndarray, *, role: str) -> None:
    """Create an NPY member once or authenticate byte-equivalent existing data."""

    if path.is_symlink():
        raise ProtocolError(f"SCEPTRE v5 {role} is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError(f"SCEPTRE v5 {role} is unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.shape != values.shape
            or observed.dtype != values.dtype
            or sha256_array(observed) != sha256_array(values)
        ):
            raise ProtocolError(f"SCEPTRE v5 {role} differs; refusing regeneration.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def persist_exact_json(path: Path, payload: Mapping[str, object]) -> None:
    """Create a JSON member once or authenticate its exact canonical value."""

    if path.is_symlink():
        raise ProtocolError("SCEPTRE v5 prediction JSON member is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(payload):
            raise ProtocolError("SCEPTRE v5 prediction JSON differs; refusing overwrite.")
        return
    atomic_json(path, payload)


__all__ = ("canonical_sha256", "persist_exact_json", "persist_exact_npy")
