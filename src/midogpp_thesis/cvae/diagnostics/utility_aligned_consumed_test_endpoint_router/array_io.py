"""Experiment-local atomic NumPy writers for transient checkpoints."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping

import numpy as np

from .artifact_io import atomic_bytes


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
    atomic_bytes(path, stream.getvalue())


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, **dict(arrays))
    atomic_bytes(path, stream.getvalue())


__all__ = ("atomic_save_npy", "atomic_save_npz")
