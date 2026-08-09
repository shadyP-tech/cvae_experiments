"""Hash-validated float32 probabilities behind the global prediction seal.

The exact-tail producer has always persisted probability bytes next to binary
predictions.  This module turns those bytes into an explicit immutable
capability without weakening the pre-label global seal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .contracts import expected_prediction_keys
from .seals import GlobalPredictionSeal


ProbabilityKey = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class SealedProbabilitySurface:
    """Canonical probability vectors whose exact bytes are globally sealed."""

    probabilities_by_key: Mapping[ProbabilityKey, np.ndarray]
    seal: GlobalPredictionSeal

    def __post_init__(self) -> None:
        self.seal.verify_complete()
        expected = expected_prediction_keys()
        observed = dict(self.probabilities_by_key)
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise ProtocolError("Exact-tail probability surface coverage drifted.")
        cell_by_key = {cell.key: cell for cell in self.seal.cells}
        normalized: dict[ProbabilityKey, np.ndarray] = {}
        for key in expected:
            values = np.asarray(observed[key])
            if (
                values.ndim != 1
                or values.size == 0
                or values.dtype != np.float32
                or not np.isfinite(values).all()
                or np.any((values < np.float32(0.0)) | (values > np.float32(1.0)))
            ):
                raise ProtocolError(
                    "Exact-tail probabilities must be finite float32 vectors in [0,1]."
                )
            if array_sha256(values) != cell_by_key[key].probability_sha256:
                raise ProtocolError(
                    "Exact-tail probability bytes drifted from their cell seal."
                )
            values.setflags(write=False)
            normalized[key] = values
        object.__setattr__(
            self, "probabilities_by_key", MappingProxyType(normalized)
        )


@dataclass(frozen=True)
class SealedSupportProbabilitySurface:
    """Label-free support probabilities sealed alongside evaluation outputs."""

    probabilities_by_key: Mapping[ProbabilityKey, np.ndarray]
    seal: GlobalPredictionSeal

    def __post_init__(self) -> None:
        self.seal.verify_complete()
        expected = expected_prediction_keys()
        observed = dict(self.probabilities_by_key)
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise ProtocolError(
                "Exact-tail support-probability surface coverage drifted."
            )
        cell_by_key = {cell.key: cell for cell in self.seal.cells}
        normalized: dict[ProbabilityKey, np.ndarray] = {}
        for key in expected:
            values = np.asarray(observed[key])
            if (
                values.ndim != 1
                or values.size == 0
                or values.dtype != np.float32
                or not np.isfinite(values).all()
                or np.any((values < np.float32(0.0)) | (values > np.float32(1.0)))
            ):
                raise ProtocolError(
                    "Exact-tail support probabilities must be finite float32 "
                    "vectors in [0,1]."
                )
            if array_sha256(values) != cell_by_key[key].support_probability_sha256:
                raise ProtocolError(
                    "Exact-tail support-probability bytes drifted from their cell seal."
                )
            values.setflags(write=False)
            normalized[key] = values
        object.__setattr__(
            self, "probabilities_by_key", MappingProxyType(normalized)
        )


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including dtype and geometry, not only raw payload bytes."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "ProbabilityKey",
    "SealedProbabilitySurface",
    "SealedSupportProbabilitySurface",
    "array_sha256",
)
