"""Scientific role enums and canonical scalar/probability serialization."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
import math
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


PROBABILITY_CLIP = 1.0e-6
BASELINE_THRESHOLD = 0.5

_FORBIDDEN_FEATURE_TOKENS = (
    "label",
    "truth",
    "outcome",
    "oracle",
    "bacc",
    "brier",
    "log_loss",
    "logloss",
    "evaluation_score",
    "center_id",
    "target_id",
)


class SurfaceRole(str, Enum):
    SOURCE_TRAIN_DEVELOPMENT = "SOURCE_TRAIN_DEVELOPMENT"
    # Ergonomic alias; canonical serialization remains the long, phase-explicit value.
    SOURCE_TRAIN = "SOURCE_TRAIN_DEVELOPMENT"
    TARGET_EVALUATION = "TARGET_EVALUATION"


class Direction(str, Enum):
    D01 = "D01"
    D10 = "D10"
    FULL = "FULL"


class CompositeKind(str, Enum):
    B = "B"
    U_FULL = "U_FULL"
    D01_ONLY = "D01_ONLY"
    D10_ONLY = "D10_ONLY"
    BOTH = "BOTH"
    SOFT_TOPK = "BOTH"  # Historical type spelling; the successor serializes BOTH.


class AdmissionStatus(str, Enum):
    ADMITTED = "ADMITTED"
    NO_NONZERO_SAFE_OOF_COVERAGE = "NO_NONZERO_SAFE_OOF_COVERAGE"
    # Source-compatibility spelling only; serialization uses the protocol term above.
    ZERO_FRONTIER = "NO_NONZERO_SAFE_OOF_COVERAGE"
    INSUFFICIENT_ROUTED_OOF = "INSUFFICIENT_ROUTED_OOF"
    APPROXIMATE_BOUNDS_FAILED = "APPROXIMATE_BOUNDS_FAILED"


def canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"HARP v21 {name} must be a canonical nonempty string.")
    return value


def finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ProtocolError(f"HARP v21 {name} must be numeric.")
    output = float(value)
    if not math.isfinite(output):
        raise ProtocolError(f"HARP v21 {name} must be finite.")
    return 0.0 if output == 0.0 else output


def canonical_probability_hex(values: Sequence[str]) -> tuple[str, ...]:
    """Validate once per immutable tuple identity before value-based parsing.

    An equality-keyed type attestation would be unsafe: a str subclass can
    equal a previously valid cell. Retaining the tuple and checking identity
    avoids that ambiguity and prevents id reuse. The FIFO bound also limits
    retention of probability vectors; no fitted or outcome data are cached.
    """
    rows = tuple(values)
    validated = getattr(canonical_probability_hex, "_validated_probability_tuples", None)
    if validated is not None:
        entry = validated.get(id(rows))
        if entry is not None and entry[0] is rows:
            return entry[1]
    if any(type(value) is not str for value in rows):
        raise ProtocolError("HARP v21 probability cells must be float32 hex strings.")
    canonical = _canonical_probability_tuple(rows)
    if validated is None:
        validated = {}
        canonical_probability_hex._validated_probability_tuples = validated
    # Both input and returned tuples contain only exact immutable strings.
    # A modified caller list is converted to a new tuple and validated again.
    for checked in (rows, canonical):
        key = id(checked)
        if key not in validated and len(validated) >= 2048:
            validated.pop(next(iter(validated)), None)
        validated[key] = (checked, canonical)
    return canonical


@lru_cache(maxsize=2048)
def _canonical_probability_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    """Bounded, pure label-free parsing cache shared by branch configurations."""
    cells: list[str] = []
    for raw in values:
        if type(raw) is not str or len(raw) != 8:
            raise ProtocolError("HARP v21 probability cells must be little-endian float32 hex.")
        try:
            packed = bytes.fromhex(raw)
        except ValueError as exc:
            raise ProtocolError("HARP v21 probability cells must be hexadecimal.") from exc
        value = struct.unpack("<f", packed)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v21 probabilities must lie in [0,1].")
        cells.append(raw.lower())
    if not cells:
        raise ProtocolError("HARP v21 probability vectors cannot be empty.")
    return tuple(cells)


def float32_probability_hex(values: Sequence[float]) -> tuple[str, ...]:
    cells: list[str] = []
    for raw in values:
        value = finite(raw, name="probability")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("HARP v21 probabilities must lie in [0,1].")
        cells.append(struct.pack("<f", value).hex())
    if not cells:
        raise ProtocolError("HARP v21 probability vectors cannot be empty.")
    return tuple(cells)


def decode_probability_hex(values: Sequence[str]) -> tuple[float, ...]:
    return _decode_probability_tuple(canonical_probability_hex(values))


@lru_cache(maxsize=2048)
def _decode_probability_tuple(values: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(
        float(struct.unpack("<f", bytes.fromhex(cell))[0])
        for cell in values
    )


