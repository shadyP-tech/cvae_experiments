"""Full-width deterministic hashes for the neutral HARP runtime."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

import numpy as np

from ...protocol import ProtocolError


_DIGEST = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_sha256(value: object) -> str:
    """Hash one JSON-like value without shortening the digest."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("HARP value is not canonically serializable.") from exc
    return hashlib.sha256(encoded).hexdigest()


def raw_array_sha256(values: np.ndarray) -> str:
    """Hash every C-order byte of an already typed array."""

    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def identity_sequence_sha256(
    values: Iterable[str], *, identity_kind: str
) -> str:
    rows = tuple(values)
    return canonical_sha256(
        {
            "schema_version": "midogpp_harp_identity_sequence_v1",
            "identity_kind": identity_kind,
            "values": list(rows),
        }
    )


def require_digest(value: object, *, name: str) -> str:
    """Accept repository lineage hashes while preserving them byte-for-byte.

    Older immutable bank and generation locks use the repository's 16-character
    semantic digest.  Newly produced HARP byte and seal identities always use a
    full SHA-256 digest.
    """

    text = str(value)
    if _DIGEST.fullmatch(text) is None:
        raise ProtocolError(f"HARP {name} must be a canonical digest.")
    return text


def require_sha256(value: object, *, name: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ProtocolError(f"HARP {name} must be a full SHA-256 digest.")
    return text


__all__ = (
    "canonical_sha256",
    "identity_sequence_sha256",
    "raw_array_sha256",
    "require_digest",
    "require_sha256",
)
