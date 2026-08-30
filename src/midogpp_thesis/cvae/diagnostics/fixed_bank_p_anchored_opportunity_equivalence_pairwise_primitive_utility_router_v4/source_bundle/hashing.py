"""Small canonical parsing and hashing helpers for immutable source bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError


_SHA256 = re.compile(r"[0-9a-f]{64}")


def text(value: object, *, role: str) -> str:
    result = str(value).strip()
    if not result:
        raise ProtocolError(f"OE-PPUR v4 historical-lineage requires non-empty {role}.")
    return result


def sha256(value: object, *, role: str) -> str:
    result = text(value, role=role).lower()
    if _SHA256.fullmatch(result) is None:
        raise ProtocolError(f"OE-PPUR v4 historical-lineage {role} is not a SHA-256 digest.")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: object, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


def _no_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"OE-PPUR v4 historical-lineage JSON contains duplicate key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"OE-PPUR v4 historical-lineage JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"OE-PPUR v4 historical-lineage JSON root is not an object: {path.name}")
    return value


def exact_keys(value: Mapping[str, object], expected: Sequence[str], *, role: str) -> None:
    if set(value) != set(expected):
        raise ProtocolError(f"OE-PPUR v4 historical-lineage {role} keys drifted.")


__all__ = ("array_sha256", "exact_keys", "file_sha256", "read_json", "sha256", "text")
