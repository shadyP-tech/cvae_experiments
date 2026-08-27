"""Path-free canonical hashes used by the SCEPTRE diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...protocol import ProtocolError


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE value is not canonically serializable.") from exc
    return text.encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash SCEPTRE member: {source.name}.") from exc
    return digest.hexdigest()


def require_sha256(value: object, role: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ProtocolError(f"SCEPTRE {role} is not a SHA-256 digest.")
    return text


__all__ = ("canonical_bytes", "canonical_hash", "file_sha256", "require_sha256")
