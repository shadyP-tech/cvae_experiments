"""Compact canonical row-table persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json
from ..identity import canonical_hash
from .safety import FORBIDDEN_KEYS, reject_forbidden_persisted_values


def persist_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    schema_version: str,
    row_role: str,
) -> dict[str, object]:
    canonical_rows = tuple(dict(row) for row in rows)
    reject_forbidden_persisted_values(canonical_rows)
    payload_base = {
        "schema_version": str(schema_version),
        "row_role": str(row_role),
        "row_count": len(canonical_rows),
        "rows": list(canonical_rows),
        "raw_labels_persisted": False,
    }
    payload = {**payload_base, "table_hash": canonical_hash(payload_base)}
    target = Path(path)
    if target.exists():
        observed = read_json(target)
        if observed != payload:
            raise ProtocolError("P-DCAPS refuses to repair a different row table.")
        return payload
    atomic_json(target, payload)
    return payload


def load_rows(path: Path) -> tuple[dict[str, object], ...]:
    payload = read_json(Path(path))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ProtocolError("P-DCAPS row table is malformed.")
    unhashed = {key: value for key, value in payload.items() if key != "table_hash"}
    if (
        payload.get("row_count") != len(rows)
        or payload.get("table_hash") != canonical_hash(unhashed)
        or payload.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("P-DCAPS row table hash drifted.")
    reject_forbidden_persisted_values(rows)
    return tuple(dict(row) for row in rows)


__all__ = ("FORBIDDEN_KEYS", "load_rows", "persist_rows")
