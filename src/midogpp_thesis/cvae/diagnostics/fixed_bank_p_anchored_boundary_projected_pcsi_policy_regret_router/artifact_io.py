"""Idempotent, nonrepairing, label/path-safe JSON persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .hashing import json_native


FORBIDDEN_KEYS = frozenset(
    {
        "label",
        "labels",
        "ground_truth",
        "true_label",
        "target_label",
        "image_path",
        "sample_path",
        "manifest_path",
    }
)


def persist_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    allow_paths: bool = False,
) -> None:
    converted = json_native(payload)
    if not isinstance(converted, dict):
        raise ProtocolError("PCSI-PARC JSON product must be an object.")
    if not allow_paths:
        reject_sensitive_persistence(converted)
    if path.is_symlink():
        raise ProtocolError("PCSI-PARC JSON path is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != converted:
            raise ProtocolError(f"PCSI-PARC refuses repair of {path.name}.")
        return
    atomic_json(path, converted)


def persist_rows(
    path: Path,
    rows: object,
    *,
    schema_version: str,
    allow_empty: bool = False,
) -> None:
    converted = json_native(rows)
    if not isinstance(converted, list) or (not converted and not allow_empty):
        raise ProtocolError("PCSI-PARC table row topology drifted.")
    reject_sensitive_persistence(converted)
    persist_json(
        path,
        {
            "schema_version": schema_version,
            "row_count": len(converted),
            "rows": converted,
        },
    )


def reject_sensitive_persistence(value: object) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            folded = str(raw_key).casefold()
            if folded in FORBIDDEN_KEYS or folded.endswith("_path"):
                raise ProtocolError(
                    f"PCSI-PARC persisted key is forbidden: {raw_key}."
                )
            reject_sensitive_persistence(nested)
        return
    if isinstance(value, (tuple, list)):
        for nested in value:
            reject_sensitive_persistence(nested)
        return
    if isinstance(value, str):
        folded = value.casefold()
        if (
            "/users/" in folded
            or "/home/" in folded
            or "file://" in folded
            or folded.startswith("/data/")
        ):
            raise ProtocolError("PCSI-PARC persisted value exposes a path.")


__all__ = ("persist_json", "persist_rows", "reject_sensitive_persistence")
