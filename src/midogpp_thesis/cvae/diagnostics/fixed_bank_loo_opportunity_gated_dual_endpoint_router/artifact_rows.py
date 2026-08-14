"""Canonical, label/path-free serialization of experiment rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .hashing import json_native


FORBIDDEN_PERSISTED_KEYS = frozenset(
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


def row_payload(value: object) -> dict[str, object]:
    converter = getattr(value, "to_payload", None)
    payload = json_native(converter() if callable(converter) else value)
    if not isinstance(payload, dict):
        raise ProtocolError("Dual-endpoint row must be a JSON object.")
    reject_forbidden_persistence(payload)
    return payload


def rows_payload(values: Sequence[object]) -> tuple[dict[str, object], ...]:
    rows = tuple(row_payload(value) for value in values)
    if not rows:
        raise ProtocolError("Dual-endpoint table cannot be empty.")
    columns = tuple(sorted(rows[0]))
    if not columns or any(tuple(sorted(row)) != columns for row in rows):
        raise ProtocolError("Dual-endpoint table row schemas differ.")
    return rows


def reject_forbidden_persistence(value: object, *, key: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            folded = str(raw_key).casefold()
            if folded in FORBIDDEN_PERSISTED_KEYS or folded.endswith("_path"):
                raise ProtocolError(
                    f"Dual-endpoint persisted key is forbidden: {raw_key}."
                )
            reject_forbidden_persistence(nested, key=str(raw_key))
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            reject_forbidden_persistence(nested, key=key)
        return
    if isinstance(value, str):
        folded = value.casefold()
        if (
            "/home/" in folded
            or "/users/" in folded
            or "file://" in folded
            or folded.startswith("/data/")
        ):
            raise ProtocolError("Dual-endpoint persisted value exposes a path.")


__all__ = (
    "FORBIDDEN_PERSISTED_KEYS",
    "reject_forbidden_persistence",
    "row_payload",
    "rows_payload",
)
