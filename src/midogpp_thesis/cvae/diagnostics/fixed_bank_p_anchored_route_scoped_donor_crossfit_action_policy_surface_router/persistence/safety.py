"""Shared persistence firewall for labels, paths, and capability handles."""

from __future__ import annotations

from typing import Mapping

from ....protocol import ProtocolError


FORBIDDEN_KEYS = {
    "raw_label",
    "raw_labels",
    "labels",
    "label_value",
    "truth",
    "targets",
    "image_path",
    "sample_path",
    "scratch_path",
    "absolute_path",
    "label_loader",
    "label_capability",
}


def reject_forbidden_persisted_values(
    value: object,
    *,
    key: str | None = None,
) -> None:
    normalized = None if key is None else key.lower()
    if normalized is not None and normalized in FORBIDDEN_KEYS:
        raise ProtocolError(f"P-DCAPS persisted forbidden field: {key}.")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            reject_forbidden_persisted_values(child, key=str(child_key))
    elif isinstance(value, (tuple, list)):
        for child in value:
            reject_forbidden_persisted_values(child)
    elif isinstance(value, str) and value.startswith(("/data/", "/home/", "/Users/")):
        raise ProtocolError("P-DCAPS persistence contains an absolute filesystem path.")


__all__ = ("FORBIDDEN_KEYS", "reject_forbidden_persisted_values")
