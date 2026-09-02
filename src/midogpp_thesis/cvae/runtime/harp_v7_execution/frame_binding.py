"""Durable, label-free frame-store identity for HARP v7 classifier tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..artifact_io import atomic_json, read_json, sha256_file
from .hash_contracts import require_sha256, require_stable_hash


@dataclass(frozen=True, slots=True)
class FrameBinding:
    """Independent semantic and byte identities for one immutable frame store."""

    array_sha256: str
    receipt_hash: str
    receipt_sha256: str


def persist_or_validate_frame_binding(
    *,
    array_path: Path,
    receipt_path: Path,
    shape: Sequence[int],
) -> FrameBinding:
    """Seal an existing frame array without reading or storing labels."""

    if (
        not array_path.is_absolute()
        or not receipt_path.is_absolute()
        or array_path.is_symlink()
        or receipt_path.is_symlink()
        or not array_path.is_file()
    ):
        raise ProtocolError("HARP v7 frame binding paths are unsafe.")
    array_sha256 = require_sha256(
        sha256_file(array_path), name="frame-array hash"
    )
    body = {
        "schema_version": "midogpp_harp_v7_scratch_frame_receipt_v1",
        "array_sha256": array_sha256,
        "shape": [int(value) for value in shape],
        "dtype": "float32",
        "labels_stored": False,
    }
    receipt_hash = require_stable_hash(
        stable_hash(body), name="frame-receipt hash"
    )
    expected = {**body, "frame_receipt_hash": receipt_hash}
    if receipt_path.exists():
        if not receipt_path.is_file() or read_json(receipt_path) != expected:
            raise ProtocolError("HARP v7 existing frame receipt drifted.")
    else:
        atomic_json(receipt_path, expected)
    if read_json(receipt_path) != expected:
        raise ProtocolError("HARP v7 frame receipt failed a durable round trip.")
    return FrameBinding(
        array_sha256=array_sha256,
        receipt_hash=receipt_hash,
        receipt_sha256=require_sha256(
            sha256_file(receipt_path), name="frame-receipt SHA-256"
        ),
    )


__all__ = ("FrameBinding", "persist_or_validate_frame_binding")
