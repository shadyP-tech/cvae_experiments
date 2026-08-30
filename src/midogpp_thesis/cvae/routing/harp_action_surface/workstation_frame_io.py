"""Closed-world readers for physical HARP label-blind frame caches."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from pathlib import Path

import numpy as np

from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ..harp_protocol.hashing import canonical_hash, require_sha256


def content_members(content: Mapping[str, object]) -> dict[str, str]:
    if isinstance(content.get("members"), Mapping):
        values = {
            str(key): require_sha256(value, name="cache member hash")
            for key, value in content["members"].items()  # type: ignore[union-attr]
        }
    elif isinstance(content.get("member_sha256"), Mapping):
        values = {
            str(key): require_sha256(value, name="cache member hash")
            for key, value in content["member_sha256"].items()  # type: ignore[union-attr]
        }
    elif isinstance(content.get("files"), list):
        values = {}
        for item in content["files"]:  # type: ignore[index]
            if not isinstance(item, Mapping):
                raise ProtocolError("HARP cache content file row is malformed.")
            relative = str(item.get("path", ""))
            if relative in values:
                raise ProtocolError("HARP cache content contains duplicate members.")
            values[relative] = require_sha256(
                item.get("sha256"), name="cache member hash"
            )
    else:
        raise ProtocolError("HARP cache content index lacks a member inventory.")
    if not values:
        raise ProtocolError("HARP cache content index is empty.")
    return values


def content_hash(content: Mapping[str, object]) -> str:
    for name in ("cache_binding_hash", "content_hash", "content_index_hash"):
        if name in content:
            observed = require_sha256(content[name], name=f"HARP {name}")
            if observed != canonical_hash(
                {key: value for key, value in content.items() if key != name}
            ):
                raise ProtocolError("HARP cache content-index hash drifted.")
            return observed
    return canonical_hash(dict(content))


def safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise ProtocolError("HARP cache member path is unsafe.")
    member = (root / value).resolve()
    try:
        member.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("HARP cache member escaped its root.") from exc
    return member


def load_float32_shard(path: Path) -> np.ndarray:
    try:
        if path.suffix == ".npy":
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"embeddings"}:
                    raise ProtocolError("HARP frame NPZ inventory drifted.")
                values = np.asarray(archive["embeddings"])
        else:
            raise ProtocolError("HARP frame shards must be NPY or NPZ.")
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"Cannot load HARP frame shard: {path}.") from exc
    if (
        values.ndim != 2
        or values.shape[1] != COMMON_OUTPUT_DIM
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("HARP frame shard values drifted.")
    return values


def read_frame_rows(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = tuple(dict(row) for row in reader)
    except (OSError, csv.Error) as exc:
        raise ProtocolError("Cannot read HARP frame row index.") from exc
    if not rows:
        raise ProtocolError("HARP frame row index is empty.")
    return columns, rows


__all__ = (
    "content_hash",
    "content_members",
    "load_float32_shard",
    "read_frame_rows",
    "safe_member",
)
