"""Durable, deterministic file publication helpers for HARP v21 preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import io
import os
from pathlib import Path

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import atomic_json, sha256_file
from .input_surfaces import CACHE_ROWS, CONTENT_INDEX, HarpCacheRow
from .preparation_contracts import V21_PREPARATION_IDENTITY


def atomic_text(path: Path, text: str) -> None:
    """Publish UTF-8 text atomically and fsync the containing directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    inventory = single_inventory(root, role="fsync tree")
    for path in inventory:
        if path.is_file():
            fsync_file(path)
    directories = tuple(
        sorted(
            (path for path in inventory if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
    )
    for path in (*directories, root):
        fsync_directory(path)


def single_inventory(root: Path, *, role: str) -> tuple[Path, ...]:
    """Return one deterministic, symlink-free recursive inventory snapshot."""

    entries = tuple(sorted(root.rglob("*")))
    if any(path.is_symlink() for path in entries):
        raise ProtocolError(f"HARP {role} inventory contains a symlink.")
    return entries


def write_cache_rows(
    path: Path,
    rows: Sequence[HarpCacheRow],
    *,
    row_schema: str = V21_PREPARATION_IDENTITY.cache_identity.row_schema,
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "schema_version",
            "row_id",
            "center",
            "case_id",
            "split_role",
            "split_row_index",
            "embedding_file",
            "embedding_row_index",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row_schema,
                row.sample_id,
                row.center,
                row.case_id,
                row.split_role,
                row.split_row_index,
                row.embedding_file,
                row.embedding_row_index,
            )
        )
    atomic_text(path, buffer.getvalue())


def write_content_index(
    path: Path,
    members: Mapping[str, str],
    *,
    content_schema: str = V21_PREPARATION_IDENTITY.cache_identity.content_schema,
) -> None:
    base: dict[str, object] = {
        "schema_version": content_schema,
        "members": dict(sorted(members.items())),
    }
    atomic_json(path, {**base, "content_index_hash": canonical_hash(base)})


def write_final_content_index(
    root: Path,
    *,
    content_schema: str = V21_PREPARATION_IDENTITY.cache_identity.content_schema,
) -> None:
    inventory = single_inventory(root, role="prepared cache")
    members = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in inventory
        if path.is_file() and path.relative_to(root) != CONTENT_INDEX
    }
    write_content_index(root / CONTENT_INDEX, members, content_schema=content_schema)


__all__ = (
    "atomic_text",
    "fsync_file",
    "fsync_directory",
    "fsync_tree",
    "single_inventory",
    "write_cache_rows",
    "write_content_index",
    "write_final_content_index",
)
