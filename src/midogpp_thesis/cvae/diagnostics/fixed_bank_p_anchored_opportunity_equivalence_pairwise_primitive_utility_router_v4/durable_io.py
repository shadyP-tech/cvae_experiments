"""Package-private durable filesystem primitives for OE-PPUR v4.

The scientific and authorization layers deliberately expose different
transactions.  This module owns only their common, fail-closed filesystem
edge: no-symlink stable reads, exclusive writes, canonical JSON bytes, and
durability barriers.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import tempfile

from ...protocol import ProtocolError
from .hashing import canonical_bytes
from .run_paths import assert_no_symlink_chain


_READ_CHUNK_BYTES = 1024 * 1024


def canonical_json_file_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the one canonical on-disk JSON representation."""

    return canonical_bytes(payload) + b"\n"


def stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Project stat metadata used to detect replacement during a read."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def read_regular_bytes_nofollow(
    path: Path,
    *,
    role: str,
) -> tuple[bytes, os.stat_result]:
    """Read one unique regular file and reject path or inode drift."""

    target = Path(path)
    assert_no_symlink_chain(target)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is unreadable.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError(f"OE-PPUR v4 {role} is unsafe.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if stat_payload(before) != stat_payload(after) or len(raw) != before.st_size:
        raise ProtocolError(f"OE-PPUR v4 {role} changed while read.")
    return raw, before


def read_json_regular_nofollow(path: Path, *, role: str) -> dict[str, object]:
    """Read an object-valued JSON file with duplicate-key rejection."""

    raw, _metadata = read_regular_bytes_nofollow(path, role=role)

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"OE-PPUR v4 {role} is malformed.")
    return payload


def fsync_directory(path: Path) -> None:
    """Durably flush one inspected, non-symlink directory."""

    target = Path(path)
    assert_no_symlink_chain(target)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 durability directory is unsafe.") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProtocolError(
                "OE-PPUR v4 durability parent is not a directory."
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_regular_file_and_parent(path: Path, *, role: str) -> None:
    """Flush a unique regular member followed by its directory entry."""

    target = Path(path)
    assert_no_symlink_chain(target)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} is unsafe.") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProtocolError(
                f"OE-PPUR v4 {role} is not a unique regular file."
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(target.parent)


def write_bytes_exclusive(
    path: Path,
    payload: bytes,
    *,
    role: str,
    mode: int = 0o600,
) -> None:
    """Create, flush, and read back one member without replacement."""

    target = Path(path)
    assert_no_symlink_chain(target, allow_missing_leaf=True)
    if not isinstance(payload, bytes):
        raise TypeError("OE-PPUR v4 durable payload must be bytes.")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(target, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(f"short OE-PPUR v4 {role} write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(target.parent)
    observed, _metadata = read_regular_bytes_nofollow(target, role=role)
    if observed != payload:
        raise ProtocolError(f"OE-PPUR v4 {role} read-back drifted.")


def write_canonical_json_exclusive(
    path: Path,
    payload: Mapping[str, object],
    *,
    role: str,
) -> None:
    """Exclusively persist the canonical JSON file representation."""

    write_bytes_exclusive(
        path,
        canonical_json_file_bytes(payload),
        role=role,
    )


def replace_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    role: str,
) -> None:
    """Atomically replace one safe regular member and verify exact bytes."""

    target = Path(path)
    assert_no_symlink_chain(target.parent)
    if not isinstance(payload, bytes):
        raise TypeError("OE-PPUR v4 durable payload must be bytes.")
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} target is unsafe.") from exc
    if metadata is not None and (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} target is unsafe.")

    descriptor, rendered = tempfile.mkstemp(
        prefix=f".{target.name}.",
        dir=target.parent,
    )
    temporary = Path(rendered)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(f"short OE-PPUR v4 {role} write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        staged, _metadata = read_regular_bytes_nofollow(
            temporary,
            role=f"temporary {role}",
        )
        if staged != payload:
            raise ProtocolError(f"OE-PPUR v4 temporary {role} read-back drifted.")
        os.replace(temporary, target)
        fsync_directory(target.parent)
        observed, _metadata = read_regular_bytes_nofollow(target, role=role)
        if observed != payload:
            raise ProtocolError(f"OE-PPUR v4 {role} read-back drifted.")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except OSError:
            pass


def replace_canonical_json_atomic(
    path: Path,
    payload: Mapping[str, object],
    *,
    role: str,
) -> None:
    """Atomically replace one member with canonical JSON bytes."""

    replace_bytes_atomic(
        path,
        canonical_json_file_bytes(payload),
        role=role,
    )


__all__ = ()
