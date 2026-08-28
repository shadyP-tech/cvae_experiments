"""No-follow, no-overwrite durability primitives for v3 preparation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError


def write_bytes_exclusive(path: Path, raw: bytes, *, role: str) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v3 {role} publication is unsafe.") from exc
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise ProtocolError(f"OE-PPUR v3 {role} write was incomplete.")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def hash_unique_regular_file(path: Path, *, role: str) -> tuple[str, int]:
    descriptor, before = _open_unique_regular(path, role=role)
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _assert_unchanged(before, after, role=role)
    return digest.hexdigest(), int(before.st_size)


def read_bounded_unique_file(
    path: Path,
    *,
    maximum_bytes: int,
    role: str,
) -> tuple[bytes, str]:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ProtocolError("OE-PPUR v3 bounded-read limit is invalid.")
    descriptor, before = _open_unique_regular(path, role=role)
    if before.st_size > maximum_bytes:
        os.close(descriptor)
        raise ProtocolError(f"OE-PPUR v3 {role} is oversized.")
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ProtocolError(f"OE-PPUR v3 {role} is oversized.")
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _assert_unchanged(before, after, role=role)
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise ProtocolError(f"OE-PPUR v3 {role} changed while read.")
    return raw, digest.hexdigest()


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 directory fsync target is unsafe.") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_unique_regular(path: Path, *, role: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v3 {role} is absent or unsafe.") from exc
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        os.close(descriptor)
        raise ProtocolError(f"OE-PPUR v3 {role} is not a unique regular file.")
    return descriptor, before


def _assert_unchanged(
    before: os.stat_result,
    after: os.stat_result,
    *,
    role: str,
) -> None:
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ProtocolError(f"OE-PPUR v3 {role} changed while read.")


__all__ = (
    "fsync_directory",
    "hash_unique_regular_file",
    "read_bounded_unique_file",
    "write_bytes_exclusive",
)
