"""Crash-safe no-overwrite persistence for OE-PPUR v3 lease journals."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import tempfile

from ...protocol import ProtocolError
from .hashing import canonical_json_bytes
from .run_paths import assert_no_symlink_chain


def publish_json_no_overwrite(
    path: Path,
    payload: Mapping[str, object],
    *,
    role: str,
) -> None:
    """Publish complete bytes atomically without ever replacing ``path``.

    A same-directory pending inode is made durable first.  Hard-link
    publication is atomic and fails when the final name already exists.  Any
    interruption leaves either a complete final member or a discoverable
    ``.pending`` entry; it never exposes partial bytes at the final name.
    """

    target = Path(path)
    parent = target.parent
    assert_no_symlink_chain(parent)
    if (
        not parent.is_dir()
        or target.exists()
        or target.is_symlink()
        or pending_publications(parent, target.name)
    ):
        raise ProtocolError(f"OE-PPUR v3 {role} publication target is unsafe.")
    descriptor, rendered = tempfile.mkstemp(
        prefix=pending_prefix(target.name),
        dir=parent,
    )
    pending = Path(rendered)
    published = False
    try:
        fsync_directory(parent)
        raw = canonical_json_bytes(payload) + b"\n"
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError(f"short OE-PPUR v3 {role} write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if read_json_regular(pending, role=f"pending {role}") != dict(payload):
            raise ProtocolError(f"OE-PPUR v3 pending {role} read-back drifted.")
        try:
            os.link(pending, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProtocolError(
                f"OE-PPUR v3 {role} refuses overwrite."
            ) from exc
        published = True
        fsync_directory(parent)
        os.unlink(pending)
        fsync_directory(parent)
        if read_json_regular(target, role=role) != dict(payload):
            raise ProtocolError(f"OE-PPUR v3 {role} read-back drifted.")
    except BaseException:
        # Never remove the pending name on failure.  It is the durable,
        # discoverable fail-closed marker for an interrupted publication.
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if published and pending.exists():
            # Reaching here means an exception occurred after atomic link
            # publication.  Keep both links so discovery rejects completion.
            pass


def read_json_regular(path: Path, *, role: str) -> dict[str, object]:
    target = Path(path)
    assert_no_symlink_chain(target)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v3 {role} is unreadable.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError(f"OE-PPUR v3 {role} is unsafe.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if stat_payload(before) != stat_payload(after) or len(raw) != before.st_size:
        raise ProtocolError(f"OE-PPUR v3 {role} changed while read.")

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
        raise ProtocolError(f"OE-PPUR v3 {role} is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"OE-PPUR v3 {role} is malformed.")
    return payload


def pending_prefix(target_name: str) -> str:
    safe = "".join(ch for ch in str(target_name) if ch.isalnum() or ch in "._-")
    if not safe or safe in {".", ".."}:
        raise ProtocolError("OE-PPUR v3 pending publication name is unsafe.")
    return f".{safe}.pending."


def pending_publications(parent: Path, target_name: str) -> tuple[Path, ...]:
    directory = Path(parent)
    assert_no_symlink_chain(directory)
    if not directory.is_dir():
        return ()
    prefix = pending_prefix(target_name)
    rows: list[Path] = []
    for candidate in directory.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        metadata = candidate.lstat()
        if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ProtocolError("OE-PPUR v3 pending publication marker is unsafe.")
        rows.append(candidate)
    return tuple(sorted(rows, key=lambda value: value.name))


def fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


__all__ = (
    "fsync_directory",
    "pending_prefix",
    "pending_publications",
    "publish_json_no_overwrite",
    "read_json_regular",
    "stat_payload",
)
