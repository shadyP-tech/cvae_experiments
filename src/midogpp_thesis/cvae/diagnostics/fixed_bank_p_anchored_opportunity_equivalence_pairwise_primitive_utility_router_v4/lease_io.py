"""Crash-safe no-overwrite persistence for OE-PPUR v4 lease journals."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import stat
import tempfile

from ...protocol import ProtocolError
from . import durable_io as _durable_io
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
        raise ProtocolError(f"OE-PPUR v4 {role} publication target is unsafe.")
    descriptor, rendered = tempfile.mkstemp(
        prefix=pending_prefix(target.name),
        dir=parent,
    )
    pending = Path(rendered)
    published = False
    try:
        fsync_directory(parent)
        raw = _durable_io.canonical_json_file_bytes(payload)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError(f"short OE-PPUR v4 {role} write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if read_json_regular(pending, role=f"pending {role}") != dict(payload):
            raise ProtocolError(f"OE-PPUR v4 pending {role} read-back drifted.")
        try:
            os.link(pending, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProtocolError(
                f"OE-PPUR v4 {role} refuses overwrite."
            ) from exc
        published = True
        fsync_directory(parent)
        os.unlink(pending)
        fsync_directory(parent)
        if read_json_regular(target, role=role) != dict(payload):
            raise ProtocolError(f"OE-PPUR v4 {role} read-back drifted.")
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
    return _durable_io.read_json_regular_nofollow(Path(path), role=role)


def pending_prefix(target_name: str) -> str:
    safe = "".join(ch for ch in str(target_name) if ch.isalnum() or ch in "._-")
    if not safe or safe in {".", ".."}:
        raise ProtocolError("OE-PPUR v4 pending publication name is unsafe.")
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
            raise ProtocolError("OE-PPUR v4 pending publication marker is unsafe.")
        rows.append(candidate)
    return tuple(sorted(rows, key=lambda value: value.name))


def fsync_directory(path: Path) -> None:
    _durable_io.fsync_directory(Path(path))


def stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return _durable_io.stat_payload(value)


__all__ = (
    "fsync_directory",
    "pending_prefix",
    "pending_publications",
    "publish_json_no_overwrite",
    "read_json_regular",
    "stat_payload",
)
