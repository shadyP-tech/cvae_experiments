"""No-follow byte evidence for immutable worker inputs and results."""

from __future__ import annotations

import hashlib
import os
from os import open as _open_descriptor
from pathlib import Path
import stat

from ..protocol import ProtocolError


_HASH_CHUNK_BYTES = 1024 * 1024


def hash_read_only_regular_file(path_text: str, *, role: str) -> str:
    """Hash one stable absolute regular file through an ``O_NOFOLLOW`` FD."""

    path = Path(path_text)
    if not path.is_absolute():
        raise ProtocolError(f"OE-PPUR {role} path is not absolute.")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR {role} file is absent.") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ProtocolError(f"OE-PPUR {role} is not a regular file.")
    descriptor = -1
    try:
        descriptor = _open_read_only_descriptor(path)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ProtocolError(f"OE-PPUR {role} changed before hashing.")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        try:
            path_after = path.lstat()
        except OSError as exc:
            raise ProtocolError(f"OE-PPUR {role} path changed after hashing.") from exc
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR {role} could not be reopened.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_before = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        identity_after != identity_before
        or path_identity_after != identity_after
        or path.is_symlink()
    ):
        raise ProtocolError(f"OE-PPUR {role} changed while hashing.")
    return digest.hexdigest()


def _open_read_only_descriptor(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return _open_descriptor(path, flags)


__all__ = ("hash_read_only_regular_file",)
