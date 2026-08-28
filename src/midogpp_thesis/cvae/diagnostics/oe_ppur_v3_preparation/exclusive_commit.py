"""Kernel-backed no-replace directory commit for OE-PPUR v3 envelopes."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

from ...protocol import ProtocolError


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one sibling directory only if destination is absent."""

    source_path = Path(source)
    destination_path = Path(destination)
    if (
        not source_path.is_absolute()
        or not destination_path.is_absolute()
        or source_path.parent != destination_path.parent
        or source_path.is_symlink()
        or not source_path.is_dir()
        or destination_path.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v3 exclusive commit topology is unsafe.")
    old = os.fsencode(source_path)
    new = os.fsencode(destination_path)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int
    if hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                _AT_FDCWD,
                old,
                _AT_FDCWD,
                new,
                _RENAME_NOREPLACE,
            )
        )
    elif hasattr(libc, "renamex_np"):
        renamex = libc.renamex_np
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
        result = int(renamex(old, new, _RENAME_EXCL))
    else:
        raise ProtocolError(
            "OE-PPUR v3 platform lacks an atomic no-replace directory commit."
        )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise ProtocolError(
                "OE-PPUR v3 exclusive commit destination already exists."
            )
        raise ProtocolError(
            "OE-PPUR v3 exclusive directory commit failed."
        ) from OSError(observed_errno, os.strerror(observed_errno))
    if source_path.exists() or source_path.is_symlink():
        raise ProtocolError("OE-PPUR v3 exclusive commit retained its source.")
    if destination_path.is_symlink() or not destination_path.is_dir():
        raise ProtocolError("OE-PPUR v3 exclusive commit destination drifted.")


__all__ = ("rename_directory_noreplace",)
