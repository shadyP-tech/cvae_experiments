"""Closed-world tree, duplicate-byte, path, and lock primitives."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterator

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .contracts import TERMINAL_JOURNAL_MEMBERS


def audit_exact_tree(
    root: Path,
    *,
    role: str,
    expected_directories: frozenset[str],
    expected_members: tuple[object, ...],
) -> list[dict[str, object]]:
    assert_regular_tree(root, role=role)
    paths = tuple(root.rglob("*"))
    directories = frozenset(
        path.relative_to(root).as_posix() for path in paths if path.is_dir()
    )
    files = frozenset(
        path.relative_to(root).as_posix() for path in paths if path.is_file()
    )
    expected_by_path = {str(row.path): row for row in expected_members}
    if directories != expected_directories or files != frozenset(expected_by_path):
        if files & TERMINAL_JOURNAL_MEMBERS:
            raise ProtocolError("CBPUPR v2 preterminal terminal journal is present.")
        raise ProtocolError(f"CBPUPR v2 preterminal {role} inventory drifted.")
    result: list[dict[str, object]] = []
    for relative in sorted(files):
        path = root / relative
        expected = expected_by_path[relative]
        actual = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if (
            actual["size_bytes"] != expected.size_bytes
            or actual["sha256"] != expected.sha256
        ):
            raise ProtocolError(
                f"CBPUPR v2 preterminal {role} member bytes drifted: {relative}."
            )
        result.append(actual)
    return result


def validate_scratch_duplicates(
    *,
    artifact_rows: list[dict[str, object]],
    scratch_rows: list[dict[str, object]],
) -> None:
    artifact = {str(row["path"]): row for row in artifact_rows}
    scratch = {str(row["path"]): row for row in scratch_rows}
    pairs = {
        "source_generation/arrays/frozen_source_streams.npy": (
            "arrays/frozen_source_streams.npy"
        ),
        "source_generation/manifests/frozen_source_stream_index.json": (
            "manifests/frozen_source_stream_index.json"
        ),
        "source_generation/manifests/frozen_source_stream_lock.json": (
            "manifests/frozen_source_stream_lock.json"
        ),
    }
    if any(
        (scratch[source]["size_bytes"], scratch[source]["sha256"])
        != (artifact[target]["size_bytes"], artifact[target]["sha256"])
        for source, target in pairs.items()
    ):
        raise ProtocolError("CBPUPR v2 preterminal scratch/source bytes differ.")


def assert_regular_tree(root: Path, *, role: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError(f"CBPUPR v2 preterminal {role} root is unsafe.")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ProtocolError(
                f"CBPUPR v2 preterminal {role} tree contains an unsafe member."
            )


def logical_path(path: Path, *, role: str) -> Path:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ProtocolError(f"CBPUPR v2 preterminal {role} path is a symlink.")
    parent = unresolved.absolute().parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError(f"CBPUPR v2 preterminal {role} parent is unsafe.")
    return parent.resolve() / unresolved.name


def is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


@contextmanager
def exclusive_existing_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("CBPUPR v2 preterminal run lock is absent or unsafe.")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProtocolError("CBPUPR v2 preterminal run lock is not regular.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            raise ProtocolError(
                "CBPUPR v2 diagnostic is active; preterminal archive is forbidden."
            ) from exc
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = (
    "audit_exact_tree",
    "exclusive_existing_run_lock",
    "is_present",
    "logical_path",
    "validate_scratch_duplicates",
)
