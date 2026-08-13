"""Run lock, phase state, and launch guards for deterministic S4 restart."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .persistence import write_run_state


def assert_launch_files(root: Path, config: object) -> None:
    required = (
        root / "config.resolved.yaml",
        root / "provenance/input_artifacts.json",
    )
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ProtocolError("S4 launch files are absent or unsafe.")
    if Path(getattr(config, "source_path")).resolve() != required[0].resolve():
        raise ProtocolError("S4 config is not bound to the run snapshot.")


def assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    paths = (
        root,
        getattr(config, "artifact_root"),
        getattr(config, "expert_bank_root"),
        getattr(config, "generation_lock_root"),
        getattr(config, "test_cache_root"),
        getattr(config, "test_manifest_path"),
        getattr(config, "test_consumption_ledger_path"),
        getattr(config, "ledger_amendment_path"),
    )
    if any(not Path(value).is_absolute() for value in paths):
        raise ProtocolError("S4 requires workspace-resolved absolute paths.")
    if root.resolve() != Path(getattr(config, "artifact_root")).resolve():
        raise ProtocolError("S4 artifact root differs from the resolved config.")


def observe(deps: object, phase: str) -> None:
    callback = getattr(deps, "phase_observer")
    if callback is not None:
        callback(phase)


def write_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    write_run_state(
        root,
        status=status,
        phase=phase,
        error=error,
        error_class=error_class,
    )


@contextmanager
def exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    if path.is_symlink():
        raise ProtocolError("S4 run lock is a symlink.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("S4 diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = (
    "assert_launch_files",
    "assert_workspace_resolved_paths",
    "exclusive_run_lock",
    "observe",
    "write_state",
)
