"""Fail-closed run locking and phase-state admission."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Iterator

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world, relative_files
from .persistence import write_run_state


def assert_launch_files(root: Path, config: object) -> None:
    required = (root / "config.resolved.yaml", root / "provenance/input_artifacts.json")
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ProtocolError("PSSCUR launch files are absent or unsafe.")
    if Path(getattr(config, "source_path")).resolve() != required[0].resolve():
        raise ProtocolError("PSSCUR config is not its persisted snapshot.")


def assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    values = (
        root,
        getattr(config, "artifact_root"),
        getattr(config, "expert_bank_root"),
        getattr(config, "generation_lock_root"),
        getattr(config, "test_cache_root"),
        getattr(config, "test_manifest_path"),
        getattr(config, "test_consumption_ledger_path"),
        getattr(config, "ledger_amendment_path"),
    )
    if any(not Path(value).is_absolute() for value in values) or root.resolve() != Path(
        getattr(config, "artifact_root")
    ).resolve():
        raise ProtocolError("PSSCUR requires workspace-resolved paths.")


def reject_existing_run_state(root: Path) -> None:
    path = root / "reports/run_state.json"
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("PSSCUR run state is unsafe.")
    state = read_json(path)
    raise ProtocolError(
        "PSSCUR cross-run recovery is forbidden; "
        f"existing status={state.get('status')}, phase={state.get('phase')}."
    )


def assert_no_partial_state(root: Path) -> None:
    assert_closed_world(root, allow_incomplete=True)
    launch = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    foreign = sorted(set(relative_files(root)) - launch)
    if foreign:
        raise ProtocolError(
            f"PSSCUR partial/cross-run state is forbidden: {foreign}."
        )


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink():
        raise ProtocolError("PSSCUR run lock is a symlink.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("PSSCUR diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


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


__all__ = (
    "assert_launch_files",
    "assert_no_partial_state",
    "assert_workspace_resolved_paths",
    "exclusive_run_lock",
    "reject_existing_run_state",
    "write_state",
)
