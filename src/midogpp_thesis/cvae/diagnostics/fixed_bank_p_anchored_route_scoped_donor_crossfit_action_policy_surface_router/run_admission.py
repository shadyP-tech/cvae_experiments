"""Filesystem lifecycle for an independently authorized P-DCAPS run."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json


RUN_STATE_MEMBER = "reports/run_state.json"
LOCK_SUFFIX = ".pdcaps-run.lock"


def assert_absent_run_root(root: Path) -> None:
    target = Path(root)
    if target.exists() or target.is_symlink():
        raise ProtocolError("P-DCAPS requires an absent output root; recovery is forbidden.")


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    target = Path(root)
    lock = target.parent / f".{target.name}{LOCK_SUFFIX}"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError("Another P-DCAPS run owns the output identity.") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def initialize_run_root(root: Path) -> None:
    target = Path(root)
    target.mkdir(parents=False, exist_ok=False)
    for member in ("arrays", "manifests", "provenance", "reports", "tables"):
        (target / member).mkdir(exist_ok=False)


def write_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    if status not in {"RUNNING", "FAILED", "COMPLETE"}:
        raise ProtocolError("P-DCAPS run-state status drifted.")
    payload = {
        "schema_version": "pdcaps_run_state_v1",
        "status": status,
        "phase": str(phase),
        "recovery_used": False,
        "cross_run_reuse_used": False,
        "error": error,
        "error_class": error_class,
    }
    atomic_json(Path(root) / RUN_STATE_MEMBER, payload)


def require_complete_state(root: Path) -> dict[str, object]:
    payload = read_json(Path(root) / RUN_STATE_MEMBER)
    if payload.get("status") != "COMPLETE" or payload.get("phase") != "COMPLETE":
        raise ProtocolError("P-DCAPS run is not complete.")
    return payload


__all__ = (
    "RUN_STATE_MEMBER",
    "assert_absent_run_root",
    "exclusive_run_lock",
    "initialize_run_root",
    "require_complete_state",
    "write_state",
)
