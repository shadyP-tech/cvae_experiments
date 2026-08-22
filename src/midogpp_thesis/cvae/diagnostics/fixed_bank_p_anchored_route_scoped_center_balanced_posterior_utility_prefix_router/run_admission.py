"""Fail-closed run locking and phase-state admission."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import stat
from typing import Iterator, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .bundle import assert_closed_world, relative_files
from .constants import SCRATCH_ROOT
from .hashing import canonical_hash
from .persistence import write_run_state


FAILED_WORKSTATION_PREFLIGHT_FILES = frozenset(
    {
        "config.resolved.yaml",
        "manifests/action_library.json",
        "manifests/policy_menu.json",
        "manifests/protocol_manifest.json",
        "provenance/input_artifacts.json",
        "reports/run_state.json",
    }
)


def assert_launch_files(root: Path, config: object) -> None:
    required = (root / "config.resolved.yaml", root / "provenance/input_artifacts.json")
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ProtocolError("CBPUPR launch files are absent or unsafe.")
    if Path(getattr(config, "source_path")).resolve() != required[0].resolve():
        raise ProtocolError("CBPUPR config is not its persisted snapshot.")


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
        raise ProtocolError("CBPUPR requires workspace-resolved paths.")


def reject_existing_run_state(root: Path) -> None:
    path = root / "reports/run_state.json"
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("CBPUPR run state is unsafe.")
    state = read_json(path)
    raise ProtocolError(
        "CBPUPR cross-run recovery is forbidden; "
        f"existing status={state.get('status')}, phase={state.get('phase')}."
    )


def audit_failed_workstation_preflight_for_quarantine(
    root: Path,
) -> Mapping[str, object]:
    """Certify only the zero-capability failed-preflight state for whole-root quarantine."""

    unresolved = Path(root)
    if unresolved.is_symlink():
        raise ProtocolError("CBPUPR failed-preflight root is a symlink.")
    resolved = unresolved.resolve()
    with _exclusive_existing_run_lock(resolved):
        return _audit_locked_failed_workstation_preflight(resolved)


def quarantine_failed_workstation_preflight(
    root: Path, *, destination: Path
) -> Mapping[str, object]:
    """Atomically preserve an audited failed root without permitting byte reuse."""

    unresolved = Path(root)
    requested_quarantine = Path(destination)
    if unresolved.is_symlink() or requested_quarantine.is_symlink():
        raise ProtocolError("CBPUPR failed-preflight quarantine path is a symlink.")
    resolved = unresolved.resolve()
    quarantine = requested_quarantine.resolve()
    expected_name = re.fullmatch(
        re.escape(resolved.name)
        + r"\.quarantine-failed-preflight-[0-9]{8}T[0-9]{6}Z",
        quarantine.name,
    )
    if (
        quarantine.parent != resolved.parent
        or expected_name is None
    ):
        raise ProtocolError("CBPUPR failed-preflight quarantine destination is unsafe.")
    with _exclusive_existing_run_lock(resolved):
        audit = _audit_locked_failed_workstation_preflight(resolved)
        if quarantine.exists() or quarantine.is_symlink():
            raise ProtocolError(
                "CBPUPR failed-preflight quarantine destination already exists."
            )
        os.rename(resolved, quarantine)
    if resolved.exists() or resolved.is_symlink() or not quarantine.is_dir():
        raise ProtocolError("CBPUPR failed-preflight quarantine move was not atomic.")
    payload: dict[str, object] = {
        "schema_version": "fixed_bank_cbpupr_failed_preflight_quarantine_receipt_v1",
        "status": "PASS",
        "source_root": str(resolved),
        "quarantine_root": str(quarantine),
        "source_root_absent_after_move": True,
        "whole_root_move_completed": True,
        "quarantined_bytes_may_feed_rerun": False,
        "fresh_workspace_prepare_required": True,
        "pre_move_audit": dict(audit),
    }
    return {**payload, "quarantine_receipt_hash": canonical_hash(payload)}


def _audit_locked_failed_workstation_preflight(
    resolved: Path,
) -> Mapping[str, object]:
    assert_closed_world(resolved, allow_incomplete=True)
    state = read_json(resolved / "reports/run_state.json")
    if (
        state.get("schema_version") != "fixed_bank_cbpupr_run_state_v1"
        or state.get("status") != "FAILED"
        or state.get("phase") != "WORKSTATION_PREFLIGHT"
        or state.get("error") != "Label-free workstation topology drifted."
        or state.get("error_class") != "ProtocolError"
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR failed-preflight quarantine state drifted.")
    observed = frozenset(relative_files(resolved))
    if observed != FAILED_WORKSTATION_PREFLIGHT_FILES:
        raise ProtocolError(
            "CBPUPR failed-preflight zero-capability inventory drifted."
        )
    scratch = Path(SCRATCH_ROOT)
    if scratch.exists() or scratch.is_symlink():
        raise ProtocolError("CBPUPR failed-preflight scratch is not absent.")
    payload: dict[str, object] = {
        "schema_version": "fixed_bank_cbpupr_failed_preflight_quarantine_audit_v1",
        "status": "PASS",
        "source_root": str(resolved),
        "source_run_status": "FAILED",
        "source_run_phase": "WORKSTATION_PREFLIGHT",
        "scratch_root": str(scratch),
        "scratch_absent": True,
        "label_capability_opened": False,
        "physical_generation_started": False,
        "cross_run_recovery_allowed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "fresh_workspace_prepare_required": True,
        "eligible_next_action": "MOVE_WHOLE_FAILED_ROOT_TO_QUARANTINE_ONLY",
        "members": [
            {
                "path": relative,
                "size_bytes": (resolved / relative).stat().st_size,
                "sha256": sha256_file(resolved / relative),
            }
            for relative in sorted(observed)
        ],
    }
    return {**payload, "quarantine_audit_hash": canonical_hash(payload)}


@contextmanager
def _exclusive_existing_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("CBPUPR failed-preflight run lock is absent or unsafe.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProtocolError("CBPUPR failed-preflight run lock is not regular.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(
                "CBPUPR diagnostic is active; failed-preflight quarantine is forbidden."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def assert_no_partial_state(root: Path) -> None:
    assert_closed_world(root, allow_incomplete=True)
    launch = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    foreign = sorted(set(relative_files(root)) - launch)
    if foreign:
        raise ProtocolError(
            f"CBPUPR partial/cross-run state is forbidden: {foreign}."
        )


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink():
        raise ProtocolError("CBPUPR run lock is a symlink.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("CBPUPR diagnostic is already running.") from exc
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
    "FAILED_WORKSTATION_PREFLIGHT_FILES",
    "audit_failed_workstation_preflight_for_quarantine",
    "assert_launch_files",
    "assert_no_partial_state",
    "assert_workspace_resolved_paths",
    "exclusive_run_lock",
    "quarantine_failed_workstation_preflight",
    "reject_existing_run_state",
    "write_state",
)
