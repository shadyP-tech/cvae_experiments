"""Run locking, launch guards, terminal recovery, and validator seams."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    assert_closed_world,
    cleanup_owned_atomic_temps,
    write_content_index,
)
from .persistence import (
    TERMINAL_CHECKPOINT_MEMBER,
    finalize_terminal_checkpoint,
    persist_validation_report,
    remove_validated_terminal_checkpoint,
    write_run_state,
)


def recover_if_possible(root: Path, *, config: object, protocol: object) -> Path | None:
    """Recover only complete bundles and the one durable terminal checkpoint."""

    state_path = root / "reports/run_state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    checkpoint = root / TERMINAL_CHECKPOINT_MEMBER
    cleanup_owned_atomic_temps(root)
    if state.get("status") == "COMPLETE":
        assert_closed_world(root, allow_incomplete=False)
        enter_cuda_free_cpu_phase()
        validate_bundle(root, config=config)
        return root
    if checkpoint.is_file():
        finalize_terminal_checkpoint(root)
        remove_validated_terminal_checkpoint(root)
    if all((root / member).is_file() for member in CONTENT_INDEX_MEMBERS):
        write_state(root, status="RUNNING", phase="FINALIZATION")
        write_content_index(
            root,
            config_contract_hash=str(getattr(config, "contract_hash")),
            protocol_contract_hash=str(getattr(protocol, "contract_hash")),
        )
        enter_cuda_free_cpu_phase()
        checks = validate_bundle(root, config=config, allow_pending_validation=True)
        persist_validation_report(root, checks)
        write_state(root, status="COMPLETE", phase="COMPLETE")
        assert_completed_binding(root, config=config, expected_checks=checks)
        return root
    return None


def enter_cuda_free_cpu_phase() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Process workers set the frozen three-thread allocation.  The orchestrator
    # itself remains single-threaded to avoid nested BLAS oversubscription.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def assert_launch_files(root: Path, config: object) -> None:
    required = (
        root / "config.resolved.yaml",
        root / "provenance/input_artifacts.json",
    )
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ProtocolError("Multi-challenger launch files are absent or unsafe.")
    if Path(getattr(config, "source_path")).resolve() != required[0].resolve():
        raise ProtocolError("Multi-challenger config is not bound to its snapshot.")


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
    if (
        any(not Path(value).is_absolute() for value in paths)
        or root.resolve() != Path(getattr(config, "artifact_root")).resolve()
    ):
        raise ProtocolError("Multi-challenger requires workspace-resolved paths.")


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
        root, status=status, phase=phase, error=error, error_class=error_class
    )


@contextmanager
def exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    if path.is_symlink():
        raise ProtocolError("Multi-challenger run lock is a symlink.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Multi-challenger diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import (
        validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle,
    )

    checks = validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle(
        root, **kwargs
    )
    if kwargs.get("allow_pending_validation") is True:
        from .fresh_process_validation import (
            require_two_fresh_process_validations,
        )

        return require_two_fresh_process_validations(
            root,
            expected_checks=checks,
        )
    return checks


def assert_completed_binding(root: Path, **kwargs: object) -> None:
    from .validation import assert_completed_bundle_binding

    assert_completed_bundle_binding(root, **kwargs)


__all__ = (
    "assert_completed_binding",
    "assert_launch_files",
    "assert_workspace_resolved_paths",
    "enter_cuda_free_cpu_phase",
    "exclusive_run_lock",
    "observe",
    "recover_if_possible",
    "validate_bundle",
    "write_state",
)
