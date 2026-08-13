"""Run locking, launch guards, terminal recovery, and validator seams."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    assert_closed_world,
    cleanup_owned_atomic_temps,
    validate_content_index,
    write_content_index,
)
from .persistence import (
    TERMINAL_CHECKPOINT_MEMBER,
    finalize_terminal_checkpoint,
    persist_validation_report,
    remove_validated_terminal_checkpoint,
    write_run_state,
)
from .recovery import (
    MultiChallengerRecoveryCapability,
    failed_finalization_schema_state,
    recovery_capability,
)
from .finalization_provenance import (
    assert_finalization_repair_repository_state_unchanged,
    finalization_recovery_audit_payload_for_root,
)
from .recovery_provenance import current_repair_repository_state


def recover_if_possible(root: Path, *, config: object, protocol: object) -> Path | None:
    """Recover only registered exact boundaries or already complete bundles."""

    state_path = root / "reports/run_state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    checkpoint = root / TERMINAL_CHECKPOINT_MEMBER
    capability = recovery_capability(root)
    if (
        capability is not None
        and capability.mode == "FINALIZATION_VALIDATION"
    ):
        return _recover_exact_finalization(
            root,
            config=config,
            protocol=protocol,
            capability=capability,
        )
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


def _recover_exact_finalization(
    root: Path,
    *,
    config: object,
    protocol: object,
    capability: MultiChallengerRecoveryCapability,
) -> Path:
    """Validate the sealed bundle without re-entering a scientific phase."""

    if (
        capability.mode != "FINALIZATION_VALIDATION"
        or not capability.validation_only
        or capability.scientific_products_may_be_recomputed
        or capability.scientific_products_may_be_persisted
    ):
        raise ProtocolError(
            "Multi-challenger finalization recovery lacks validation-only authority."
        )
    failed_state = failed_finalization_schema_state(root)
    if read_json(root / "reports/run_state.json") != failed_state:
        raise ProtocolError(
            "Multi-challenger finalization state changed after capability admission."
        )
    content_before = _indexed_content_fingerprints(root)
    validation_report = root / "reports/validation_report.json"
    validation_report_before = (
        _file_fingerprint(validation_report)
        if validation_report.is_file() and not validation_report.is_symlink()
        else None
    )

    try:
        cleanup_owned_atomic_temps(root)
        validate_content_index(
            root,
            config_contract_hash=str(getattr(config, "contract_hash")),
            protocol_contract_hash=str(getattr(protocol, "contract_hash")),
        )
        _assert_indexed_content_unchanged(root, content_before)
        runtime_summary = read_json(root / "reports/runtime_summary.json")
        mappingproxy_audit = runtime_summary.get("mappingproxy_recovery")
        if not isinstance(mappingproxy_audit, Mapping):
            raise ProtocolError(
                "Multi-challenger finalization recovery lacks its B audit."
            )
        finalization_audit = finalization_recovery_audit_payload_for_root(
            root,
            failed_state=failed_state,
            mappingproxy_recovery_audit=mappingproxy_audit,
            current_repository_state=current_repair_repository_state(),
        )
        enter_cuda_free_cpu_phase()
        checks = validate_bundle(
            root,
            config=config,
            allow_pending_validation=True,
            finalization_recovery_audit=finalization_audit,
        )
        _assert_indexed_content_unchanged(root, content_before)
        persist_validation_report(root, checks)
        _assert_indexed_content_unchanged(root, content_before)
        write_state(root, status="COMPLETE", phase="COMPLETE")
        assert_completed_binding(root, config=config, expected_checks=checks)
        assert_finalization_repair_repository_state_unchanged(
            finalization_audit
        )
        _assert_indexed_content_unchanged(root, content_before)
        return root
    except BaseException as exc:
        write_state(
            root,
            status="FAILED",
            phase="FINALIZATION",
            error=str(failed_state["error"]),
            error_class="ProtocolError",
        )
        if validation_report_before is None:
            _remove_attempt_validation_report(validation_report)
            report_error = None
        elif _file_fingerprint(validation_report) != validation_report_before:
            report_error = ProtocolError(
                "Multi-challenger recovery changed a pre-existing validation report."
            )
        else:
            report_error = None
        _assert_indexed_content_unchanged(root, content_before)
        if report_error is not None:
            raise report_error from exc
        raise


def _indexed_content_fingerprints(
    root: Path,
) -> Mapping[str, tuple[int, str]]:
    """Capture immutable indexed members plus the index that binds them."""

    members = (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json")
    fingerprints: dict[str, tuple[int, str]] = {}
    for member in members:
        path = root / member
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(
                f"Multi-challenger finalization content member is unsafe: {member}."
            )
        fingerprints[member] = (path.stat().st_size, sha256_file(path))
    return fingerprints


def _assert_indexed_content_unchanged(
    root: Path, expected: Mapping[str, tuple[int, str]]
) -> None:
    observed = _indexed_content_fingerprints(root)
    if observed != dict(expected):
        raise ProtocolError(
            "Multi-challenger validation-only recovery changed indexed content."
        )


def _file_fingerprint(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(
            "Multi-challenger recovery report is absent or unsafe."
        )
    return path.stat().st_size, sha256_file(path)


def _remove_attempt_validation_report(path: Path) -> None:
    """Remove only this attempt's excluded validation product after failure."""

    if path.is_symlink():
        raise ProtocolError(
            "Multi-challenger recovery validation report became a symlink."
        )
    if path.exists():
        if not path.is_file():
            raise ProtocolError(
                "Multi-challenger recovery validation report became unsafe."
            )
        path.unlink()


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
