"""Run locking, recovery, CUDA transition, and scoped scratch cleanup."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import (
    assert_closed_world,
    assert_terminal_phase_complete,
    write_content_index,
)
from .persistence import persist_validation_report, write_run_state


def recover_if_possible(
    root: Path, *, config: object, deps: object, protocol: object
) -> Path | None:
    state_path = root / "reports/run_state.json"
    if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
        assert_closed_world(root, allow_incomplete=False)
        enter_cuda_free_cpu_phase()
        _validator(deps)(root, config=config)
        cleanup_after_recovery_if_used(root, config=config, deps=deps)
        return root
    terminal = root / "manifests/sealed_terminal_evaluation.json"
    index = root / "manifests/content_index.json"
    if terminal.is_file() and not index.is_file():
        assert_terminal_phase_complete(root)
        write_state(
            deps,
            root,
            status="RUNNING",
            phase="TERMINAL_PHASE_VALIDATION_RECOVERY",
        )
        (getattr(deps, "write_index") or write_content_index)(
            root,
            config_contract_hash=str(getattr(config, "contract_hash")),
            protocol_contract_hash=str(getattr(protocol, "contract_hash")),
        )
        enter_cuda_free_cpu_phase()
        checks = _validator(deps)(root, config=config)
        (getattr(deps, "persist_validation") or persist_validation_report)(
            root, checks
        )
        write_state(deps, root, status="COMPLETE", phase="COMPLETE")
        _validator(deps)(root, config=config)
        cleanup_after_recovery_if_used(root, config=config, deps=deps)
        return root
    if index.is_file():
        assert_closed_world(
            root,
            allow_incomplete=False,
            allow_pending_validation=not (
                root / "reports/validation_report.json"
            ).is_file(),
        )
        write_state(
            deps,
            root,
            status="RUNNING",
            phase="CLOSED_WORLD_CONTENT_FIRST_VALIDATION_RECOVERY",
        )
        enter_cuda_free_cpu_phase()
        checks = _validator(deps)(root, config=config)
        (getattr(deps, "persist_validation") or persist_validation_report)(
            root, checks
        )
        write_state(deps, root, status="COMPLETE", phase="COMPLETE")
        _validator(deps)(root, config=config)
        cleanup_after_recovery_if_used(root, config=config, deps=deps)
        return root
    return None


def validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_fixed_bank_actionability_recoverability_bundle

    return validate_fixed_bank_actionability_recoverability_bundle(root, **kwargs)


def enter_cuda_free_cpu_phase() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def cleanup_validated_local_stage(
    config: object, *, canonical_source: object | None = None
) -> None:
    """Delete only the exact validated package-owned source staging directory."""

    from .execution_adapter import SCRATCH_ROOT, load_frozen_source_streams

    expected = Path(SCRATCH_ROOT).resolve() / "source_cache"
    scratch = tuple(
        str(value) for value in getattr(config, "runtime")["scratch_preference"]
    )
    if scratch != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("Refusing to clean a noncanonical actionability scratch path.")
    if not expected.exists():
        return
    canonical = canonical_source
    if canonical is None:
        canonical = load_frozen_source_streams(
            Path(getattr(config, "artifact_root")),
            expected_config_hash=str(getattr(config, "contract_hash")),
        )
    staged = load_frozen_source_streams(
        expected,
        expected_config_hash=str(getattr(config, "contract_hash")),
    )
    if dict(getattr(staged, "lock_payload")) != dict(
        getattr(canonical, "lock_payload")
    ):
        raise ProtocolError("Refusing to clean an unvalidated staged source cache.")
    shutil.rmtree(expected)


def cleanup_after_recovery_if_used(
    root: Path, *, config: object, deps: object
) -> None:
    summary_path = root / "reports/runtime_summary.json"
    if not summary_path.is_file():
        return
    staging = read_json(summary_path).get("local_source_staging")
    if isinstance(staging, Mapping) and staging.get("used") is True:
        (getattr(deps, "cleanup_staging") or cleanup_validated_local_stage)(config)


def assert_persisted_prelabel(
    root: Path, *, prediction: object, prelabel: object
) -> None:
    observed = read_json(root / "manifests/prelabel_feature_seal.json")
    if (
        observed.get("feature_surface_hash")
        != getattr(prelabel, "feature_surface_hash")
        or observed.get("permutation_provenance_hash")
        != getattr(prelabel, "permutation_provenance_hash")
        or read_json(root / "manifests/sealed_probability_surface.json").get(
            "global_prediction_seal_hash"
        )
        != getattr(prediction, "seal_hash")
    ):
        raise ProtocolError("Prelabel durable seal differs from in-memory products.")


def observe(deps: object, phase: str) -> None:
    callback = getattr(deps, "phase_observer")
    if callback is not None:
        callback(phase)


def write_state(
    deps: object,
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    (getattr(deps, "write_state") or write_run_state)(
        root, status=status, phase=phase, error=error
    )


@contextmanager
def exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(
                "Actionability/recoverability diagnostic is already running."
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": getattr(config, "artifact_root"),
        "expert-bank root": getattr(config, "expert_bank_root"),
        "generation-lock root": getattr(config, "generation_lock_root"),
        "test-cache root": getattr(config, "test_cache_root"),
        "test manifest": getattr(config, "test_manifest_path"),
        "test-consumption ledger": getattr(config, "test_consumption_ledger_path"),
        "ledger amendment": getattr(config, "ledger_amendment_path"),
    }
    unresolved = [role for role, value in paths.items() if not Path(value).is_absolute()]
    if unresolved or root.resolve() != Path(getattr(config, "artifact_root")).resolve():
        raise ProtocolError(
            "Actionability runner requires workspace-resolved paths; "
            f"unresolved={unresolved}."
        )


def assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Actionability launch files are absent: {missing}.")


def _validator(deps: object):
    return getattr(deps, "validate_bundle") or validate_bundle


__all__ = (
    "assert_launch_files",
    "assert_persisted_prelabel",
    "assert_workspace_resolved_paths",
    "cleanup_validated_local_stage",
    "enter_cuda_free_cpu_phase",
    "exclusive_run_lock",
    "observe",
    "recover_if_possible",
    "validate_bundle",
    "write_state",
)
