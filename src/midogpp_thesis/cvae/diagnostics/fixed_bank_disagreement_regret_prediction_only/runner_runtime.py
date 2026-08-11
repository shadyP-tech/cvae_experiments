"""Run lock, recovery, path validation, and CUDA-free CPU transition."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world
from .persistence import persist_validation_report, write_run_state


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


@contextmanager
def exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Prediction-only diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Prediction-only launch files are absent: {missing}.")


def assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    names = (
        "artifact_root",
        "expert_bank_root",
        "generation_lock_root",
        "train_cache_root",
        "test_cache_root",
        "test_consumption_ledger_path",
        "ledger_amendment_path",
    )
    unresolved = [name for name in names if not Path(getattr(config, name)).is_absolute()]
    resolved_root = root.resolve()
    expected_config = resolved_root / "config.resolved.yaml"
    source_path = Path(getattr(config, "source_path")).resolve()
    if (
        unresolved
        or resolved_root != Path(getattr(config, "artifact_root")).resolve()
        or source_path != expected_config
    ):
        raise ProtocolError(
            "Prediction-only runner requires workspace-resolved paths; "
            f"unresolved={unresolved}, config_snapshot_matches="
            f"{source_path == expected_config}."
        )


def observe(dependencies: object, phase: str) -> None:
    callback = getattr(dependencies, "phase_observer")
    if callback is not None:
        callback(phase)


def write_state(
    dependencies: object,
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    (getattr(dependencies, "write_state") or write_run_state)(
        root, status=status, phase=phase, error=error
    )


def recover_complete(root: Path, *, config: object, dependencies: object) -> Path | None:
    state = root / "reports/run_state.json"
    if not state.is_file() or read_json(state).get("status") != "COMPLETE":
        return None
    assert_closed_world(root, allow_incomplete=False)
    enter_cuda_free_cpu_phase()
    validator = getattr(dependencies, "validate_bundle") or _validator
    validator(root, config=config)
    return root


def _validator(root: Path, *, config: object):
    from .validation import (
        validate_fixed_bank_disagreement_regret_prediction_only_bundle,
    )

    return validate_fixed_bank_disagreement_regret_prediction_only_bundle(
        root, config=config
    )


__all__ = (
    "assert_launch_files",
    "assert_workspace_resolved_paths",
    "enter_cuda_free_cpu_phase",
    "exclusive_run_lock",
    "observe",
    "recover_complete",
    "write_state",
)
