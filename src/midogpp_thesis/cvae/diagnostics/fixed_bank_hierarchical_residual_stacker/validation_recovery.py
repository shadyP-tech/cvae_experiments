"""Byte-preserving recovery for the terminal validation-only failure."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .artifact_io import read_json, sha256_file
from .bundle import CONTENT_INDEX_MEMBERS, assert_closed_world, validate_content_index
from .persistence import persist_validation_report, write_run_state
from .reports import run_state_payload


VALIDATION_PHASE = "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
RECOVERABLE_VALIDATOR_ERROR = (
    "ProtocolError: Residual-stacker bundle persisted per-case BACC."
)


def recover_fixed_bank_hierarchical_residual_stacker_validation(
    config: object,
    *,
    artifact_root: str | Path,
) -> Path:
    """Complete only the excluded validation controls of one exact failed bundle.

    This path intentionally has no dependency on the scientific runner or any
    persistence phase that owns indexed content.  It is valid only for the
    known validator-only failure after the content index was durably sealed.
    """

    root = Path(artifact_root)
    _assert_exact_resolved_config_binding(config, root=root)
    with _exclusive_recovery_lock(root):
        _assert_recoverable_state(root)
        assert_closed_world(
            root,
            allow_incomplete=False,
            allow_pending_validation=True,
        )

        # Validate the already sealed byte inventory before any semantic read.
        validate_content_index(
            root,
            config_contract_hash=str(getattr(config, "contract_hash")),
        )
        snapshot = _snapshot_indexed_bytes(root)
        validation_path = root / "reports/validation_report.json"
        recovery_started = False
        try:
            write_run_state(
                root,
                status="RUNNING",
                phase=VALIDATION_PHASE,
            )
            recovery_started = True
            checks = _validate_bundle(root, config=config)
            _assert_snapshot_unchanged(root, snapshot)

            persist_validation_report(root, checks)
            write_run_state(root, status="COMPLETE", phase="COMPLETE")
            repeated = _validate_bundle(root, config=config)
            if dict(repeated) != dict(checks):
                raise ProtocolError(
                    "Residual-stacker validation-only replay was not deterministic."
                )
            _assert_snapshot_unchanged(root, snapshot)
            return root
        except BaseException as exc:
            # The validation report was absent at admission.  Roll it back if a
            # later excluded-control check fails, leaving only FAILED run state.
            validation_path.unlink(missing_ok=True)
            if recovery_started:
                write_run_state(
                    root,
                    status="FAILED",
                    phase=VALIDATION_PHASE,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise


def _validate_bundle(root: Path, *, config: object) -> Mapping[str, object]:
    from .validation import (  # noqa: PLC0415
        validate_fixed_bank_hierarchical_residual_stacker_bundle,
    )

    return validate_fixed_bank_hierarchical_residual_stacker_bundle(
        root,
        config=config,
    )


def _assert_exact_resolved_config_binding(config: object, *, root: Path) -> None:
    if not root.is_absolute() or root != root.resolve():
        raise ProtocolError(
            "Residual-stacker validation recovery requires the exact absolute "
            "artifact root."
        )
    source = Path(getattr(config, "source_path", ""))
    configured_root = Path(getattr(config, "artifact_root", ""))
    expected_source = root / "config.resolved.yaml"
    if (
        source != expected_source
        or configured_root != root
        or source.resolve() != expected_source.resolve()
        or configured_root.resolve() != root.resolve()
    ):
        raise ProtocolError(
            "Residual-stacker validation recovery config/root binding drifted."
        )


def _assert_recoverable_state(root: Path) -> None:
    state_path = root / "reports/run_state.json"
    validation_path = root / "reports/validation_report.json"
    content_index_path = root / "manifests/content_index.json"
    expected_state = run_state_payload(
        "FAILED",
        VALIDATION_PHASE,
        error=RECOVERABLE_VALIDATOR_ERROR,
    )
    if not state_path.is_file() or read_json(state_path) != expected_state:
        raise ProtocolError(
            "Residual-stacker validation recovery requires the exact known "
            "FAILED validator state."
        )
    if validation_path.exists():
        raise ProtocolError(
            "Residual-stacker validation recovery requires an absent validation report."
        )
    if not content_index_path.is_file():
        raise ProtocolError(
            "Residual-stacker validation recovery requires the sealed content index."
        )


def _snapshot_indexed_bytes(root: Path) -> dict[str, tuple[int, str]]:
    members = ("manifests/content_index.json", *CONTENT_INDEX_MEMBERS)
    snapshot: dict[str, tuple[int, str]] = {}
    for member in members:
        path = root / member
        if not path.is_file():
            raise ProtocolError(
                f"Residual-stacker recovery snapshot member is absent: {member}."
            )
        snapshot[member] = (path.stat().st_size, sha256_file(path))
    return snapshot


def _assert_snapshot_unchanged(
    root: Path,
    expected: Mapping[str, tuple[int, str]],
) -> None:
    observed = _snapshot_indexed_bytes(root)
    if dict(observed) != dict(expected):
        raise ProtocolError(
            "Residual-stacker validation recovery changed indexed bytes."
        )


@contextmanager
def _exclusive_recovery_lock(root: Path):
    path = root / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError("Residual-stacker diagnostic is already running.") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


__all__ = (
    "RECOVERABLE_VALIDATOR_ERROR",
    "VALIDATION_PHASE",
    "recover_fixed_bank_hierarchical_residual_stacker_validation",
)
