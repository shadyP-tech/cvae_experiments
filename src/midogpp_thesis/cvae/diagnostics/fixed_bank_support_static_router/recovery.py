"""Exact, validation-only recovery for the observed S4 finalization defect.

This module is deliberately not a general resume facility.  It recognizes one
failed validator state, one exact durable inventory, and grants authority only
to replay validation and publish the two excluded validation reports/state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .bundle import CONTENT_INDEX_MEMBERS, REQUIRED_FILES, validate_content_index


RECOVERABLE_ERROR = "S4 G_static seal is not reconstructive."
FAILED_FINALIZATION_STATE: dict[str, object] = {
    "schema_version": "fixed_bank_support_static_router_run_state_v1",
    "status": "FAILED",
    "phase": "FINALIZATION",
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_supported": False,
    "deterministic_restart_from_admission_requires_hash_validation": True,
    "terminal_checkpoint_recovery_supported": False,
    "terminal_checkpoint_is_atomicity_boundary_only": True,
    "error": RECOVERABLE_ERROR,
    "error_class": "ProtocolError",
}
FINALIZATION_RECOVERABLE_INVENTORY = frozenset(REQUIRED_FILES) - {
    "reports/fresh_process_validation.json",
    "reports/validation_report.json",
}
_STATE_MEMBER = "reports/run_state.json"
_ATOMIC_REMNANT = re.compile(r".+\.[1-9][0-9]*\.tmp")
_ALLOWED_DIRECTORIES = frozenset(
    parent.as_posix()
    for member in FINALIZATION_RECOVERABLE_INVENTORY
    for parent in Path(member).parents
    if parent.as_posix() != "."
)


@dataclass(frozen=True)
class SupportStaticRouterRecoveryCapability:
    """Narrow authority produced by the exact failed state and inventory."""

    mode: str = "FINALIZATION_VALIDATION"
    validation_only: bool = True
    labels_may_be_reopened_for_validation: bool = True
    scientific_products_may_be_recomputed: bool = False
    scientific_products_may_be_persisted: bool = False

    def __post_init__(self) -> None:
        if (
            self.mode != "FINALIZATION_VALIDATION"
            or not self.validation_only
            or not self.labels_may_be_reopened_for_validation
            or self.scientific_products_may_be_recomputed
            or self.scientific_products_may_be_persisted
        ):
            raise ProtocolError("S4 recovery capability is not validation-only.")


def recovery_capability(root: Path) -> SupportStaticRouterRecoveryCapability | None:
    """Recognize only the exact workstation failure without changing the root."""

    path = Path(root)
    if path.is_symlink():
        raise ProtocolError("S4 recovery root cannot be a symlink.")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ProtocolError("S4 recovery root is not a directory.")
    state_path = path / _STATE_MEMBER
    if state_path.is_symlink():
        raise ProtocolError("S4 recovery state cannot be a symlink.")
    if not state_path.exists():
        return None
    if not state_path.is_file():
        raise ProtocolError("S4 recovery state is unsafe.")
    state = _read_state(state_path)
    if (
        state.get("status") in {"FAILED", "RUNNING"}
        and state != FAILED_FINALIZATION_STATE
    ):
        raise ProtocolError(
            "S4 existing partial run is not an exact recovery boundary."
        )
    if state != FAILED_FINALIZATION_STATE:
        return None
    observed = _exact_inventory(path)
    if observed != FINALIZATION_RECOVERABLE_INVENTORY:
        missing = sorted(FINALIZATION_RECOVERABLE_INVENTORY - observed)
        extras = sorted(observed - FINALIZATION_RECOVERABLE_INVENTORY)
        raise ProtocolError(
            "S4 finalization recovery inventory drifted: "
            f"missing={missing}, extras={extras}."
        )
    return SupportStaticRouterRecoveryCapability()


def detect_registered_support_static_router_recovery(root: Path) -> bool:
    """Workspace-dispatch facade for the exact package-local capability."""

    return recovery_capability(Path(root)) is not None


def recover_exact_finalization(
    root: Path,
    *,
    config: object,
    protocol: object,
    capability: SupportStaticRouterRecoveryCapability,
) -> Path:
    """Replay only parent/fresh validation and publish excluded reports/state."""

    if capability != SupportStaticRouterRecoveryCapability():
        raise ProtocolError("S4 finalization recovery lacks exact authority.")
    path = Path(root)
    if read_json(path / _STATE_MEMBER) != FAILED_FINALIZATION_STATE:
        raise ProtocolError("S4 failed state changed after recovery admission.")

    # The index is the first scientific object validated.  Only after it binds
    # every member do we take the byte snapshot used as the recovery invariant.
    validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=str(getattr(protocol, "contract_hash")),
    )
    immutable_before = _indexed_bytes(path)

    from .fresh_process_validation import run_two_fresh_process_replays
    from .persistence import (
        persist_fresh_process_report,
        persist_validation_report,
        write_run_state,
    )
    from .recovery_provenance import (
        assert_repair_repository_state_unchanged,
        current_repair_repository_state,
        finalization_recovery_audit_payload,
    )
    from .validation import (
        assert_completed_bundle_binding,
        validate_fixed_bank_support_static_router_bundle,
    )

    repair_state = current_repair_repository_state()
    # Fail before any label-bearing scientific replay unless the repair is a
    # clean, changed checkout whose exact state and indexed input surface can
    # be bound into the finalization audit.
    finalization_recovery_audit_payload(
        path, current_repository_state=repair_state
    )
    try:
        parent_checks = validate_fixed_bank_support_static_router_bundle(
            path,
            config=config,
            allow_pending_validation=True,
            skip_fresh_process_report=True,
        )
        _assert_indexed_bytes(path, immutable_before)
        fresh = run_two_fresh_process_replays(
            path, config_path=Path(getattr(config, "source_path"))
        )
        if fresh.get("validation_result") != dict(parent_checks):
            raise ProtocolError("S4 recovery fresh validation disagreed with parent.")
        persist_fresh_process_report(path, fresh)
        _assert_indexed_bytes(path, immutable_before)
        checks = validate_fixed_bank_support_static_router_bundle(
            path,
            config=config,
            allow_pending_validation=True,
        )
        persist_validation_report(path, checks)
        _assert_indexed_bytes(path, immutable_before)
        write_run_state(path, status="COMPLETE", phase="COMPLETE")
        assert_completed_bundle_binding(path, config=config, expected_checks=checks)
        assert_repair_repository_state_unchanged(repair_state)
        _assert_indexed_bytes(path, immutable_before)
        return path
    except BaseException as exc:
        rollback_error = _rollback_attempt(path)
        _assert_indexed_bytes(path, immutable_before)
        if read_json(path / _STATE_MEMBER) != FAILED_FINALIZATION_STATE:
            raise ProtocolError(
                "S4 recovery could not restore the failed state."
            ) from exc
        if rollback_error is not None:
            raise rollback_error from exc
        raise


def _rollback_attempt(root: Path) -> ProtocolError | None:
    error: ProtocolError | None = None
    for member in (
        "reports/fresh_process_validation.json",
        "reports/validation_report.json",
    ):
        path = root / member
        if path.is_symlink() or (path.exists() and not path.is_file()):
            error = ProtocolError("S4 recovery report became unsafe during rollback.")
            continue
        if path.is_file():
            path.unlink()
    from .persistence import write_run_state

    write_run_state(
        root,
        status="FAILED",
        phase="FINALIZATION",
        error=RECOVERABLE_ERROR,
        error_class="ProtocolError",
    )
    return error


def _indexed_bytes(root: Path) -> dict[str, tuple[int, str]]:
    return {
        member: _file_fingerprint(root / member)
        for member in (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json")
    }


def _assert_indexed_bytes(
    root: Path, expected: Mapping[str, tuple[int, str]]
) -> None:
    if _indexed_bytes(root) != dict(expected):
        raise ProtocolError("S4 validation-only recovery changed indexed bytes.")


def _file_fingerprint(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"S4 recovery member is absent or unsafe: {path}.")
    return path.stat().st_size, sha256_file(path)


def _read_state(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("S4 recovery state is unreadable.") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError("S4 recovery state is malformed.")
    return dict(value)


def _exact_inventory(root: Path) -> frozenset[str]:
    observed: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in (*names, *files):
            if (parent / name).is_symlink():
                raise ProtocolError("S4 recovery boundary contains a symlink.")
        for name in names:
            directory_path = parent / name
            relative = directory_path.relative_to(root).as_posix()
            if not directory_path.is_dir() or relative not in _ALLOWED_DIRECTORIES:
                raise ProtocolError(
                    f"S4 recovery contains an extra directory: {relative}."
                )
        for name in files:
            candidate = parent / name
            relative = candidate.relative_to(root).as_posix()
            if relative == ".run.lock":
                continue
            if not candidate.is_file() or _ATOMIC_REMNANT.fullmatch(relative):
                raise ProtocolError(
                    f"S4 recovery contains an unsafe member: {relative}."
                )
            observed.add(relative)
    return frozenset(observed)


__all__ = (
    "FAILED_FINALIZATION_STATE",
    "FINALIZATION_RECOVERABLE_INVENTORY",
    "RECOVERABLE_ERROR",
    "SupportStaticRouterRecoveryCapability",
    "detect_registered_support_static_router_recovery",
    "recover_exact_finalization",
    "recovery_capability",
)
