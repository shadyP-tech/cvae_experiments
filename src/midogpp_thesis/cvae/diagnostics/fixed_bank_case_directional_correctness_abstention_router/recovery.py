"""Exact validation-only recovery for the observed CDCA finalization failure.

This module is deliberately not a general resume facility.  It recognizes one
exact failed state and either the original exact 42-file inventory or the exact
43-file atomic-report hard-crash boundary.  The resulting capability authorizes
only reconstructive validation and publication of the two content-index-
excluded validation/state products.  A pre-existing report is never rewritten
and is reused only after its full attestation matches a fresh parent replay.
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
from .hashing import canonical_json
from .reports import run_state_payload


RECOVERABLE_ERROR = (
    "Case-directional persisted table is not reconstructive: "
    "tables/route_model_fits.csv."
)
FAILED_FINALIZATION_STATE: dict[str, object] = run_state_payload(
    "FAILED",
    "CLOSED_WORLD_TWO_FRESH_PROCESS_VALIDATION",
    error=RECOVERABLE_ERROR,
    error_class="ProtocolError",
)
FINALIZATION_RECOVERABLE_INVENTORY = frozenset(REQUIRED_FILES) - {
    "reports/validation_report.json"
}
FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY = frozenset(REQUIRED_FILES)
FINALIZATION_RETRY_INVENTORIES = (
    FINALIZATION_RECOVERABLE_INVENTORY,
    FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY,
)
_STATE_MEMBER = "reports/run_state.json"
_VALIDATION_REPORT_MEMBER = "reports/validation_report.json"
_ATOMIC_REMNANT = re.compile(r".+\.[1-9][0-9]*\.tmp")
_ALLOWED_DIRECTORIES = frozenset(
    parent.as_posix()
    for member in FINALIZATION_RECOVERABLE_INVENTORY
    for parent in Path(member).parents
    if parent.as_posix() != "."
)


@dataclass(frozen=True)
class CaseDirectionalFinalizationRecoveryCapability:
    """Narrow authority granted by the exact failed state and inventory."""

    mode: str = "FINALIZATION_VALIDATION"
    validation_only: bool = True
    labels_may_be_reopened_for_validation: bool = True
    scientific_products_may_be_reconstructed_for_validation: bool = True
    scientific_products_may_be_persisted: bool = False
    terminal_products_may_be_persisted: bool = False
    policy_may_be_mutated: bool = False
    validation_report_present: bool = False

    def __post_init__(self) -> None:
        if (
            self.mode != "FINALIZATION_VALIDATION"
            or not self.validation_only
            or not self.labels_may_be_reopened_for_validation
            or not self.scientific_products_may_be_reconstructed_for_validation
            or self.scientific_products_may_be_persisted
            or self.terminal_products_may_be_persisted
            or self.policy_may_be_mutated
            or not isinstance(self.validation_report_present, bool)
        ):
            raise ProtocolError(
                "Case-directional recovery capability is not validation-only."
            )


def recovery_capability(
    root: Path,
) -> CaseDirectionalFinalizationRecoveryCapability | None:
    """Recognize only the exact observed workstation finalization boundary."""

    path = Path(root)
    if path.is_symlink():
        raise ProtocolError("Case-directional recovery root cannot be a symlink.")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ProtocolError("Case-directional recovery root is not a directory.")
    state_path = path / _STATE_MEMBER
    if state_path.is_symlink():
        raise ProtocolError("Case-directional recovery state cannot be a symlink.")
    if not state_path.exists():
        return None
    if not state_path.is_file():
        raise ProtocolError("Case-directional recovery state is unsafe.")
    state = _read_state(state_path)
    if state.get("status") in {"FAILED", "RUNNING"} and state != (
        FAILED_FINALIZATION_STATE
    ):
        raise ProtocolError(
            "Case-directional existing partial run is not the exact registered "
            "finalization recovery boundary."
        )
    if state != FAILED_FINALIZATION_STATE:
        return None
    observed = _exact_inventory(path)
    if observed not in FINALIZATION_RETRY_INVENTORIES:
        missing = sorted(FINALIZATION_RECOVERABLE_INVENTORY - observed)
        extras = sorted(observed - FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY)
        raise ProtocolError(
            "Case-directional finalization recovery inventory drifted: "
            f"missing={missing}, extras={extras}."
        )
    return CaseDirectionalFinalizationRecoveryCapability(
        validation_report_present=(
            observed == FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY
        )
    )


def detect_registered_case_directional_correctness_abstention_router_recovery(
    root: Path,
) -> bool:
    """Workspace-dispatch facade for the exact package-local capability."""

    return recovery_capability(Path(root)) is not None


def recover_exact_finalization(
    root: Path,
    *,
    config: object,
    protocol: object,
    capability: CaseDirectionalFinalizationRecoveryCapability,
) -> Path:
    """Run only the parent/two-child validation transaction and finalize it."""

    if capability != CaseDirectionalFinalizationRecoveryCapability(
        validation_report_present=capability.validation_report_present
    ):
        raise ProtocolError(
            "Case-directional finalization recovery lacks exact authority."
        )
    path = Path(root)
    if recovery_capability(path) != capability:
        raise ProtocolError(
            "Case-directional finalization recovery boundary changed after "
            "capability admission."
        )
    if read_json(path / _STATE_MEMBER) != FAILED_FINALIZATION_STATE:
        raise ProtocolError(
            "Case-directional failed state changed after recovery admission."
        )

    # The content index must be the first scientific object opened.  Only after
    # it authenticates all 40 indexed members may the recovery fingerprint and
    # subsequently reopen labels for read-only reconstruction.
    validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=str(getattr(protocol, "protocol_hash")),
    )
    immutable_before = _indexed_bytes(path)
    validation_report = path / _VALIDATION_REPORT_MEMBER
    validation_report_before = (
        _file_fingerprint(validation_report)
        if capability.validation_report_present
        else None
    )

    from .fresh_process_validation import require_two_fresh_process_validations
    from .persistence import persist_validation_report, write_run_state
    from .recovery_provenance import (
        assert_repair_repository_state_unchanged,
        current_repair_repository_state,
        finalization_recovery_audit_payload,
    )
    from .runner_runtime import enter_cuda_free_cpu_phase
    from .validation import (
        validate_fixed_bank_case_directional_correctness_abstention_router_bundle,
    )

    repair_state = current_repair_repository_state()
    audit = finalization_recovery_audit_payload(
        path, current_repository_state=repair_state
    )
    try:
        enter_cuda_free_cpu_phase()
        parent_checks = (
            validate_fixed_bank_case_directional_correctness_abstention_router_bundle(
                path,
                config=config,
                allow_pending_validation=True,
                finalization_recovery_audit=audit,
            )
        )
        _assert_indexed_bytes(path, immutable_before)
        assert_repair_repository_state_unchanged(repair_state)

        checks = (
            _validated_existing_validation_report(path, parent_checks)
            if validation_report_before is not None
            else require_two_fresh_process_validations(
                path, expected_checks=parent_checks
            )
        )
        if (
            validation_report_before is not None
            and _file_fingerprint(validation_report) != validation_report_before
        ):
            raise ProtocolError(
                "Case-directional recovery changed the pre-existing validation "
                "report."
            )
        _assert_indexed_bytes(path, immutable_before)
        assert_repair_repository_state_unchanged(repair_state)

        if validation_report_before is None:
            persist_validation_report(path, checks)
        _assert_indexed_bytes(path, immutable_before)
        assert_repair_repository_state_unchanged(repair_state)

        write_run_state(path, status="COMPLETE", phase="COMPLETE")
        _assert_completed_validation_binding(path, checks)
        _assert_indexed_bytes(path, immutable_before)
        if _exact_inventory(path) != FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY:
            raise ProtocolError(
                "Case-directional recovered final inventory is not exact."
            )
        assert_repair_repository_state_unchanged(repair_state)
        return path
    except BaseException as exc:
        rollback_error = _rollback_attempt(
            path, validation_report_before=validation_report_before
        )
        _assert_indexed_bytes(path, immutable_before)
        if read_json(path / _STATE_MEMBER) != FAILED_FINALIZATION_STATE:
            raise ProtocolError(
                "Case-directional recovery could not restore the exact failed state."
            ) from exc
        expected_inventory = (
            FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY
            if validation_report_before is not None
            else FINALIZATION_RECOVERABLE_INVENTORY
        )
        if _exact_inventory(path) != expected_inventory:
            raise ProtocolError(
                "Case-directional failed recovery inventory is not exact."
            ) from exc
        if rollback_error is not None:
            raise rollback_error from exc
        raise


def _assert_completed_validation_binding(
    root: Path, checks: Mapping[str, object]
) -> None:
    """Read back only the two excluded finalization products, not science."""

    from .fresh_process_validation import (
        ATTESTATION_KEY,
        verify_attested_validation_checks,
    )

    reconstructed = {
        key: value for key, value in checks.items() if key != ATTESTATION_KEY
    }
    verified = verify_attested_validation_checks(
        checks, expected_reconstructed_checks=reconstructed
    )
    expected_report = dict(verified)
    if "schema_version" in expected_report:
        raise ProtocolError(
            "Case-directional recovered validation checks contain a report header."
        )
    expected_report["schema_version"] = "fixed_bank_cdca_validation_report_v1"
    report_path = root / _VALIDATION_REPORT_MEMBER
    state_path = root / _STATE_MEMBER
    observed_report = read_json(report_path)
    observed_state = read_json(state_path)
    if (
        observed_report != expected_report
        or report_path.read_bytes() != canonical_json(observed_report) + b"\n"
        or observed_state != run_state_payload("COMPLETE", "COMPLETE")
        or state_path.read_bytes() != canonical_json(observed_state) + b"\n"
    ):
        raise ProtocolError(
            "Case-directional recovered validation/state binding is not exact."
        )


def _validated_existing_validation_report(
    root: Path, parent_checks: Mapping[str, object]
) -> Mapping[str, object]:
    """Authenticate an atomic hard-crash report before reusing its attestation."""

    from .fresh_process_validation import verify_attested_validation_checks

    report_path = root / _VALIDATION_REPORT_MEMBER
    if report_path.is_symlink() or not report_path.is_file():
        raise ProtocolError(
            "Case-directional retry validation report is absent or unsafe."
        )
    report = read_json(report_path)
    if report_path.read_bytes() != canonical_json(report) + b"\n":
        raise ProtocolError(
            "Case-directional retry validation report is not canonical."
        )
    if report.get("schema_version") != "fixed_bank_cdca_validation_report_v1":
        raise ProtocolError(
            "Case-directional retry validation report header drifted."
        )
    persisted_checks = {
        key: value for key, value in report.items() if key != "schema_version"
    }
    verified = verify_attested_validation_checks(
        persisted_checks,
        expected_reconstructed_checks=parent_checks,
    )
    if report != {
        "schema_version": "fixed_bank_cdca_validation_report_v1",
        **dict(verified),
    }:
        raise ProtocolError(
            "Case-directional retry validation report is not reconstructive."
        )
    return verified


def _rollback_attempt(
    root: Path,
    *,
    validation_report_before: tuple[int, str] | None,
) -> ProtocolError | None:
    """Remove only this attempt's excluded report and restore the exact state."""

    report = root / _VALIDATION_REPORT_MEMBER
    rollback_error: ProtocolError | None = None
    if report.is_symlink() or (report.exists() and not report.is_file()):
        rollback_error = ProtocolError(
            "Case-directional recovery validation report became unsafe."
        )
    elif validation_report_before is None and report.is_file():
        report.unlink()
    elif validation_report_before is not None:
        if not report.is_file():
            rollback_error = ProtocolError(
                "Case-directional recovery removed a pre-existing validation "
                "report."
            )
        elif _file_fingerprint(report) != validation_report_before:
            rollback_error = ProtocolError(
                "Case-directional recovery changed a pre-existing validation "
                "report."
            )
    from .persistence import write_run_state

    write_run_state(
        root,
        status="FAILED",
        phase="CLOSED_WORLD_TWO_FRESH_PROCESS_VALIDATION",
        error=RECOVERABLE_ERROR,
        error_class="ProtocolError",
    )
    return rollback_error


def _indexed_bytes(root: Path) -> dict[str, tuple[int, str]]:
    return {
        member: _file_fingerprint(root / member)
        for member in (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json")
    }


def _assert_indexed_bytes(
    root: Path, expected: Mapping[str, tuple[int, str]]
) -> None:
    if _indexed_bytes(root) != dict(expected):
        raise ProtocolError(
            "Case-directional validation-only recovery changed indexed bytes."
        )


def _file_fingerprint(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(
            f"Case-directional recovery member is absent or unsafe: {path}."
        )
    return path.stat().st_size, sha256_file(path)


def _read_state(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Case-directional recovery state is unreadable."
        ) from exc
    if not isinstance(value, Mapping):
        raise ProtocolError("Case-directional recovery state is malformed.")
    if path.read_bytes() != canonical_json(value) + b"\n":
        raise ProtocolError(
            "Case-directional recovery state bytes are not canonical."
        )
    return dict(value)


def _exact_inventory(root: Path) -> frozenset[str]:
    observed: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in (*names, *files):
            if (parent / name).is_symlink():
                raise ProtocolError(
                    "Case-directional recovery boundary contains a symlink."
                )
        for name in names:
            directory_path = parent / name
            relative = directory_path.relative_to(root).as_posix()
            if not directory_path.is_dir() or relative not in _ALLOWED_DIRECTORIES:
                raise ProtocolError(
                    "Case-directional recovery contains an extra directory: "
                    f"{relative}."
                )
        for name in files:
            candidate = parent / name
            relative = candidate.relative_to(root).as_posix()
            if relative == ".run.lock":
                continue
            if not candidate.is_file() or _ATOMIC_REMNANT.fullmatch(relative):
                raise ProtocolError(
                    "Case-directional recovery contains an unsafe member: "
                    f"{relative}."
                )
            observed.add(relative)
    return frozenset(observed)


__all__ = (
    "CaseDirectionalFinalizationRecoveryCapability",
    "FAILED_FINALIZATION_STATE",
    "FINALIZATION_RECOVERABLE_INVENTORY",
    "FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY",
    "FINALIZATION_RETRY_INVENTORIES",
    "RECOVERABLE_ERROR",
    "detect_registered_case_directional_correctness_abstention_router_recovery",
    "recover_exact_finalization",
    "recovery_capability",
)
