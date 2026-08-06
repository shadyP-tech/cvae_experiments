"""Strict projection of the canonical manifest through the Stage-70 firewall."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

from .contracts import (
    CANONICAL_MANIFEST_SHA256,
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    EXPECTED_TEST_ROWS_BY_CENTER,
    ManifestAccessEvent,
    TargetEvaluationContractError,
    TargetEvaluationReservation,
    TargetEvaluationRow,
    evaluation_row_id,
    reservation_id,
    reservation_protocol_payload,
    semantic_sha256,
    validate_sha256,
)
from .validation import validate_target_evaluation_reservation


class AccessSink(Protocol):
    def __call__(self, event: ManifestAccessEvent, /) -> object: ...


AccessLog = AccessSink | list[ManifestAccessEvent] | None


def project_target_evaluation_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_rows_by_center: Mapping[str, int] = EXPECTED_TEST_ROWS_BY_CENTER,
    access_log: AccessLog = None,
    allow_test_fixture: bool = False,
) -> TargetEvaluationReservation:
    """Project one hash-bound manifest without observing outcome or path fields.

    ``allow_test_fixture`` exists only so focused tests can exercise the same
    firewall with a tiny manifest.  Such reservations are explicitly stamped
    ``test_fixture_only`` and cannot validate as canonical artifacts.
    """

    path = Path(manifest_path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_manifest_sha256:
        raise TargetEvaluationContractError(
            "Stage-70 scoring manifest SHA-256 drifted: "
            f"observed={actual_sha256}, expected={expected_manifest_sha256}."
        )
    validate_sha256(actual_sha256, role="manifest")
    if not allow_test_fixture:
        if actual_sha256 != CANONICAL_MANIFEST_SHA256:
            raise TargetEvaluationContractError(
                "Canonical Stage-70 projection requires the frozen manifest digest."
            )
        if dict(expected_rows_by_center) != dict(EXPECTED_TEST_ROWS_BY_CENTER):
            raise TargetEvaluationContractError(
                "Canonical Stage-70 projection requires exact nine-center coverage."
            )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "center", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise TargetEvaluationContractError(
                "Stage-70 manifest lacks a required projected field."
            )
        reservation = project_target_evaluation_rows(
            reader,
            manifest_sha256=actual_sha256,
            expected_rows_by_center=expected_rows_by_center,
            access_log=access_log,
            coverage_scope="test_fixture_only" if allow_test_fixture else "canonical",
        )
    validate_target_evaluation_reservation(
        reservation,
        expected_manifest_sha256=actual_sha256,
        expected_rows_by_center=expected_rows_by_center,
        allow_test_fixture=allow_test_fixture,
    )
    return reservation


def project_target_evaluation_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    manifest_sha256: str,
    expected_rows_by_center: Mapping[str, int],
    access_log: AccessLog = None,
    coverage_scope: str = "canonical",
) -> TargetEvaluationReservation:
    """Project mappings while touching only ``split``, ``center``, and ``case_id``.

    This lower-level surface is intentionally sentinel-friendly: callers may
    supply mappings whose ``__getitem__`` records every access, and an
    additional value-free access log may be supplied for protocol tests.
    """

    validate_sha256(manifest_sha256, role="manifest")
    expected = _normalized_counts(expected_rows_by_center)
    projected: list[TargetEvaluationRow] = []
    for contract_row_index, source in enumerate(rows):
        split = _projected_value(
            source,
            field="split",
            contract_row_index=contract_row_index,
            access_log=access_log,
        )
        center = _projected_value(
            source,
            field="center",
            contract_row_index=contract_row_index,
            access_log=access_log,
        )
        if split != EVALUATION_SPLIT or center not in expected:
            continue
        case_id = _projected_value(
            source,
            field="case_id",
            contract_row_index=contract_row_index,
            access_log=access_log,
        )
        if not case_id:
            raise TargetEvaluationContractError(
                f"Stage-70 projected case identity is empty at row {contract_row_index}."
            )
        projected.append(
            TargetEvaluationRow(
                evaluation_row_id=evaluation_row_id(
                    manifest_sha256,
                    contract_row_index,
                ),
                contract_row_index=contract_row_index,
                case_id=case_id,
                center=center,
                split=split,
            )
        )

    ordered_rows = tuple(projected)
    observed = {
        center: sum(row.center == center for row in ordered_rows) for center in expected
    }
    if observed != expected:
        raise TargetEvaluationContractError(
            "Stage-70 eligible test coverage drifted: "
            f"observed={observed}, expected={expected}."
        )
    protocol_payload = reservation_protocol_payload(
        manifest_sha256=manifest_sha256,
        expected_rows_by_center=expected,
        coverage_scope=coverage_scope,
    )
    protocol_hash = semantic_sha256(protocol_payload)
    return TargetEvaluationReservation(
        manifest_sha256=manifest_sha256,
        protocol_hash=protocol_hash,
        reservation_id=reservation_id(protocol_hash, ordered_rows),
        rows=ordered_rows,
        coverage_scope=coverage_scope,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TargetEvaluationContractError(
            f"Cannot read Stage-70 scoring manifest: {Path(path)}."
        ) from exc
    return digest.hexdigest()


def _projected_value(
    source: Mapping[str, object],
    *,
    field: str,
    contract_row_index: int,
    access_log: AccessLog,
) -> str:
    _record_access(
        access_log,
        ManifestAccessEvent(
            phase="reservation_projection",
            field=field,
            contract_row_index=contract_row_index,
        ),
    )
    try:
        value = source[field]
    except (KeyError, TypeError) as exc:
        raise TargetEvaluationContractError(
            f"Stage-70 manifest row {contract_row_index} lacks projected field {field!r}."
        ) from exc
    return str(value)


def _record_access(access_log: AccessLog, event: ManifestAccessEvent) -> None:
    if access_log is None:
        return
    if callable(access_log):
        access_log(event)
        return
    append = getattr(access_log, "append", None)
    if not callable(append):
        raise TypeError("Stage-70 manifest access log must be callable or appendable.")
    append(event)


def _normalized_counts(values: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise TargetEvaluationContractError(
            "Stage-70 expected center coverage must be a non-empty mapping."
        )
    normalized: dict[str, int] = {}
    for center, raw_count in values.items():
        rendered_center = str(center)
        if rendered_center not in ELIGIBLE_CENTERS or rendered_center in normalized:
            raise TargetEvaluationContractError(
                "Stage-70 expected center coverage contains an ineligible center."
            )
        if isinstance(raw_count, bool):
            raise TargetEvaluationContractError(
                "Stage-70 expected center row counts must be positive integers."
            )
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise TargetEvaluationContractError(
                "Stage-70 expected center row counts must be positive integers."
            ) from exc
        if count <= 0:
            raise TargetEvaluationContractError(
                "Stage-70 expected center row counts must be positive integers."
            )
        normalized[rendered_center] = count
    canonical_order = {
        center: normalized[center]
        for center in ELIGIBLE_CENTERS
        if center in normalized
    }
    if list(canonical_order) != list(normalized):
        raise TargetEvaluationContractError(
            "Stage-70 expected center coverage must use canonical center order."
        )
    return canonical_order


__all__ = (
    "AccessLog",
    "AccessSink",
    "file_sha256",
    "project_target_evaluation_manifest",
    "project_target_evaluation_rows",
)
