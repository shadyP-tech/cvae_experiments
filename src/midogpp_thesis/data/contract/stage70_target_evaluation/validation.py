"""Independent validation for Stage-70 target-evaluation reservations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    CANONICAL_MANIFEST_SHA256,
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FORBIDDEN_IDENTITY_FIELD_NAMES,
    FRESH_EVIDENCE,
    LEGACY_OUTCOME_PATTERN,
    PURPOSE,
    RESERVATION_PROTOCOL_ID,
    RESERVATION_ROW_FIELDS,
    RESERVATION_SCHEMA_VERSION,
    TargetEvaluationContractError,
    TargetEvaluationReservation,
    evaluation_row_id,
    reservation_id,
    reservation_protocol_payload,
    semantic_sha256,
)


def validate_target_evaluation_reservation(
    reservation: TargetEvaluationReservation,
    *,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_rows_by_center: Mapping[str, int] = EXPECTED_TEST_ROWS_BY_CENTER,
    allow_test_fixture: bool = False,
) -> dict[str, object]:
    """Recompute every identity and reject any field outside the firewall."""

    expected_counts = {str(key): int(value) for key, value in expected_rows_by_center.items()}
    expected_scope = "test_fixture_only" if allow_test_fixture else "canonical"
    if (
        reservation.schema_version != RESERVATION_SCHEMA_VERSION
        or reservation.protocol_id != RESERVATION_PROTOCOL_ID
        or reservation.authorized_consumer_experiment_id
        != AUTHORIZED_CONSUMER_EXPERIMENT_ID
        or reservation.purpose != PURPOSE
        or reservation.fresh_evidence is not FRESH_EVIDENCE
        or reservation.coverage_scope != expected_scope
        or reservation.manifest_sha256 != expected_manifest_sha256
    ):
        raise TargetEvaluationContractError(
            "Stage-70 reservation protocol identity drifted."
        )
    if not allow_test_fixture and (
        expected_manifest_sha256 != CANONICAL_MANIFEST_SHA256
        or expected_counts != dict(EXPECTED_TEST_ROWS_BY_CENTER)
    ):
        raise TargetEvaluationContractError(
            "Canonical Stage-70 reservation validation requires exact frozen coverage."
        )
    if tuple(expected_counts) != tuple(
        center for center in ELIGIBLE_CENTERS if center in expected_counts
    ):
        raise TargetEvaluationContractError(
            "Stage-70 reservation center order drifted."
        )
    if not expected_counts or any(
        center not in ELIGIBLE_CENTERS or count <= 0
        for center, count in expected_counts.items()
    ):
        raise TargetEvaluationContractError(
            "Stage-70 reservation center coverage is invalid."
        )

    rows = reservation.rows
    indices = tuple(row.contract_row_index for row in rows)
    row_ids = tuple(row.evaluation_row_id for row in rows)
    if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
        raise TargetEvaluationContractError(
            "Stage-70 reservation rows are not unique canonical-manifest order."
        )
    if len(row_ids) != len(set(row_ids)):
        raise TargetEvaluationContractError(
            "Stage-70 reservation row identities are duplicated."
        )
    for row in rows:
        if row.evaluation_row_id != evaluation_row_id(
            reservation.manifest_sha256,
            row.contract_row_index,
        ):
            raise TargetEvaluationContractError(
                "Stage-70 evaluation row identity was not derived from manifest "
                "digest and row index."
            )
        if row.split != EVALUATION_SPLIT or row.center not in expected_counts:
            raise TargetEvaluationContractError(
                "Stage-70 reservation contains an unauthorized split or center."
            )
        if set(row.to_dict()) != RESERVATION_ROW_FIELDS:
            raise TargetEvaluationContractError(
                "Stage-70 reservation row schema crossed the identity firewall."
            )

    observed_counts = {
        center: sum(row.center == center for row in rows) for center in expected_counts
    }
    if observed_counts != expected_counts or len(rows) != sum(expected_counts.values()):
        raise TargetEvaluationContractError(
            "Stage-70 reservation row coverage drifted."
        )
    protocol_payload = reservation_protocol_payload(
        manifest_sha256=reservation.manifest_sha256,
        expected_rows_by_center=expected_counts,
        coverage_scope=expected_scope,
    )
    expected_protocol_hash = semantic_sha256(protocol_payload)
    if reservation.protocol_hash != expected_protocol_hash:
        raise TargetEvaluationContractError(
            "Stage-70 reservation protocol hash drifted."
        )
    if reservation.reservation_id != reservation_id(expected_protocol_hash, rows):
        raise TargetEvaluationContractError(
            "Stage-70 reservation identity drifted."
        )
    scan_reservation_firewall(reservation.to_dict())
    return {
        "status": "PASS",
        "manifest_sha256": reservation.manifest_sha256,
        "target_evaluation_reservation_id": reservation.reservation_id,
        "target_evaluation_reservation_protocol_hash": reservation.protocol_hash,
        "row_count": len(rows),
        "rows_by_center": observed_counts,
        "row_order_hash": reservation.row_order_hash,
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
    }


def validate_target_evaluation_reservation_against_manifest(
    manifest_path: str | Path,
    reservation: TargetEvaluationReservation,
    *,
    access_log: object = None,
    expected_rows_by_center: Mapping[str, int] = EXPECTED_TEST_ROWS_BY_CENTER,
    allow_test_fixture: bool = False,
) -> dict[str, object]:
    """Independently re-project the hash-bound manifest and compare all rows."""

    # Local import prevents the projector's build-time validation dependency
    # from turning into a module import cycle.
    from .projector import project_target_evaluation_manifest

    reconstructed = project_target_evaluation_manifest(
        manifest_path,
        expected_manifest_sha256=reservation.manifest_sha256,
        expected_rows_by_center=expected_rows_by_center,
        access_log=access_log,  # type: ignore[arg-type]
        allow_test_fixture=allow_test_fixture,
    )
    if reconstructed != reservation:
        raise TargetEvaluationContractError(
            "Stage-70 reservation does not match an independent manifest projection."
        )
    return validate_target_evaluation_reservation(
        reservation,
        expected_manifest_sha256=reservation.manifest_sha256,
        expected_rows_by_center=expected_rows_by_center,
        allow_test_fixture=allow_test_fixture,
    )


def scan_reservation_firewall(payload: object) -> None:
    """Reject forbidden identity keys and legacy outcome-bearing strings."""

    def visit(value: object, *, location: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key)
                normalized = key.casefold()
                if normalized in FORBIDDEN_IDENTITY_FIELD_NAMES:
                    raise TargetEvaluationContractError(
                        f"Stage-70 reservation contains forbidden identity field at {location}."
                    )
                visit(nested, location=f"{location}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, location=f"{location}[{index}]")
            return
        if isinstance(value, str):
            lowered = value.casefold()
            if LEGACY_OUTCOME_PATTERN.search(lowered):
                raise TargetEvaluationContractError(
                    f"Stage-70 reservation contains a legacy outcome encoding at {location}."
                )

    visit(payload, location="reservation")


def assert_serialized_reservation_is_sealed(path: str | Path) -> None:
    """Read a serialized reservation and apply the independent content scan."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetEvaluationContractError(
            f"Cannot scan Stage-70 reservation JSON: {Path(path)}."
        ) from exc
    scan_reservation_firewall(payload)


__all__ = (
    "assert_serialized_reservation_is_sealed",
    "scan_reservation_firewall",
    "validate_target_evaluation_reservation",
    "validate_target_evaluation_reservation_against_manifest",
)
