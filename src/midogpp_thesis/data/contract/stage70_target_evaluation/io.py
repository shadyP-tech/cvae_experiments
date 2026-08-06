"""Strict JSON I/O for the label-sealed Stage-70 reservation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

from .contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    CANONICAL_MANIFEST_SHA256,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FRESH_EVIDENCE,
    PURPOSE,
    RESERVATION_PROTOCOL_ID,
    RESERVATION_ROW_FIELDS,
    RESERVATION_SCHEMA_VERSION,
    TargetEvaluationContractError,
    TargetEvaluationReservation,
    TargetEvaluationRow,
)
from .validation import (
    assert_serialized_reservation_is_sealed,
    scan_reservation_firewall,
    validate_target_evaluation_reservation,
)


RESERVATION_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "authorized_consumer_experiment_id",
        "purpose",
        "fresh_evidence",
        "coverage_scope",
        "manifest_sha256",
        "protocol_hash",
        "reservation_id",
        "row_count",
        "rows_by_center",
        "row_order_hash",
        "rows",
    }
)


def write_target_evaluation_reservation(
    reservation: TargetEvaluationReservation,
    path: str | Path,
    *,
    allow_test_fixture: bool = False,
    expected_rows_by_center: Mapping[str, int] = EXPECTED_TEST_ROWS_BY_CENTER,
) -> Path:
    """Atomically write a validated reservation without extra identity fields."""

    validate_target_evaluation_reservation(
        reservation,
        expected_manifest_sha256=reservation.manifest_sha256,
        expected_rows_by_center=expected_rows_by_center,
        allow_test_fixture=allow_test_fixture,
    )
    payload = reservation.to_dict()
    scan_reservation_firewall(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite immutable Stage-70 reservation: {output}."
        )
    temporary = output.with_name(f".{output.name}.staging-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert_serialized_reservation_is_sealed(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_target_evaluation_reservation(
    path: str | Path,
    *,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_rows_by_center: Mapping[str, int] = EXPECTED_TEST_ROWS_BY_CENTER,
    allow_test_fixture: bool = False,
) -> TargetEvaluationReservation:
    """Load a reservation only after exact-schema and identity validation."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise TargetEvaluationContractError(
            f"Stage-70 reservation is missing or unsafe: {source}."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetEvaluationContractError(
            f"Stage-70 reservation JSON is unreadable: {source}."
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != RESERVATION_DOCUMENT_FIELDS:
        raise TargetEvaluationContractError(
            "Stage-70 reservation document schema drifted."
        )
    scan_reservation_firewall(payload)
    rows_payload = payload.get("rows")
    if isinstance(rows_payload, (str, bytes)) or not isinstance(rows_payload, Sequence):
        raise TargetEvaluationContractError(
            "Stage-70 reservation rows must be a sequence."
        )
    rows: list[TargetEvaluationRow] = []
    for raw in rows_payload:
        if not isinstance(raw, Mapping) or set(raw) != RESERVATION_ROW_FIELDS:
            raise TargetEvaluationContractError(
                "Stage-70 reservation row schema drifted."
            )
        try:
            row_index = int(raw["contract_row_index"])
        except (TypeError, ValueError) as exc:
            raise TargetEvaluationContractError(
                "Stage-70 reservation row index is invalid."
            ) from exc
        if isinstance(raw["contract_row_index"], bool):
            raise TargetEvaluationContractError(
                "Stage-70 reservation row index is invalid."
            )
        rows.append(
            TargetEvaluationRow(
                evaluation_row_id=str(raw["evaluation_row_id"]),
                contract_row_index=row_index,
                case_id=str(raw["case_id"]),
                center=str(raw["center"]),
                split=str(raw["split"]),
            )
        )
    raw_fresh_evidence = payload["fresh_evidence"]
    if not isinstance(raw_fresh_evidence, bool):
        raise TargetEvaluationContractError(
            "Stage-70 reservation freshness flag must be boolean."
        )
    reservation = TargetEvaluationReservation(
        schema_version=str(payload["schema_version"]),
        protocol_id=str(payload["protocol_id"]),
        authorized_consumer_experiment_id=str(
            payload["authorized_consumer_experiment_id"]
        ),
        purpose=str(payload["purpose"]),
        fresh_evidence=raw_fresh_evidence,
        coverage_scope=str(payload["coverage_scope"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        protocol_hash=str(payload["protocol_hash"]),
        reservation_id=str(payload["reservation_id"]),
        rows=tuple(rows),
    )
    if (
        payload["row_count"] != reservation.row_count
        or payload["rows_by_center"] != reservation.rows_by_center
        or payload["row_order_hash"] != reservation.row_order_hash
    ):
        raise TargetEvaluationContractError(
            "Stage-70 reservation summary does not match its projected rows."
        )
    if (
        reservation.schema_version != RESERVATION_SCHEMA_VERSION
        or reservation.protocol_id != RESERVATION_PROTOCOL_ID
        or reservation.authorized_consumer_experiment_id
        != AUTHORIZED_CONSUMER_EXPERIMENT_ID
        or reservation.purpose != PURPOSE
        or reservation.fresh_evidence is not FRESH_EVIDENCE
    ):
        raise TargetEvaluationContractError(
            "Stage-70 reservation fixed protocol fields drifted."
        )
    validate_target_evaluation_reservation(
        reservation,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_rows_by_center=expected_rows_by_center,
        allow_test_fixture=allow_test_fixture,
    )
    return reservation


__all__ = (
    "RESERVATION_DOCUMENT_FIELDS",
    "load_target_evaluation_reservation",
    "write_target_evaluation_reservation",
)
