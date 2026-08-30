"""Label-journal chain and role-access reconstruction for SCEPTRE v5."""

from __future__ import annotations

from typing import Mapping

from ....protocol import ProtocolError
from ...fixed_bank_sceptre_router.hashing import canonical_hash
from ...fixed_bank_sceptre_router.seals import EXPECTED_DECISION_KEYS
from ..support_posterior import SupportPosteriorDecision


def validate_label_journal(payload: Mapping[str, object]) -> None:
    """Replay the append-only hash chain without accepting persisted labels."""

    events = payload.get("events")
    if not isinstance(events, list):
        raise ProtocolError("SCEPTRE v5 persisted label journal is malformed.")
    predecessor = None
    hashes = []
    for ordinal, row in enumerate(events):
        if not isinstance(row, Mapping):
            raise ProtocolError("SCEPTRE v5 persisted label event is malformed.")
        body = {key: value for key, value in row.items() if key != "event_hash"}
        if (
            row.get("event_ordinal") != ordinal
            or row.get("predecessor_event_hash") != predecessor
            or row.get("prediction_store_hash")
            != payload.get("prediction_store_hash")
            or row.get("authorization_lease_hash")
            != payload.get("authorization_lease_hash")
            or row.get("raw_labels_persisted") is not False
            or row.get("event_hash") != canonical_hash(body)
        ):
            raise ProtocolError("SCEPTRE v5 persisted label event chain drifted.")
        predecessor = str(row["event_hash"])
        hashes.append(predecessor)
    journal_body = {
        "schema_version": "sceptre_v5_label_journal_chain_v1",
        "partition_hash": payload.get("partition_hash"),
        "prediction_store_hash": payload.get("prediction_store_hash"),
        "authorization_lease_hash": payload.get("authorization_lease_hash"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "event_hashes": hashes,
        "raw_labels_persisted": False,
    }
    if (
        payload.get("journal_hash") != canonical_hash(journal_body)
        or payload.get("raw_labels_persisted") is not False
        or payload.get("sample_paths_persisted") is not False
    ):
        raise ProtocolError("SCEPTRE v5 persisted label journal hash drifted.")


def validate_preterminal_journal(
    payload: Mapping[str, object],
    support_by_key: Mapping[tuple[str, int], SupportPosteriorDecision],
) -> None:
    """Bind selection/calibration events to the exact 45 support decisions."""

    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 2 * len(EXPECTED_DECISION_KEYS):
        raise ProtocolError("SCEPTRE v5 preterminal label-event inventory drifted.")
    selection = events[: len(EXPECTED_DECISION_KEYS)]
    calibration = events[len(EXPECTED_DECISION_KEYS) :]
    for key, event in zip(EXPECTED_DECISION_KEYS, selection, strict=True):
        support = support_by_key[key]
        if (
            not isinstance(event, Mapping)
            or event.get("event") != "SELECTION_LABELS_DECODED"
            or (event.get("target_center"), event.get("fold_ordinal")) != key
            or event.get("case_set_hash") != support.selection_case_set_hash
            or not isinstance(event.get("row_count"), int)
            or int(event["row_count"]) <= 0
            or event.get("manifest_rows_decoded") != event.get("row_count")
        ):
            raise ProtocolError("SCEPTRE v5 selection journal semantics drifted.")
    for key, event in zip(EXPECTED_DECISION_KEYS, calibration, strict=True):
        support = support_by_key[key]
        expected_event = (
            "CALIBRATION_SKIPPED_SUPPORT_FALLBACK"
            if support.fallback_required
            else "CALIBRATION_LABELS_DECODED"
        )
        if (
            not isinstance(event, Mapping)
            or event.get("event") != expected_event
            or (event.get("target_center"), event.get("fold_ordinal")) != key
            or event.get("case_set_hash") != support.calibration_case_set_hash
            or (
                support.fallback_required
                and (
                    event.get("row_count") != 0
                    or event.get("manifest_rows_decoded") != 0
                    or event.get("support_decision_hash") != support.decision_hash
                )
            )
            or (
                not support.fallback_required
                and (
                    not isinstance(event.get("row_count"), int)
                    or int(event["row_count"]) <= 0
                    or event.get("manifest_rows_decoded") != event.get("row_count")
                )
            )
        ):
            raise ProtocolError("SCEPTRE v5 calibration journal semantics drifted.")


__all__ = ("validate_label_journal", "validate_preterminal_journal")
