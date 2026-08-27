"""Durable validation of the v2 hash-only label-capability journal."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..identity import CENTERS, EXPECTED_CASE_COUNT
from ..protocol import GovernanceError
from .hashing import canonical_hash, require_sha256
from .io import atomic_json, member_path, read_json_object


def persist_label_capability_journal(
    root: str | Path,
    payload: Mapping[str, object],
    *,
    phase: str,
) -> dict[str, object]:
    """Persist an immutable preterminal or final hash-only journal snapshot."""

    normalized = _validated_journal(payload, phase=phase)
    member = f"reports/label_capability_journal_{phase}.json"
    atomic_json(member_path(root, member), normalized)
    return normalized


def validate_persisted_label_capability_journal(
    root: str | Path, *, phase: str
) -> dict[str, object]:
    member = f"reports/label_capability_journal_{phase}.json"
    return _validated_journal(
        read_json_object(member_path(root, member)), phase=phase
    )


def _validated_journal(
    payload: Mapping[str, object], *, phase: str
) -> dict[str, object]:
    if phase not in {"preterminal", "final"}:
        raise GovernanceError("SCALE-BP v2 label-journal phase drifted.")
    row = dict(payload)
    events = row.get("events")
    counts = row.get("closed_scope_counts")
    expected_state = "DECISIONS_SEALED" if phase == "preterminal" else "CLOSED"
    if (
        row.get("schema_version") != "scale_bp_v2_label_capability_journal_v1"
        or row.get("phase") != expected_state
        or row.get("active_capability") is not None
        or not isinstance(events, list)
        or row.get("event_count") != len(events)
        or not isinstance(counts, Mapping)
        or set(counts) != {"DONOR", "SUPPORT", "TERMINAL"}
        or type(counts.get("DONOR")) is not int
        or type(counts.get("SUPPORT")) is not int
        or row.get("raw_labels_persisted") is not False
        or row.get("row_level_label_values_persisted") is not False
        or row.get("historical_capability_state_imported") is not False
        or row.get("audit_hash")
        != canonical_hash({key: value for key, value in row.items() if key != "audit_hash"})
    ):
        raise GovernanceError("SCALE-BP v2 label-capability journal drifted.")
    delegated_count = row.get("delegated_worker_count", 0)
    accepted_count = row.get("accepted_worker_audit_count", 0)
    delegated_hashes = row.get("delegated_worker_audit_hashes", {})
    local_complete = (
        delegated_count == 0
        and accepted_count == 0
        and int(counts["DONOR"]) > 0
        and int(counts["SUPPORT"]) > 0
    )
    delegated_complete = (
        delegated_count == len(CENTERS)
        and accepted_count == len(CENTERS)
        and int(counts["DONOR"]) == 0
        and int(counts["SUPPORT"]) == 0
        and isinstance(delegated_hashes, Mapping)
        and set(delegated_hashes) == set(CENTERS)
    )
    if not (local_complete or delegated_complete):
        raise GovernanceError("SCALE-BP v2 parent/worker label audit topology drifted.")
    if delegated_complete:
        for center, digest in delegated_hashes.items():  # type: ignore[union-attr]
            require_sha256(digest, f"worker audit {center}")
    require_sha256(row.get("journal_id"), "label journal id")
    require_sha256(row.get("run_identity_hash"), "label journal run identity")
    decision_seal = require_sha256(
        row.get("decision_seal_hash"), "label journal decision seal"
    )
    previous: str | None = None
    terminal_open_count = 0
    terminal_close_count = 0
    for sequence, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            raise GovernanceError("SCALE-BP v2 label journal event is malformed.")
        item = dict(event)
        event_hash = item.pop("event_hash", None)
        transition = str(item.get("transition", ""))
        if (
            item.get("schema_version") != "scale_bp_v2_label_capability_event_v1"
            or item.get("journal_id") != row.get("journal_id")
            or item.get("sequence") != sequence
            or item.get("previous_event_hash") != previous
            or item.get("raw_labels_persisted") is not False
            or event_hash != canonical_hash(item)
        ):
            raise GovernanceError("SCALE-BP v2 label journal hash chain drifted.")
        previous = require_sha256(event_hash, "label journal event hash")
        terminal_open_count += transition == "OPEN_TERMINAL"
        terminal_close_count += transition == "CLOSE_TERMINAL"
    terminal_closed = counts.get("TERMINAL")
    if phase == "preterminal":
        if terminal_open_count or terminal_close_count or terminal_closed != 0:
            raise GovernanceError(
                "SCALE-BP v2 terminal labels opened before the durable seal."
            )
    elif terminal_open_count != 1 or terminal_close_count != 1 or terminal_closed != 1:
        raise GovernanceError("SCALE-BP v2 terminal capability topology drifted.")
    if not any(
        isinstance(event, Mapping)
        and event.get("transition") == "SEAL_DECISIONS"
        and event.get("decision_seal_hash") == decision_seal
        and event.get("route_count") == EXPECTED_CASE_COUNT
        for event in events
    ):
        raise GovernanceError("SCALE-BP v2 journal does not bind the decision seal.")
    return row


__all__ = (
    "persist_label_capability_journal",
    "validate_persisted_label_capability_journal",
)
