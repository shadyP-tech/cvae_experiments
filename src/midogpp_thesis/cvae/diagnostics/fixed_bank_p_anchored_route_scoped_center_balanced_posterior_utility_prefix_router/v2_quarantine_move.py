"""Retry-safe same-parent moves and immutable receipt for v2 quarantine."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import os
from pathlib import Path
import re
import stat
from typing import Iterable

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_io import persist_json
from .constants import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .hashing import canonical_hash
from .v2_quarantine_audit import (
    audit_locked_failed_v2_terminal_or_final,
    exclusive_existing_v2_terminal_run_lock,
    is_present,
    logical_source_root,
    v2_scratch_root,
)
from .v2_quarantine_contracts import (
    ELIGIBLE_NEXT_ACTION,
    QUARANTINE_SUFFIX,
    RECEIPT_SCHEMA,
)


def quarantine_failed_v2_terminal_or_final(
    root: Path,
    *,
    artifact_destination: Path,
    scratch_destination: Path,
) -> Mapping[str, object]:
    logical_root = logical_source_root(root)
    logical_scratch = v2_scratch_root()
    quarantine_root, quarantine_scratch, quarantine_tag = quarantine_destinations(
        logical_root,
        Path(artifact_destination),
        logical_scratch,
        Path(scratch_destination),
    )
    receipt_path = Path(f"{quarantine_root}.receipt.json")
    paths = (logical_root, logical_scratch, quarantine_root, quarantine_scratch)
    if any(path.is_symlink() for path in (*paths, receipt_path)):
        raise ProtocolError("CBPUPR v2 terminal/final quarantine path is a symlink.")

    initial_with_scratch = (True, True, False, False)
    scratch_moved = (True, False, False, True)
    completed_with_scratch = (False, False, True, True)
    initial_without_scratch = (True, False, False, False)
    completed_without_scratch = (False, False, True, False)
    initial_states = {initial_with_scratch, scratch_moved, initial_without_scratch}
    completed_states = {completed_with_scratch, completed_without_scratch}
    state = tuple(is_present(path) for path in paths)
    if state not in initial_states | completed_states:
        raise ProtocolError("CBPUPR v2 terminal/final quarantine state is unsafe.")
    if state in initial_states and receipt_path.exists():
        raise ProtocolError("CBPUPR v2 quarantine receipt appeared before both moves.")

    if state in completed_states:
        observed_scratch = quarantine_scratch if state == completed_with_scratch else None
        with exclusive_existing_v2_terminal_run_lock(quarantine_root):
            audit = audit_locked_failed_v2_terminal_or_final(
                logical_root=logical_root,
                observed_root=quarantine_root,
                logical_scratch=logical_scratch,
                observed_scratch=observed_scratch,
            )
        receipt = quarantine_receipt(
            source_root=logical_root,
            quarantine_root=quarantine_root,
            source_scratch=logical_scratch,
            quarantine_scratch=quarantine_scratch,
            quarantine_tag=quarantine_tag,
            audit=audit,
        )
        _persist_or_validate_receipt(receipt_path, receipt)
        return receipt

    with exclusive_existing_v2_terminal_run_lock(logical_root):
        locked_state = tuple(is_present(path) for path in paths)
        if locked_state not in initial_states:
            raise ProtocolError("CBPUPR v2 terminal/final quarantine state changed.")
        observed_scratch = (
            logical_scratch
            if locked_state == initial_with_scratch
            else quarantine_scratch
            if locked_state == scratch_moved
            else None
        )
        audit = audit_locked_failed_v2_terminal_or_final(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=logical_scratch,
            observed_scratch=observed_scratch,
        )
        artifact_snapshot = tree_stat_snapshot(
            logical_root,
            (row["path"] for row in audit["artifact_members"]),
        )
        scratch_snapshot = (
            tree_stat_snapshot(
                observed_scratch,
                (row["path"] for row in audit["scratch_members"]),
            )
            if observed_scratch is not None
            else {}
        )
        scratch_was_present = observed_scratch is not None
        if locked_state == initial_with_scratch:
            if is_present(quarantine_scratch):
                raise ProtocolError(
                    "CBPUPR v2 terminal/final scratch destination appeared."
                )
            os.rename(logical_scratch, quarantine_scratch)
            assert_tree_stat_snapshot(quarantine_scratch, scratch_snapshot)
        if is_present(quarantine_root):
            raise ProtocolError(
                "CBPUPR v2 terminal/final artifact destination appeared."
            )
        os.rename(logical_root, quarantine_root)
        if (
            is_present(logical_root)
            or is_present(logical_scratch)
            or not quarantine_root.is_dir()
            or scratch_was_present != quarantine_scratch.is_dir()
        ):
            raise ProtocolError(
                "CBPUPR v2 terminal/final quarantine move was incomplete."
            )
        assert_tree_stat_snapshot(quarantine_root, artifact_snapshot)
        if scratch_was_present:
            assert_tree_stat_snapshot(quarantine_scratch, scratch_snapshot)
        post_move_audit = audit_locked_failed_v2_terminal_or_final(
            logical_root=logical_root,
            observed_root=quarantine_root,
            logical_scratch=logical_scratch,
            observed_scratch=quarantine_scratch if scratch_was_present else None,
        )
        if post_move_audit != audit:
            raise ProtocolError("CBPUPR v2 terminal/final moved bytes drifted.")

    receipt = quarantine_receipt(
        source_root=logical_root,
        quarantine_root=quarantine_root,
        source_scratch=logical_scratch,
        quarantine_scratch=quarantine_scratch,
        quarantine_tag=quarantine_tag,
        audit=post_move_audit,
    )
    _persist_or_validate_receipt(receipt_path, receipt)
    return receipt


def quarantine_receipt(
    *,
    source_root: Path,
    quarantine_root: Path,
    source_scratch: Path,
    quarantine_scratch: Path,
    quarantine_tag: str,
    audit: Mapping[str, object],
) -> Mapping[str, object]:
    scratch_was_present = audit.get("scratch_state") != "ABSENT_AFTER_FINAL_REPORT"
    artifact_members = list(audit["artifact_members"])
    scratch_members = list(audit["scratch_members"])
    payload: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "quarantine_tag": quarantine_tag,
        "source_root": str(source_root),
        "quarantine_root": str(quarantine_root),
        "source_scratch_root": str(source_scratch),
        "quarantine_scratch_root": (
            str(quarantine_scratch) if scratch_was_present else None
        ),
        "receipt_path": str(Path(f"{quarantine_root}.receipt.json")),
        "source_root_absent_after_move": True,
        "source_scratch_absent_after_move": True,
        "whole_artifact_move_completed": True,
        "whole_scratch_move_completed": scratch_was_present,
        "scratch_absent_before_quarantine": not scratch_was_present,
        "move_order": ["scratch", "artifact"] if scratch_was_present else ["artifact"],
        "same_parent_atomic_rename_per_present_root": True,
        "pre_move_artifact_members": artifact_members,
        "post_move_artifact_members": artifact_members,
        "pre_move_scratch_members": scratch_members,
        "post_move_scratch_members": scratch_members,
        "config_contract_hash": audit["config_contract_hash"],
        "protocol_contract_hash": audit["protocol_contract_hash"],
        "protocol_manifest_hash": audit["protocol_manifest_hash"],
        "preterminal_content_index_hash": audit[
            "preterminal_content_index_hash"
        ],
        "preterminal_validation_checks_hash": audit[
            "preterminal_validation_checks_hash"
        ],
        "preterminal_hash": audit["preterminal_hash"],
        "repair_source_manifest_sha256": audit[
            "repair_source_manifest_sha256"
        ],
        "repair_source_tree_sha256": audit["repair_source_tree_sha256"],
        "repair_source_member_count": audit["repair_source_member_count"],
        "test_manifest_sha256": audit["test_manifest_sha256"],
        "test_consumption_ledger_sha256": audit[
            "test_consumption_ledger_sha256"
        ],
        "ledger_amendment_sha256": audit["ledger_amendment_sha256"],
        "terminal_access_journal_status": audit[
            "terminal_access_journal_status"
        ],
        "quarantined_bytes_may_feed_rerun": False,
        "v2_rerun_authorized": False,
        "v1_output_scratch_or_capability_history_may_be_used": False,
        "quarantined_v2_results_may_be_promoted": False,
        "eligible_next_action": ELIGIBLE_NEXT_ACTION,
        "pre_move_audit": dict(audit),
        "post_move_audit": dict(audit),
    }
    return {**payload, "quarantine_receipt_hash": canonical_hash(payload)}


def _persist_or_validate_receipt(
    path: Path, receipt: Mapping[str, object]
) -> None:
    if path.is_symlink():
        raise ProtocolError("CBPUPR v2 quarantine receipt is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(receipt):
            raise ProtocolError("CBPUPR v2 quarantine receipt drifted.")
        return
    persist_json(path, receipt, allow_paths=True)
    if read_json(path) != dict(receipt):
        raise ProtocolError("CBPUPR v2 quarantine receipt persistence drifted.")


def quarantine_destinations(
    source_root: Path,
    artifact_destination: Path,
    source_scratch: Path,
    scratch_destination: Path,
) -> tuple[Path, Path, str]:
    requested = (artifact_destination, scratch_destination)
    if any(path.is_symlink() for path in requested):
        raise ProtocolError("CBPUPR v2 terminal/final quarantine path is a symlink.")
    if any(path.parent.is_symlink() or not path.parent.is_dir() for path in requested):
        raise ProtocolError("CBPUPR v2 terminal/final quarantine parent is unsafe.")
    artifact = artifact_destination.parent.resolve() / artifact_destination.name
    scratch = scratch_destination.parent.resolve() / scratch_destination.name
    if artifact.parent != source_root.parent or scratch.parent != source_scratch.parent:
        raise ProtocolError(
            "CBPUPR v2 terminal/final quarantine destination is not same-parent."
        )
    artifact_match = re.fullmatch(
        re.escape(source_root.name) + QUARANTINE_SUFFIX.pattern, artifact.name
    )
    scratch_match = re.fullmatch(
        re.escape(source_scratch.name) + QUARANTINE_SUFFIX.pattern, scratch.name
    )
    if artifact_match is None or scratch_match is None:
        raise ProtocolError("CBPUPR v2 terminal/final quarantine name is unsafe.")
    tag = artifact_match.group(1)
    try:
        datetime.strptime(tag, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ProtocolError(
            "CBPUPR v2 terminal/final quarantine timestamp is invalid."
        ) from exc
    if tag != scratch_match.group(1):
        raise ProtocolError(
            "CBPUPR v2 terminal/final quarantine timestamps differ."
        )
    return artifact, scratch, tag


def tree_stat_snapshot(
    root: Path, members: Iterable[object]
) -> dict[str, tuple[int, int, int]]:
    snapshot: dict[str, tuple[int, int, int]] = {}
    for relative in members:
        path = root / str(relative)
        value = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise ProtocolError("CBPUPR v2 quarantine snapshot member is unsafe.")
        snapshot[str(relative)] = (value.st_dev, value.st_ino, value.st_size)
    return snapshot


def assert_tree_stat_snapshot(
    root: Path, snapshot: Mapping[str, tuple[int, int, int]]
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("CBPUPR v2 quarantine destination is absent or unsafe.")
    for relative, expected in snapshot.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("CBPUPR v2 quarantine moved member is unsafe.")
        value = path.stat(follow_symlinks=False)
        if (value.st_dev, value.st_ino, value.st_size) != expected:
            raise ProtocolError("CBPUPR v2 quarantine moved bytes drifted.")


__all__ = ("quarantine_failed_v2_terminal_or_final",)
