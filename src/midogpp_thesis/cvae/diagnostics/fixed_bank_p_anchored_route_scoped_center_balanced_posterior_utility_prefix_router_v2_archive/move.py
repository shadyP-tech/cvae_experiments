"""Retry-safe same-parent quarantine and immutable archive receipt."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat

from ...protocol import ProtocolError
from . import contracts as archive_contracts
from .audit import audit_locked_failed_v2_preterminal
from .contracts import (
    ELIGIBLE_NEXT_ACTION,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    QUARANTINE_SUFFIX,
    RECEIPT_SCHEMA,
)
from .hashing import canonical_hash
from .tree import exclusive_existing_run_lock, is_present, logical_path


def quarantine_failed_v2_preterminal_for_archive(
    root: Path,
    *,
    artifact_destination: Path,
    scratch_root: Path | None,
    scratch_destination: Path | None,
) -> Mapping[str, object]:
    """Move scratch first, then the artifact, without authorizing byte reuse."""

    logical_root = logical_path(root, role="artifact")
    logical_scratch = (
        None if scratch_root is None else logical_path(scratch_root, role="scratch")
    )
    if (logical_scratch is None) != (scratch_destination is None):
        raise ProtocolError(
            "CBPUPR v2 preterminal scratch source/destination must be supplied together."
        )
    quarantine_root, quarantine_scratch, tag = _destinations(
        logical_root,
        Path(artifact_destination),
        logical_scratch,
        None if scratch_destination is None else Path(scratch_destination),
    )
    receipt_path = Path(f"{quarantine_root}.receipt.json")
    pending_receipt = Path(f"{receipt_path}.pending")
    all_paths = [logical_root, quarantine_root, receipt_path, pending_receipt]
    if logical_scratch is not None and quarantine_scratch is not None:
        all_paths.extend((logical_scratch, quarantine_scratch))
    if any(path.is_symlink() for path in all_paths):
        raise ProtocolError("CBPUPR v2 preterminal archive path is a symlink.")

    if logical_scratch is None:
        return _quarantine_without_scratch(
            logical_root=logical_root,
            quarantine_root=quarantine_root,
            receipt_path=receipt_path,
            tag=tag,
        )
    assert quarantine_scratch is not None
    return _quarantine_with_scratch(
        logical_root=logical_root,
        logical_scratch=logical_scratch,
        quarantine_root=quarantine_root,
        quarantine_scratch=quarantine_scratch,
        receipt_path=receipt_path,
        tag=tag,
    )


def _quarantine_with_scratch(
    *,
    logical_root: Path,
    logical_scratch: Path,
    quarantine_root: Path,
    quarantine_scratch: Path,
    receipt_path: Path,
    tag: str,
) -> Mapping[str, object]:
    paths = (logical_root, logical_scratch, quarantine_root, quarantine_scratch)
    initial = (True, True, False, False)
    scratch_moved = (True, False, False, True)
    completed = (False, False, True, True)
    state = tuple(is_present(path) for path in paths)
    if state not in {initial, scratch_moved, completed}:
        raise ProtocolError("CBPUPR v2 preterminal archive state is unsafe.")
    if state != completed and is_present(receipt_path):
        raise ProtocolError("CBPUPR v2 preterminal receipt appeared before both moves.")

    if state == completed:
        with exclusive_existing_run_lock(quarantine_root):
            audit = audit_locked_failed_v2_preterminal(
                logical_root=logical_root,
                observed_root=quarantine_root,
                logical_scratch=logical_scratch,
                observed_scratch=quarantine_scratch,
            )
        receipt = _receipt(
            source_root=logical_root,
            quarantine_root=quarantine_root,
            source_scratch=logical_scratch,
            quarantine_scratch=quarantine_scratch,
            tag=tag,
            audit=audit,
        )
        _persist_or_validate_immutable_receipt(receipt_path, receipt)
        return receipt

    with exclusive_existing_run_lock(logical_root):
        locked_state = tuple(is_present(path) for path in paths)
        if locked_state not in {initial, scratch_moved}:
            raise ProtocolError("CBPUPR v2 preterminal archive state changed.")
        observed_scratch = (
            logical_scratch if locked_state == initial else quarantine_scratch
        )
        audit = audit_locked_failed_v2_preterminal(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=logical_scratch,
            observed_scratch=observed_scratch,
        )
        artifact_snapshot = _tree_stat_snapshot(
            logical_root, (row["path"] for row in audit["artifact_members"])
        )
        scratch_snapshot = _tree_stat_snapshot(
            observed_scratch, (row["path"] for row in audit["scratch_members"])
        )
        if locked_state == initial:
            if is_present(quarantine_scratch):
                raise ProtocolError(
                    "CBPUPR v2 preterminal scratch destination appeared."
                )
            os.rename(logical_scratch, quarantine_scratch)
            _assert_tree_stat_snapshot(quarantine_scratch, scratch_snapshot)
        if is_present(quarantine_root):
            raise ProtocolError(
                "CBPUPR v2 preterminal artifact destination appeared."
            )
        os.rename(logical_root, quarantine_root)
        if (
            is_present(logical_root)
            or is_present(logical_scratch)
            or not quarantine_root.is_dir()
            or not quarantine_scratch.is_dir()
        ):
            raise ProtocolError("CBPUPR v2 preterminal archive move was incomplete.")
        _assert_tree_stat_snapshot(quarantine_root, artifact_snapshot)
        _assert_tree_stat_snapshot(quarantine_scratch, scratch_snapshot)
        post_move_audit = audit_locked_failed_v2_preterminal(
            logical_root=logical_root,
            observed_root=quarantine_root,
            logical_scratch=logical_scratch,
            observed_scratch=quarantine_scratch,
        )
        if post_move_audit != audit:
            raise ProtocolError("CBPUPR v2 preterminal moved bytes drifted.")

    receipt = _receipt(
        source_root=logical_root,
        quarantine_root=quarantine_root,
        source_scratch=logical_scratch,
        quarantine_scratch=quarantine_scratch,
        tag=tag,
        audit=post_move_audit,
    )
    _persist_or_validate_immutable_receipt(receipt_path, receipt)
    return receipt


def _quarantine_without_scratch(
    *,
    logical_root: Path,
    quarantine_root: Path,
    receipt_path: Path,
    tag: str,
) -> Mapping[str, object]:
    state = (is_present(logical_root), is_present(quarantine_root))
    if state not in {(True, False), (False, True)}:
        raise ProtocolError("CBPUPR v2 preterminal no-scratch archive state is unsafe.")
    if state == (True, False) and is_present(receipt_path):
        raise ProtocolError("CBPUPR v2 preterminal receipt appeared before the move.")
    if state == (False, True):
        with exclusive_existing_run_lock(quarantine_root):
            audit = audit_locked_failed_v2_preterminal(
                logical_root=logical_root,
                observed_root=quarantine_root,
                logical_scratch=None,
                observed_scratch=None,
            )
        receipt = _receipt(
            source_root=logical_root,
            quarantine_root=quarantine_root,
            source_scratch=None,
            quarantine_scratch=None,
            tag=tag,
            audit=audit,
        )
        _persist_or_validate_immutable_receipt(receipt_path, receipt)
        return receipt

    with exclusive_existing_run_lock(logical_root):
        if not is_present(logical_root) or is_present(quarantine_root):
            raise ProtocolError("CBPUPR v2 preterminal no-scratch state changed.")
        audit = audit_locked_failed_v2_preterminal(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=None,
            observed_scratch=None,
        )
        snapshot = _tree_stat_snapshot(
            logical_root, (row["path"] for row in audit["artifact_members"])
        )
        os.rename(logical_root, quarantine_root)
        if is_present(logical_root) or not quarantine_root.is_dir():
            raise ProtocolError("CBPUPR v2 preterminal artifact move was incomplete.")
        _assert_tree_stat_snapshot(quarantine_root, snapshot)
        post_move_audit = audit_locked_failed_v2_preterminal(
            logical_root=logical_root,
            observed_root=quarantine_root,
            logical_scratch=None,
            observed_scratch=None,
        )
        if post_move_audit != audit:
            raise ProtocolError("CBPUPR v2 preterminal moved bytes drifted.")

    receipt = _receipt(
        source_root=logical_root,
        quarantine_root=quarantine_root,
        source_scratch=None,
        quarantine_scratch=None,
        tag=tag,
        audit=post_move_audit,
    )
    _persist_or_validate_immutable_receipt(receipt_path, receipt)
    return receipt


def _receipt(
    *,
    source_root: Path,
    quarantine_root: Path,
    source_scratch: Path | None,
    quarantine_scratch: Path | None,
    tag: str,
    audit: Mapping[str, object],
) -> Mapping[str, object]:
    scratch_present = source_scratch is not None
    payload: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "archive_contract_hash": archive_contracts.CANONICAL_ARCHIVE_CONTRACT.contract_hash,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "quarantine_tag": tag,
        "source_root": str(source_root),
        "quarantine_root": str(quarantine_root),
        "source_scratch_root": (
            str(source_scratch) if source_scratch is not None else None
        ),
        "quarantine_scratch_root": (
            str(quarantine_scratch) if quarantine_scratch is not None else None
        ),
        "receipt_path": str(Path(f"{quarantine_root}.receipt.json")),
        "source_root_absent_after_move": True,
        "source_scratch_absent_after_move": True if scratch_present else None,
        "whole_artifact_move_completed": True,
        "whole_scratch_move_completed": scratch_present,
        "explicit_no_scratch_api_state": not scratch_present,
        "move_order": ["scratch", "artifact"] if scratch_present else ["artifact"],
        "same_parent_atomic_rename_per_present_root": True,
        "pre_move_artifact_members": list(audit["artifact_members"]),
        "post_move_artifact_members": list(audit["artifact_members"]),
        "pre_move_scratch_members": list(audit["scratch_members"]),
        "post_move_scratch_members": list(audit["scratch_members"]),
        "terminal_access_journal_status": "ABSENT_NOT_OPENED",
        "terminal_labels_opened": False,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "quarantined_bytes_may_feed_successor": False,
        "v2_rerun_authorized": False,
        "v3_input_reuse_authorized": False,
        "routing_success_claim_authorized": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "eligible_next_action": ELIGIBLE_NEXT_ACTION,
        "pre_move_audit": dict(audit),
        "post_move_audit": dict(audit),
    }
    return {**payload, "archive_receipt_hash": canonical_hash(payload)}


def _persist_or_validate_immutable_receipt(
    path: Path, receipt: Mapping[str, object]
) -> None:
    encoded = (
        json.dumps(dict(receipt), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    pending = Path(f"{path}.pending")
    for candidate in (path, pending):
        if candidate.is_symlink():
            raise ProtocolError("CBPUPR v2 preterminal receipt path is a symlink.")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise ProtocolError("CBPUPR v2 preterminal immutable receipt drifted.")
        if pending.exists():
            if not pending.is_file() or pending.read_bytes() != encoded:
                raise ProtocolError("CBPUPR v2 preterminal pending receipt drifted.")
            pending.unlink()
            _fsync_directory(path.parent)
        return
    if pending.exists():
        if not pending.is_file() or pending.read_bytes() != encoded:
            raise ProtocolError("CBPUPR v2 preterminal pending receipt drifted.")
    else:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(pending, flags, 0o444)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    try:
        os.link(pending, path, follow_symlinks=False)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != encoded:
            raise ProtocolError("CBPUPR v2 preterminal immutable receipt raced.")
    if path.read_bytes() != encoded:
        raise ProtocolError("CBPUPR v2 preterminal receipt persistence drifted.")
    pending.unlink()
    _fsync_directory(path.parent)


def _destinations(
    source_root: Path,
    artifact_destination: Path,
    source_scratch: Path | None,
    scratch_destination: Path | None,
) -> tuple[Path, Path | None, str]:
    requested = [artifact_destination]
    if scratch_destination is not None:
        requested.append(scratch_destination)
    if any(path.is_symlink() for path in requested) or any(
        path.parent.is_symlink() or not path.parent.is_dir() for path in requested
    ):
        raise ProtocolError("CBPUPR v2 preterminal quarantine parent is unsafe.")
    artifact = artifact_destination.parent.resolve() / artifact_destination.name
    if artifact.parent != source_root.parent:
        raise ProtocolError(
            "CBPUPR v2 preterminal artifact quarantine is not same-parent."
        )
    artifact_match = re.fullmatch(
        re.escape(source_root.name) + QUARANTINE_SUFFIX.pattern,
        artifact.name,
    )
    if artifact_match is None:
        raise ProtocolError("CBPUPR v2 preterminal quarantine name is unsafe.")
    tag = artifact_match.group(1)
    try:
        datetime.strptime(tag, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ProtocolError(
            "CBPUPR v2 preterminal quarantine timestamp is invalid."
        ) from exc
    if source_scratch is None:
        if scratch_destination is not None:
            raise ProtocolError("CBPUPR v2 preterminal unexpected scratch destination.")
        return artifact, None, tag
    if scratch_destination is None:
        raise ProtocolError("CBPUPR v2 preterminal scratch destination is absent.")
    scratch = scratch_destination.parent.resolve() / scratch_destination.name
    if scratch.parent != source_scratch.parent:
        raise ProtocolError(
            "CBPUPR v2 preterminal scratch quarantine is not same-parent."
        )
    scratch_match = re.fullmatch(
        re.escape(source_scratch.name) + QUARANTINE_SUFFIX.pattern,
        scratch.name,
    )
    if scratch_match is None or scratch_match.group(1) != tag:
        raise ProtocolError("CBPUPR v2 preterminal scratch quarantine name drifted.")
    return artifact, scratch, tag


def _tree_stat_snapshot(
    root: Path, members: Iterable[object]
) -> dict[str, tuple[int, int, int]]:
    result: dict[str, tuple[int, int, int]] = {}
    for relative in members:
        path = root / str(relative)
        value = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise ProtocolError("CBPUPR v2 preterminal snapshot member is unsafe.")
        result[str(relative)] = (value.st_dev, value.st_ino, value.st_size)
    return result


def _assert_tree_stat_snapshot(
    root: Path, snapshot: Mapping[str, tuple[int, int, int]]
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("CBPUPR v2 preterminal moved root is unsafe.")
    for relative, expected in snapshot.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("CBPUPR v2 preterminal moved member is unsafe.")
        value = path.stat(follow_symlinks=False)
        if (value.st_dev, value.st_ino, value.st_size) != expected:
            raise ProtocolError("CBPUPR v2 preterminal moved bytes drifted.")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ("quarantine_failed_v2_preterminal_for_archive",)
