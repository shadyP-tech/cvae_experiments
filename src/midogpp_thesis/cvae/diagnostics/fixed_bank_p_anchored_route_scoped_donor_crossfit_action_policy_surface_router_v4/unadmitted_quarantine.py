"""Fail-closed quarantine for a workspace-prepared, unadmitted v4 root."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace
from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID, canonical_hash
from .run_admission import LAUNCH_FILES, WORKSPACE_DIRECTORIES, assert_no_partial_state
from .scratch import CANONICAL_SCRATCH_ROOT


QUARANTINE_PREPARED_SCHEMA = (
    "pdcaps_v4_unadmitted_workspace_quarantine_prepared_v1"
)
QUARANTINE_RECEIPT_SCHEMA = "pdcaps_v4_unadmitted_workspace_quarantine_v1"
QUARANTINE_EVENT = "UNADMITTED_WORKSPACE_SKELETON_QUARANTINED"


def quarantine_unadmitted_workspace_skeleton(
    root: Path,
    *,
    destination: Path,
) -> Mapping[str, object]:
    """Preserve only the exact pre-admission skeleton by a same-parent move.

    The transition is retry-safe around the directory move and receipt write,
    but it is ineligible for experiment recovery: any lock, run state, scratch,
    or science member fails closed.
    """

    source, quarantine, receipt_path, prepared_path = _safe_paths(
        root,
        destination,
    )
    scratch_paths = _scratch_paths(source)
    _assert_scratch_absent(scratch_paths)
    state = _transition_state(source, quarantine, receipt_path, prepared_path)

    if state[2]:
        return _validate_completed_transition(
            source,
            quarantine,
            receipt_path,
            prepared_path,
            scratch_paths,
        )
    if state == (False, True, False, True):
        return _finalize_moved_transition(
            source,
            quarantine,
            receipt_path,
            prepared_path,
            scratch_paths,
        )
    if state not in {
        (True, False, False, False),
        (True, False, False, True),
    }:
        raise ProtocolError("P-DCAPS v4 quarantine transition state is unsafe.")

    before = _audit_skeleton(source)
    expected_prepared = _prepared_receipt(
        source,
        quarantine,
        receipt_path,
        scratch_paths,
        before,
    )
    if state[3]:
        if read_json(prepared_path) != expected_prepared:
            raise ProtocolError("P-DCAPS v4 prepared quarantine receipt drifted.")
    else:
        atomic_json(prepared_path, expected_prepared)
        if read_json(prepared_path) != expected_prepared:
            raise ProtocolError(
                "P-DCAPS v4 prepared quarantine receipt replay drifted."
            )
        _fsync_directory(prepared_path.parent)

    if _audit_skeleton(source) != before:
        raise ProtocolError("P-DCAPS v4 unadmitted skeleton changed during audit.")
    _assert_scratch_absent(scratch_paths)
    if quarantine.exists() or quarantine.is_symlink() or receipt_path.exists():
        raise ProtocolError("P-DCAPS v4 quarantine destination appeared before move.")
    try:
        os.rename(source, quarantine)
    except OSError as exc:
        raise ProtocolError("P-DCAPS v4 unadmitted skeleton move failed.") from exc
    _fsync_directory(quarantine.parent)
    return _finalize_moved_transition(
        source,
        quarantine,
        receipt_path,
        prepared_path,
        scratch_paths,
    )


def _safe_paths(
    root: Path,
    destination: Path,
) -> tuple[Path, Path, Path, Path]:
    unresolved = Path(root)
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise ProtocolError("P-DCAPS v4 quarantine root is not absolute and real.")
    source = unresolved.resolve()
    expected_source = _canonical_output_root()
    if source != unresolved or source != expected_source:
        raise ProtocolError("P-DCAPS v4 quarantine root identity drifted.")
    if (
        source.parent.is_symlink()
        or source.parent.resolve() != source.parent
        or not source.parent.is_dir()
    ):
        raise ProtocolError("P-DCAPS v4 quarantine root parent is unsafe.")
    if source.exists() and (
        not source.is_dir() or source.stat().st_dev != source.parent.stat().st_dev
    ):
        raise ProtocolError("P-DCAPS v4 quarantine root is unsafe.")

    unresolved_destination = Path(destination)
    if (
        not unresolved_destination.is_absolute()
        or unresolved_destination.is_symlink()
    ):
        raise ProtocolError("P-DCAPS v4 quarantine destination is unsafe.")
    quarantine = unresolved_destination.resolve()
    expected_name = re.fullmatch(
        re.escape(source.name)
        + r"\.quarantine-unadmitted-provenance-header-[0-9]{8}T[0-9]{6}Z",
        quarantine.name,
    )
    if (
        quarantine != unresolved_destination
        or quarantine.parent != source.parent
        or expected_name is None
    ):
        raise ProtocolError("P-DCAPS v4 quarantine destination is not a safe sibling.")
    receipt_path = Path(f"{quarantine}.receipt.json")
    prepared_path = Path(f"{quarantine}.receipt.pending.json")
    if any(path.is_symlink() for path in (quarantine, receipt_path, prepared_path)):
        raise ProtocolError("P-DCAPS v4 quarantine transition path is a symlink.")
    return source, quarantine, receipt_path, prepared_path


def _canonical_output_root() -> Path:
    try:
        workspace = MidogppWorkspace.load()
        root = workspace.resolve_artifact(
            OUTPUT_ARTIFACT_ID,
            for_output=True,
            require_exists=False,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("P-DCAPS v4 canonical output binding failed.") from exc
    return Path(root).resolve()


def _transition_state(
    source: Path,
    quarantine: Path,
    receipt_path: Path,
    prepared_path: Path,
) -> tuple[bool, bool, bool, bool]:
    values = tuple(
        path.exists() or path.is_symlink()
        for path in (source, quarantine, receipt_path, prepared_path)
    )
    return values  # type: ignore[return-value]


def _prepared_receipt(
    source: Path,
    quarantine: Path,
    receipt_path: Path,
    scratch_paths: tuple[Path, Path],
    inventory: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": QUARANTINE_PREPARED_SCHEMA,
        "status": "PREPARED",
        "event": QUARANTINE_EVENT,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "source_root": str(source),
        "quarantine_root": str(quarantine),
        "receipt_path": str(receipt_path),
        "whole_root_move_completed": False,
        "run_lock_present": False,
        "run_state_present": False,
        "scratch_present": False,
        "authorization_consumed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "fresh_workspace_prepare_required": True,
        "scratch_paths": [str(path) for path in scratch_paths],
        "inventory": dict(inventory),
    }
    return {**payload, "prepared_receipt_hash": canonical_hash(payload)}


def _finalize_moved_transition(
    source: Path,
    quarantine: Path,
    receipt_path: Path,
    prepared_path: Path,
    scratch_paths: tuple[Path, Path],
) -> Mapping[str, object]:
    if (
        source.exists()
        or source.is_symlink()
        or quarantine.is_symlink()
        or not quarantine.is_dir()
        or prepared_path.is_symlink()
        or not prepared_path.is_file()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise ProtocolError("P-DCAPS v4 moved quarantine transition is unsafe.")
    inventory = _audit_skeleton(quarantine)
    expected_prepared = _prepared_receipt(
        source,
        quarantine,
        receipt_path,
        scratch_paths,
        inventory,
    )
    if read_json(prepared_path) != expected_prepared:
        raise ProtocolError("P-DCAPS v4 moved prepared receipt drifted.")
    _assert_scratch_absent(scratch_paths)
    receipt = _completed_receipt(expected_prepared, inventory)
    atomic_json(receipt_path, receipt)
    if read_json(receipt_path) != receipt:
        raise ProtocolError("P-DCAPS v4 quarantine receipt replay drifted.")
    _fsync_directory(receipt_path.parent)
    try:
        prepared_path.unlink()
    except OSError as exc:
        raise ProtocolError("P-DCAPS v4 prepared receipt cleanup failed.") from exc
    _fsync_directory(receipt_path.parent)
    return receipt


def _completed_receipt(
    prepared: Mapping[str, object],
    inventory: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": QUARANTINE_RECEIPT_SCHEMA,
        "status": "PASS",
        "event": QUARANTINE_EVENT,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "source_root": prepared["source_root"],
        "quarantine_root": prepared["quarantine_root"],
        "receipt_path": prepared["receipt_path"],
        "source_root_absent_after_move": True,
        "whole_root_move_completed": True,
        "run_lock_present": False,
        "run_state_present": False,
        "scratch_present": False,
        "authorization_consumed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "fresh_workspace_prepare_required": True,
        "scratch_paths": prepared["scratch_paths"],
        "inventory": dict(inventory),
        "prepared_receipt_hash": prepared["prepared_receipt_hash"],
    }
    return {**payload, "receipt_hash": canonical_hash(payload)}


def _validate_completed_transition(
    source: Path,
    quarantine: Path,
    receipt_path: Path,
    prepared_path: Path,
    scratch_paths: tuple[Path, Path],
) -> Mapping[str, object]:
    if (
        source.exists()
        or source.is_symlink()
        or quarantine.is_symlink()
        or not quarantine.is_dir()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        raise ProtocolError("P-DCAPS v4 completed quarantine transition is unsafe.")
    inventory = _audit_skeleton(quarantine)
    prepared = _prepared_receipt(
        source,
        quarantine,
        receipt_path,
        scratch_paths,
        inventory,
    )
    receipt = _completed_receipt(prepared, inventory)
    if read_json(receipt_path) != receipt:
        raise ProtocolError("P-DCAPS v4 completed quarantine receipt drifted.")
    _assert_scratch_absent(scratch_paths)
    if prepared_path.exists() or prepared_path.is_symlink():
        if prepared_path.is_symlink() or read_json(prepared_path) != prepared:
            raise ProtocolError("P-DCAPS v4 residual prepared receipt drifted.")
        try:
            prepared_path.unlink()
        except OSError as exc:
            raise ProtocolError("P-DCAPS v4 prepared receipt cleanup failed.") from exc
        _fsync_directory(prepared_path.parent)
    return receipt


def _scratch_paths(source: Path) -> tuple[Path, Path]:
    return (
        Path(CANONICAL_SCRATCH_ROOT),
        source.parent / f".{source.name}.pdcaps-v4-scratch",
    )


def _assert_scratch_absent(paths: tuple[Path, Path]) -> None:
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ProtocolError("P-DCAPS v4 scratch exists; skeleton is not unadmitted.")


def _audit_skeleton(root: Path) -> dict[str, object]:
    try:
        assert_no_partial_state(root)
        members = tuple(root.rglob("*"))
        files = tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in members
                if path.is_file()
            )
        )
        directories = tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in members
                if path.is_dir()
            )
        )
        if files != tuple(sorted(LAUNCH_FILES)) or directories != tuple(
            sorted(WORKSPACE_DIRECTORIES)
        ):
            raise ProtocolError("P-DCAPS v4 unadmitted skeleton inventory drifted.")
        return {
            "files": list(files),
            "directories": list(directories),
            "launch_members": [
                {
                    "path": relative,
                    "size_bytes": (root / relative).stat().st_size,
                    "sha256": sha256_file(root / relative),
                }
                for relative in files
            ],
        }
    except ProtocolError:
        raise
    except OSError as exc:
        raise ProtocolError("P-DCAPS v4 skeleton audit failed.") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProtocolError("P-DCAPS v4 quarantine directory fsync failed.") from exc


__all__ = (
    "QUARANTINE_EVENT",
    "QUARANTINE_PREPARED_SCHEMA",
    "QUARANTINE_RECEIPT_SCHEMA",
    "quarantine_unadmitted_workspace_skeleton",
)
