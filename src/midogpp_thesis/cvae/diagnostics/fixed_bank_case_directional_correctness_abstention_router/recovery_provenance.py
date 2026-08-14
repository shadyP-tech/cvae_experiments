"""Hash-bound provenance for CDCA validator-only finalization recovery."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace, WorkspaceError
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .bundle import CONTENT_INDEX_MEMBERS
from .hashing import canonical_hash
from .recovery import FAILED_FINALIZATION_STATE
from .reports import run_state_payload


_SCHEMA = "fixed_bank_cdca_finalization_recovery_audit_v1"
_EMPTY_STATUS_HASH = hashlib.sha256(b"").hexdigest()
_PENDING_PHASE = "CLOSED_WORLD_TWO_FRESH_PROCESS_VALIDATION"
_REPORT_MEMBER = "reports/validation_report.json"
_RECONSTRUCTED_SURFACES = (
    "source_streams",
    "probability_surface",
    "held_case_plans_and_features",
    "support_responses",
    "donor_priors",
    "route_models",
    "route_candidate_scores",
    "route_decisions",
    "terminal_evaluation",
)


def fresh_finalization_audit_payload() -> Mapping[str, object]:
    """Canonical marker for an uninterrupted finalization."""

    unhashed = {
        "schema_version": _SCHEMA,
        "finalization_recovery_used": False,
        "indexed_scientific_bytes_changed_during_recovery": False,
        **_validation_surface_semantics(),
        "labels_reopened_only_for_read_only_validation": True,
        "policy_mutated_during_validation": False,
        "terminal_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    return {**unhashed, "audit_hash": canonical_hash(unhashed)}


def finalization_recovery_audit_payload(
    root: Path,
    *,
    current_repository_state: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Bind the exact failure, clean changed checkout, index, and all bytes."""

    original = original_repository_state_from_provenance(root)
    repair = validated_repository_state(
        current_repository_state or current_repair_repository_state(),
        role="repair",
        require_clean=True,
    )
    if repair["repository_revision"] == original["repository_revision"]:
        raise ProtocolError(
            "Case-directional finalization repair revision did not change."
        )
    bindings = indexed_surface_bindings(root)
    unhashed = {
        "schema_version": _SCHEMA,
        "finalization_recovery_used": True,
        "failed_run_state_hash": canonical_hash(FAILED_FINALIZATION_STATE),
        "failed_status": "FAILED",
        "failed_phase": _PENDING_PHASE,
        "failed_error": FAILED_FINALIZATION_STATE["error"],
        "failed_error_class": "ProtocolError",
        "original_repository_revision": original["repository_revision"],
        "original_repository_dirty": original["repository_dirty"],
        "original_repository_status_hash": original["repository_status_hash"],
        "repair_repository_revision": repair["repository_revision"],
        "repair_repository_dirty": repair["repository_dirty"],
        "repair_repository_status_hash": repair["repository_status_hash"],
        **bindings,
        "indexed_scientific_bytes_changed_during_recovery": False,
        **_validation_surface_semantics(),
        "labels_reopened_only_for_read_only_validation": True,
        "excluded_products_persisted_during_recovery": [
            "reports/validation_report.json",
            "reports/run_state.json",
        ],
        "raw_labels_persisted": False,
        "policy_mutated_during_validation": False,
        "terminal_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    return {**unhashed, "audit_hash": canonical_hash(unhashed)}


def audit_for_validation(
    root: Path,
    *,
    allow_pending_validation: bool,
    explicit_audit: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Admit one exact state and reconstruct/verify its finalization audit.

    Callers invoke this only after the content index has been validated.  The
    state gate therefore runs before labels or any other scientific replay.
    """

    path = Path(root)
    state_path = path / "reports/run_state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise ProtocolError("Case-directional validation run state is unsafe.")
    state = read_json(state_path)
    report_path = path / _REPORT_MEMBER
    if allow_pending_validation:
        if state == FAILED_FINALIZATION_STATE:
            if report_path.is_symlink() or (
                report_path.exists() and not report_path.is_file()
            ):
                raise ProtocolError(
                    "Case-directional retry validation report is unsafe."
                )
            expected = finalization_recovery_audit_payload(path)
        elif state == run_state_payload("RUNNING", _PENDING_PHASE):
            if report_path.exists():
                raise ProtocolError(
                    "Case-directional fresh pending validation report must be absent."
                )
            expected = fresh_finalization_audit_payload()
        else:
            raise ProtocolError(
                "Case-directional pending validation state is not reconstructive."
            )
    else:
        if state != run_state_payload("COMPLETE", "COMPLETE"):
            raise ProtocolError(
                "Case-directional completed run state is not reconstructive."
            )
        if report_path.is_symlink() or not report_path.is_file():
            raise ProtocolError(
                "Case-directional completed validation report is unsafe."
            )
        observed = read_json(report_path).get("finalization_recovery")
        if not isinstance(observed, Mapping):
            raise ProtocolError(
                "Case-directional validation report lacks its finalization audit."
            )
        if observed.get("finalization_recovery_used") is True:
            repair = {
                "repository_revision": observed.get("repair_repository_revision"),
                "repository_dirty": observed.get("repair_repository_dirty"),
                "repository_status_hash": observed.get(
                    "repair_repository_status_hash"
                ),
            }
            expected = finalization_recovery_audit_payload(
                path, current_repository_state=repair
            )
            assert_repair_repository_state_unchanged(repair)
        elif observed.get("finalization_recovery_used") is False:
            expected = fresh_finalization_audit_payload()
        else:
            raise ProtocolError(
                "Case-directional finalization recovery mode is invalid."
            )
        if dict(observed) != dict(expected):
            raise ProtocolError(
                "Case-directional finalization recovery audit drifted."
            )
    if explicit_audit is not None and dict(explicit_audit) != dict(expected):
        raise ProtocolError(
            "Case-directional explicit finalization recovery audit drifted."
        )
    return dict(expected)


def indexed_surface_bindings(root: Path) -> Mapping[str, object]:
    """Bind the content index file and every indexed member byte-for-byte."""

    path = Path(root)
    index_path = path / "manifests/content_index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ProtocolError(
            "Case-directional recovery content index is absent or unsafe."
        )
    index = read_json(index_path)
    rows = index.get("members")
    if not isinstance(rows, list) or len(rows) != len(CONTENT_INDEX_MEMBERS):
        raise ProtocolError(
            "Case-directional recovery content-index rows are malformed."
        )
    fingerprints: dict[str, dict[str, object]] = {}
    for expected_member, row in zip(CONTENT_INDEX_MEMBERS, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"member", "size_bytes", "sha256"}
            or row.get("member") != expected_member
            or not isinstance(row.get("size_bytes"), int)
            or isinstance(row.get("size_bytes"), bool)
            or int(row["size_bytes"]) < 0
            or not _is_hex(row.get("sha256"), length=64)
        ):
            raise ProtocolError(
                "Case-directional recovery content-index binding drifted."
            )
        member_path = path / expected_member
        if member_path.is_symlink() or not member_path.is_file():
            raise ProtocolError(
                "Case-directional recovery indexed member is absent or unsafe."
            )
        observed = {
            "size_bytes": member_path.stat().st_size,
            "sha256": sha256_file(member_path),
        }
        if observed != {
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }:
            raise ProtocolError(
                "Case-directional recovery indexed member binding drifted."
            )
        fingerprints[expected_member] = observed
    if not _is_hex(index.get("content_hash"), length=64):
        raise ProtocolError(
            "Case-directional recovery content hash is invalid."
        )
    return {
        "content_index_hash": index["content_hash"],
        "content_index_file_size_bytes": index_path.stat().st_size,
        "content_index_file_sha256": sha256_file(index_path),
        "indexed_member_fingerprints": fingerprints,
    }


def original_repository_state_from_provenance(
    root: Path,
) -> Mapping[str, object]:
    """Read revision A from the immutable workspace-rendered input manifest."""

    payload = read_json(Path(root) / "provenance/input_artifacts.json")
    return validated_repository_state(
        {
            "repository_revision": payload.get("repository_revision"),
            "repository_dirty": payload.get("repository_dirty"),
            "repository_status_hash": payload.get("repository_status_hash"),
        },
        role="original",
        require_clean=False,
    )


def current_repair_repository_state() -> Mapping[str, object]:
    """Read the active checkout revision and complete porcelain status."""

    try:
        repo_root = MidogppWorkspace.load().repo_root
    except (OSError, ValueError, WorkspaceError) as exc:
        raise ProtocolError(
            "Cannot locate the case-directional repair checkout."
        ) from exc
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(
            "Cannot attest the case-directional repair checkout."
        ) from exc
    status_text = status.stdout
    return validated_repository_state(
        {
            "repository_revision": revision.stdout.strip(),
            "repository_dirty": bool(status_text.strip()),
            "repository_status_hash": hashlib.sha256(
                status_text.encode("utf-8")
            ).hexdigest(),
        },
        role="repair",
        require_clean=False,
    )


def assert_repair_repository_state_unchanged(
    expected: Mapping[str, object],
) -> None:
    expected_state = validated_repository_state(
        expected, role="repair", require_clean=True
    )
    if dict(current_repair_repository_state()) != expected_state:
        raise ProtocolError(
            "Case-directional repair checkout changed during recovery."
        )


def validated_repository_state(
    value: Mapping[str, object], *, role: str, require_clean: bool
) -> dict[str, object]:
    revision = value.get("repository_revision")
    dirty = value.get("repository_dirty")
    status_hash = value.get("repository_status_hash")
    if (
        not _is_hex(revision, length=40)
        or not isinstance(dirty, bool)
        or not _is_hex(status_hash, length=64)
        or (dirty is False and status_hash != _EMPTY_STATUS_HASH)
        or (require_clean and dirty is not False)
    ):
        raise ProtocolError(
            f"Case-directional {role} repository state is invalid."
        )
    return {
        "repository_revision": revision,
        "repository_dirty": dirty,
        "repository_status_hash": status_hash,
    }


def _is_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validation_surface_semantics() -> dict[str, object]:
    """State exactly what validation reconstructs in memory and never writes."""

    return {
        "scientific_reconstruction_performed_for_validation": True,
        **{
            f"{surface}_reconstructed_for_validation": True
            for surface in _RECONSTRUCTED_SURFACES
        },
        **{
            f"{surface}_persisted_during_recovery": False
            for surface in _RECONSTRUCTED_SURFACES
        },
        "scientific_products_persisted_during_recovery": False,
        "terminal_products_persisted_during_recovery": False,
    }


__all__ = (
    "assert_repair_repository_state_unchanged",
    "audit_for_validation",
    "current_repair_repository_state",
    "finalization_recovery_audit_payload",
    "fresh_finalization_audit_payload",
    "indexed_surface_bindings",
    "original_repository_state_from_provenance",
    "validated_repository_state",
)
