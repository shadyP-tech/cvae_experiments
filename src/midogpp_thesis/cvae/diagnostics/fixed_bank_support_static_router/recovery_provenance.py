"""Hash-bound provenance for S4 validator-only finalization recovery."""

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


_SCHEMA = "fixed_bank_support_static_router_finalization_recovery_audit_v1"


def fresh_finalization_audit_payload() -> Mapping[str, object]:
    """Canonical marker for an uninterrupted finalization."""

    unhashed = {
        "schema_version": _SCHEMA,
        "finalization_recovery_used": False,
        "indexed_scientific_bytes_changed_during_finalization": False,
        "scientific_products_recomputed_during_finalization": False,
        "terminal_consumed_test_diagnostic_only": True,
        "promotion_eligible": False,
    }
    return {**unhashed, "audit_hash": canonical_hash(unhashed)}


def finalization_recovery_audit_payload(
    root: Path,
    *,
    current_repository_state: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Bind the exact failed state, both checkouts, index, and member hashes."""

    original = original_repository_state_from_provenance(root)
    repair = validated_repository_state(
        current_repository_state or current_repair_repository_state(),
        role="repair",
        require_clean=True,
    )
    if repair["repository_revision"] == original["repository_revision"]:
        raise ProtocolError("S4 finalization repair revision did not change.")
    bindings = indexed_surface_bindings(root)
    unhashed = {
        "schema_version": _SCHEMA,
        "finalization_recovery_used": True,
        "failed_run_state_hash": canonical_hash(FAILED_FINALIZATION_STATE),
        "failed_phase": "FINALIZATION",
        "failed_error": FAILED_FINALIZATION_STATE["error"],
        "failed_error_class": "ProtocolError",
        "original_repository_revision": original["repository_revision"],
        "original_repository_dirty": original["repository_dirty"],
        "original_repository_status_hash": original["repository_status_hash"],
        "repair_repository_revision": repair["repository_revision"],
        "repair_repository_dirty": repair["repository_dirty"],
        "repair_repository_status_hash": repair["repository_status_hash"],
        **bindings,
        "indexed_scientific_bytes_changed_during_finalization": False,
        "scientific_products_recomputed_during_finalization": False,
        "source_generation_recomputed_during_finalization": False,
        "predictions_recomputed_during_finalization": False,
        "route_decisions_recomputed_during_finalization": False,
        "terminal_evaluation_recomputed_during_finalization": False,
        "reports_rewritten_during_finalization": [
            "reports/fresh_process_validation.json",
            "reports/validation_report.json",
            "reports/run_state.json",
        ],
        "terminal_consumed_test_diagnostic_only": True,
        "promotion_eligible": False,
    }
    return {**unhashed, "audit_hash": canonical_hash(unhashed)}


def audit_for_validation(root: Path) -> Mapping[str, object]:
    """Reconstruct the pending audit, or verify the persisted completed audit."""

    path = Path(root)
    state = read_json(path / "reports/run_state.json")
    if state == FAILED_FINALIZATION_STATE:
        return finalization_recovery_audit_payload(path)
    report_path = path / "reports/validation_report.json"
    if report_path.is_file() and not report_path.is_symlink():
        observed = read_json(report_path).get("finalization_recovery")
        if not isinstance(observed, Mapping):
            raise ProtocolError("S4 validation report lacks its finalization audit.")
        if observed.get("finalization_recovery_used") is True:
            expected = finalization_recovery_audit_payload(
                path,
                current_repository_state={
                    "repository_revision": observed.get("repair_repository_revision"),
                    "repository_dirty": observed.get("repair_repository_dirty"),
                    "repository_status_hash": observed.get(
                        "repair_repository_status_hash"
                    ),
                },
            )
            assert_repair_repository_state_unchanged(
                {
                    "repository_revision": observed.get("repair_repository_revision"),
                    "repository_dirty": observed.get("repair_repository_dirty"),
                    "repository_status_hash": observed.get(
                        "repair_repository_status_hash"
                    ),
                }
            )
        else:
            expected = fresh_finalization_audit_payload()
        if dict(observed) != dict(expected):
            raise ProtocolError("S4 finalization recovery audit drifted.")
        return dict(observed)
    return fresh_finalization_audit_payload()


def indexed_surface_bindings(root: Path) -> Mapping[str, object]:
    index_path = Path(root) / "manifests/content_index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ProtocolError("S4 recovery content index is absent or unsafe.")
    index = read_json(index_path)
    rows = index.get("members")
    if not isinstance(rows, list):
        raise ProtocolError("S4 recovery content index rows are malformed.")
    hashes = {
        str(row.get("member")): str(row.get("sha256"))
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(hashes) != set(CONTENT_INDEX_MEMBERS) or any(
        sha256_file(Path(root) / member) != digest
        for member, digest in hashes.items()
    ):
        raise ProtocolError("S4 recovery indexed hash binding drifted.")
    return {
        "content_index_hash": index.get("content_hash"),
        "content_index_file_sha256": sha256_file(index_path),
        "indexed_member_sha256": hashes,
    }


def original_repository_state_from_provenance(root: Path) -> Mapping[str, object]:
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
    try:
        repo_root = MidogppWorkspace.load().repo_root
    except (OSError, ValueError, WorkspaceError) as exc:
        raise ProtocolError("Cannot locate the S4 repair checkout.") from exc
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
        raise ProtocolError("Cannot attest the S4 repair checkout.") from exc
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
        raise ProtocolError("S4 repair checkout changed during recovery.")


def validated_repository_state(
    value: Mapping[str, object], *, role: str, require_clean: bool
) -> dict[str, object]:
    revision = value.get("repository_revision")
    dirty = value.get("repository_dirty")
    status_hash = value.get("repository_status_hash")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision.lower())
        or not isinstance(dirty, bool)
        or not isinstance(status_hash, str)
        or len(status_hash) != 64
        or any(char not in "0123456789abcdef" for char in status_hash.lower())
        or (require_clean and dirty)
    ):
        raise ProtocolError(f"S4 {role} repository state is invalid.")
    return {
        "repository_revision": revision,
        "repository_dirty": dirty,
        "repository_status_hash": status_hash,
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
