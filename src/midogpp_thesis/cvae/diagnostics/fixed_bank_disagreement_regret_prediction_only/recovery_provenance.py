"""Closed, hash-bound provenance for the post-test-seal code transition."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace, WorkspaceError
from ...protocol import ProtocolError
from .hashing import canonical_hash
from .recovery_contracts import FAILED_INFERENCE_STATE


_AUDIT_SCHEMA = "midogpp_post_test_seal_recovery_audit_v1"


def fresh_recovery_audit_payload() -> Mapping[str, object]:
    unhashed = _audit_payload(
        recovery_used=False,
        failed_run_state_hash=None,
        failed_phase=None,
        failed_error=None,
        reused_model_bank_seal_hash=None,
        reused_test_prediction_seal_hash=None,
        original_repository_state=None,
        repair_repository_state=None,
    )
    return {**unhashed, "recovery_audit_hash": canonical_hash(unhashed)}


def recovery_audit_payload(
    *,
    original_repository_state: Mapping[str, object],
    repair_repository_state: Mapping[str, object],
    model_bank_seal_hash: str,
    test_prediction_seal_hash: str,
) -> Mapping[str, object]:
    if not _is_hex(model_bank_seal_hash, length=64) or not _is_hex(
        test_prediction_seal_hash, length=64
    ):
        raise ProtocolError("Recovery reused-seal hash is invalid.")
    original = _validated_repository_state(
        original_repository_state, role="original", require_clean=False
    )
    repair = _validated_repository_state(
        repair_repository_state, role="repair", require_clean=True
    )
    if repair["repository_revision"] == original["repository_revision"]:
        raise ProtocolError("Recovery repair revision did not change.")
    unhashed = _audit_payload(
        recovery_used=True,
        failed_run_state_hash=canonical_hash(FAILED_INFERENCE_STATE),
        failed_phase=str(FAILED_INFERENCE_STATE["phase"]),
        failed_error=str(FAILED_INFERENCE_STATE["error"]),
        reused_model_bank_seal_hash=model_bank_seal_hash,
        reused_test_prediction_seal_hash=test_prediction_seal_hash,
        original_repository_state=original,
        repair_repository_state=repair,
    )
    return {**unhashed, "recovery_audit_hash": canonical_hash(unhashed)}


def validate_recovery_audit_payload(
    value: object,
    *,
    model_bank_seal_hash: str,
    test_prediction_seal_hash: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError("Runtime recovery audit is not an object.")
    observed = dict(value)
    if observed.get("recovery_used") is False:
        if observed != fresh_recovery_audit_payload():
            raise ProtocolError("Fresh-run recovery audit drifted.")
        return observed
    original = _validated_repository_state(
        {
            "repository_revision": observed.get("original_repository_revision"),
            "repository_dirty": observed.get("original_repository_dirty"),
            "repository_status_hash": observed.get(
                "original_repository_status_hash"
            ),
        },
        role="original",
        require_clean=False,
    )
    repair = _validated_repository_state(
        {
            "repository_revision": observed.get("repair_repository_revision"),
            "repository_dirty": observed.get("repair_repository_dirty"),
            "repository_status_hash": observed.get(
                "repair_repository_status_hash"
            ),
        },
        role="repair",
        require_clean=True,
    )
    expected = recovery_audit_payload(
        original_repository_state=original,
        repair_repository_state=repair,
        model_bank_seal_hash=model_bank_seal_hash,
        test_prediction_seal_hash=test_prediction_seal_hash,
    )
    if observed != expected:
        raise ProtocolError("Post-test-seal recovery audit drifted.")
    return observed


def current_repair_repository_state() -> Mapping[str, object]:
    try:
        repo_root = MidogppWorkspace.load().repo_root
    except (OSError, ValueError, WorkspaceError) as exc:
        raise ProtocolError("Cannot locate the recovery repair checkout.") from exc
    return _read_git_state(repo_root)


def assert_repair_repository_state_unchanged(
    audit: Mapping[str, object],
) -> None:
    """Require the clean repair checkout to remain byte-identical through recovery."""

    expected = {
        "repository_revision": audit.get("repair_repository_revision"),
        "repository_dirty": audit.get("repair_repository_dirty"),
        "repository_status_hash": audit.get("repair_repository_status_hash"),
    }
    observed = dict(current_repair_repository_state())
    if observed != expected:
        raise ProtocolError("Recovery repair checkout changed during continuation.")


def _read_git_state(repo_root: Path) -> Mapping[str, object]:
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
        raise ProtocolError("Cannot bind recovery to a Git checkout.") from exc
    status_text = status.stdout
    return {
        "repository_revision": revision.stdout.strip(),
        "repository_dirty": bool(status_text.strip()),
        "repository_status_hash": hashlib.sha256(
            status_text.encode("utf-8")
        ).hexdigest(),
    }


def _audit_payload(
    *,
    recovery_used: bool,
    failed_run_state_hash: object,
    failed_phase: object,
    failed_error: object,
    reused_model_bank_seal_hash: object,
    reused_test_prediction_seal_hash: object,
    original_repository_state: Mapping[str, object] | None,
    repair_repository_state: Mapping[str, object] | None,
) -> dict[str, object]:
    original = original_repository_state or {}
    repair = repair_repository_state or {}
    return {
        "schema_version": _AUDIT_SCHEMA,
        "recovery_used": recovery_used,
        "failed_run_state_hash": failed_run_state_hash,
        "failed_phase": failed_phase,
        "failed_error": failed_error,
        "reused_model_bank_seal_hash": reused_model_bank_seal_hash,
        "reused_test_prediction_seal_hash": reused_test_prediction_seal_hash,
        "production_models_refit": False,
        "production_test_predictions_recomputed": False,
        "validator_source_models_refit": True,
        "test_labels_opened": False,
        "test_scores_computed": False,
        "original_repository_revision": original.get("repository_revision"),
        "original_repository_dirty": original.get("repository_dirty"),
        "original_repository_status_hash": original.get(
            "repository_status_hash"
        ),
        "repair_repository_revision": repair.get("repository_revision"),
        "repair_repository_dirty": repair.get("repository_dirty"),
        "repair_repository_status_hash": repair.get("repository_status_hash"),
    }


def _validated_repository_state(
    value: Mapping[str, object], *, role: str, require_clean: bool
) -> dict[str, object]:
    revision = value.get("repository_revision")
    dirty = value.get("repository_dirty")
    status_hash = value.get("repository_status_hash")
    if (
        not _is_hex(revision, length=40)
        or not isinstance(dirty, bool)
        or not _is_hex(status_hash, length=64)
        or (require_clean and dirty is not False)
    ):
        raise ProtocolError(f"Recovery {role} repository state is invalid.")
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


__all__ = (
    "assert_repair_repository_state_unchanged",
    "current_repair_repository_state",
    "fresh_recovery_audit_payload",
    "recovery_audit_payload",
    "validate_recovery_audit_payload",
)
