"""Hash-bound provenance for the mappingproxy-only recovery path.

The recovery reuses the already sealed label-free source, prediction, feature,
and fold-plan surfaces.  It repeats deterministic donor fitting, downstream
decision construction, and terminal reconstruction under the repaired checkout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Mapping

from ....workspace.runtime import MidogppWorkspace, WorkspaceError
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .hashing import canonical_hash


_AUDIT_SCHEMA = (
    "midogpp_fixed_bank_multi_challenger_mappingproxy_recovery_audit_v1"
)


def fresh_recovery_audit_payload() -> Mapping[str, object]:
    """Return the canonical audit carried by an uninterrupted fresh run."""

    unhashed = _audit_payload(
        recovery_used=False,
        failed_run_state_hash=None,
        failed_phase=None,
        failed_error=None,
        reused_source_stream_lock_hash=None,
        reused_global_prediction_seal_hash=None,
        reused_prelabel_feature_surface_hash=None,
        reused_fold_plan_surface_hash=None,
        original_repository_state=None,
        repair_repository_state=None,
    )
    return {**unhashed, "recovery_audit_hash": canonical_hash(unhashed)}


def recovery_audit_payload(
    *,
    original_repository_state: Mapping[str, object],
    repair_repository_state: Mapping[str, object],
    source_stream_lock_hash: str,
    global_prediction_seal_hash: str,
    prelabel_feature_surface_hash: str,
    fold_plan_surface_hash: str,
) -> Mapping[str, object]:
    """Bind one allowed continuation to its failed state, seals, and checkouts."""

    _validate_reused_hashes(
        source_stream_lock_hash=source_stream_lock_hash,
        global_prediction_seal_hash=global_prediction_seal_hash,
        prelabel_feature_surface_hash=prelabel_feature_surface_hash,
        fold_plan_surface_hash=fold_plan_surface_hash,
    )
    original = _validated_repository_state(
        original_repository_state, role="original", require_clean=False
    )
    repair = _validated_repository_state(
        repair_repository_state, role="repair", require_clean=True
    )
    if repair["repository_revision"] == original["repository_revision"]:
        raise ProtocolError("Mappingproxy recovery repair revision did not change.")

    failed_state = _failed_mappingproxy_state()
    unhashed = _audit_payload(
        recovery_used=True,
        failed_run_state_hash=canonical_hash(failed_state),
        failed_phase=str(failed_state["phase"]),
        failed_error=str(failed_state["error"]),
        reused_source_stream_lock_hash=source_stream_lock_hash,
        reused_global_prediction_seal_hash=global_prediction_seal_hash,
        reused_prelabel_feature_surface_hash=prelabel_feature_surface_hash,
        reused_fold_plan_surface_hash=fold_plan_surface_hash,
        original_repository_state=original,
        repair_repository_state=repair,
    )
    return {**unhashed, "recovery_audit_hash": canonical_hash(unhashed)}


def validate_recovery_audit_payload(
    value: object,
    *,
    source_stream_lock_hash: str,
    global_prediction_seal_hash: str,
    prelabel_feature_surface_hash: str,
    fold_plan_surface_hash: str,
    original_repository_state: Mapping[str, object],
    current_repository_state: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate a fresh or recovery-used audit against the active sealed inputs."""

    if not isinstance(value, Mapping):
        raise ProtocolError("Multi-challenger recovery audit is not an object.")
    _validate_reused_hashes(
        source_stream_lock_hash=source_stream_lock_hash,
        global_prediction_seal_hash=global_prediction_seal_hash,
        prelabel_feature_surface_hash=prelabel_feature_surface_hash,
        fold_plan_surface_hash=fold_plan_surface_hash,
    )
    original = _validated_repository_state(
        original_repository_state, role="original", require_clean=False
    )
    current = _validated_repository_state(
        current_repository_state, role="current", require_clean=False
    )
    observed = dict(value)
    recovery_required = original != current
    if observed.get("recovery_used") is not recovery_required:
        raise ProtocolError(
            "Multi-challenger recovery mode disagrees with repository states."
        )
    if not recovery_required:
        if observed != fresh_recovery_audit_payload():
            raise ProtocolError("Fresh-run multi-challenger recovery audit drifted.")
        return observed

    expected = recovery_audit_payload(
        original_repository_state=original,
        repair_repository_state=current,
        source_stream_lock_hash=source_stream_lock_hash,
        global_prediction_seal_hash=global_prediction_seal_hash,
        prelabel_feature_surface_hash=prelabel_feature_surface_hash,
        fold_plan_surface_hash=fold_plan_surface_hash,
    )
    if observed != expected:
        raise ProtocolError("Mappingproxy recovery audit drifted.")
    return observed


def original_repository_state_from_provenance(
    root: Path,
) -> Mapping[str, object]:
    """Read revision A only from the immutable prepared input manifest."""

    payload = read_json(Path(root) / "provenance/input_artifacts.json")
    return _validated_repository_state(
        {
            "repository_revision": payload.get("repository_revision"),
            "repository_dirty": payload.get("repository_dirty"),
            "repository_status_hash": payload.get("repository_status_hash"),
        },
        role="original",
        require_clean=False,
    )


def sealed_recovery_input_hashes(root: Path) -> Mapping[str, str]:
    """Bind recovery to the four upstream products sealed before donor fitting."""

    path = Path(root)
    source_lock_path = path / "manifests/frozen_source_stream_lock.json"
    prediction_seal_path = path / "manifests/fixed_bank_a1_prediction_seal.json"
    feature_seal_path = path / "manifests/prelabel_feature_seal.json"
    fold_plan_path = path / "manifests/fold_plan_seals.json"
    for member in (
        source_lock_path,
        prediction_seal_path,
        feature_seal_path,
        fold_plan_path,
    ):
        if member.is_symlink() or not member.is_file():
            raise ProtocolError(
                "Mappingproxy recovery upstream seal is absent or unsafe."
            )
    feature = read_json(feature_seal_path)
    fold_plan = read_json(fold_plan_path)
    hashes = {
        "source_stream_lock_hash": sha256_file(source_lock_path),
        "global_prediction_seal_hash": sha256_file(prediction_seal_path),
        "prelabel_feature_surface_hash": str(feature.get("feature_surface_hash", "")),
        "fold_plan_surface_hash": str(fold_plan.get("fold_plan_surface_hash", "")),
    }
    _validate_reused_hashes(**hashes)
    return hashes


def current_repair_repository_state() -> Mapping[str, object]:
    """Read the revision and complete porcelain status of the active checkout."""

    try:
        repo_root = MidogppWorkspace.load().repo_root
    except (OSError, ValueError, WorkspaceError) as exc:
        raise ProtocolError("Cannot locate the mappingproxy repair checkout.") from exc
    return _read_git_state(repo_root)


def assert_repair_repository_state_unchanged(
    audit: Mapping[str, object],
) -> None:
    """Require the clean repair checkout to remain unchanged through recovery."""

    expected = _validated_repository_state(
        {
            "repository_revision": audit.get("repair_repository_revision"),
            "repository_dirty": audit.get("repair_repository_dirty"),
            "repository_status_hash": audit.get("repair_repository_status_hash"),
        },
        role="repair",
        require_clean=True,
    )
    observed = dict(current_repair_repository_state())
    if observed != expected:
        raise ProtocolError(
            "Mappingproxy recovery repair checkout changed during continuation."
        )


def _failed_mappingproxy_state() -> dict[str, object]:
    # Import lazily: recovery orchestration imports these audit helpers.
    from .recovery import FAILED_MAPPINGPROXY_STATE

    if not isinstance(FAILED_MAPPINGPROXY_STATE, Mapping):
        raise ProtocolError("Mappingproxy recovery failure state is invalid.")
    state = dict(FAILED_MAPPINGPROXY_STATE)
    if (
        state.get("status") != "FAILED"
        or not isinstance(state.get("phase"), str)
        or not state["phase"]
        or not isinstance(state.get("error"), str)
        or not state["error"]
    ):
        raise ProtocolError("Mappingproxy recovery failure state is invalid.")
    return state


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
        raise ProtocolError(
            "Cannot bind mappingproxy recovery to a Git checkout."
        ) from exc
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
    reused_source_stream_lock_hash: object,
    reused_global_prediction_seal_hash: object,
    reused_prelabel_feature_surface_hash: object,
    reused_fold_plan_surface_hash: object,
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
        "reused_source_stream_lock_hash": reused_source_stream_lock_hash,
        "reused_global_prediction_seal_hash": reused_global_prediction_seal_hash,
        "reused_prelabel_feature_surface_hash": (
            reused_prelabel_feature_surface_hash
        ),
        "reused_fold_plan_surface_hash": reused_fold_plan_surface_hash,
        "donor_models_refit_during_recovery": recovery_used,
        "source_generation_recomputed_during_recovery": False,
        "predictions_recomputed_during_recovery": False,
        "decisions_recomputed_during_recovery": recovery_used,
        "terminal_evaluation_recomputed_during_recovery": recovery_used,
        "labels_reopened_only_for_deterministic_reconstruction_during_recovery": (
            recovery_used
        ),
        "terminal_consumed_test_diagnostic_only": True,
        "policy_promotion_authorized": False,
        "original_repository_revision": original.get("repository_revision"),
        "original_repository_dirty": original.get("repository_dirty"),
        "original_repository_status_hash": original.get(
            "repository_status_hash"
        ),
        "repair_repository_revision": repair.get("repository_revision"),
        "repair_repository_dirty": repair.get("repository_dirty"),
        "repair_repository_status_hash": repair.get("repository_status_hash"),
    }


def _validate_reused_hashes(
    *,
    source_stream_lock_hash: object,
    global_prediction_seal_hash: object,
    prelabel_feature_surface_hash: object,
    fold_plan_surface_hash: object,
) -> None:
    if (
        not _is_hex(source_stream_lock_hash, length=64)
        or not _is_hex(global_prediction_seal_hash, length=64)
        or not _is_hex(prelabel_feature_surface_hash, length=64)
        or not _is_hex(fold_plan_surface_hash, length=64)
    ):
        raise ProtocolError("Mappingproxy recovery reused-seal hash is invalid.")


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
        raise ProtocolError(
            f"Mappingproxy recovery {role} repository state is invalid."
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


__all__ = (
    "assert_repair_repository_state_unchanged",
    "current_repair_repository_state",
    "fresh_recovery_audit_payload",
    "original_repository_state_from_provenance",
    "recovery_audit_payload",
    "sealed_recovery_input_hashes",
    "validate_recovery_audit_payload",
)
